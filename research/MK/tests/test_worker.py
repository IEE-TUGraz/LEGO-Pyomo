"""pool.Worker with real subprocesses (fake tasks): packing, dependencies, failures, guards, chaining, low priority."""
import threading
import time

import cluster
import pool


def test_local_runs_in_parallel_respects_deps_and_skips_after_failure(mk):
    plan = dict([mk.task("A/prepare", 1.5), mk.task("B/prepare", 1.5), mk.task("C/prepare", 0.5, rc=1),
                 mk.task("A/main", 0.5, deps=["A/prepare"]), mk.task("C/main", 0.5, deps=["C/prepare"]),
                 mk.task("C/regret", 0.5, deps=["C/main"]), mk.task("B/main", 0.5, deps=["B/prepare"])])
    mk.use(plan)
    t0 = time.time()
    assert mk.cli("local", "x", "*", "-j", "3", "--keep-going") == 1
    assert time.time() - t0 < 10  # the three prepares ran in parallel
    logs = sorted(p.name for p in (cluster.RUNS_DIR / "fake-local" / "logs").iterdir())
    assert "C__prepare.1.log" in logs and "C__main.1.log" not in logs and "C__regret.1.log" not in logs
    assert {"A__main.1.log", "B__main.1.log"} <= set(logs)


def test_two_workers_share_a_pool_and_evict_a_task_that_outgrows_its_estimate(mk, capsys):
    """Each task runs exactly once across both workers; the growing task is evicted, retried with a measured estimate
    and then runs alone."""
    plan = dict([mk.task(f"D/sd1/c3/base/main/E{i}", 3, mb=60, mem="120M") for i in range(6)]
                + [mk.task("D/sd1/c3/base/regret/GROW", 4, mb=50, mb_end=700, mem="120M")])
    cfg = mk.use(plan)
    store = mk.store()
    pool._write_json_atomic(store.root / "meta.json", {"config": "x"})
    workers = [mk.worker(cfg, plan, store, mem_mb=600, evict_grace_s=1) for _ in range(2)]
    for i, (w, _) in enumerate(workers):
        w.id += f"-w{i}"
    threads = [threading.Thread(target=w.run) for w, _ in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    results = store.results()
    done = [r["key"] for r in results if r["outcome"] == pool.DONE]
    assert sorted(done) == sorted(plan)  # every task done exactly once
    evicted = [r for r in results if r["outcome"] == pool.EVICTED]
    assert [r["key"].split("/")[-1] for r in evicted] == ["GROW"]
    grow_starts = {int(line.split("(attempt ")[1].split(",")[0]): line  # either worker may have started an attempt
                   for _, logs in workers for line in logs if " start " in line and "GROW" in line}
    assert "(prior)" in grow_starts[1] and "(prior)" not in grow_starts[max(grow_starts)]  # retried with a measured estimate

    capsys.readouterr()
    mk.cli("status", "fake")
    out = capsys.readouterr().out
    assert "WORKERS (2)" in out and "1 evicted" in out
    mk.cli("resources", "fake")
    out = capsys.readouterr().out
    assert "evicted at" in out and "Source: pool results" in out


def test_stale_claim_failure_restart_walltime_interrupt(mk):
    plan = dict([mk.task("F/sd1/c3/base/main/OK", 0.5), mk.task("F/sd1/c3/base/main/BAD", 0.5, rc=3),
                 mk.task("F/sd1/c3/base/regret/AFTERBAD", 0.5, deps=["F/sd1/c3/base/main/BAD"]), mk.task("F/sd1/c3/base/main/LONG", 6)])
    cfg = mk.use(plan)
    store = mk.store()
    pool._write_json_atomic(store.root / "meta.json", {"config": "x"})
    store.claim("F/sd1/c3/base/main/OK", 1, "deadworker")  # a dead worker holds attempt 1
    pool._write_json_atomic(store.root / "workers" / "deadworker.json", {"beat": time.time() - 2 * pool.STALE_S})
    w, _ = mk.worker(cfg, plan, store, end_time=time.time() + 4, interrupt_margin_s=1)
    w.run()
    snap = store.snapshot(plan)
    assert snap["F/sd1/c3/base/main/OK"][:2] == (pool.DONE, 2)
    assert snap["F/sd1/c3/base/main/BAD"][0] == pool.FAILED
    assert snap["F/sd1/c3/base/main/LONG"][0] == pool.RETRY and snap["F/sd1/c3/base/main/LONG"][2]["outcome"] == pool.INTERRUPTED
    assert cluster.pool_status(plan, store)["F/sd1/c3/base/regret/AFTERBAD"][0] == cluster.BLOCKED

    assert mk.cli("restart", "fake", "--failed") == 0
    assert cluster.pool_status(plan, store)["F/sd1/c3/base/main/BAD"][0] == cluster.PENDING
    w, logs = mk.worker(cfg, plan, store)
    w.run()
    snap = store.snapshot(plan)
    assert snap["F/sd1/c3/base/main/BAD"][1] == 2 and snap["F/sd1/c3/base/regret/AFTERBAD"][0] == pool.NEW  # fails again
    assert snap["F/sd1/c3/base/main/LONG"][0] == pool.DONE
    assert any("nothing left to run" in line for line in logs)


def test_idle_worker_waits_for_a_task_another_worker_runs(mk):
    plan = dict([mk.task("G/sd1/c3/base/main/X"), mk.task("G/sd1/c3/base/regret/Y", deps=["G/sd1/c3/base/main/X"])])
    cfg = mk.use(plan)
    store = mk.store()
    store.claim("G/sd1/c3/base/main/X", 1, "other")
    pool._write_json_atomic(store.root / "workers" / "other.json", {"beat": time.time() + 3600})  # alive
    w, logs = mk.worker(cfg, plan, store, idle_s=2)
    t0 = time.time()
    w.run()
    assert 2 <= time.time() - t0 < 8 and any("idle for" in line for line in logs)


def test_noop_runs_are_not_measurements(mk):
    plan = dict([mk.task("N/sd1/c3/base/main/A", 0.3, noop=True, mem="5G")])
    cfg = mk.use(plan)
    store = mk.store()
    w, logs = mk.worker(cfg, plan, store)
    w.run()
    (record,) = store.results()
    assert record["outcome"] == pool.DONE and record["noop"] is True
    assert w.est.estimate("N/sd1/c3/base/main/A")["mem_src"] == "prior"
    assert any("not used as a measurement" in line for line in logs)


def test_walltime_check_uses_measured_runtimes_only(mk):
    plan = dict([mk.task("W/sd1/c3/base/main/SLOW"), mk.task("W/sd1/c3/base/main/NEW")])
    cfg = mk.use(plan)
    store = mk.store()
    store.claim("W/sd1/c3/base/main/SLOW", 1, "gone")
    store.write_result("W/sd1/c3/base/main/SLOW", 1, {"outcome": pool.INTERRUPTED, "elapsed_s": 7200, "peak_rss_mb": 10})
    w, _ = mk.worker(cfg, plan, store, end_time=time.time() + 3600, interrupt_margin_s=60)
    w.run()
    snap = store.snapshot(plan)
    assert snap["W/sd1/c3/base/main/SLOW"][0] == pool.RETRY  # measured >= 2.5 h x 1.5 does not fit into 1 h
    assert snap["W/sd1/c3/base/main/NEW"][0] == pool.DONE  # TOML prior (10 min) never blocks


def test_large_task_is_not_starved_by_smaller_ones(mk):
    small = dict(mem="300M")
    plan = dict([mk.task("H/sd1/c3/base/main/S0", 1.5, **small), mk.task("H/sd1/c3/base/main/S1", 1.5, **small),
                 mk.task("H/sd1/c3/base/main/BIG", 0.5, mem="900M")]
                + [mk.task(f"H/sd1/c3/base/regret/S{i}", 1.5, **small) for i in range(2, 6)])
    cfg = mk.use(plan)
    store = mk.store()
    w, logs = mk.worker(cfg, plan, store, mem_mb=1000, mem_fraction=1.0, reserve_after_s=0.5)
    w.specs["H/sd1/c3/base/main/BIG"]["rank"] = 5  # first round: S0 and S1 start
    w.refresh_estimates()
    w.admit(store.snapshot(w.specs))
    w.specs["H/sd1/c3/base/main/BIG"]["rank"] = -1  # now BIG is the top task but does not fit
    w.run()
    starts = [k for e, k in mk.events(logs) if e == "start"]
    assert starts.index("BIG") < starts.index("S5")


def test_low_disk_space_pauses_new_tasks(mk):
    plan = dict([mk.task("K/sd1/c3/base/main/A")])
    cfg = mk.use(plan)
    store = mk.store()
    w, logs = mk.worker(cfg, plan, store, disk_min_free_mb=10 ** 12)
    w.run()
    assert any("free on the node-file disk" in line for line in logs)
    assert store.snapshot(plan)["K/sd1/c3/base/main/A"][0] == pool.NEW


def test_chaining_retries_a_failed_submission_and_chains_once(mk):
    plan = dict([mk.task("C/sd1/c3/base/main/A", 2), mk.task("C/sd1/c3/base/main/B", 2)])
    cfg = mk.use(plan)
    calls = []

    def chain():
        calls.append(1)
        return None if len(calls) == 1 else f"JOB{len(calls)}"

    w, _ = mk.worker(cfg, plan, mk.store(), chain_after_s=0.3, chain=chain, chain_retry_s=0.3)
    w.run()
    assert len(calls) == 2 and w.chained == "JOB2"


def test_no_successor_when_nothing_remains(mk):
    plan = dict([mk.task("C/sd1/c3/base/main/QUICK", 0.3)])
    cfg = mk.use(plan)
    calls = []
    w, _ = mk.worker(cfg, plan, mk.store(), chain_after_s=2, chain=lambda: calls.append(1) or "J")
    w.run()
    assert not calls


def test_evaluate_runs_last_even_after_failures(mk):
    plan = dict([mk.task("E/sd1/c3/base/main/OK"), mk.task("E/sd1/c3/base/main/BAD", rc=2),
                 mk.task("E/sd1/c3/base/regret/AFTERBAD", deps=["E/sd1/c3/base/main/BAD"]), mk.task("evaluate")])
    cfg = mk.use(plan)
    store = mk.store()
    w, logs = mk.worker(cfg, plan, store)
    w.run()
    starts = [k for e, k in mk.events(logs) if e == "start"]
    assert starts[-1] == "evaluate" and store.snapshot(plan)["evaluate"][0] == pool.DONE


def test_low_priority_tasks_only_fill_leftover_cores(mk):
    plan = dict([mk.task("A/sd1/c3/base/main/N1", 1.5), mk.task("A/sd1/c3/base/main/N2", 1.5)]
                + [mk.task(f"A/sd1/c3/perturbTM1/main/L{i}", 1.5) for i in range(1, 4)])
    cfg = mk.use(plan, rank_offset_hours=12)
    w, logs = mk.worker(cfg, plan, mk.store(), low=[k for k in plan if "perturb" in k])
    w.run()
    first = [k for e, k in mk.events(logs) if e == "start"][:4]
    assert first[:2] == ["N1", "N2"] and set(first[2:]) <= {"L1", "L2", "L3"}


def test_running_low_priority_tasks_are_never_stopped_for_normal_ones(mk):
    plan = dict([mk.task("B/sd1/c3/base/main/P", 0.5), mk.task("B/sd1/c3/base/regret/N", 0.5, cpus=4, deps=["B/sd1/c3/base/main/P"]),
                 mk.task("B/sd1/c3/perturbTM1/main/L1", 2), mk.task("B/sd1/c3/perturbTM1/main/L2", 3),
                 mk.task("B/sd1/c3/perturbTM1/main/L3", 4)])
    cfg = mk.use(plan, rank_offset_hours=12)
    store = mk.store()
    w, logs = mk.worker(cfg, plan, store, low=[k for k in plan if "perturb" in k])
    w.run()
    assert all(r["outcome"] == pool.DONE for r in store.results())  # nothing stopped
    ev = mk.events(logs)
    assert ev.index(("done", "L1")) < ev.index(("start", "N"))  # N waited for free cores


def test_memory_guard_evicts_low_priority_tasks_first(mk):
    plan = dict([mk.task("D/sd1/c3/perturbTM1/main/L", 4, mb=150, mb_end=450), mk.task("D/sd1/c3/base/main/P", 0.3),
                 mk.task("D/sd1/c3/base/regret/N", 4, mb=250, deps=["D/sd1/c3/base/main/P"])])
    cfg = mk.use(plan, rank_offset_hours=12)
    store = mk.store()
    w, _ = mk.worker(cfg, plan, store, low=["D/sd1/c3/perturbTM1/main/L"], mem_mb=600, evict_grace_s=1)
    w.run()
    evicted = {r["key"].split("/")[-1] for r in store.results() if r["outcome"] == pool.EVICTED}
    assert evicted == {"L"}  # N was started later, but L is low priority
    assert all(s[0] == pool.DONE for s in store.snapshot(plan).values())
