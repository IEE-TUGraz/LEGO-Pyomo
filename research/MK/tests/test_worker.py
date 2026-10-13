"""pool.Worker with real subprocesses (fake tasks): packing, dependencies, failures, guards, chaining, low priority."""
import socket
import subprocess
import sys
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


def test_two_workers_share_a_pool_and_preempt_tasks_for_one_that_outgrows_its_estimate(mk, capsys):
    """Each task runs exactly once across both workers; the growing task uses the most memory, so the guard preempts the
    smaller ones next to it (retried later) instead of the growing task, which finishes in its first attempt."""
    plan = dict([mk.task(f"D/sd1/c3/base/main/E{i}", 8, mb=60, mem="120M") for i in range(6)]
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
    assert not [r for r in results if r["outcome"] == pool.EVICTED]
    preempted = [r["key"].split("/")[-1] for r in results if r["outcome"] == pool.PREEMPTED]
    assert preempted and all(k.startswith("E") for k in preempted)
    assert [r["attempt"] for r in results if r["key"].endswith("/GROW")] == [1]  # never stopped

    capsys.readouterr()
    mk.cli("status", "fake")
    out = capsys.readouterr().out
    assert "WORKERS (2)" in out and f"{len(preempted)} preempted" in out
    mk.cli("resources", "fake")
    out = capsys.readouterr().out
    assert "preempted at" in out and "Source: pool results" in out


def test_memory_guard_evicts_a_task_beyond_its_own_estimate(mk):
    """BIG stays within its estimate and uses the most memory; GROW (estimated far too low) pushes the machine over
    the limit: it is evicted, retried with a measured estimate and then waits for BIG to finish."""
    plan = dict([mk.task("D/sd1/c3/base/main/BIG", 12, mb=400, mem="480M"),
                 mk.task("D/sd1/c3/base/main/GROW", 6, mb=50, mb_end=390, mem="120M")])
    cfg = mk.use(plan)
    store = mk.store()
    w, logs = mk.worker(cfg, plan, store, mem_mb=700, evict_grace_s=1)
    w.run()
    results = store.results()
    assert sorted(r["key"].split("/")[-1] for r in results if r["outcome"] == pool.DONE) == ["BIG", "GROW"]
    assert [r["key"].split("/")[-1] for r in results if r["outcome"] == pool.EVICTED] == ["GROW"]
    assert not [r for r in results if r["outcome"] == pool.PREEMPTED]
    grow_starts = [line for line in logs if " start " in line and "GROW" in line]
    assert "(prior)" in grow_starts[0] and "(prior)" not in grow_starts[-1]  # retried with a measured estimate
    ev = mk.events(logs)
    assert ev.index(("done", "BIG")) < ev.index(("done", "GROW"))  # the retry did not fit next to BIG


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
    plan = dict([mk.task("N/sd1/c3/base/main/A", 0.3, noop=True, mem="3G")])  # a prior far above the measurement, below the RAM
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


def test_stop_request_drains_or_interrupts_only_workers_started_before_it(mk):
    plan = dict([mk.task("S/sd1/c3/base/main/A", 2), mk.task("S/sd1/c3/base/main/B", 2, cpus=8)])
    cfg = mk.use(plan)
    store = mk.store()
    pool._write_json_atomic(store.root / "meta.json", {"config": "x"})
    w, logs = mk.worker(cfg, plan, store, idle_s=30)
    threading.Timer(0.8, lambda: mk.cli("stop", "fake", "--host", w.host)).start()  # drain: A finishes, B never starts
    w.run()
    snap = store.snapshot(plan)
    assert snap["S/sd1/c3/base/main/A"][0] == pool.DONE and snap["S/sd1/c3/base/main/B"][0] == pool.NEW
    assert any("stopped on request" in line for line in logs)

    w, logs = mk.worker(cfg, plan, store, idle_s=30)  # started after the request: not affected; --now interrupts B
    threading.Timer(1.0, lambda: mk.cli("stop", "fake", "--now")).start()
    w.run()
    snap = store.snapshot(plan)
    assert snap["S/sd1/c3/base/main/B"][0] == pool.RETRY and snap["S/sd1/c3/base/main/B"][2]["outcome"] == pool.INTERRUPTED


def test_workers_of_this_host_whose_process_is_gone_are_retried_at_once(mk):
    plan = dict([mk.task("R/sd1/c3/base/main/A", 0.3)])
    cfg = mk.use(plan)
    store = mk.store()
    w, logs = mk.worker(cfg, plan, store)
    store.claim("R/sd1/c3/base/main/A", 1, "crashed")
    pool._write_json_atomic(store.root / "workers" / "crashed.json", {"host": w.host, "pid": 4_000_000, "beat": time.time()})
    w.run()
    assert store.snapshot(plan)["R/sd1/c3/base/main/A"][:2] == (pool.DONE, 2)
    assert store.workers()["crashed"]["exit_reason"] == "process gone"


def test_tasks_needing_more_cpus_than_cores_are_never_started(mk):
    plan = dict([mk.task("T/sd1/c3/base/main/BIG", 0.3, cpus=16), mk.task("T/sd1/c3/base/main/SMALL", 0.3, cpus=2)])
    cfg = mk.use(plan)
    store = mk.store()
    w, logs = mk.worker(cfg, plan, store)  # 8 cores
    w.run()
    snap = store.snapshot(plan)
    assert snap["T/sd1/c3/base/main/SMALL"][0] == pool.DONE and snap["T/sd1/c3/base/main/BIG"][0] == pool.NEW
    assert sum("skipping T/sd1/c3/base/main/BIG" in line for line in logs) == 1


def test_tasks_estimated_above_the_ram_are_never_started(mk):
    # other datasets: SMALL's measurement would otherwise lower HUGE's estimate (shared levels) and let it start
    plan = dict([mk.task("H/sd1/c3/base/main/HUGE", 0.3, mem="2G"), mk.task("S/sd1/c3/base/main/SMALL", 0.3)])
    cfg = mk.use(plan)
    store = mk.store()
    w, logs = mk.worker(cfg, plan, store, mem_mb=1000)
    w.run()
    snap = store.snapshot(plan)
    assert snap["S/sd1/c3/base/main/SMALL"][0] == pool.DONE and snap["H/sd1/c3/base/main/HUGE"][0] == pool.NEW
    assert sum("skipping H/sd1/c3/base/main/HUGE" in line for line in logs) == 1


def test_small_tasks_start_opportunistically_next_to_a_large_task_and_finish(mk):
    """BIG reserves 700 of 900 MB usable but uses ~100 MB: the small tasks (300 MB each) do not fit its reservation but
    start opportunistically next to its actual use and finish before it."""
    # One dataset per task: a finished task's measurement would otherwise lower the others' estimates (shared levels)
    plan = dict([mk.task("B/sd1/c3/base/main/BIG", 10, mb=100, mem="700M")]
                + [mk.task(f"O{i}/sd1/c3/base/regret/S{i}", 1, mem="300M") for i in range(3)])
    cfg = mk.use(plan)
    store = mk.store()
    w, logs = mk.worker(cfg, plan, store, mem_mb=1000, opportunistic_after_s=0)
    w.run()
    results = store.results()
    assert all(r["outcome"] == pool.DONE for r in results) and len(results) == len(plan)
    assert {r["key"].split("/")[-1] for r in results if r["opportunistic"]} == {"S0", "S1", "S2"}
    ev = mk.events(logs)
    assert all(ev.index(("done", f"S{i}")) < ev.index(("done", "BIG")) for i in range(3))


def test_opportunistic_tasks_are_preempted_when_the_large_task_grows(mk):
    """BIG grows towards its estimate: the opportunistic tasks next to it are preempted (not BIG), retried after it."""
    plan = dict([mk.task("B/sd1/c3/base/main/BIG", 6, mb=50, mb_end=700, mem="800M")]
                + [mk.task(f"O/sd1/c3/base/regret/S{i}", 10, mem="200M") for i in range(3)])
    cfg = mk.use(plan)
    store = mk.store()
    w, logs = mk.worker(cfg, plan, store, mem_mb=1000, opportunistic_after_s=0)
    w.run()
    results = store.results()
    preempted = [r for r in results if r["outcome"] == pool.PREEMPTED]
    assert preempted and all(r["opportunistic"] and "/S" in r["key"] for r in preempted)
    assert [r["attempt"] for r in results if r["key"].endswith("/BIG")] == [1]
    assert all(s[0] == pool.DONE for s in store.snapshot(plan).values())


def test_opportunistic_starts_off(mk):
    plan = dict([mk.task("B/sd1/c3/base/main/BIG", 3, mb=100, mem="700M"), mk.task("O/sd1/c3/base/regret/S0", 1, mem="300M")])
    cfg = mk.use(plan, opportunistic=False)
    w, logs = mk.worker(cfg, plan, mk.store(), mem_mb=1000, opportunistic_after_s=0)
    w.run()
    ev = mk.events(logs)
    assert ev.index(("done", "BIG")) < ev.index(("start", "S0"))


def test_tasks_of_another_worker_on_this_machine_count_as_occupied(mk):
    """A (draining) worker of another process on this host runs a task estimated at 800 MB on 6 cores: A (300 MB) waits
    until that process is gone instead of being packed as if the machine were empty - and the waiting worker does not
    exit as idle meanwhile (idle_s is shorter than the other worker's task)."""
    plan = dict([mk.task("X/sd1/c3/base/main/A", 0.5, mem="300M")])
    cfg = mk.use(plan)
    store = mk.store()
    other = subprocess.Popen([sys.executable, str(mk.fake), "3", "0", "50"])
    store.beat("other-worker", {"host": socket.gethostname(), "pid": other.pid, "slurm_job": None,
                                "tasks": {"Y/sd1/c3/base/main/F": {"pid": other.pid, "cpus": 6, "est_mb": 800, "peak_mb": 0,
                                                                   "rss_mb": 0, "start": time.time(), "opp": False}}})
    w, logs = mk.worker(cfg, plan, store, mem_mb=1000, idle_s=1)
    w.run()
    other.wait()
    counted = [i for i, line in enumerate(logs) if "other workers on this machine run 1 task(s)" in line]
    started = [i for i, line in enumerate(logs) if " start " in line and "/A " in line]
    assert counted and started and counted[0] < started[0]
    assert any("other workers on this machine run 0 task(s)" in line for line in logs[:started[0]])


def test_handover_drains_the_worker(mk):
    """After the handover callback reports a successor, the worker starts nothing new and exits after its own tasks."""
    plan = dict([mk.task("H/sd1/c3/base/main/T1", 3), mk.task("H/sd1/c3/base/main/T2", 3), mk.task("H/sd1/c3/base/main/T3", 0.3)])
    cfg = mk.use(plan)
    store = mk.store()
    calls = []

    def handover():
        calls.append(1)
        return "pid 12345, code abc" if len(calls) == 2 else None

    w, logs = mk.worker(cfg, plan, store, cores=4, handover=handover, reload_s=0.2)
    w.run()
    snap = store.snapshot(plan)
    assert len(calls) == 2  # never asked again once draining
    assert snap["H/sd1/c3/base/main/T1"][0] == snap["H/sd1/c3/base/main/T2"][0] == pool.DONE
    assert snap["H/sd1/c3/base/main/T3"][0] == pool.NEW  # left to the successor
    assert any("handed over to a new worker (pid 12345" in line for line in logs)
    assert any("handed over: own tasks finished" in line for line in logs)
    assert store.workers()[w.id].get("exited")
