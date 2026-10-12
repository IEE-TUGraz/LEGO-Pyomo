"""Fast unit tests without subprocesses: store, estimator, priority order, estimate levels."""
import time

import pytest

import cluster
import pool


def test_claim_is_exclusive_and_states_follow_highest_attempt(tmp_path):
    store = pool.Store(tmp_path)
    keys = ["a", "b", "c", "d"]
    assert store.claim("a", 1, "w1") and not store.claim("a", 1, "w2")
    store.beat("w1", {})
    store.write_result("b", 1, {"outcome": pool.DONE})
    store.claim("b", 1, "w1")
    store.write_result("c", 1, {"outcome": pool.FAILED})
    store.claim("c", 1, "w1")
    store.claim("d", 1, "dead")
    pool._write_json_atomic(store.root / "workers" / "dead.json", {"beat": time.time() - 2 * pool.STALE_S})
    snap = store.snapshot(keys)
    assert snap["a"][:2] == (pool.RUNNING, 1)
    assert snap["b"][0] == pool.DONE and snap["c"][0] == pool.FAILED
    assert snap["d"][0] == pool.RETRY  # claim of a worker without heartbeat
    store.reset("c", 1)
    assert store.snapshot(keys)["c"][0] == pool.RETRY


def _estimator(keys, **kw):
    specs = {k: {"prior_mem_mb": 1000, "prior_time_s": 3600} for k in keys}
    return pool.Estimator(specs, cluster.estimate_levels, time_level=lambda level: "/class:" not in level, **kw)


def test_estimator_levels_lower_bounds_and_noop():
    a, b, r = "T/sd1/c28/base/main/Markov", "T/sd0.5/c21/shiftTM1/main/Markov", "T/sd1/c28/base/regret/Markov"
    est = _estimator([a, b, r], mem_safety=1.2, time_safety=1.5)
    est.add({"key": a, "outcome": pool.DONE, "peak_rss_mb": 50000, "elapsed_s": 7200, "source": "slurm:pilot", "job": "1"})
    est.add({"key": r, "outcome": "oom", "peak_rss_mb": 99000, "limit_mb": 100000, "limit_s": 86400, "source": "slurm:pilot", "job": "2"})
    est.add({"key": b, "outcome": pool.DONE, "peak_rss_mb": 1, "elapsed_s": 1, "noop": True, "attempt": 1})
    e_a, e_b, e_r = est.estimate(a), est.estimate(b), est.estimate(r)
    assert e_a["mem_mb"] == 60000 and e_a["mem_src"] == a and e_a["time_s"] == 10800
    assert e_b["mem_src"] == "T/main/Markov" and e_b["mem_mb"] == 60000  # other demand / TM / clusters; the no-op is ignored
    assert e_r["mem_mb"] == 100000 * 1.25 * 1.2  # OOM: the limit (not the slightly lower peak) is the lower bound
    assert e_r["time_src"] == "prior"  # runtime never falls back to the class level


def test_estimate_levels_and_resource_class():
    assert cluster.estimate_levels("TX-123BT/sd0.5/c21/shiftTM1/regret/Markov") == [
        "TX-123BT/sd0.5/c21/shiftTM1/regret/Markov", "TX-123BT/regret/Markov/c21", "TX-123BT/regret/Markov", "TX-123BT/class:full"]
    assert cluster.estimate_levels("TX-123BT/sd1/prepare") == ["TX-123BT/sd1/prepare", "TX-123BT/prepare"]
    assert cluster.estimate_levels("evaluate") == ["evaluate"]
    assert cluster.resource_class("X/sd1/c3/base/main/Markov") == "rp"
    assert cluster.resource_class("X/sd1/c3/base/main/Truth") == "full"
    assert cluster.resource_class("X/sd1/c3/base/operational/NoEnf") == "rp"
    assert cluster.resource_class("X/sd1/c3/base/invest-regret/NoEnf") == "full"


def test_soft_priority_order(tmp_path):
    """Critical path minus 12 h per dataset position; low-priority tasks always last."""
    specs = {"A": {"rank": 0, "dataset_rank": 0, "low": False}, "B": {"rank": 1, "dataset_rank": 1, "low": False},
             "C": {"rank": 1, "dataset_rank": 1, "low": False}, "L": {"rank": 3, "dataset_rank": 0, "low": True}}
    for spec in specs.values():
        spec.update(deps=[], cpus=1, prior_mem_mb=1, prior_time_s=1)
    w = pool.Worker(specs, pool.Store(tmp_path), pool.Estimator(specs, lambda k: [k]), cwd=tmp_path, cores=1, mem_mb=1,
                    rank_offset_s=12 * 3600, out=lambda m: None)
    w.critical = {"A": 30 * 3600, "B": 50 * 3600, "C": 35 * 3600, "L": 999 * 3600}
    w.estimates = {k: {"mem_mb": 1} for k in specs}
    assert sorted(specs, key=w.priority) == ["B", "A", "C", "L"]
    w.rank_offset_s = None  # strict: rank first
    assert sorted(specs, key=w.priority) == ["A", "B", "C", "L"]


def test_lower_bounds_never_lower_the_prior():
    """A task that crashes after seconds (failed, tiny peak) keeps its prior; a completed run may go below it."""
    f, d = "T/sd1/c28/base/main/Markov", "U/sd1/c28/base/main/Markov"
    est = _estimator([f, d])
    est.add({"key": f, "outcome": pool.FAILED, "peak_rss_mb": 200, "elapsed_s": 3, "attempt": 1})
    est.add({"key": d, "outcome": pool.DONE, "peak_rss_mb": 200, "elapsed_s": 3, "attempt": 1})
    assert est.estimate(f)["mem_mb"] == 1000 and est.estimate(f)["mem_src"] == "prior"
    assert est.estimate(d)["mem_mb"] == 200 * 1.2 and est.estimate(d)["mem_src"] == d


def test_own_completed_runs_take_precedence_and_preemptions_are_memory_minima():
    """Own completed runs lower what a seed run's lower bound (or another task's) says; a seed's completed run is raised
    by own lower bounds; a preemption is an unscaled memory minimum without runtime."""
    t1, t2, t3 = "R/sd1/c28/base/main/Truth", "R/sd0.5/c28/base/main/Truth", "R/sd0.7/c28/base/main/Truth"
    est = _estimator([t1, t2, t3], mem_safety=1.0)
    est.add({"key": t1, "outcome": pool.EVICTED, "peak_rss_mb": 8000, "source": "pool:old", "attempt": 1})
    assert est.estimate(t3)["mem_mb"] == 8000 * 1.25  # only the seed's lower bound
    est.add({"key": t1, "outcome": pool.DONE, "peak_rss_mb": 6000, "elapsed_s": 100, "attempt": 1})
    est.add({"key": t2, "outcome": pool.EVICTED, "peak_rss_mb": 9000, "attempt": 1})
    assert est.estimate(t3)["mem_mb"] == 6000  # own completed run wins over the seed's and t2's lower bounds
    assert est.estimate(t2)["mem_mb"] == 9000 * 1.25  # t2 keeps its own lower bound

    seeded = _estimator([t1, t2, t3], mem_safety=1.0)
    seeded.add({"key": t1, "outcome": pool.DONE, "peak_rss_mb": 5000, "elapsed_s": 100, "source": "pool:old", "attempt": 1})
    seeded.add({"key": t2, "outcome": pool.PREEMPTED, "peak_rss_mb": 7000, "elapsed_s": 5, "attempt": 1})
    assert seeded.estimate(t3)["mem_mb"] == 7000  # seed's completed run raised by the own preemption's peak (unscaled)
    assert seeded.estimate(t2)["time_s"] == 100 * 1.5  # a preemption says nothing about the runtime: the sibling's


def test_weighted_priority_order(tmp_path):
    """Critical path x dataset weight: a 10x lower weighted task goes first only with a more than 10x longer chain;
    low-priority tasks always last; [pool] dataset_weights is validated and excludes rank_offset_hours."""
    specs = {k: {"rank": 0, "dataset_rank": 0, "low": k.endswith("L"), "deps": [], "cpus": 1, "prior_mem_mb": 1, "prior_time_s": 1}
             for k in ("R/a", "T/b", "T/c", "N/d", "R/L")}
    weights = {"R/a": 100, "T/b": 10, "T/c": 10, "N/d": 1, "R/L": 100}
    w = pool.Worker(specs, pool.Store(tmp_path), pool.Estimator(specs, lambda k: [k]), cwd=tmp_path, cores=1, mem_mb=1,
                    weights=weights, rank_offset_s=12 * 3600, out=lambda m: None)
    w.critical = {"R/a": 10, "T/b": 101, "T/c": 99, "N/d": 980, "R/L": 999}
    w.estimates = {k: {"mem_mb": 1} for k in specs}
    assert sorted(specs, key=w.priority) == ["T/b", "R/a", "T/c", "N/d", "R/L"]  # weights win over rank_offset_s

    cfg = {"name": "x", "grid": {"datasets": ["data/R", "data/T"]}, "pool": {"dataset_weights": {"R": 10}}}
    worker = cluster.make_worker(cfg, specs, pool.Store(tmp_path / "p"), cores=1, mem_mb=1, node_base=tmp_path)
    assert worker.weights == {"R/a": 10, "T/b": 1, "T/c": 1, "N/d": 1, "R/L": 10}  # unlisted datasets: 1
    for bad in ({"dataset_weights": {"X": 2}}, {"dataset_weights": {"R": 0}}, {"dataset_weights": {"R": 2}, "rank_offset_hours": 12}):
        with pytest.raises(SystemExit):
            cluster.make_worker({**cfg, "pool": bad}, specs, pool.Store(tmp_path / "p"), cores=1, mem_mb=1, node_base=tmp_path)
