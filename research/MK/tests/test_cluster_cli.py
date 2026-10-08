"""cluster.py: config validation on the real experiment.toml, submit-workers dry run, resources of a Slurm run, sbatch."""
import json
import os
import tomllib

import pytest

import cluster
import pool


def test_experiment_config_node_file_start_and_cpus_per_dataset():
    _, cfg = cluster._load_config("experiment")
    plan = cluster.build_plan(cfg)
    for dataset in cfg["grid"]["datasets"]:
        name = dataset.split("/")[-1]
        solves = [t for k, t in plan.items() if k.startswith(f"{name}/") and not k.endswith("/prepare")]
        values = {t["cmd"].split("--node-file-start ")[1].split(" ")[0] for t in solves}
        assert len(values) == 1, f"{name}: NodefileStart must be the same for all its solves ({values})"
        assert len({t["res"]["cpus"] for t in solves}) == 1
    assert "@NODEFILESTART@" not in json.dumps(plan)


def test_node_file_start_fraction_is_rejected():
    _, cfg = cluster._load_config("experiment")
    markov = {k: v for k, v in cfg["markov"].items() if k != "node_file_start"}
    with pytest.raises(ValueError, match="node_file_start_fraction was replaced"):
        cluster.build_plan({**cfg, "markov": {**markov, "node_file_start_fraction": 0.5}})


def test_submit_workers_dry_run_creates_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cluster, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("MK_MAIL_USER", "someone@example.org")
    monkeypatch.setattr("sys.argv", ["cluster.py", "submit-workers", "experiment", "--dry-run"])
    cluster.main()
    out = capsys.readouterr().out
    for expected in ("#SBATCH --exclusive", "#SBATCH --mem=0", "#SBATCH --mail-type=BEGIN,END,FAIL", "--chain-after-hours 24",
                     "--cores 192 --mem 740G", "Estimated work"):
        assert expected in out
    assert out.count("sbatch --parsable") == 3
    assert not (tmp_path / "runs").exists()


def test_resources_of_a_slurm_run_saves_measurements(mk, monkeypatch, capsys):
    """Done tasks (except no-op ones) and OOM limits become measurements.json, the seed of pool runs."""
    keys = ["R/sd1/c3/base/main/Markov", "R/sd1/c3/base/main/Truth", "R/sd1/c3/base/regret/Markov"]
    rdir = cluster.RUNS_DIR / "slurmrun"
    (rdir / "logs").mkdir(parents=True)
    state = {"tasks": {}}
    for i, key in enumerate(keys):
        log = rdir / "logs" / f"{i}.out"
        log.write_text("MK-NOOP: output existed\n" if key.endswith("main/Truth") else "solved\n")
        state["tasks"][key] = {"key": key, "deps": [], "attempts": [{"job": f"9_{i}", "log": str(log), "res": {"cpus": 8, "mem": "64G", "time": "1-00:00:00"}}]}
    (rdir / "state.json").write_text(json.dumps(state))
    info = {"9_0": {"state": "COMPLETED", "elapsed": "02:00:00", "maxrss_mb": 30000},
            "9_1": {"state": "COMPLETED", "elapsed": "00:00:05", "maxrss_mb": 300},
            "9_2": {"state": "OUT_OF_MEMORY", "elapsed": "05:00:00", "maxrss_mb": 64000}}
    monkeypatch.setattr(cluster.Slurm, "query", staticmethod(lambda jobs: info))
    assert mk.cli("resources", "slurmrun", "--save") == 0
    out = capsys.readouterr().out
    assert "needs more than 64G" in out
    saved = {m["key"]: m for m in json.loads((rdir / "measurements.json").read_text())}
    assert set(saved) == {keys[0], keys[2]}  # the no-op Truth run is not a measurement
    assert saved[keys[0]]["outcome"] == pool.DONE and saved[keys[0]]["elapsed_s"] == 7200
    assert saved[keys[2]]["outcome"] == "oom" and saved[keys[2]]["limit_mb"] == 64 * 1024


def test_sbatch_from_a_job_drops_the_job_variables(monkeypatch):
    seen = {}

    class Result:
        returncode, stdout, stderr = 0, "12345;cluster\n", ""

    monkeypatch.setattr(cluster.subprocess, "run", lambda cmd, **kw: seen.update(env=kw["env"], cmd=cmd) or Result())
    for k, v in {"SLURM_MEM_PER_NODE": "1000", "SBATCH_TIMELIMIT": "1", "SRUN_CPUS_PER_TASK": "4", "KEEP_ME": "yes"}.items():
        monkeypatch.setenv(k, v)
    assert cluster._sbatch_from_job("/x/worker.sbatch") == "12345"
    assert seen["cmd"] == ["sbatch", "--parsable", "/x/worker.sbatch"]
    assert seen["env"]["KEEP_ME"] == "yes" and not any(k.startswith(("SLURM_", "SBATCH_", "SRUN_")) for k in seen["env"])


def test_work_estimate_uses_measurements(mk, capsys):
    plan = dict([mk.task(f"W/sd1/c3/base/main/T{i}", cpus=12, mem="64G", time="3-00:00:00") for i in range(4)])
    cfg = mk.use(plan)
    store = mk.store()
    store.claim("W/sd1/c3/base/main/T0", 1, "x")
    store.write_result("W/sd1/c3/base/main/T0", 1, {"outcome": pool.DONE, "elapsed_s": 3600, "peak_rss_mb": 100 * 1024})
    settings = {"cores": 192, "mem": "740G", "mem_mb": 740 * 1024, "mem_fraction": 0.9}
    cluster.work_estimate(cfg, plan, store, settings)
    out = capsys.readouterr().out
    assert "Open tasks: 3 of 4" in out and "measured for memory 3, runtime 0" in out and "node-hours" in out


def test_pool_toml_settings_are_complete():
    with open(cluster.HERE / "experiments" / "experiment.toml", "rb") as f:
        cfg = tomllib.load(f)
    settings = cluster._pool_settings(cfg)
    assert settings["cores"] == 192 and settings["chain_hours"] < 72 and settings["workers"] == 3
    assert os.path.basename(str(cluster.HERE)) == "MK"
