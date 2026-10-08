"""Fixtures for the cluster.py / pool.py tests: fake tasks (sleep, allocate memory, exit code, optional MK-NOOP marker)
and fake plans, run in a temporary runs/ folder. Run: pytest research/MK/tests"""
import sys
from pathlib import Path

import pytest

MK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MK))
import cluster  # noqa: E402
import pool  # noqa: E402

# fake_task.py <seconds> <exit code> <MB> [<MB at the end>] [--noop]: grows to the memory in 10 steps (touching every page,
# so it shows up in the RSS), prints pool.NOOP_MARKER with --noop
FAKE_TASK = '''import sys, time
args = [a for a in sys.argv[1:] if a != "--noop"]
secs, rc, mb = float(args[0]), int(args[1]), float(args[2])
mb_end = float(args[3]) if len(args) > 3 else mb
blocks, steps = [], 10
for i in range(steps):
    target = mb + (mb_end - mb) * (i + 1) / steps
    have = sum(len(b) for b in blocks) / 2 ** 20
    if target > have:
        b = bytearray(int((target - have) * 2 ** 20)); b[::4096] = b"\\x01" * len(b[::4096]); blocks.append(b)
    time.sleep(secs / steps)
if "--noop" in sys.argv:
    print("MK-NOOP: fake task had nothing to do", flush=True)
sys.exit(rc)
'''


@pytest.fixture(scope="session")
def fake_task(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("fake") / "fake_task.py"
    path.write_text(FAKE_TASK)
    return path


class MK:
    """Helpers bound to one test: temporary runs/ folder, fake plans, workers with captured logs."""

    def __init__(self, tmp_path: Path, monkeypatch, fake: Path):
        self.tmp, self.monkeypatch, self.fake = tmp_path, monkeypatch, fake
        monkeypatch.setattr(cluster, "RUNS_DIR", tmp_path / "runs")

    def task(self, key, secs=0.5, rc=0, mb=20, mb_end=None, deps=(), cpus=2, mem="100M", time="00:10:00", noop=False, rank=0):
        cmd = f"python {self.fake} {secs} {rc} {mb}" + (f" {mb_end}" if mb_end else "") + (" --noop" if noop else "")
        return key, {"key": key, "cmd": cmd, "res": {"cpus": cpus, "mem": mem, "time": time}, "deps": list(deps), "rank": rank}

    def use(self, plan: dict, name: str = "fake", **pool_cfg) -> dict:
        """Make cluster.build_plan / _load_config return this plan and a minimal config."""
        cfg = {"name": name, "pool": pool_cfg}
        self.monkeypatch.setattr(cluster, "build_plan", lambda cfg_: plan)
        self.monkeypatch.setattr(cluster, "_load_config", lambda _: (self.tmp / f"{name}.toml", cfg))
        return cfg

    def store(self, name: str = "fake") -> "pool.Store":
        return pool.Store(cluster.RUNS_DIR / name / "pool")

    def worker(self, cfg, plan, store, low=(), **kw):
        """A pool.Worker over the whole plan (tasks in `low` are low priority); returns (worker, log lines)."""
        specs = cluster.pool_specs(plan, list(plan))
        for key in low:
            specs[key]["low"] = True
        logs = []
        kw = {"cores": 8, "mem_mb": 4000, "idle_s": 1, "poll_s": 0.3, **kw}
        return cluster.make_worker(cfg, specs, store, out=logs.append, **kw), logs

    @staticmethod
    def events(logs) -> list[tuple[str, str]]:
        """(event, last key part) for start / done / failed / evicted / interrupted log lines."""
        out = []
        for line in logs:
            parts = line.split(" ")
            if len(parts) > 2 and parts[1] in ("start", "done", "failed", "evicted", "interrupted") and not parts[2].endswith(":"):
                out.append((parts[1], parts[2].split("/")[-1]))
        return out

    @staticmethod
    def cli(*argv) -> int:
        sys_argv = sys.argv
        sys.argv = ["cluster.py", *argv]
        try:
            cluster.main()
            return 0
        except SystemExit as e:
            return e.code or 0
        finally:
            sys.argv = sys_argv


@pytest.fixture
def mk(tmp_path, monkeypatch, fake_task) -> MK:
    return MK(tmp_path, monkeypatch, fake_task)
