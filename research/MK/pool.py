"""Task pool on a shared filesystem and the worker that runs it (see README.md, section "Cluster runs", and CLAUDE.md).

Workers (one per node, or one on a laptop) pull tasks from a pool in runs/<name>/pool/ and pack them onto their machine by
estimated memory and cores. Estimates start from the TOML resources and improve with every finished, evicted or
interrupted attempt of this and the seed runs (this pool's completed runs take precedence over the seed runs'). Standard library only; independent of cluster.py (which builds the specs).

Store layout (all files immutable once written, except the worker heartbeats):
    claims/<task>.<n>     attempt n of a task, created with O_EXCL (exactly one worker wins); content: worker id
    results/<task>.<n>.json   outcome of attempt n (done / failed / oom / evicted / preempted / interrupted) + measurements
    resets/<task>.<n>     attempt n failed (or: an evaluate that ran too early is done), but may be retried (cluster.py restart)
    workers/<id>.json     heartbeat (rewritten every HEARTBEAT_S)
    jobs/<slurm id>.json  submitted worker jobs (submit-workers, chained successors)
    stop/<target>.json    stop request (cluster.py stop) for the workers of a host / one worker / all that started before it
    logs/<task>.<n>.log   output of attempt n
A task's state follows from its highest attempt: no claim -> new; result done/failed -> done/failed (+ reset ->
retry); oom/evicted/preempted/interrupted -> retry; claim without result -> running while its worker's heartbeat is fresh, else retry.
"""
import ctypes
import datetime
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

NOOP_MARKER = "MK-NOOP:"  # printed by Markov.py when a task had nothing to do (output existed): not a measurement
HEARTBEAT_S = 30
STALE_S = 600  # a worker without heartbeat for this long is dead; its claims are retried
# The last live workers on other hosts going silent together (last beats within STALE_VIEW_SPREAD_S of each other) means
# this machine more likely reads an outdated copy of the store (seen on Windows after standby) than that they all died:
# their claims are never retried from here (Store.stale_view) - if they really died, that needs manual intervention
STALE_VIEW_SPREAD_S = 3 * HEARTBEAT_S
DONE, FAILED, EVICTED, INTERRUPTED = "done", "failed", "evicted", "interrupted"
PREEMPTED = "preempted"  # stopped by the memory guard to make room for another task (it stayed within its own estimate)
NEW, RUNNING, RETRY = "new", "running", "retry"
# Out of memory: a task that exits non-zero with a peak of >= NEAR_RAM_FRACTION of its machine's RAM (Gurobi's own OOM:
# the process peaks just below the RAM, the rest went to the page file). Not a failure: the attempt is a lower bound of
# max(peak, RAM) x lower_bound_factor, the task goes back into the pool and only a worker whose RAM exceeds the new
# estimate starts it (exceeds_ram). Also the Slurm outcome of an OUT_OF_MEMORY job (cluster.resources).
OOM = "oom"
NEAR_RAM_FRACTION = 0.85


def near_ram(peak_mb: float | None, ram_mb: float | None) -> bool:
    return bool(peak_mb and ram_mb and peak_mb >= NEAR_RAM_FRACTION * ram_mb)


def exceeds_ram(est_mb: float, mem_mb: float, attempt: int, last: dict) -> bool:
    """A worker with mem_mb of RAM never starts a task estimated above it - unless the estimate is a scaled-up eviction
    (the task may fit alone: it runs alone). Applies before the first attempt and after an out-of-memory attempt."""
    return est_mb > mem_mb and (attempt == 0 or (last.get("outcome") == OOM and not last.get("discarded")))


def _safe(key: str) -> str:
    return key.replace("/", "__")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _write_json_atomic(path: Path, data) -> None:
    """Write via a temp file + rename, so readers never see a half-written file. Windows refuses the rename while another
    process reads the target: retry briefly."""
    tmp = path.with_name(path.name + f".tmp{os.getpid()}-{id(data)}")
    tmp.write_text(json.dumps(data, indent=1))
    for i in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 19:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05 * (i + 1))


########################################################################################################################
# Machine: memory and process RSS (Linux /proc, Windows via ctypes; elsewhere unknown)
########################################################################################################################

class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def machine_memory_mb() -> tuple[float | None, float | None]:
    """(total, available) physical memory of this machine in MB; None where unknown."""
    try:
        if sys.platform == "win32":
            status = _MemoryStatus(dwLength=ctypes.sizeof(_MemoryStatus))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / 1024 ** 2, status.ullAvailPhys / 1024 ** 2
            return None, None
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                name, _, rest = line.partition(":")
                info[name] = float(rest.split()[0]) / 1024  # kB -> MB
        return info.get("MemTotal"), info.get("MemAvailable")
    except (OSError, AttributeError, ValueError):
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 2, None
        except (AttributeError, OSError, ValueError):
            return None, None


def process_rss_mb(proc: subprocess.Popen) -> tuple[float | None, float | None]:
    """(current, peak) resident memory of a child process in MB (Linux VmRSS/VmHWM, Windows working set)."""
    if sys.platform == "win32":
        return _rss_windows(int(proc._handle))
    return _rss_linux(proc.pid)


def pid_rss_mb(pid: int) -> tuple[float | None, float | None]:
    """(current, peak) resident memory in MB of any process of this machine by id (another worker's task)."""
    if sys.platform != "win32":
        return _rss_linux(pid)
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
    except (OSError, AttributeError):
        return None, None
    if not handle:
        return None, None
    try:
        return _rss_windows(handle)
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def _rss_windows(handle: int) -> tuple[float | None, float | None]:
    try:
        counters = _ProcessMemoryCounters(cb=ctypes.sizeof(_ProcessMemoryCounters))
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1024 ** 2, counters.PeakWorkingSetSize / 1024 ** 2
        return None, None
    except (OSError, AttributeError, ValueError):
        return None, None


def _rss_linux(pid: int) -> tuple[float | None, float | None]:
    try:
        rss = hwm = None
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = float(line.split()[1]) / 1024
                elif line.startswith("VmHWM:"):
                    hwm = float(line.split()[1]) / 1024
        return rss, hwm
    except (OSError, AttributeError, ValueError):
        return None, None


def physical_cores() -> int | None:
    """Physical cores of this Windows machine (os.cpu_count() counts hyperthreads); None elsewhere / if unknown."""
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        size = ctypes.c_ulong(0)
        kernel32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # RelationProcessorCore: query the size
        buf = ctypes.create_string_buffer(size.value)
        if not kernel32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
            return None
        count, offset = 0, 0
        while offset < size.value:  # variable-size records: DWORD Relationship, DWORD Size, ...
            count += 1
            offset += ctypes.c_ulong.from_buffer(buf, offset + 4).value or size.value
        return count or None
    except (OSError, AttributeError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    """Whether a process with this id runs on this machine (a reused id counts as alive: callers fall back to timeouts)."""
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: exists (other user); invalid parameter: no such process
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # e.g. EPERM: exists, owned by someone else
    return True


class _JobObjectLimits(ctypes.Structure):
    """JOBOBJECT_EXTENDED_LIMIT_INFORMATION (only LimitFlags is set; the rest stays zero)."""
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", ctypes.c_ulong),
                ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_ulong),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_ulong), ("SchedulingClass", ctypes.c_ulong),
                ("IoCounters", ctypes.c_ulonglong * 6), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _kill_on_close_job():
    """Windows job object whose processes are killed when the worker's handle closes - i.e. when the worker dies (closed
    window, Task Manager, reboot). Without it, orphaned tasks would keep running while another worker retries them
    (POSIX: Slurm kills the job's cgroup). None if unavailable."""
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        job = kernel32.CreateJobObjectW(None, None)
        limits = _JobObjectLimits(LimitFlags=0x2000)  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if job and kernel32.SetInformationJobObject(ctypes.c_void_p(job), 9, ctypes.byref(limits), ctypes.sizeof(limits)):  # ExtendedLimitInformation
            return job
    except (OSError, AttributeError):
        pass
    return None


def _assign_to_job(job, proc: subprocess.Popen) -> bool:
    try:
        return bool(ctypes.windll.kernel32.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(int(proc._handle))))
    except (OSError, AttributeError):
        return False


def _dir_size_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total / 1024 ** 2


def _log_has_marker(path: Path, marker: str, tail_bytes: int = 256 * 1024) -> bool:
    """Whether the end of a log contains a marker line (logs of solves can be large: read only the tail)."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - tail_bytes))
            return marker.encode() in f.read()
    except OSError:
        return False


def _terminate(proc: subprocess.Popen) -> None:
    """Ask a task to stop (whole process group on POSIX); _kill after a grace period."""
    try:
        if sys.platform == "win32":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass


def _kill(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


########################################################################################################################
# Store
########################################################################################################################

class Store:
    def __init__(self, root: Path, log_dir: Path | None = None):
        self.root = Path(root)
        self.log_dir = Path(log_dir) if log_dir else self.root / "logs"
        for sub in ("claims", "results", "resets", "workers", "jobs", "stop"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._results = {}  # file name -> record (results are immutable: read each once)

    def claim(self, key: str, attempt: int, worker: str) -> bool:
        """Atomically create claim <task>.<attempt>; False if another worker was faster."""
        try:
            fd = os.open(self.root / "claims" / f"{_safe(key)}.{attempt}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            f.write(worker)
        return True

    def write_result(self, key: str, attempt: int, record: dict) -> None:
        _write_json_atomic(self.root / "results" / f"{_safe(key)}.{attempt}.json", {"key": key, "attempt": attempt, **record})

    def reset(self, key: str, attempt: int, drop_args: list[str] | None = None) -> None:
        """Make attempt n retryable. drop_args: command-line arguments the next attempt runs without (e.g. a re-run of
        a done task without --no-overwrite, which would otherwise skip it)."""
        _write_json_atomic(self.root / "resets" / f"{_safe(key)}.{attempt}", {"time": _now(), "drop_args": drop_args or []})

    def drop_args(self, key: str, attempt: int) -> list[str]:
        """The drop_args of the reset of attempt n ([] if none; old resets hold a plain timestamp)."""
        try:
            return list(json.loads((self.root / "resets" / f"{_safe(key)}.{attempt}").read_text()).get("drop_args", []))
        except (OSError, ValueError, AttributeError):
            return []

    def log_path(self, key: str, attempt: int) -> Path:
        return self.log_dir / f"{_safe(key)}.{attempt}.log"

    def beat(self, worker: str, info: dict) -> None:
        """Heartbeat; a failed write is skipped (the next one follows in HEARTBEAT_S, staleness needs STALE_S)."""
        try:
            _write_json_atomic(self.root / "workers" / f"{worker}.json", {**info, "beat": time.time()})
        except OSError as e:
            print(f"{_now()} warning: heartbeat not written ({e})", flush=True)

    def add_job(self, job: str, info: dict) -> None:
        """Record a submitted worker job (Slurm id) - for status and cancel."""
        _write_json_atomic(self.root / "jobs" / f"{job}.json", {"job": job, "submitted": _now(), **info})

    def jobs(self) -> dict:
        out = {}
        for path in (self.root / "jobs").glob("*.json"):
            try:
                out[path.stem] = json.loads(path.read_text())
            except (OSError, ValueError):
                pass
        return out

    def workers(self) -> dict:
        out = {}
        for path in (self.root / "workers").glob("*.json"):
            try:
                out[path.stem] = json.loads(path.read_text())
            except (OSError, ValueError):
                pass
        return out

    def request_stop(self, target: str, now: bool) -> None:
        """Stop request for the workers that started before it: target = 'all', a host name or a worker id. Drain (finish
        the running tasks, start no new ones) or, with now, interrupt the running tasks (they are retried)."""
        _write_json_atomic(self.root / "stop" / f"{target.lower()}.json", {"target": target, "now": now, "time": time.time(), "requested": _now()})

    def stop_request(self, worker: str, host: str, started: float) -> dict | None:
        """The newest stop request that applies to this worker (requested after it started), else None."""
        found = None
        for target in ("all", host.lower(), worker.lower()):
            try:
                req = json.loads((self.root / "stop" / f"{target}.json").read_text())
            except (OSError, ValueError):
                continue
            if req.get("time", 0) > started and (found is None or req["time"] > found["time"]):
                found = req
        return found

    def mark_dead_workers(self, host: str) -> list[str]:
        """Mark this host's workers whose process no longer exists as exited (e.g. after a crash or reboot), so their
        claims are retried at once instead of after STALE_S."""
        dead = []
        for wid, info in self.workers().items():
            if info.get("exited") or info.get("host", "").lower() != host.lower() or not info.get("pid"):
                continue
            if not pid_alive(int(info["pid"])):
                try:
                    _write_json_atomic(self.root / "workers" / f"{wid}.json", {**info, "exited": _now(), "exit_reason": "process gone"})
                    dead.append(wid)
                except OSError:
                    pass
        return dead

    def results(self) -> list[dict]:
        """All result records (new files are read once and cached). Records written before the out-of-memory outcome (no
        `mem_mb`) are read as OOM if they failed near their machine's RAM (from the worker's heartbeat)."""
        workers = None
        for path in (self.root / "results").iterdir():
            if path.suffix == ".json" and path.name not in self._results:
                try:
                    record = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue  # being written (atomic replace makes this rare); read next time
                if record.get("outcome") == FAILED and "mem_mb" not in record and record.get("peak_rss_mb"):
                    workers = self.workers() if workers is None else workers
                    ram = (workers.get(record.get("worker")) or {}).get("mem_mb")
                    if near_ram(record["peak_rss_mb"], ram):
                        record = {**record, "outcome": OOM, "limit_mb": ram, "oom_inferred": True}
                self._results[path.name] = record
        # A reset out-of-memory attempt (cluster.py restart <pattern>, e.g. it ran out of memory next to another solve)
        # is no measurement any more: no lower bound, no RAM rule (Estimator.add / exceeds_ram skip `discarded`)
        resets, out = None, []
        for name, record in self._results.items():
            if record.get("outcome") == OOM:
                resets = set(os.listdir(self.root / "resets")) if resets is None else resets
                if name.removesuffix(".json") in resets:
                    record = {**record, "discarded": True}
            out.append(record)
        return out

    def stale_view(self, host: str, now: float | None = None) -> tuple[str, set[str]] | None:
        """(description, worker ids) if this host's view of the store looks frozen, else None: no non-exited worker on
        another host is fresh, and at least two of them went silent (STALE_S) within STALE_VIEW_SPREAD_S of the last
        beat seen. Workers that went silent earlier died before the freeze and are not held. No time limit: a real
        simultaneous death of all of them is released by hand (the workers marked exited). Workers on this host are no
        evidence: they read through the same (possibly outdated) view."""
        now = time.time() if now is None else now
        beats = {wid: w["beat"] for wid, w in self.workers().items()
                 if not w.get("exited") and "beat" in w and w.get("host", "").lower() != host.lower()}
        if not beats or now - max(beats.values()) < STALE_S:
            return None
        last = max(beats.values())
        held = {wid for wid, b in beats.items() if last - b <= STALE_VIEW_SPREAD_S}
        if len(held) < 2:
            return None
        return f"the last {len(held)} live workers on other hosts went silent together {(now - last) / 60:.0f} min ago", held

    def snapshot(self, keys, now: float | None = None, hold: set[str] = frozenset()) -> dict:
        """{key: (state, attempt, info)}: state NEW/RUNNING/RETRY/DONE/FAILED; attempt = highest claimed attempt (0 = none);
        info = the result record, or {'worker': id} for RUNNING. hold: the worker ids of stale_view() (a worker whose
        view looks frozen) - their stale claims stay RUNNING ({'unverified': True}), not retried."""
        now = time.time() if now is None else now
        claims = {}
        for name in os.listdir(self.root / "claims"):
            safe, _, n = name.rpartition(".")
            if n.isdigit():
                claims[safe] = max(claims.get(safe, 0), int(n))
        results = {(r["key"], r["attempt"]): r for r in self.results()}
        resets = set(os.listdir(self.root / "resets"))
        workers = self.workers()
        out = {}
        for key in keys:
            n = claims.get(_safe(key), 0)
            if n == 0:
                out[key] = (NEW, 0, {})
                continue
            result = results.get((key, n))
            if result is None:
                try:
                    worker = (self.root / "claims" / f"{_safe(key)}.{n}").read_text().strip()
                except OSError:
                    worker = ""
                w = workers.get(worker)
                alive = w is not None and now - w.get("beat", 0) < STALE_S and not w.get("exited")
                if alive or not worker:
                    out[key] = (RUNNING, n, {"worker": worker})
                elif worker in hold and not w.get("exited"):
                    out[key] = (RUNNING, n, {"worker": worker, "unverified": True})
                else:
                    out[key] = (RETRY, n, {"worker": worker, "stale": True})
            elif result["outcome"] == DONE:
                out[key] = (RETRY if f"{_safe(key)}.{n}" in resets else DONE, n, result)
            elif result["outcome"] == FAILED:
                out[key] = (RETRY if f"{_safe(key)}.{n}" in resets else FAILED, n, result)
            else:
                out[key] = (RETRY, n, result)
        return out


########################################################################################################################
# Estimates
########################################################################################################################

class Estimator:
    """Memory / runtime per task from measurements, falling back through `levels(key)` (most specific first) to the
    prior (TOML resources). Memory of a completed run is its peak as measured, capped at its machine's RAM (no safety
    factor: user decision Oct 2026 - a run that finished with paging needs no more than the RAM); memory lower bounds and
    all runtimes are scaled by the safety factors. Per level: the max of this pool's completed runs if there are any
    (they may lower what the seed runs or other tasks' lower bounds say), else the max of the seed runs' completed runs
    (raised by this pool's lower bounds), else the largest lower bound. Lower bounds: failed attempts (peak), evictions
    (scaled up once more), preemptions (peak, no runtime), interruptions (runtime scaled up). Seed records carry a
    `source` (cluster.seed_records), this pool's do not. Runtimes use only the levels `time_level(level)` accepts: they
    vary too much across a coarse level (one slow task would make all of them look too long for a job's rest)."""

    def __init__(self, specs: dict, levels, mem_safety: float = 1.2, time_safety: float = 1.5, lower_bound_factor: float = 1.25,
                 time_level=lambda level: True):
        self.specs, self.levels, self.time_level = specs, levels, time_level
        self.mem_safety, self.time_safety, self.lb = mem_safety, time_safety, lower_bound_factor
        # level -> {(origin, kind): max observed, safety factors applied}; origin own / seed, kind done / lb
        self.mem, self.time = {}, {}
        self.seen = set()

    def add(self, record: dict) -> None:
        """record: key, outcome, peak_rss_mb, elapsed_s (results of pool attempts or imported Slurm measurements)."""
        ident = (record.get("source", "pool"), record["key"], record.get("attempt"), record.get("job"))
        if ident in self.seen or record.get("noop") or record.get("discarded"):
            return
        self.seen.add(ident)
        outcome, peak, elapsed = record["outcome"], record.get("peak_rss_mb"), record.get("elapsed_s")
        slot = ("seed" if "source" in record else "own", "done" if outcome == DONE else "lb")
        mem = time_ = None
        if outcome == DONE:
            mem, time_ = peak, elapsed
            if mem and record.get("mem_mb"):  # finished with paging: it needs no more than that machine's RAM
                mem = min(mem, record["mem_mb"])
        elif outcome == FAILED:
            mem = peak  # whatever it used is a lower bound; the runtime of a failure says nothing
        elif outcome == EVICTED or outcome == OOM:  # the need is above what it reached (an OOM: above the limit)
            mem = max(peak or 0, record.get("limit_mb") or 0) * self.lb or None
        elif outcome == PREEMPTED:
            mem = peak  # made room for another task: it needs at least what it reached; its runtime says nothing
        elif outcome == INTERRUPTED or outcome == "timeout":
            mem = peak
            time_ = max(elapsed or 0, record.get("limit_s") or 0) * self.lb or None
        mem = mem and mem * (1.0 if outcome == DONE else self.mem_safety)
        time_ = time_ and time_ * self.time_safety
        for level in self.levels(record["key"]):
            if mem:
                values = self.mem.setdefault(level, {})
                values[slot] = max(values.get(slot, 0), mem)
            if time_ and self.time_level(level):
                values = self.time.setdefault(level, {})
                values[slot] = max(values.get(slot, 0), time_)

    def clear(self) -> None:
        """Forget all records (a measurement was discarded: the maxima cannot be undone, so the caller re-adds the rest)."""
        self.mem, self.time, self.seen = {}, {}, set()

    @staticmethod
    def _pick(values: dict) -> tuple[float, bool]:
        """(value, measured = may go below the prior) of one level, see the class docstring."""
        if ("own", "done") in values:
            return values[("own", "done")], True
        if ("seed", "done") in values:
            return max(values[("seed", "done")], values.get(("own", "lb"), 0)), True
        return max(values.values()), False

    def estimate(self, key: str) -> dict:
        """{mem_mb, mem_src, time_s, time_src}; src = the level the value comes from, or 'prior'. The most specific level
        with completed runs sets the value (it may go below the prior); levels with only lower bounds (failed / evicted /
        preempted / interrupted) before it are floors that only ever raise it - a task that crashes after seconds says
        nothing about its need, and a task's own preemption must not hide the coarser levels' measurements."""
        spec = self.specs[key]
        mem, mem_src = self._resolve(self.mem, key, spec["prior_mem_mb"])
        time_s, time_src = self._resolve(self.time, key, spec["prior_time_s"])
        return {"mem_mb": mem, "mem_src": mem_src, "time_s": time_s, "time_src": time_src}

    def _resolve(self, table: dict, key: str, prior: float) -> tuple[float, str]:
        value, src, floor, floor_src = prior, "prior", 0.0, None
        for level in self.levels(key):
            if level not in table:
                continue
            level_value, measured = self._pick(table[level])
            if measured:
                value, src = level_value, level
                break
            if level_value > floor:
                floor, floor_src = level_value, level
        if floor > value:
            value, src = floor, floor_src
        return value, src


########################################################################################################################
# Worker
########################################################################################################################

class Worker:
    """Runs tasks of the pool on this machine until nothing is left (or idle too long / walltime ends).

    specs: {key: {cmd: [argv], deps: [key], cpus, prior_mem_mb, prior_time_s, rank}} in dependency order (deps first).
    Selection: only these keys are run; dependencies outside the specs count as satisfied.
    reload: optional callback, called every reload_s: returns None (no change) or {specs, weights, plan_id} - the new
    task set (see update_specs), so tasks can be added / removed while the worker runs.
    """

    def __init__(self, specs: dict, store: Store, estimator: Estimator, *, cwd: Path, cores: int, mem_mb: float,
                 mem_fraction: float = 0.9, mem_packing: bool = True, end_time: float | None = None, idle_s: float = 3 * 3600,
                 max_tasks: int | None = None, thread_cap: int | None = None, keep_going: bool = True, console: bool = False,
                 node_base: Path | None = None, disk_min_free_mb: float = 10 * 1024, seed_records=(), poll_s: float = 15,
                 guard_available_fraction: float = 0.04, guard_rss_fraction: float = 0.97, evict_grace_s: float = 60,
                 interrupt_margin_s: float = 600, reserve_after_s: float = 1800, rank_offset_s: float | None = None,
                 weights: dict | None = None, opportunistic: bool = True, opportunistic_after_s: float = 600,
                 chain_after_s: float | None = None,
                 chain=None, chain_retry_s: float = 600, info: dict | None = None, reload=None, reload_s: float = 60, handover=None,
                 plan_id: str | None = None, out=print):
        self.store, self.est = store, estimator
        self.info = info or {}  # extra heartbeat fields (e.g. the worker's log file)
        self.cwd, self.cores, self.mem_mb = cwd, cores, mem_mb
        self.pack_mb = mem_fraction * mem_mb
        self.mem_packing, self.end_time, self.idle_s = mem_packing, end_time, idle_s
        self.max_tasks, self.thread_cap, self.keep_going, self.console = max_tasks, thread_cap, keep_going, console
        self.node_base = Path(node_base or os.environ.get("TMPDIR") or tempfile.gettempdir())
        self.node_base.mkdir(parents=True, exist_ok=True)  # the free-space check needs an existing directory
        self.disk_min_free_mb, self.poll_s = disk_min_free_mb, poll_s
        self.guard_available_fraction, self.guard_rss_fraction, self.evict_grace_s = guard_available_fraction, guard_rss_fraction, evict_grace_s
        self.interrupt_margin_s, self.reserve_after_s, self.rank_offset_s = interrupt_margin_s, reserve_after_s, rank_offset_s
        self.weights = weights  # {key: weight} (weighted order; takes precedence over rank_offset_s) or None
        # Opportunistic starts (see admit), only with memory packing. A firm task counts its full reservation as in use
        # for its first opportunistic_after_s (fast early growth), then only its peak so far
        self.opportunistic, self.opportunistic_after_s = opportunistic and mem_packing, opportunistic_after_s
        # Chaining: once, chain_after_s after the start, call chain() (submits a successor job, returns its id or None)
        # if tasks remain; retried every chain_retry_s until it succeeds
        self.chain_after_s, self.chain, self.chain_retry_s = chain_after_s, chain, chain_retry_s
        self.chained, self.last_chain_try = None, 0.0
        self.out = out
        job = os.environ.get("SLURM_JOB_ID")
        self.host = socket.gethostname()
        self.id = f"{self.host}-{('j' + job) if job else 'local'}-{os.getpid()}"
        self.job_object = _kill_on_close_job()  # Windows: tasks die with the worker
        self.stopping = None  # the stop request being followed (cluster.py stop)
        self.too_large = set()  # tasks needing more cpus than this worker's cores (logged once, never started here)
        # tasks whose memory estimate exceeds the RAM (exceeds_ram; logged once, re-checked every admission)
        self.too_big = set()
        self.running = {}  # key -> {proc, attempt, start, est, cpus, peak_mb, rss_mb, node_dir, node_peak_mb, log, stop, opp, opp_start}
        self.reload, self.reload_s, self.plan_id = reload, reload_s, plan_id
        # Code handover: called every reload_s; starts a successor running the changed code and returns its description
        # (then this worker drains), or None
        self.handover = handover
        # Tasks of other live workers on this machine (e.g. one draining after a handover), shaped like `running`
        # entries: they occupy cores and memory here (see refresh_foreign)
        self.foreign, self.foreign_peaks = {}, {}
        self._set_specs(specs)
        self.seed_records, self.discarded = list(seed_records), set()  # discarded: (key, attempt) of reset OOM attempts
        for record in self.seed_records:
            self.est.add(record)
        self.estimates, self.critical = {}, {}
        self.head_blocked = (None, 0.0)
        self.last_evict = 0.0
        self.last_disk_warning = 0.0
        self.stale_view = None  # Store.stale_view() of the last poll: (note, worker ids whose claims are not retried)
        self.any_failed = False
        self.counts = {DONE: 0, FAILED: 0, OOM: 0, EVICTED: 0, PREEMPTED: 0, INTERRUPTED: 0}

    def log(self, msg: str) -> None:
        self.out(f"{_now()} {msg}")

    # -- task set -------------------------------------------------------------------------------------------------------
    def _set_specs(self, specs: dict) -> None:
        self.specs = specs
        self.est.specs = specs
        self.children = {}
        for key, spec in specs.items():
            for dep in spec["deps"]:
                if dep in specs:
                    self.children.setdefault(dep, []).append(key)

    def update_specs(self, specs: dict, weights: dict | None = None, plan_id: str | None = None) -> None:
        """Switch to a new task set (a changed config): added tasks become schedulable, removed ones are no longer
        started. A removed task that runs here keeps running to its end (its result is recorded as usual) and is
        dropped when it finishes ('retired'). Changed specs (command, resources, deps) apply to the next attempt."""
        old = self.specs
        added = [k for k in specs if k not in old]
        removed = [k for k in old if k not in specs and not old[k].get("retired")]
        changed = [k for k in specs if k in old and specs[k] != old[k]]
        retired = {k: {**old[k], "retired": True} for k in self.running if k not in specs}
        self._set_specs({**specs, **retired})  # retired last: nothing in the new plan depends on them
        self.weights = weights
        self.plan_id = plan_id
        self.head_blocked = (None, 0.0)
        self.refresh_estimates()
        self.log(f"plan reloaded ({plan_id}): {len(added)} task(s) added, {len(removed)} removed, {len(changed)} changed, "
                 f"{len(specs)} selected" + (f"; {len(retired)} removed task(s) still running here, they finish normally" if retired else ""))

    # -- estimates and priorities ---------------------------------------------------------------------------------------
    def refresh_estimates(self) -> None:
        records = self.store.results()
        discarded = {(r["key"], r.get("attempt")) for r in records if r.get("discarded")}
        if discarded - self.discarded:  # a reset OOM attempt was already added as a lower bound: rebuild without it
            self.discarded = discarded
            self.est.clear()
            for record in self.seed_records:
                self.est.add(record)
        for record in records:
            self.est.add(record)
        self.estimates = {key: self.est.estimate(key) for key in self.specs}
        self.critical = {}
        for key in reversed(list(self.specs)):  # children come after their parents
            self.critical[key] = self.estimates[key]["time_s"] + max((self.critical[c] for c in self.children.get(key, [])), default=0)

    def priority(self, key: str) -> tuple:
        """Sort key of a ready task (smaller = first). Strict (rank_offset_s None): rank, then the longest chain of
        dependent tasks (critical path), then memory. Soft: low-priority tasks still last; otherwise the critical path
        minus dataset_rank x rank_offset_s, so a later dataset's task goes first if its chain is that much longer.
        Weighted (weights set): low-priority tasks still last; otherwise the critical path x the task's weight, so a task
        of a dataset weighted 10x lower goes first only if its chain is more than 10x longer."""
        spec = self.specs[key]
        if self.weights is not None:
            return spec.get("low", False), -self.critical.get(key, 0) * self.weights.get(key, 1.0), -self.estimates[key]["mem_mb"]
        if self.rank_offset_s is None:
            return spec["rank"], -self.critical.get(key, 0), -self.estimates[key]["mem_mb"]
        return (spec.get("low", False), -(self.critical.get(key, 0) - spec.get("dataset_rank", spec["rank"]) * self.rank_offset_s),
                -self.estimates[key]["mem_mb"])

    def threads(self, key: str) -> int:
        cpus = min(self.specs[key]["cpus"], self.cores)
        return min(cpus, self.thread_cap) if self.thread_cap else cpus

    # -- task lifecycle -------------------------------------------------------------------------------------------------
    def start(self, key: str, attempt: int, opp: bool = False) -> None:
        spec, est = self.specs[key], self.estimates[key]
        node_dir = self.node_base / f"gurobi-nodes-{self.id}-{_safe(key)}-{attempt}"
        node_dir.mkdir(parents=True, exist_ok=True)
        cmd = [a.replace("@THREADS@", str(self.threads(key))).replace("@NODEDIR@", str(node_dir)) for a in spec["cmd"]]
        dropped = self.store.drop_args(key, attempt - 1) if attempt > 1 else []
        if dropped:  # manual re-run of a done task (cluster.py restart)
            cmd = [a for a in cmd if a not in dropped]
        log = None if self.console else self.store.log_path(key, attempt)
        handle = open(log, "w", encoding="utf-8", errors="replace") if log else None
        # Windows: no console window per task when logging to a file (the worker may run without a desktop, e.g. as a
        # scheduled task); stdin from NUL so no task inherits an invalid console handle
        kwargs = {"start_new_session": True} if sys.platform != "win32" else \
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | (subprocess.CREATE_NO_WINDOW if handle else 0)}
        if handle:
            handle.write(f"# task {key}, attempt {attempt}, worker {self.id}, {_now()}\n# estimate: {est}\n# {' '.join(cmd)}\n")
            handle.flush()
        proc = subprocess.Popen(cmd, cwd=self.cwd, stdin=subprocess.DEVNULL if handle else None, stdout=handle,
                                stderr=subprocess.STDOUT if handle else None, **kwargs)
        if self.job_object and not _assign_to_job(self.job_object, proc):
            self.log(f"warning: {key} could not be tied to the worker's lifetime (job object) - kill it by hand if the worker dies")
        self.running[key] = {"proc": proc, "attempt": attempt, "start": time.time(), "est": est, "cpus": self.threads(key),
                             "peak_mb": 0.0, "rss_mb": 0.0, "node_dir": node_dir, "node_peak_mb": 0.0, "log": log, "handle": handle,
                             "stop": None, "opp": opp, "opp_start": opp}
        src = "" if est["mem_src"] != "prior" else " (prior)"
        self.log(f"start {key} (" + ("opportunistic, " if opp else "") + f"attempt {attempt}, {self.threads(key)} threads, est. {est['mem_mb'] / 1024:.1f} GB{src}, "
                 f"{est['time_s'] / 3600:.1f} h)" + (f", re-run without {' '.join(dropped)}" if dropped else "") + (f" -> {log}" if log else ""))

    def stop(self, key: str, outcome: str, reason: str) -> None:
        """Evict / interrupt a running task: TERM now, KILL after 30 s (checked in poll)."""
        run = self.running[key]
        if run["stop"] is None:
            run["stop"] = (outcome, time.time())
            self.log(f"{outcome} {key}: {reason}")
            _terminate(run["proc"])

    def finish(self, key: str, rc: int) -> None:
        run = self.running.pop(key)
        if run["handle"]:
            run["handle"].close()
        shutil.rmtree(run["node_dir"], ignore_errors=True)
        outcome = run["stop"][0] if run["stop"] else (DONE if rc == 0 else FAILED)
        # Out of memory only with memory packing: without it (local) nothing would keep the retry off this machine
        if outcome == FAILED and self.mem_packing and near_ram(run["peak_mb"], self.mem_mb):
            outcome = OOM
        elapsed = time.time() - run["start"]
        noop = outcome == DONE and run["log"] is not None and _log_has_marker(run["log"], NOOP_MARKER)
        record = {"outcome": outcome, "noop": noop, "rc": rc, "elapsed_s": round(elapsed, 1), "peak_rss_mb": round(run["peak_mb"], 1) or None,
                  "node_files_peak_mb": round(run["node_peak_mb"], 1), "threads": run["cpus"], "worker": self.id, "mem_mb": round(self.mem_mb, 1),
                  "host": socket.gethostname(), "start": datetime.datetime.fromtimestamp(run["start"]).isoformat(timespec="seconds"),
                  "end": _now(), "estimate": run["est"], "opportunistic": run["opp_start"], "log": str(run["log"]) if run["log"] else None}
        if outcome == OOM:
            record["limit_mb"] = round(self.mem_mb, 1)  # the estimator's lower bound: max(peak, limit) x lower_bound_factor
        self.store.write_result(key, run["attempt"], record)
        self.est.add({"key": key, "attempt": run["attempt"], **record})
        self.counts[outcome] += 1
        if outcome == FAILED:
            self.any_failed = True
        peak = f", peak {run['peak_mb'] / 1024:.1f} GB" if run["peak_mb"] else ""
        peak += " (nothing to do - not used as a measurement)" if noop else ""
        self.log(f"{outcome} {key} after {datetime.timedelta(seconds=int(elapsed))}" + (f" (exit {rc})" if outcome in (FAILED, OOM) else "") + peak
                 + (f" of {self.mem_mb / 1024:.0f} GB RAM: out of memory - back in the pool for a worker with more RAM" if outcome == OOM else ""))
        if self.specs[key].get("retired"):  # removed from the plan while it ran: not started again
            self._set_specs({k: s for k, s in self.specs.items() if k != key})
            self.estimates.pop(key, None)
            self.critical.pop(key, None)

    def poll(self) -> bool:
        """Sample memory / node files, kill tasks that ignore TERM, collect finished ones. True if one finished."""
        finished = False
        for key, run in list(self.running.items()):
            rss, hwm = process_rss_mb(run["proc"])
            if rss is not None:
                run["rss_mb"] = rss
                run["peak_mb"] = max(run["peak_mb"], hwm or rss)
            rc = run["proc"].poll()
            if rc is not None:
                self.finish(key, rc)
                finished = True
            elif run["stop"] and time.time() - run["stop"][1] > 30:
                _kill(run["proc"])
        return finished

    def disk_free_gb(self) -> str:
        try:
            return f"{shutil.disk_usage(self.node_base).free / 1024 ** 3:.0f}"
        except OSError:
            return "?"

    def check_disk(self) -> bool:
        """Node-file usage per task and free disk; False (= start nothing new) while free space is below the minimum."""
        for run in self.running.values():
            run["node_peak_mb"] = max(run["node_peak_mb"], _dir_size_mb(run["node_dir"]))
        try:
            free = shutil.disk_usage(self.node_base).free / 1024 ** 2
        except OSError:
            return True
        if free < self.disk_min_free_mb:
            if time.time() - self.last_disk_warning > 600:
                used = sum(r["node_peak_mb"] for r in self.running.values())
                self.log(f"WARNING: only {free / 1024:.0f} GB free on the node-file disk {self.node_base} (node files of running "
                         f"tasks: {used / 1024:.1f} GB) - starting no new tasks until it recovers")
                self.last_disk_warning = time.time()
            return False
        return True

    def task_details(self) -> dict:
        """Heartbeat field `tasks`: what another worker on this machine needs to account for them (refresh_foreign)."""
        return {k: {"pid": r["proc"].pid, "cpus": r["cpus"], "est_mb": round(r["est"]["mem_mb"], 1), "peak_mb": round(r["peak_mb"], 1),
                    "rss_mb": round(r["rss_mb"], 1), "start": r["start"], "opp": r["opp"]} for k, r in self.running.items()}

    def check_stale_view(self) -> None:
        """Update self.stale_view (Store.stale_view), logging when this machine's view of the pool freezes / recovers."""
        view = self.store.stale_view(self.host)
        if view and not self.stale_view:
            self.log(f"warning: {view[0]} - this machine probably reads an outdated copy of the pool (e.g. after standby);"
                     " their tasks are not retried until their heartbeats come through again (or they are marked exited)")
        elif self.stale_view and not view:
            self.log("heartbeats of other hosts come through again (or the silent workers were marked exited) - retrying stale claims normally")
        self.stale_view = view

    def refresh_foreign(self) -> None:
        """Tasks of the other live workers on this machine (outside Slurm only) from their heartbeats,
        memory sampled here by pid (heartbeats are up to HEARTBEAT_S old). Workers in this process are skipped: they only
        exist in tests, where they stand for separate machines. Heartbeats without task details (workers predating them)
        count each task with this worker's estimate, as just started."""
        if os.environ.get("SLURM_JOB_ID"):
            return  # a Slurm job has one worker (successors are new jobs with their own allocation)
        foreign, now = {}, time.time()
        for wid, w in self.store.workers().items():
            if (wid == self.id or w.get("exited") or now - w.get("beat", 0) > STALE_S or not w.get("pid") or w.get("pid") == os.getpid()
                    or w.get("host", "").lower() != self.host.lower() or w.get("slurm_job") or not pid_alive(int(w["pid"]))):
                continue
            for key, t in (w.get("tasks") or {k: {} for k in w.get("running", [])}).items():
                fid = f"{wid}/{key}"
                rss, hwm = pid_rss_mb(int(t["pid"])) if t.get("pid") else (None, None)
                if t.get("pid") and rss is None and not pid_alive(int(t["pid"])):
                    continue  # finished since the heartbeat
                est = t.get("est_mb") or (self.estimates[key]["mem_mb"] if key in self.estimates else 0.0)
                foreign[fid] = {"est": {"mem_mb": est}, "cpus": t.get("cpus") or (self.threads(key) if key in self.specs else 1),
                                "peak_mb": max(self.foreign_peaks.get(fid, 0.0), t.get("peak_mb") or 0.0, hwm or rss or 0.0),
                                "rss_mb": rss if rss is not None else t.get("rss_mb") or 0.0, "start": t.get("start") or now,
                                "opp": bool(t.get("opp")), "stop": None}
        if len(foreign) != len(self.foreign):
            self.log(f"other workers on this machine run {len(foreign)} task(s) - counted as occupied here"
                     + (f" ({sum(r['cpus'] for r in foreign.values())} threads, est. {sum(r['est']['mem_mb'] for r in foreign.values()) / 1024:.1f} GB)" if foreign else ""))
        self.foreign = foreign
        self.foreign_peaks = {fid: r["peak_mb"] for fid, r in foreign.items()}

    def _reservation(self, run: dict) -> float:
        """Memory a firm task holds for admission: its estimate, or more if it already uses more."""
        return max(run["est"]["mem_mb"], run["rss_mb"] * 1.1)

    def _in_use(self, run: dict) -> float:
        """Memory a task counts as actually using (opportunistic admission and guard): an opportunistic task its estimate
        (or its peak, if higher), a firm one its peak so far - after an initial opportunistic_after_s with its full
        reservation. Peaks only grow, so a stopped opportunistic task is not restarted into the same gap."""
        if run["opp"]:
            return max(run["est"]["mem_mb"], run["peak_mb"] * 1.1)
        if time.time() - run["start"] < self.opportunistic_after_s:
            return max(self._reservation(run), run["peak_mb"] * 1.1)
        return run["peak_mb"] * 1.1

    def guard_opportunistic(self) -> None:
        """Stop the newest opportunistic tasks while the memory the tasks actually use (RSS) exceeds the packing share, i.e.
        once the firm tasks have really grown into the gap the opportunistic ones were started in - not merely because of
        reservations (a young firm task's full estimate, `_in_use`), which only gate new starts in `admit`. A task above
        its estimate survives as long as the RAM suffices. Runs every second (cheap: uses the RSS sampled by poll), so the
        firm tasks keep the headroom up to the RAM. No restart into the same gap: admit counts at least 1.1 x peak >= RSS."""
        active = [k for k, r in self.running.items() if r["stop"] is None]
        opps = sorted((k for k in active if self.running[k]["opp"]), key=lambda k: self.running[k]["start"], reverse=True)
        if not opps:
            return
        used = sum(self.running[k]["rss_mb"] for k in active) + sum(r["rss_mb"] for r in self.foreign.values())
        for victim in opps:
            if used <= self.pack_mb:
                break
            run = self.running[victim]
            why = f"tasks use {used / 1024:.1f} GB of {self.pack_mb / 1024:.1f} GB usable (opportunistic start)"
            if run["rss_mb"] > run["est"]["mem_mb"]:
                self.stop(victim, EVICTED, f"{why}, it uses {run['rss_mb'] / 1024:.1f} GB > its estimate "
                                           f"{run['est']['mem_mb'] / 1024:.1f} GB - will be retried with a higher estimate")
            else:
                self.stop(victim, PREEMPTED, f"{why} - makes room for the firm tasks, will be retried")
            used -= run["rss_mb"]

    def guard_memory(self) -> None:
        """When memory runs out, stop one task: the most recently started opportunistic one, else the most recently started
        low-priority one, else the most recently started one (least work lost) except the one using the most memory (it may
        really need it; the others make room). A task beyond its
        own estimate is evicted (retried with a higher estimate), any other one preempted (retried; its peak is a
        minimum, its runtime is not used). Tasks of other workers on this machine count for the memory used and as the
        largest one, but only own tasks are stopped (the other worker's guard handles its own)."""
        active = [k for k, r in self.running.items() if r["stop"] is None]
        tasks = {**{k: self.running[k] for k in active}, **self.foreign}
        if not active or len(tasks) < 2 or time.time() - self.last_evict < self.evict_grace_s:
            return
        total, available = machine_memory_mb()
        used = sum(r["rss_mb"] for r in tasks.values())
        low_system = available is not None and total and available < self.guard_available_fraction * total
        over_capacity = used > self.guard_rss_fraction * self.mem_mb
        if low_system or over_capacity:
            largest = max(tasks, key=lambda k: tasks[k]["rss_mb"])
            # opportunistic, then low-priority tasks go first (even the largest one)
            opps = [k for k in active if self.running[k]["opp"]]
            lows = [k for k in active if self.specs[k].get("low")]
            candidates = opps or lows or [k for k in active if k != largest]
            if not candidates:  # the only own task is the largest one: the other worker has to make room
                return
            victim = max(candidates, key=lambda k: self.running[k]["start"])
            run = self.running[victim]
            why = (f"only {available / 1024:.1f} GB of {total / 1024:.0f} GB available" if low_system
                   else f"tasks use {used / 1024:.1f} GB of {self.mem_mb / 1024:.0f} GB")
            if run["rss_mb"] > run["est"]["mem_mb"]:
                self.stop(victim, EVICTED, f"{why}, it uses {run['rss_mb'] / 1024:.1f} GB > its estimate "
                                           f"{run['est']['mem_mb'] / 1024:.1f} GB - will be retried with a higher estimate")
            else:
                self.stop(victim, PREEMPTED, f"{why} - makes room for {largest} ({tasks[largest]['rss_mb'] / 1024:.1f} GB), "
                                             f"will be retried")
            self.last_evict = time.time()

    def guard_walltime(self) -> bool:
        """Interrupt everything shortly before the job ends. True if the worker must stop."""
        if self.end_time is None or time.time() < self.end_time - self.interrupt_margin_s:
            return False
        for key in list(self.running):
            self.stop(key, INTERRUPTED, "walltime ends - will be retried by another worker")
        return True

    # -- scheduling -----------------------------------------------------------------------------------------------------
    def ready(self, snap: dict) -> tuple[list[str], int]:
        """Tasks that can start now (by priority) and the number of tasks still to do (not done / failed / blocked).
        A spec with `after_all` (evaluate) becomes ready once every other task is done, failed or blocked."""
        blocked, ready, remaining, last = set(), [], 0, []
        for key, spec in self.specs.items():  # dependency order: blocked parents are known before their children
            if spec.get("retired"):  # removed from the plan, only finishing here
                continue
            state = snap[key][0]
            deps = [d for d in spec["deps"] if d in self.specs and not self.specs[d].get("retired")]
            if state in (DONE,):
                continue
            if state == FAILED or any(d in blocked for d in deps):
                blocked.add(key)
                continue
            remaining += 1
            if spec.get("after_all"):
                last.append(key)
            elif state in (NEW, RETRY) and key not in self.running and all(snap[d][0] == DONE for d in deps):
                ready.append(key)
        if remaining == len(last):  # only after_all tasks left
            ready += [k for k in last if snap[k][0] in (NEW, RETRY) and k not in self.running]
        ready.sort(key=self.priority)
        return ready, remaining

    def admit(self, snap: dict) -> int:
        """Start ready tasks that fit (cores, memory estimate, walltime); returns the number started.

        Low-priority tasks (low_priority TM variants) only use spare capacity: they start once every ready normal task
        is placed. A started task is never stopped to make room (only the memory guards and the walltime stop tasks).
        A running firm task reserves its estimate (or more, if it already uses more); a task estimated above the packing
        share starts only when no other firm task runs.

        Opportunistic starts: a ready task that does not fit by these reservations may still start if it fits next to
        the memory the running tasks actually use (`_in_use`: firm tasks their peak so far) - e.g. next to a large task
        that has not grown to its estimate yet, whose idle cores would otherwise wait. Opportunistic tasks are ignored by
        the firm reservations (they never keep a firm task from starting), go shortest-estimate first (best chance to
        finish), never take the cores of a top task in head-of-line reservation, are stopped first when the firm tasks
        grow into their gap (`guard_opportunistic`) and become firm once their estimate fits (not during a reservation)."""
        if self.any_failed and not self.keep_going:
            return 0
        ready, _ = self.ready(snap)
        started = 0
        occupied = [*self.running.values(), *self.foreign.values()]  # other workers' tasks on this machine count too
        firm = [r for r in occupied if not r["opp"]]
        reserved = sum(self._reservation(r) for r in firm)
        in_use = sum(self._in_use(r) for r in occupied)
        cores_used = sum(r["cpus"] for r in occupied)
        head = ready[0] if ready else None
        if head != self.head_blocked[0]:
            self.head_blocked = (head, time.time())
        waiting_normal = None  # the first ready normal-priority task that did not fit
        reserving = False  # head-of-line reservation: the top task has waited too long, no firm backfilling
        candidates = []  # tasks that may run here but did not start firmly: opportunistic candidates
        for key in ready:
            low = self.specs[key].get("low", False)
            if self.max_tasks and len(self.running) >= self.max_tasks:
                if not low and waiting_normal is None:
                    waiting_normal = key
                break
            if self.specs[key]["cpus"] > self.cores and not self.thread_cap:
                if key not in self.too_large:  # never with fewer threads (work units / solver times would not be comparable)
                    self.too_large.add(key)
                    self.log(f"skipping {key}: needs {self.specs[key]['cpus']} cpus, this worker has {self.cores} cores - left to other workers")
                continue
            est = self.estimates[key]
            # Above the RAM it would only swap / run out of memory here (before the first attempt and after an out-of-memory
            # attempt; an evicted task's estimate is a scaled-up lower bound, so its retry still runs alone)
            if self.mem_packing and exceeds_ram(est["mem_mb"], self.mem_mb, snap[key][1], snap[key][2]):
                if key not in self.too_big:
                    self.too_big.add(key)
                    self.log(f"skipping {key}: estimated at {est['mem_mb'] / 1024:.1f} GB, this machine has {self.mem_mb / 1024:.0f} GB "
                             + ("(out of memory before) " if snap[key][2].get("outcome") == OOM else "") + "- left to other workers")
                continue
            remaining_s = None if self.end_time is None else self.end_time - self.interrupt_margin_s - time.time()
            if remaining_s is not None and est["time_src"] != "prior" and est["time_s"] > remaining_s:
                continue  # measured runtime does not fit into the rest of this job; another worker will take it
            if reserving or (low and waiting_normal is not None):
                # the top task waits too long (keep the freed resources for it) / low-priority tasks come last in
                # `ready`: none of them while a normal task waits for space - but both may still start opportunistically
                candidates.append(key)
                continue
            fits_cores = cores_used + self.threads(key) <= self.cores
            fits_mem = not self.mem_packing or reserved + est["mem_mb"] <= self.pack_mb
            alone = not firm  # a task above the packing share (but within the RAM) still runs, without other firm tasks
            if fits_cores and (fits_mem or alone):
                if not self.store.claim(key, snap[key][1] + 1, self.id):
                    continue  # another worker took it
                if alone and not fits_mem:
                    self.log(f"note: {key} is estimated at {est['mem_mb'] / 1024:.1f} GB > {self.pack_mb / 1024:.1f} GB usable - running it alone")
                self.start(key, snap[key][1] + 1)
                firm.append(self.running[key])
                reserved += est["mem_mb"]
                in_use += est["mem_mb"]
                cores_used += self.threads(key)
                started += 1
                if key == head:
                    self.head_blocked = (None, 0.0)
            else:
                candidates.append(key)
                if not low and waiting_normal is None:
                    waiting_normal = key
                if key == head and time.time() - self.head_blocked[1] > self.reserve_after_s:
                    reserving = True
        if self.opportunistic:
            started += self._admit_opportunistic(snap, candidates, in_use, cores_used, head if reserving else None)
            if not reserving:
                self._promote(reserved)
        return started

    def _admit_opportunistic(self, snap: dict, candidates: list[str], in_use: float, cores_used: int, head: str | None) -> int:
        """Start candidates that fit next to the memory in use (see admit); `head` = the top task in reservation, whose
        cores stay free."""
        started = 0
        keep_cores = self.threads(head) if head else 0
        for key in sorted(candidates, key=lambda k: (self.specs[k].get("low", False), self.estimates[k]["time_s"])):
            if self.max_tasks and len(self.running) >= self.max_tasks:
                break
            est = self.estimates[key]["mem_mb"]
            if est > self.pack_mb or cores_used + self.threads(key) + keep_cores > self.cores or in_use + est > self.pack_mb:
                continue
            if not self.store.claim(key, snap[key][1] + 1, self.id):
                continue
            self.start(key, snap[key][1] + 1, opp=True)
            in_use += est
            cores_used += self.threads(key)
            started += 1
        return started

    def _promote(self, reserved: float) -> None:
        """Opportunistic tasks become firm (oldest first) once their reservation fits next to the firm ones."""
        for key, run in sorted(self.running.items(), key=lambda kr: kr[1]["start"]):
            if run["opp"] and run["stop"] is None and reserved + self._reservation(run) <= self.pack_mb:
                run["opp"] = False
                reserved += self._reservation(run)
                self.log(f"firm {key}: its estimate fits now - no longer stopped first")

    def check_stop(self, started: float) -> None:
        """Follow a stop request (cluster.py stop): drain = start nothing new; now = also interrupt the running tasks."""
        req = self.store.stop_request(self.id, self.host, started)
        if req is None or (self.stopping is not None and self.stopping["time"] >= req["time"]):
            return
        self.stopping = req
        if req.get("now"):
            self.log(f"stop requested ({req.get('requested')}): interrupting {len(self.running)} running task(s), they will be retried")
            for key in list(self.running):
                self.stop(key, INTERRUPTED, "stop requested - will be retried by another worker")
        else:
            self.log(f"stop requested ({req.get('requested')}): starting no new tasks, exiting after the {len(self.running)} running one(s)")

    def maybe_chain(self, started: float, remaining: int) -> None:
        """Submit the successor job once, chain_after_s after the start, if tasks remain."""
        if (self.chain is None or self.chain_after_s is None or self.chained or remaining == 0 or self.stopping
                or time.time() - started < self.chain_after_s or time.time() - self.last_chain_try < self.chain_retry_s):
            return
        self.last_chain_try = time.time()
        try:
            job = self.chain()
        except Exception as e:  # noqa: BLE001 - a failed submission must not stop the running tasks
            job = None
            self.log(f"ERROR: submitting the successor job raised {e!r}")
        if job:
            self.chained = job
            self.log(f"submitted successor job {job} ({remaining} task(s) remain)")
        else:
            self.log(f"ERROR: could not submit the successor job - retrying in {self.chain_retry_s / 60:.0f} min "
                     f"(without it, this worker chain ends with this job)")

    def maybe_reload(self) -> None:
        """Ask the reload callback for a changed task set; a failing callback must not stop the running tasks."""
        try:
            new = self.reload()
        except Exception as e:  # noqa: BLE001
            self.log(f"WARNING: reloading the plan raised {e!r} - keeping the current plan")
            return
        if new:
            self.update_specs(new["specs"], new.get("weights"), new.get("plan_id"))

    def maybe_handover(self) -> bool:
        """Ask the handover callback whether the code changed and a successor took over; if so, drain (start nothing new,
        exit after the own tasks - the successor counts them as occupied, see refresh_foreign). True if handed over."""
        if not self.handover or self.stopping:
            return False
        try:
            successor = self.handover()
        except Exception as e:  # noqa: BLE001
            self.log(f"WARNING: the code handover raised {e!r} - this worker keeps going")
            return False
        if not successor:
            return False
        self.stopping = {"time": time.time(), "requested": f"code handover to {successor}", "handover": successor}
        self.log(f"code changed: handed over to a new worker ({successor}) - starting no new tasks, exiting after the "
                 f"{len(self.running)} running one(s)")
        return True

    def run(self) -> int:
        """Main loop; returns 0 if nothing failed in this worker, 1 otherwise."""
        self.log(f"worker {self.id}: {self.cores} cores, {self.mem_mb / 1024:.0f} GB ({self.pack_mb / 1024:.0f} GB for packing"
                 + ("" if self.mem_packing else ", memory packing off") + (", opportunistic starts" if self.opportunistic else "") + f"), {len(self.specs)} task(s) selected, node files in {self.node_base}"
                 + f" ({self.disk_free_gb()} GB free; new tasks pause below {self.disk_min_free_mb / 1024:g} GB)"
                 + (f", ends {datetime.datetime.fromtimestamp(self.end_time).isoformat(timespec='minutes')}" if self.end_time else "")
                 + (f", plan {self.plan_id} (reloaded when the config changes)" if self.reload else ""))
        started = time.time()
        info = {"host": self.host, "pid": os.getpid(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "cores": self.cores,
                "mem_mb": self.mem_mb, "started": _now(), "started_ts": started, "end_time": self.end_time, **self.info}
        for wid in self.store.mark_dead_workers(self.host):
            self.log(f"worker {wid} on this host is gone (crash / reboot) - its tasks are retried now")
        last_beat = last_full = 0.0
        last_reload = time.time()
        idle_since = None
        snap = {}
        try:
            while True:
                now = time.time()
                # Beat before a handover check: a task started in the previous iteration must be in the heartbeat
                # the successor reads (refresh_foreign)
                if now - last_beat > HEARTBEAT_S:
                    beat = {**info, "running": sorted(self.running), "tasks": self.task_details(), "stopping": bool(self.stopping)}
                    if self.stopping and self.stopping.get("handover"):
                        beat["handover"] = self.stopping["handover"]
                    if self.reload:  # status compares it with the config's current plan
                        beat["plan"] = self.plan_id
                    self.store.beat(self.id, beat)
                    last_beat = now
                if (self.reload or self.handover) and now - last_reload > self.reload_s:
                    last_reload = now
                    if self.maybe_handover():
                        last_beat = 0.0  # status shows the drain at once
                    elif self.reload:
                        self.maybe_reload()
                finished = self.poll()
                self.guard_opportunistic()
                if self.guard_walltime():
                    if not self.running:
                        break
                    time.sleep(1)
                    continue
                if finished or now - last_full > self.poll_s:
                    last_full = now
                    self.check_stop(started)
                    # Estimates before refresh_foreign: a foreign task without est_mb (heartbeat predating task details)
                    # takes this worker's estimate - on the first poll, with no estimates yet, it counted as 0 GB and a
                    # full-year solve was started next to another one (Oct 2026)
                    self.refresh_estimates()
                    self.refresh_foreign()
                    self.guard_memory()
                    disk_ok = self.check_disk()
                    self.check_stale_view()
                    snap = self.store.snapshot(self.specs, hold=self.stale_view[1] if self.stale_view else frozenset())
                    if disk_ok and not self.stopping and self.admit(snap) or finished:
                        last_beat = 0.0  # status shows started / finished tasks at once, not up to HEARTBEAT_S later
                    ready, remaining = self.ready(snap)
                    self.maybe_chain(started, remaining)
                    if self.stopping and not self.running:
                        self.log("handed over: own tasks finished" if self.stopping.get("handover") else "stopped on request")
                        break
                    if self.running:
                        idle_since = None
                    elif remaining == 0 or (self.any_failed and not self.keep_going):
                        self.log("nothing left to run" if remaining == 0 else "stopping after a failure (keep_going off)")
                        break
                    elif self.foreign:
                        # Other workers on this machine (e.g. one draining after a handover) hold cores / memory that
                        # free up when their tasks end: not idle - exiting now could leave the machine without a worker
                        idle_since = None
                    else:
                        idle_since = idle_since or now
                        if now - idle_since > self.idle_s:
                            self.log(f"idle for {datetime.timedelta(seconds=int(self.idle_s))} with {remaining} task(s) still waiting for others - exiting")
                            break
                time.sleep(1)
        except KeyboardInterrupt:
            self.log("interrupted - stopping running tasks (they will be retried)")
            for key in list(self.running):
                self.stop(key, INTERRUPTED, "worker interrupted")
            while self.running:
                self.poll()
                time.sleep(1)
        finally:
            self.store.beat(self.id, {**info, "running": [], "exited": _now()})
        self.log(f"worker {self.id} exits: " + ", ".join(f"{n} {k}" for k, n in self.counts.items() if n))
        return 1 if self.any_failed else 0
