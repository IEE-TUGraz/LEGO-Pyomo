"""Check whether a machine produces the same Gurobi work units (and solutions) as the reference machines.

Runs 9 RTS-GMLC solves through Markov.py (one at a time, 8 threads, the experiment's solver options) and
compares input data, Gurobi version, work units, B&B nodes and objective with `workunit_reference.json`. Identical
work units mean an identical solution path, so work units of this machine can be compared with the reference
machines' results. Seconds per work unit are printed as well, but they are only meaningful on an otherwise idle
machine.

Usage (from the repo root, in the conda env; ~20 min on an otherwise idle current desktop CPU):
    python research/MK/experiments/workunitChecks/check_workunits.py
    python research/MK/experiments/workunitChecks/check_workunits.py --write-reference   # regenerate the reference after model code changes

Needs the clustered RTS-GMLC folders (data/RTS-GMLC/[stretchDemandX/]N clusters/). Copy them from a machine that
has them, or create them with `Markov.py data/RTS-GMLC --prepare-only ...` - the input hash check then tells whether
the clustering was reproduced identically.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
MARKOV = Path(__file__).resolve().parents[2] / "Markov.py"
REFERENCE_FILE = Path(__file__).resolve().parent / "workunit_reference.json"

# Options of the experiment runs (experiment-win-gap0.5.toml / experiment.toml), threads fixed for comparability
COMMON_ARGS = ["--reuse-inputfiles", "--network", "TP", "--threads", "8", "--node-file-start", "8", "--task", "main"]
# (name, stretch demand, clusters, shift TM, edge, MIP gap): every edge handling and TM shift (0, 1, 2) at least once,
# from a root-node-only solve up to B&B trees with ~7000 nodes; shortest first. ~20 min on an otherwise idle current
# desktop CPU - a different solution path shows already in the small cases, more cases add little certainty.
CASES = [
    ("sd0.9-c5-base-Markov", 0.9, 5, None, "Markov", 0.005),
    ("sd1-c5-base-NoEnf", 1.0, 5, None, "NoEnf", 0.005),
    ("sd0.5-c5-shiftTM1-Markov", 0.5, 5, 1, "Markov", 0.005),
    ("sd1-c7-base-Cyclic", 1.0, 7, None, "Cyclic", 0.005),
    ("sd0.5-c14-shiftTM2-Cyclic", 0.5, 14, 2, "Cyclic", 0.005),
    ("sd0.9-c28-base-Cyclic", 0.9, 28, None, "Cyclic", 0.005),
    ("sd0.7-c7-base-NoEnf", 0.7, 7, None, "NoEnf", 0.005),
    ("sd0.7-c14-shiftTM2-NoEnf", 0.7, 14, 2, "NoEnf", 0.005),
    ("sd1-c14-base-Markov", 1.0, 14, None, "Markov", 0.005),
]
WU_RTOL = 1e-6
OBJ_RTOL = 1e-6


def case_folder(data: Path, stretch: float, clusters: int) -> Path:
    folder = data if stretch == 1.0 else data / f"stretchDemand{stretch:g}"
    return folder / f"{clusters} clusters"


def input_hash(folder: Path) -> str:
    """Hash of all dataframes of the case study (independent of xlsx metadata such as timestamps)."""
    import pandas as pd
    from InOutModule.CaseStudy import CaseStudy
    cs = CaseStudy(str(folder))
    h = hashlib.sha256()
    for name, value in sorted(vars(cs).items()):
        if isinstance(value, pd.DataFrame):
            h.update(name.encode())
            h.update(repr(list(value.columns)).encode())
            h.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
    return h.hexdigest()


def copy_case_inputs(data: Path, workdir: Path, stretch: float, clusters: int) -> Path:
    """Copy the inputs of one case into `workdir/data/<name>/` (the files of the base and stretch folders and the
    cluster folder) and return the copy. A local copy instead of a link: Windows junctions cannot point to network
    drives, and every machine works on its own files, so several machines can run the check from one shared clone."""
    dst = workdir / "data" / data.name
    folders = [(data, dst)] + ([(data / f"stretchDemand{stretch:g}", dst / f"stretchDemand{stretch:g}")] if stretch != 1.0 else [])
    for src_dir, dst_dir in folders:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_dir / f.name)
    shutil.copytree(case_folder(data, stretch, clusters), case_folder(dst, stretch, clusters))
    return dst


def run_case(stretch: float, clusters: int, shift: int | None, edge: str, gap: float, workdir: Path) -> dict:
    """Solve one case with Markov.py in `workdir` (inputs copied to `workdir/data/` by `copy_case_inputs`) and return
    the measured values. Markov.py builds its output file name from the data path and writes into the cwd, so the
    data is passed as a relative path (an absolute Windows path breaks the file name)."""
    data_rel = next((workdir / "data").iterdir()).relative_to(workdir).as_posix()
    args = [sys.executable, "-u", str(MARKOV), data_rel, *COMMON_ARGS, "--mip-gap", f"{gap:g}",
            "--node-file-dir", str(workdir / "nodes"), "--clusters", str(clusters), "--edge", edge]
    if stretch != 1.0:
        args += ["--stretch-demand", f"{stretch:g}"]
    if shift is not None:
        args += ["--shift-tm", str(shift)]
    log_file = workdir / "markov.log"
    start = time.time()
    with open(log_file, "w", encoding="utf-8") as log:
        rc = subprocess.run(args, cwd=workdir, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
    elapsed = time.time() - start
    text = log_file.read_text(encoding="utf-8", errors="replace")

    def last(pattern):
        found = re.findall(pattern, text)
        return found[-1] if found else None

    result = {"rc": rc, "elapsed_s": elapsed, "log": str(log_file),
              "gurobi": last(r"Gurobi Optimizer version (\S+)"), "cpu": last(r"CPU model: (.*?),"),
              "nodes": int(n) if (n := last(r"Explored (\d+) nodes")) is not None else None,
              "objective": float(o) if (o := last(r"Optimal .* solution: ([-\d.]+)")) is not None else None,
              "solver_s": float(s) if (s := last(r"Solving .* took ([\d.]+) seconds")) is not None else None,
              "work_units": None, "termination": None}
    sqlite_files = list(workdir.glob("MK-*.sqlite"))
    if sqlite_files:
        with sqlite3.connect(sqlite_files[0]) as cnx:
            row = cnx.execute("SELECT work_units, termination_condition FROM solver_statistics").fetchone()
        result["work_units"], result["termination"] = row
    return result


def same(a, b, rtol) -> bool:
    return a is not None and b is not None and abs(a - b) <= rtol * max(abs(a), abs(b), 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "RTS-GMLC", help="RTS-GMLC data folder (default: data/RTS-GMLC of this repo)")
    parser.add_argument("--write-reference", action="store_true", help=f"Store this machine's results as the new reference ({REFERENCE_FILE.name})")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary folder with the logs and result files")
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated case names to run (default: all). With "
                        "--write-reference, only these cases are replaced in the existing reference")
    args = parser.parse_args()
    data = args.data.resolve()
    cases = CASES
    if args.cases:
        names = [n.strip() for n in args.cases.split(",")]
        unknown = set(names) - {c[0] for c in CASES}
        if unknown:
            sys.exit(f"Unknown cases {sorted(unknown)}; available: {[c[0] for c in CASES]}")
        cases = [c for c in CASES if c[0] in names]

    from InOutModule.SQLiteWriter import machine_info
    info = machine_info()
    print(f"Machine: {info}  |  Python {platform.python_version()}")
    reference = None
    if not args.write_reference:
        if not REFERENCE_FILE.exists():
            sys.exit(f"No reference file '{REFERENCE_FILE}' - create one on a reference machine with --write-reference")
        reference = json.loads(REFERENCE_FILE.read_text())
        print(f"Reference: Gurobi {reference['gurobi']}, created {reference['created']}")

    missing = sorted({str(case_folder(data, sd, c)) for _, sd, c, _, _, _ in cases if not case_folder(data, sd, c).is_dir()})
    if missing:
        sys.exit("Missing clustered input folders (copy them or create them with Markov.py --prepare-only):\n  " + "\n  ".join(missing))

    tmp = Path(tempfile.mkdtemp(prefix="check_workunits-"))
    results = {}
    try:
        for i, (name, sd, clusters, shift, edge, gap) in enumerate(cases, 1):
            print(f"\n[{i}/{len(cases)}] {name}: copying and hashing inputs ...", flush=True)
            workdir = tmp / name
            workdir.mkdir()
            h = input_hash(case_folder(copy_case_inputs(data, workdir, sd, clusters), sd, clusters))
            print(f"{name}: solving ...", flush=True)
            r = run_case(sd, clusters, shift, edge, gap, workdir)
            r["input_hash"] = h
            results[name] = r
            print(f"{name}: rc={r['rc']}, {r['termination']}, {r['work_units']} WU, {r['nodes']} nodes, objective {r['objective']}, "
                  f"{r['solver_s']} s", flush=True)
    finally:
        if args.keep:
            print(f"\nLogs and results kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    failed = [n for n, r in results.items() if r["rc"] != 0 or r["termination"] != "optimal"]
    if args.write_reference:
        if failed:
            sys.exit(f"Not writing a reference: {failed} did not solve to optimality")
        gurobi = {r["gurobi"] for r in results.values()}
        if len(gurobi) != 1:
            sys.exit(f"Not writing a reference: several Gurobi versions {gurobi}")
        gurobi = gurobi.pop()
        stored = {}
        if args.cases and REFERENCE_FILE.exists():  # partial run: replace only these cases
            old = json.loads(REFERENCE_FILE.read_text())
            if old["gurobi"] != gurobi:
                sys.exit(f"Not merging into the reference: it was made with Gurobi {old['gurobi']}, this machine has {gurobi}")
            stored = old["cases"]
        stored.update({n: {k: r[k] for k in ("input_hash", "work_units", "nodes", "objective", "solver_s")} for n, r in results.items()})
        stored = {c[0]: stored[c[0]] for c in CASES if c[0] in stored}  # CASES order, drops removed cases
        REFERENCE_FILE.write_text(json.dumps({
            "created": datetime.date.today().isoformat(), "gurobi": gurobi,  # no machine specs (versioned file)
            "cases": stored,
        }, indent=1) + "\n")
        print(f"\nReference written to {REFERENCE_FILE} ({len(stored)} of {len(CASES)} cases)")
        return

    # Comparison with the reference
    print(f"\n{'case':28} {'inputs':>7} {'ref WU':>9} {'WU':>9} {'nodes':>13} {'objective':>9} {'ref s/WU':>8} {'s/WU':>6}")
    problems = []
    for name, r in results.items():
        ref = reference["cases"].get(name)
        if ref is None:
            problems.append(f"{name}: no reference values")
            continue
        ok_in = r["input_hash"] == ref["input_hash"]
        ok_wu = same(r["work_units"], ref["work_units"], WU_RTOL)
        ok_nodes = r["nodes"] == ref["nodes"]
        ok_obj = same(r["objective"], ref["objective"], OBJ_RTOL)
        spwu = f"{r['solver_s'] / r['work_units']:.2f}" if r["solver_s"] and r["work_units"] else "-"
        nodes = "same" if ok_nodes else f"{r['nodes']} vs {ref['nodes']}"
        print(f"{name:28} {'same' if ok_in else 'DIFF':>7} {ref['work_units']:9.2f} {r['work_units'] or float('nan'):9.2f} "
              f"{nodes:>13} {'same' if ok_obj else 'DIFF':>9} {ref['solver_s'] / ref['work_units']:8.2f} {spwu:>6}")
        if not ok_in:
            problems.append(f"{name}: input data differs from the reference (different clustering or data version) - not conclusive")
        elif not (ok_wu and ok_nodes and ok_obj):
            problems.append(f"{name}: different solution path (work units {r['work_units']} vs {ref['work_units']})")

    gurobi = {r["gurobi"] for r in results.values()}
    print(f"\nGurobi {', '.join(map(str, gurobi))} (reference {reference['gurobi']}), CPU (Gurobi): {', '.join({str(r['cpu']) for r in results.values()})}")
    if gurobi != {reference["gurobi"]}:
        problems.append(f"Gurobi version differs from the reference ({reference['gurobi']}) - work units are only comparable within one version")
    if failed:
        problems.append(f"Not solved to optimality: {failed}")

    if problems:
        print("\nNOT ELIGIBLE - work units of this machine are not comparable with the reference machines:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nELIGIBLE - identical work units, B&B nodes and objectives: work units of this machine are comparable "
          "with the reference machines.\n(Seconds per work unit compare speed only if both machines were otherwise idle.)")


if __name__ == "__main__":
    main()
