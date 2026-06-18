#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NonBinarityMarkov.py - Overview of how often the integrality-constrained
variables in a folder of MK-*.sqlite files actually take non-integer values.

The model has five variables that are declared Binary or Integer (and therefore
written to SQLite as integrality-constrained):

    vLineInvest   Binary       transmission line investment
    vGenInvest    Integer      generation investment (units)
    vCommit       Binary*      unit commitment
    vStartup      Binary*      start-up
    vShutdown     Binary*      shut-down

(* the Markov edge handling relaxes vCommit/vStartup/vShutdown to a continuous
domain for the non-edge timesteps, so for those runs fractional values are
*expected* — this script quantifies exactly that.)

For every file it counts, per variable, how many stored values deviate from the
nearest integer by more than --tol, and the maximum such deviation. Results are
aggregated into a table per run kind (main / operational / invest-regret /
regret / operational-regret), with one row per (shift_tm, perturb_tm) × edge
handling — the same axes CompareMarkov.py uses.

The discovery and filtering (--nrOfClusters, --separateClusters, --tm,
--include-nonoptimal) are shared with CompareMarkov.py (imported directly), so a
folder filtered here matches the same folder filtered there.

Usage
-----
python research/MK/NonBinarityMarkov.py [folder]
python research/MK/NonBinarityMarkov.py results/ --per-file
python research/MK/NonBinarityMarkov.py results/ --nrOfClusters 3,5,7 --separateClusters
python research/MK/NonBinarityMarkov.py results/ --tm none:0.2 --tm 1:none --tm base
python research/MK/NonBinarityMarkov.py results/ --tol 1e-4 --include-nonoptimal
"""

import argparse
import glob
import os
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich_argparse import RichHelpFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse CompareMarkov's filtering / grouping helpers verbatim so the same folder
# filters identically in both tools.
from CompareMarkov import (  # noqa: E402
    EDGE_ORDER,
    EDGE_STRICT,
    _tm_label,
    _tm_sort,
    filter_entries,
    parse_tm_spec,
)
from EvaluateMarkov import _load_metadata_from_conn  # noqa: E402  (metadata parsing)
from InOutModule.printer import Printer  # noqa: E402

printer = Printer.getInstance()
printer.set_width(200)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The integrality-constrained variables, in the order they are reported. Some may
# be absent from a given file (e.g. no thermal generators) — missing tables are
# skipped silently.
INTEGRALITY_VARS = ['vLineInvest', 'vGenInvest', 'vCommit', 'vStartup', 'vShutdown']

# Edge-handling display order (Truth only appears for main/operational kinds).
EDGE_DISPLAY_ORDER = ['Truth'] + list(EDGE_ORDER) + [EDGE_STRICT]

# Run kinds in report order, with their file-suffix routing. Order matters:
# '-invest-regret' and '-operational-regret' both end in '-regret', so they are
# tested before the bare '-regret'; '-operational' is tested before plain main.
KIND_ORDER = ['main', 'operational', 'invest_regret', 'regret', 'operational_regret']
KIND_LABELS = {
    'main': 'Investment (main) runs',
    'operational': 'Operational runs',
    'invest_regret': 'Invest-regret runs',
    'regret': 'Regret runs',
    'operational_regret': 'Operational-regret runs',
}


# ---------------------------------------------------------------------------
# SQLite reading
# ---------------------------------------------------------------------------

def _classify_kind(basename: str) -> str:
    if basename.endswith("-invest-regret.sqlite"):
        return 'invest_regret'
    if basename.endswith("-operational-regret.sqlite"):
        return 'operational_regret'
    if basename.endswith("-regret.sqlite"):
        return 'regret'
    if basename.endswith("-operational.sqlite"):
        return 'operational'
    return 'main'


def _table_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _nonbinarity_for_table(conn, table: str, tol: float) -> dict | None:
    """Per-variable integrality stats, computed entirely in SQL.

    Returns {'total', 'fractional', 'sum_dev', 'max_dev'} or None when the table
    is absent. 'fractional' counts rows whose value deviates from the nearest
    integer by more than *tol*; 'sum_dev' is the summed deviation Σ|v-round(v)|
    over those same fractional rows; 'max_dev' is the largest such deviation.
    """
    if not _table_exists(conn, table):
        return None
    row = conn.execute(
        f'SELECT COUNT(*), '
        f'       SUM(CASE WHEN ABS("values" - ROUND("values")) > ? THEN 1 ELSE 0 END), '
        f'       SUM(CASE WHEN ABS("values" - ROUND("values")) > ? '
        f'                THEN ABS("values" - ROUND("values")) ELSE 0 END), '
        f'       MAX(ABS("values" - ROUND("values"))) '
        f'FROM "{table}"', (tol, tol)
    ).fetchone()
    total = int(row[0] or 0)
    fractional = int(row[1] or 0)
    sum_dev = float(row[2]) if row[2] is not None else 0.0
    max_dev = float(row[3]) if row[3] is not None else 0.0
    return {'total': total, 'fractional': fractional, 'sum_dev': sum_dev, 'max_dev': max_dev}


def _load_file(sqlite_file: str, tol: float):
    """Load metadata + per-variable non-binarity for one MK sqlite file.

    Returns ('skip', file, None) on failure, else ('ok', file, entry) where
    entry has the metadata fields plus 'file', 'kind', and 'nb' (a
    {var: stats|None} map).
    """
    basename = os.path.basename(sqlite_file)
    try:
        conn = sqlite3.connect(sqlite_file)
        meta = _load_metadata_from_conn(conn, basename)
        nb = {var: _nonbinarity_for_table(conn, var, tol) for var in INTEGRALITY_VARS}
        conn.close()
    except Exception:
        return ('skip', sqlite_file, None)
    entry = {**meta, 'file': sqlite_file, 'kind': _classify_kind(basename), 'nb': nb}
    return ('ok', sqlite_file, entry)


def load_all(folder: str, tol: float, include_nonoptimal: bool = False) -> list[dict]:
    """Discover and load all MK-*.sqlite files in *folder* concurrently."""
    pattern = os.path.join(folder, "MK-*.sqlite")
    files = sorted(f for f in glob.glob(pattern) if os.path.basename(f).startswith("MK-"))
    if not files:
        return []
    max_workers = min(len(files), (os.cpu_count() or 4), 60)
    printer.information(f"Found {len(files)} MK-*.sqlite files in '{folder}', loading with up to {max_workers} threads...")

    entries: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_load_file, f, tol) for f in files]
        for future in as_completed(futures):
            status, _f, entry = future.result()
            if status == 'ok':
                entries.append(entry)

    if not include_nonoptimal:
        entries = [e for e in entries if e.get('termination_condition') == 'optimal']
    return entries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _edge_sort_key(edge):
    try:
        return (0, EDGE_DISPLAY_ORDER.index(edge))
    except ValueError:
        return (1, str(edge))


def _fmt_dev(dev: float) -> str:
    return "0" if dev <= 0 else f"{dev:.2e}"


def _aggregate_group(entries: list[dict]) -> dict:
    """Sum per-variable stats over a list of entries.

    Returns {'files', 'files_with_frac', 'per_var': {var: {total, fractional}},
    'total_frac', 'total_vals', 'sum_dev', 'max_dev'}.
    """
    per_var = {var: {'total': 0, 'fractional': 0} for var in INTEGRALITY_VARS}
    files_with_frac = 0
    max_dev = sum_dev = 0.0
    total_frac = total_vals = 0
    for e in entries:
        any_frac = False
        for var in INTEGRALITY_VARS:
            stats = e['nb'].get(var)
            if not stats:
                continue
            per_var[var]['total'] += stats['total']
            per_var[var]['fractional'] += stats['fractional']
            total_vals += stats['total']
            total_frac += stats['fractional']
            sum_dev += stats['sum_dev']
            max_dev = max(max_dev, stats['max_dev'])
            if stats['fractional'] > 0:
                any_frac = True
        if any_frac:
            files_with_frac += 1
    return {
        'files': len(entries), 'files_with_frac': files_with_frac, 'per_var': per_var,
        'total_frac': total_frac, 'total_vals': total_vals,
        'sum_dev': sum_dev, 'max_dev': max_dev,
    }


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    """Print a simple right-aligned (left for first two cols) table via the printer."""
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.ljust(widths[i]) if i < 2 else cell.rjust(widths[i]))
        return "  ".join(parts)

    printer.information(fmt_row(headers))
    printer.information("  ".join("-" * w for w in widths))
    for r in rows:
        printer.information(fmt_row(r))


def report_kind(kind: str, entries: list[dict]) -> tuple[int, int]:
    """Print the table for one run kind. Returns (n_files, n_files_with_frac)."""
    groups: dict = defaultdict(list)
    for e in entries:
        groups[((e.get('shift_tm'), e.get('perturb_tm')), e.get('edge_handling'))].append(e)

    headers = (['TM', 'Edge', 'Files']
               + INTEGRALITY_VARS  # fractional count per variable
               + ['TotalFrac', 'TotalVals', '%Frac', 'FilesWithFrac', 'SumDev', 'MaxDev'])
    rows: list[list[str]] = []
    for (tm_key, edge) in sorted(groups, key=lambda g: (_tm_sort(g[0]), _edge_sort_key(g[1]))):
        agg = _aggregate_group(groups[(tm_key, edge)])
        pct = (agg['total_frac'] / agg['total_vals'] * 100) if agg['total_vals'] else 0.0
        rows.append(
            [_tm_label(*tm_key), str(edge)] + [str(agg['files'])]
            + [str(agg['per_var'][v]['fractional']) for v in INTEGRALITY_VARS]
            + [str(agg['total_frac']), str(agg['total_vals']), f"{pct:.3f}",
               f"{agg['files_with_frac']}/{agg['files']}",
               _fmt_dev(agg['sum_dev']), _fmt_dev(agg['max_dev'])]
        )

    total = _aggregate_group(entries)
    printer.information("")
    printer.rule(f"{KIND_LABELS[kind]}  (n={total['files']}, "
                 f"files with non-binarity: {total['files_with_frac']})")
    _print_table(rows, headers)
    return total['files'], total['files_with_frac']


def report_per_file(entries: list[dict]) -> None:
    """One row per file: total fractional values and max deviation."""
    headers = ['Kind', 'Edge', 'TM'] + INTEGRALITY_VARS + ['TotalFrac', 'SumDev', 'MaxDev', 'File']
    rows: list[list[str]] = []
    for e in sorted(entries, key=lambda x: (KIND_ORDER.index(x['kind']) if x['kind'] in KIND_ORDER else 99,
                                            _tm_sort((x.get('shift_tm'), x.get('perturb_tm'))),
                                            _edge_sort_key(x.get('edge_handling')))):
        total_frac = 0
        sum_dev = max_dev = 0.0
        var_cells = []
        for var in INTEGRALITY_VARS:
            stats = e['nb'].get(var)
            if stats:
                total_frac += stats['fractional']
                sum_dev += stats['sum_dev']
                max_dev = max(max_dev, stats['max_dev'])
                var_cells.append(str(stats['fractional']))
            else:
                var_cells.append("-")
        rows.append([e['kind'], str(e.get('edge_handling')),
                     _tm_label(e.get('shift_tm'), e.get('perturb_tm'))]
                    + var_cells + [str(total_frac), _fmt_dev(sum_dev), _fmt_dev(max_dev),
                                   os.path.basename(e['file'])])
    printer.information("")
    printer.rule("Per-file non-binarity")
    _print_table(rows, headers)


def report(entries: list[dict], per_file: bool, label_suffix: str = "") -> None:
    if label_suffix:
        printer.information("")
        printer.rule(label_suffix.strip(" —"), style="cyan")

    if per_file:
        report_per_file(entries)

    grand_files = grand_frac = 0
    for kind in KIND_ORDER:
        kind_entries = [e for e in entries if e['kind'] == kind]
        if not kind_entries:
            continue
        n, nf = report_kind(kind, kind_entries)
        grand_files += n
        grand_frac += nf

    printer.information("")
    printer.success(f"Overall{label_suffix}: {grand_files} file(s); "
                    f"{grand_frac} with at least one non-binary value "
                    f"({(grand_frac / grand_files * 100) if grand_files else 0:.1f}%).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=("Overview of the non-binarity (integrality violation) of the binary/integer "
                     "variables (vLineInvest, vGenInvest, vCommit, vStartup, vShutdown) across a "
                     "folder of MK-*.sqlite files. Shares CompareMarkov.py's filtering "
                     "(--nrOfClusters, --separateClusters, --tm, --include-nonoptimal)."),
        formatter_class=RichHelpFormatter)
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing MK-*.sqlite files (default: current directory)")
    parser.add_argument("--tol", type=float, default=1e-6, help="A value counts as non-binary if it deviates from the nearest integer by more than this (default: 1e-6).")
    parser.add_argument("--per-file", action="store_true", help="Also print a row per file (total fractional values + max deviation), not just the aggregated tables.")
    parser.add_argument("--include-nonoptimal", action="store_true", help="Include runs whose termination_condition is not 'optimal' (default: optimal only).")
    parser.add_argument("--nrOfClusters", default=None, metavar="N,N,...",
                        help="Only include runs whose 'clusters' run-parameter is in this comma-separated list (e.g. 3,5,7).")
    parser.add_argument("--separateClusters", action="store_true",
                        help="Emit the full report once per cluster count found in the (filtered) data instead of aggregating all cluster counts together.")
    parser.add_argument("--tm", action="append", default=None, metavar="SHIFT:PERTURB",
                        help="Only include the selected (shift_tm, perturb_tm) combinations. Repeatable and/or comma-separated. Each side is a number, 'none' (parameter unset) or '*' (any); 'base' is shorthand for none:none. Examples: --tm none:0.2 --tm 1:none --tm base ; --tm \"1:*,2:*\".")
    args = parser.parse_args()

    try:
        cluster_filter = ({int(t) for t in args.nrOfClusters.split(',') if t.strip()}
                          if args.nrOfClusters else None)
    except ValueError:
        parser.error(f"--nrOfClusters expects a comma-separated list of integers, got '{args.nrOfClusters}'")
    try:
        tm_specs = [parse_tm_spec(s) for raw in (args.tm or []) for s in raw.split(',') if s.strip()]
    except ValueError as exc:
        parser.error(f"--tm: {exc}")

    printer.information(f"Loading MK-*.sqlite files from '{args.folder}' (tol={args.tol:g}) ...")
    entries = load_all(args.folder, args.tol, include_nonoptimal=args.include_nonoptimal)
    entries = filter_entries(entries, cluster_filter, tm_specs)

    if not entries:
        reasons = [] if args.include_nonoptimal else ["filtering to optimal runs"]
        if cluster_filter is not None:
            reasons.append(f"--nrOfClusters {sorted(cluster_filter)}")
        if tm_specs:
            reasons.append("--tm filter")
        printer.warning(f"No usable MK-*.sqlite files found in '{args.folder}'"
                        + (f" (after {', '.join(reasons)})" if reasons else ""))
        return

    filt = "all" if args.include_nonoptimal else "optimal-only"
    printer.information(f"Loaded [{filt}]: {len(entries)} file(s).")

    if args.separateClusters:
        cluster_values = sorted({e.get('clusters') for e in entries},
                                key=lambda v: (v is None, v))
        for value in cluster_values:
            label = f"{value} clusters" if value is not None else "no clustering"
            report([e for e in entries if e.get('clusters') == value],
                   args.per_file, label_suffix=f" — {label}")
    else:
        report(entries, args.per_file)


if __name__ == "__main__":
    main()
