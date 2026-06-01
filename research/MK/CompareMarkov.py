#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CompareMarkov.py - Boxplots comparing edge-handling strategies (NoEnf, Cyclic,
Markov) against the Truth model across shift-tm / perturb-tm combinations.

Produces 9 logical plots from a folder of MK-*.sqlite files; each is emitted
twice — once with all strategies and once with NoEnf excluded ('_noNoEnf'
suffix, since NoEnf's large deviations otherwise compress the scale) — for up
to 18 PNGs:

  A  compare_workunits_operational_absolute.png   Work units, operational runs
     compare_workunits_operational_relative.png   Work units as % of Truth, operational
  B  compare_vshutdown_operational_relative.png   vShutdown dev. vs Truth-op [%]
     compare_vshutdown_operational_absolute.png   vShutdown dev. vs Truth-op (abs)
  C  compare_workunits_investment_absolute.png    Work units, investment (main) runs
     compare_workunits_investment_relative.png    Work units as % of Truth, investment
  D  compare_vshutdown_investment_relative.png    vShutdown dev. vs Truth-main [%]
     compare_vshutdown_investment_absolute.png    vShutdown dev. vs Truth-main (abs)
  E  compare_invest_regret.png                    Invest-regret [%] vs Truth-main obj.

Each figure has one subplot per (shift_tm, perturb_tm) combination (shared
y-axis), and within each subplot one boxplot per edge handling (NoEnf, Cyclic,
Markov; +Markov-Strict with --markov-strict). Every box aggregates over the
*sub-cases* that share that TM combination — i.e. the other run parameters that
vary (clusters, stretch_demand, ...). Truth is the reference for
deviation/regret, never a box.

"Operational runs" are the `--operational` runs (vGenInvest fixed to Truth's
investment); "investment runs" are the regular main runs. Work-units plots need
a solver that reports work units (Gurobi) and are empty otherwise (e.g. HiGHS).

By default only runs with termination_condition == 'optimal' are included
(--include-nonoptimal to override).

Usage
-----
python research/MK/CompareMarkov.py [folder]
python research/MK/CompareMarkov.py results/ --output-dir plots/ --no-show
python research/MK/CompareMarkov.py results/ --include-nonoptimal
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

import matplotlib.pyplot as plt

from EvaluateMarkov import _load_metadata_from_conn  # re-use metadata parsing
from InOutModule.printer import Printer

printer = Printer.getInstance()
printer.set_width(300)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Run parameters that define a "sub-case" — everything except the TM perturbation
# params (shift_tm / perturb_tm), which form the per-subplot axis. Note: 'shift'
# is the time-series shift and is a sub-case key, NOT the subplot axis 'shift_tm'.
SUB_CASE_KEYS = [
    'case_study_directory', 'filter_zone', 'limit_k', 'clusters', 'shift',
    'stretch_demand', 'scale_vres', 'scale_invest_cost', 'thermal_invest_only',
    'merge_generators', 'relax_count', 'no_investment', 'rmip', 'no_crossover',
    'force_barrier', 'mip_gap', 'network', 'commit_consumption', 'startup_consumption',
]

# Edge-handling strategies shown as boxes (Truth is the reference, never a box).
# Markov-Strict is included only when --markov-strict is passed. The exact strings
# match the `edge_handling` value stored in run_parameters.
EDGE_ORDER = ['NoEnf', 'Cyclic', 'Markov']
EDGE_STRICT = 'Markov-Strict'
EDGE_COLORS = {
    'NoEnf': '#9467bd',  # purple
    'Cyclic': '#ff7f0e',  # orange
    'Markov': '#1f77b4',  # blue
    'Markov-Strict': '#2ca02c',  # green
}

# Base filename per logical plot (the "(excl. NoEnf)" variant inserts "_noNoEnf").
OUTPUT_NAMES = {
    'workunits_operational_abs': 'compare_workunits_operational_absolute.png',
    'workunits_operational_rel': 'compare_workunits_operational_relative.png',
    'vshutdown_operational_rel': 'compare_vshutdown_operational_relative.png',
    'vshutdown_operational_abs': 'compare_vshutdown_operational_absolute.png',
    'workunits_investment_abs': 'compare_workunits_investment_absolute.png',
    'workunits_investment_rel': 'compare_workunits_investment_relative.png',
    'vshutdown_investment_rel': 'compare_vshutdown_investment_relative.png',
    'vshutdown_investment_abs': 'compare_vshutdown_investment_absolute.png',
    'invest_regret': 'compare_invest_regret.png',
}


# ---------------------------------------------------------------------------
# SQLite readers — read only what each plot needs
# ---------------------------------------------------------------------------

def _read_objective(conn) -> float | None:
    try:
        row = conn.execute('SELECT "values" FROM objective LIMIT 1').fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def _read_termination(conn) -> str | None:
    try:
        row = conn.execute(
            'SELECT termination_condition FROM solver_statistics LIMIT 1'
        ).fetchone()
        return str(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def _weighted_var_sum(conn, var_name: str) -> float | None:
    """Annual-weighted sum of a (rp, k)-indexed variable, computed in SQL.

    Mirrors EvaluateMarkov's approach: build temp weight tables and aggregate via
    JOINs so the (potentially large) variable table is never transferred to Python.
    """
    try:
        wrp_rows = conn.execute('SELECT * FROM pWeight_rp').fetchall()
        wk_rows = conn.execute('SELECT * FROM pWeight_k').fetchall()
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _wrp (rp TEXT PRIMARY KEY, w REAL)")
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _wk (k TEXT PRIMARY KEY, w REAL)")
        conn.executemany("INSERT OR IGNORE INTO _wrp VALUES (?,?)",
                         [(str(r[0]), float(r[1])) for r in wrp_rows])
        conn.executemany("INSERT OR IGNORE INTO _wk VALUES (?,?)",
                         [(str(r[0]), float(r[1])) for r in wk_rows])
        val = conn.execute(
            f'SELECT SUM(v."values" * wrp.w * wk.w) '
            f'FROM "{var_name}" v '
            f'JOIN _wrp wrp ON CAST(v.rp AS TEXT) = wrp.rp '
            f'JOIN _wk wk ON CAST(v.k AS TEXT) = wk.k'
        ).fetchone()[0]
        return float(val) if val is not None else None
    except Exception:
        return None


def _load_compare_file(sqlite_file: str):
    """Load just the fields needed for the comparison plots from one MK sqlite.

    Returns one of:
        ('skip', file, None, None)
        ('invest_regret', file, termination, objective)
        ('main'|'operational', file, meta, payload)
    where payload = {'vShutdown': float|None, 'Objective': float|None}.
    """
    basename = os.path.basename(sqlite_file)

    # invest-regret MUST be tested before regret (the suffix "-invest-regret.sqlite"
    # also ends with "-regret.sqlite").
    if basename.endswith("-invest-regret.sqlite"):
        try:
            conn = sqlite3.connect(sqlite_file)
            obj = _read_objective(conn)
            term = _read_termination(conn)
            conn.close()
        except Exception:
            obj, term = None, None
        return ('invest_regret', sqlite_file, term, obj)

    # -regret.sqlite (operational regret from --calculate-regret) is not needed.
    if basename.endswith("-regret.sqlite"):
        return ('skip', sqlite_file, None, None)

    kind = 'operational' if basename.endswith("-operational.sqlite") else 'main'
    try:
        conn = sqlite3.connect(sqlite_file)
        meta = _load_metadata_from_conn(conn, basename)
        vshut = _weighted_var_sum(conn, 'vShutdown')
        obj = _read_objective(conn) if kind == 'main' else None
        conn.close()
    except Exception:
        return ('skip', sqlite_file, None, None)

    return (kind, sqlite_file, meta, {'vShutdown': vshut, 'Objective': obj})


def load_all(folder: str, include_nonoptimal: bool = False):
    """Discover and load all MK-*.sqlite files in *folder* concurrently.

    Returns (main_entries, operational_entries, invest_regret_map) where the map is
    keyed by the *base* main file path -> invest-regret objective.
    """
    pattern = os.path.join(folder, "MK-*.sqlite")
    files = sorted(f for f in glob.glob(pattern) if os.path.basename(f).startswith("MK-"))
    max_workers = min(len(files), (os.cpu_count() or 4), 60)

    printer.information(f"Found {len(files)} MK-*.sqlite files in '{folder}', loading with up to {max_workers} threads...")

    main_entries: list[dict] = []
    operational_entries: list[dict] = []
    invest_regret_obj: dict[str, float] = {}
    invest_regret_term: dict[str, str | None] = {}

    if not files:
        return main_entries, operational_entries, invest_regret_obj

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_load_compare_file, f) for f in files]
        for future in as_completed(futures):
            kind, f, meta, payload = future.result()
            if kind == 'skip':
                continue
            if kind == 'invest_regret':
                base = f.replace("-invest-regret.sqlite", ".sqlite")
                invest_regret_obj[base] = payload
                invest_regret_term[base] = meta  # termination condition
            else:
                entry = {**meta, **payload, 'file': f}
                (operational_entries if kind == 'operational' else main_entries).append(entry)

    if not include_nonoptimal:
        main_entries = [e for e in main_entries if e.get('termination_condition') == 'optimal']
        operational_entries = [e for e in operational_entries if e.get('termination_condition') == 'optimal']
        invest_regret_obj = {b: o for b, o in invest_regret_obj.items()
                             if invest_regret_term.get(b) == 'optimal'}

    return main_entries, operational_entries, invest_regret_obj


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def _tm_key(entry: dict) -> tuple:
    return (entry.get('shift_tm'), entry.get('perturb_tm'))


def _subcase_key(entry: dict) -> tuple:
    return tuple(entry.get(k) for k in SUB_CASE_KEYS)


def _tm_sort(tm_key: tuple):
    """Sort TM combinations by shift first, then perturb; None (unset) sorts first.

    Gives: base, perturb0.2, shift1, shift1+perturb0.2, shift2, shift2+perturb0.2, ...
    """
    shift_tm, perturb_tm = tm_key
    try:
        s_val = float(shift_tm) if shift_tm not in (None, 'None') else float('-inf')
    except (TypeError, ValueError):
        s_val = float('-inf')
    try:
        p_val = float(perturb_tm) if perturb_tm not in (None, 'None') else float('-inf')
    except (TypeError, ValueError):
        p_val = float('-inf')
    return (s_val, p_val)


def _tm_label(shift_tm, perturb_tm) -> str:
    parts = []
    if shift_tm not in (None, 'None'):
        parts.append(f"shiftTM={shift_tm}")
    if perturb_tm not in (None, 'None'):
        try:
            parts.append(f"perturbTM={float(perturb_tm):g}")
        except (TypeError, ValueError):
            parts.append(f"perturbTM={perturb_tm}")
    return ', '.join(parts) if parts else '(base)'


# ---------------------------------------------------------------------------
# Box builders — each returns {tm_key: {edge: [values]}}
# ---------------------------------------------------------------------------

def build_runtime_boxes(entries: list[dict], edges: list[str]) -> dict:
    """Absolute work units per (tm_key, edge), aggregated over sub-cases."""
    boxes: dict = defaultdict(lambda: defaultdict(list))
    for e in entries:
        edge = e.get('edge_handling')
        if edge not in edges:
            continue
        wu = e.get('work_units')
        if wu is None:
            continue
        boxes[_tm_key(e)][edge].append(float(wu))
    return boxes


def build_runtime_relative_boxes(entries: list[dict], edges: list[str]) -> dict:
    """Work units as a percentage of the Truth run of the same (tm, sub-case).

    value = work_units / truth_work_units * 100   (100% == as expensive as Truth)
    Requires Truth to report work units (Gurobi); skips sub-cases where it doesn't.
    """
    grouped: dict = defaultdict(lambda: defaultdict(dict))
    for e in entries:
        edge = e.get('edge_handling')
        if edge is None:
            continue
        grouped[_tm_key(e)][_subcase_key(e)][edge] = e.get('work_units')

    boxes: dict = defaultdict(lambda: defaultdict(list))
    for tm_key, subcases in grouped.items():
        for _sk, eh_map in subcases.items():
            truth = eh_map.get('Truth')
            if truth in (None, 0):
                continue
            for edge in edges:
                val = eh_map.get(edge)
                if val is None:
                    continue
                boxes[tm_key][edge].append(float(val) / float(truth) * 100)
    return boxes


def build_deviation_boxes(entries: list[dict], mode: str, edges: list[str]) -> dict:
    """vShutdown deviation from the Truth entry of the same (tm, sub-case).

    mode='relative': (val - truth) / |truth| * 100  [%]   (skips truth==0)
    mode='absolute': val - truth                          [native weighted units]
    """
    # tm_key -> subcase_key -> {edge: vShutdown}
    grouped: dict = defaultdict(lambda: defaultdict(dict))
    for e in entries:
        edge = e.get('edge_handling')
        if edge is None:
            continue
        grouped[_tm_key(e)][_subcase_key(e)][edge] = e.get('vShutdown')

    boxes: dict = defaultdict(lambda: defaultdict(list))
    for tm_key, subcases in grouped.items():
        for _sk, eh_map in subcases.items():
            truth = eh_map.get('Truth')
            if truth is None:
                continue
            for edge in edges:
                val = eh_map.get(edge)
                if val is None:
                    continue
                diff = float(val) - float(truth)
                if mode == 'relative':
                    if truth == 0:
                        continue
                    boxes[tm_key][edge].append(diff / abs(float(truth)) * 100)
                else:
                    boxes[tm_key][edge].append(diff)
    return boxes


def build_regret_boxes(main_entries: list[dict], invest_regret_obj: dict,
                       edges: list[str]) -> dict:
    """Invest-regret as % cost increase over Truth's objective, per (tm, sub-case).

    regret% = (invest_regret_obj - truth_obj) / |truth_obj| * 100
    """
    grouped: dict = defaultdict(lambda: defaultdict(dict))
    for e in main_entries:
        edge = e.get('edge_handling')
        if edge is None:
            continue
        grouped[_tm_key(e)][_subcase_key(e)][edge] = e

    boxes: dict = defaultdict(lambda: defaultdict(list))
    for tm_key, subcases in grouped.items():
        for _sk, eh_map in subcases.items():
            truth = eh_map.get('Truth')
            truth_obj = truth.get('Objective') if truth else None
            if truth_obj in (None, 0, -1):
                continue
            for edge in edges:
                ent = eh_map.get(edge)
                if ent is None:
                    continue
                iregret = invest_regret_obj.get(ent['file'])
                if iregret is None:
                    continue
                boxes[tm_key][edge].append((iregret - truth_obj) / abs(truth_obj) * 100)
    return boxes


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _has_data(boxes: dict, edges: list[str]) -> bool:
    return any(boxes[t].get(e) for t in boxes for e in edges)


def make_boxplot_figure(boxes: dict, title: str, ylabel: str,
                        output: str | None, no_show: bool, edges: list[str],
                        logy: bool = False, ref_line: float | None = None,
                        symmetric_y: bool = False):
    """Render one figure: a subplot per TM combination, one edge box per subplot.

    *edges* selects which strategies to draw (and their left-to-right order).
    *ref_line*, if given, draws a horizontal reference line at that y value.
    *symmetric_y*, if True, centres the (shared) y-axis on 0 so the limits are
    equidistant from 0 in both directions — used for plots whose values can be
    negative or positive (deviation / regret).
    """
    if not _has_data(boxes, edges):
        printer.information(f"  [skip] {title}: no data")
        return None

    tm_keys = sorted(boxes.keys(), key=_tm_sort)
    n = len(tm_keys)
    fig, axes = plt.subplots(1, n, figsize=(max(2.6 * len(edges) * n / 3, 4.0), 5.0),
                             sharey=True, squeeze=False)
    axes = axes[0]

    positions = {edge: i + 1 for i, edge in enumerate(edges)}

    # Decide log scale only if every drawn value is strictly positive.
    use_log = logy and all(
        v > 0 for edge_map in boxes.values()
        for edge in edges for v in edge_map.get(edge, [])
    )

    for ax, tm_key in zip(axes, tm_keys):
        edge_map = boxes[tm_key]
        for edge in edges:
            data = edge_map.get(edge, [])
            if not data:
                continue
            bp = ax.boxplot([data], positions=[positions[edge]], widths=0.6,
                            patch_artist=True, showfliers=True)
            for patch in bp['boxes']:
                patch.set_facecolor(EDGE_COLORS[edge])
                patch.set_alpha(0.75)
            for med in bp['medians']:
                med.set_color('black')
            # Point count beneath the x-axis label region (top of axes, in axes fraction)
            ax.annotate(f"n={len(data)}", xy=(positions[edge], 0.99),
                        xycoords=('data', 'axes fraction'),
                        ha='center', va='top', fontsize=7, color='dimgray')

        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels(edges, fontsize=8)
        ax.set_xlim(0.5, len(edges) + 0.5)
        ax.set_title(_tm_label(*tm_key), fontsize=9, fontweight='bold')
        if use_log:
            ax.set_yscale('log')
        if ref_line is not None:
            ax.axhline(ref_line, color='black', linewidth=0.9, alpha=0.6)
        ax.grid(axis='y', alpha=0.3, linestyle=':')

    # Centre the shared y-axis on 0 for signed plots so equal magnitudes above
    # and below the reference read the same. (Log scale is always all-positive.)
    if symmetric_y and not use_log:
        all_vals = [v for tm_key in tm_keys for edge in edges
                    for v in boxes[tm_key].get(edge, [])]
        ymax = max((abs(v) for v in all_vals), default=0.0)
        if ymax > 0:
            axes[0].set_ylim(-ymax * 1.05, ymax * 1.05)  # sharey propagates to all

    axes[0].set_ylabel(ylabel)

    # No legend: each subplot's x-axis already labels the boxes and the colors
    # are consistent across all subplots.
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        printer.information(f"  Saved {output}")
    if not no_show:
        plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description=("Boxplots of edge-handling strategies (NoEnf, Cyclic, Markov) vs. the "
                                                  "Truth model across shift-tm / perturb-tm combinations.\n\n"
                                                  "For both operational and investment runs: work-units (absolute + as %% "
                                                  "of Truth) and vShutdown-deviation (relative + absolute), plus "
                                                  "invest-regret. Each plot is emitted twice: with all strategies and with "
                                                  "NoEnf excluded (suffix '_noNoEnf')."),
                                     formatter_class=RichHelpFormatter)
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing MK-*.sqlite files (default: current directory)")
    parser.add_argument("--output-dir", default=None, help="Directory to save the PNG files in (default: the input folder)")
    parser.add_argument("--no-show", action="store_true", help="Don't display the figures (useful for headless/batch runs)")
    parser.add_argument("--include-nonoptimal", action="store_true", help="Include runs whose termination_condition is not 'optimal' (default: optimal only)")
    parser.add_argument("--logscale", action="store_true", help="Use a log-scale y-axis for the work-units plots (default: linear). Has no effect on the deviation/regret plots (those can be negative).")
    parser.add_argument("--markov-strict", action="store_true", help="Also include the Markov-Strict strategy as a box in every plot (only meaningful if runs were produced with --enable-strict-markov).")
    args = parser.parse_args()

    out_dir = args.output_dir or args.folder
    os.makedirs(out_dir, exist_ok=True)

    edges = list(EDGE_ORDER) + ([EDGE_STRICT] if args.markov_strict else [])

    printer.information(f"Loading MK-*.sqlite files from '{args.folder}' ...")
    main_entries, operational_entries, invest_regret_obj = load_all(args.folder, include_nonoptimal=args.include_nonoptimal)

    if not main_entries and not operational_entries:
        printer.warning(f"  No usable MK-*.sqlite files found in '{args.folder}'"
                        + ("" if args.include_nonoptimal else " (after filtering to optimal runs)"))
        return

    def _count(entries, edge):
        return sum(1 for e in entries if e.get('edge_handling') == edge)

    filt = "all" if args.include_nonoptimal else "optimal-only"
    strict_note = (f", Markov-Strict={_count(main_entries, EDGE_STRICT)}"
                   if args.markov_strict else "")
    printer.information(f"  Loaded [{filt}]: "
                        f"main={len(main_entries)} (Truth={_count(main_entries, 'Truth')}, "
                        f"NoEnf={_count(main_entries, 'NoEnf')}, Cyclic={_count(main_entries, 'Cyclic')}, "
                        f"Markov={_count(main_entries, 'Markov')}{strict_note}); "
                        f"operational={len(operational_entries)}; "
                        f"invest-regret={len(invest_regret_obj)}")

    def emit(boxes, title, ylabel, name_key, logy=False, ref_line=None,
             symmetric_y=False):
        """Render the full figure and a NoEnf-excluded twin ('_noNoEnf' suffix)."""
        fname = OUTPUT_NAMES[name_key]
        stem, ext = os.path.splitext(fname)
        make_boxplot_figure(boxes, title, ylabel, os.path.join(out_dir, fname),
                            args.no_show, edges, logy=logy, ref_line=ref_line,
                            symmetric_y=symmetric_y)
        sub_edges = [e for e in edges if e != 'NoEnf']
        make_boxplot_figure(boxes, f"{title} (excl. NoEnf)", ylabel,
                            os.path.join(out_dir, f"{stem}_noNoEnf{ext}"),
                            args.no_show, sub_edges, logy=logy, ref_line=ref_line,
                            symmetric_y=symmetric_y)

    # --- Operational runs ---
    emit(build_runtime_boxes(operational_entries, edges),
         "Work units — Operational runs", "Work units",
         'workunits_operational_abs', logy=args.logscale)
    emit(build_runtime_relative_boxes(operational_entries, edges),
         "Work units (% of Truth) — Operational runs", "Work units [% of Truth]",
         'workunits_operational_rel', logy=args.logscale)
    emit(build_deviation_boxes(operational_entries, 'relative', edges),
         "vShutdown deviation vs Truth — Operational runs",
         "Relative deviation from Truth [%]",
         'vshutdown_operational_rel', ref_line=0, symmetric_y=True)
    emit(build_deviation_boxes(operational_entries, 'absolute', edges),
         "vShutdown deviation vs Truth — Operational runs",
         "Absolute deviation from Truth",
         'vshutdown_operational_abs', ref_line=0, symmetric_y=True)

    # --- Investment (main) runs ---
    emit(build_runtime_boxes(main_entries, edges),
         "Work units — Investment runs", "Work units",
         'workunits_investment_abs', logy=args.logscale)
    emit(build_runtime_relative_boxes(main_entries, edges),
         "Work units (% of Truth) — Investment runs", "Work units [% of Truth]",
         'workunits_investment_rel', logy=args.logscale)
    emit(build_deviation_boxes(main_entries, 'relative', edges),
         "vShutdown deviation vs Truth — Investment runs",
         "Relative deviation from Truth [%]",
         'vshutdown_investment_rel', ref_line=0, symmetric_y=True)
    emit(build_deviation_boxes(main_entries, 'absolute', edges),
         "vShutdown deviation vs Truth — Investment runs",
         "Absolute deviation from Truth",
         'vshutdown_investment_abs', ref_line=0, symmetric_y=True)

    # --- Invest-regret (investment runs) ---
    emit(build_regret_boxes(main_entries, invest_regret_obj, edges),
         "Invest-regret vs Truth — Investment runs",
         "Invest-regret over Truth objective [%]",
         'invest_regret', ref_line=0, symmetric_y=True)


if __name__ == "__main__":
    main()
