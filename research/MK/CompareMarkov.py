#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CompareMarkov.py - Boxplots comparing edge-handling strategies (NoEnf, Cyclic,
Markov) against the Truth model across shift-tm / perturb-tm combinations.

Produces 18 logical plots from a folder of MK-*.sqlite files; each is emitted
twice — once with all strategies and once with NoEnf excluded ('_noNoEnf'
suffix, since NoEnf's large deviations otherwise compress the scale) — for up
to 36 PNGs:

  A  compare_workunits_operational_absolute.png   Work units, operational runs
     compare_workunits_operational_relative.png   Work units as % of Truth, operational
  B  compare_vshutdown_operational_absolute.png   vShutdown dev. vs Truth-op (abs)
     compare_vshutdown_operational_relative.png   vShutdown dev. vs Truth-op [%]
     compare_vshutdown_operational_absolute_magnitude.png  |vShutdown dev.| vs Truth-op (abs)
     compare_vshutdown_operational_relative_magnitude.png  |vShutdown dev.| vs Truth-op [%]
  C  compare_workunits_investment_absolute.png    Work units, investment (main) runs
     compare_workunits_investment_relative.png    Work units as % of Truth, investment
  D  compare_vshutdown_investment_absolute.png    vShutdown dev. vs Truth-main (abs)
     compare_vshutdown_investment_relative.png    vShutdown dev. vs Truth-main [%]
     compare_vshutdown_investment_absolute_magnitude.png   |vShutdown dev.| vs Truth-main (abs)
     compare_vshutdown_investment_relative_magnitude.png   |vShutdown dev.| vs Truth-main [%]
  E  compare_invest_regret_absolute.png           Invest-regret (abs) vs Truth-main obj.
     compare_invest_regret_relative.png           Invest-regret [%] vs Truth-main obj.
  F  compare_regret_absolute.png                  Regret (abs) vs Truth-main obj. (invest+commit fixed)
     compare_regret_relative.png                  Regret [%] vs Truth-main obj.
  G  compare_operational_regret_absolute.png      Operational-regret (abs) vs Truth-operational obj.
     compare_operational_regret_relative.png      Operational-regret [%] vs Truth-operational obj.

Each figure has one subplot per (shift_tm, perturb_tm) combination (shared
y-axis), and within each subplot one boxplot per edge handling (NoEnf, Cyclic,
Markov; +Markov-Strict with --markov-strict). Every box aggregates over the
*sub-cases* that share that TM combination — i.e. the other run parameters that
vary (clusters, stretch_demand, ...). Truth is the reference for
deviation/regret, never a box.

The B/D "_magnitude" plots show |deviation| (or |relative deviation|) instead
of the signed value — useful for comparing how much Cyclic vs. Markov deviate
from Truth in size, without sign direction obscuring the comparison. Their
y-axis is clamped to [0, max*1.05] (never negative), unlike the signed B/D
plots, whose y-axis is symmetric around 0. The E/F/G regret plots' y-axis is
"tight" instead: [min(0, min*1.05), max*1.05] — regret is expected to be
non-negative (fixing variables can only raise the objective), so the axis only
dips as far negative as needed to show actual solver/MIP-gap noise, rather
than mirroring the full positive range.

"Operational runs" are the `--operational` runs (vGenInvest fixed to Truth's
investment); "investment runs" are the regular main runs. Work-units plots need
a solver that reports work units (Gurobi) and are empty otherwise (e.g. HiGHS).

By default only runs with termination_condition == 'optimal' are included
(--include-nonoptimal to override).

Results table
-------------
Alongside the PNGs, the mean/median/min/max *behind* each boxplot are aggregated
per (TM_variant, method) cell into a numeric results table (for a paper). It is
printed to the terminal and saved as 'compare_markov_results.txt' (+ '.csv') in
the output dir. TM_variant is the (shift_tm, perturb_tm) axis ('Original' = both
unset, 'ShiftN' = shift_tm=N); each cell aggregates over the sub-cases sharing
that combination (clusters/RPs x demand x ...). Columns: operational vShutdown
deviation [%], invest-regret [M EUR] (the LEGO objective is already in M EUR),
and operational/investment work units [% of Truth]. Disable with
--no-results-table.

Filtering / splitting
---------------------
--nrOfClusters 3,5,7      keep only runs whose `clusters` run-parameter is 3, 5 or 7
--separateClusters        emit the full plot set once per cluster count
                          (filename suffix `_clusters{N}`)
--tm SHIFT:PERTURB        show only the selected (shift_tm, perturb_tm) subplots;
                          repeatable and/or comma-separated, each side a number,
                          'none' (parameter unset) or '*' (any); 'base' is
                          shorthand for none:none

Usage
-----
python research/MK/CompareMarkov.py [folder]
python research/MK/CompareMarkov.py results/ --output-dir plots/ --no-show
python research/MK/CompareMarkov.py results/ --include-nonoptimal
python research/MK/CompareMarkov.py results/ --nrOfClusters 3,5,7 --separateClusters
python research/MK/CompareMarkov.py results/ --tm none:0.2 --tm 1:none --tm base
python research/MK/CompareMarkov.py results/ --tm "1:*,2:*"
"""

import argparse
import csv
import glob
import os
import sqlite3
import statistics
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

# Pretty x-axis labels for the boxes (the internal edge keys stay as-is for
# lookups/colors); only entries listed here are relabelled, the rest show verbatim.
EDGE_LABELS = {'NoEnf': 'No Enf.'}

# Colour of the MIP-gap noise band/line drawn on the invest-regret plots.
GAP_COLOR = '#ff6666'  # light red

# Base filename per logical plot (the "(excl. NoEnf)" variant inserts "_noNoEnf").
OUTPUT_NAMES = {
    'workunits_operational_abs': 'compare_workunits_operational_absolute.png',
    'workunits_operational_rel': 'compare_workunits_operational_relative.png',
    'vshutdown_operational_abs': 'compare_vshutdown_operational_absolute.png',
    'vshutdown_operational_rel': 'compare_vshutdown_operational_relative.png',
    'vshutdown_operational_abs_mag': 'compare_vshutdown_operational_absolute_magnitude.png',
    'vshutdown_operational_rel_mag': 'compare_vshutdown_operational_relative_magnitude.png',
    'workunits_investment_abs': 'compare_workunits_investment_absolute.png',
    'workunits_investment_rel': 'compare_workunits_investment_relative.png',
    'vshutdown_investment_abs': 'compare_vshutdown_investment_absolute.png',
    'vshutdown_investment_rel': 'compare_vshutdown_investment_relative.png',
    'vshutdown_investment_abs_mag': 'compare_vshutdown_investment_absolute_magnitude.png',
    'vshutdown_investment_rel_mag': 'compare_vshutdown_investment_relative_magnitude.png',
    'invest_regret_abs': 'compare_invest_regret_absolute.png',
    'invest_regret_rel': 'compare_invest_regret_relative.png',
    'regret_abs': 'compare_regret_absolute.png',
    'regret_rel': 'compare_regret_relative.png',
    'operational_regret_abs': 'compare_operational_regret_absolute.png',
    'operational_regret_rel': 'compare_operational_regret_relative.png',
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
        ('invest_regret'|'regret'|'operational_regret', file, termination, objective)
        ('main'|'operational', file, meta, payload)
    where payload = {'vShutdown': float|None, 'Objective': float|None}.
    """
    basename = os.path.basename(sqlite_file)

    # Regret-objective files. Ordering matters: "-invest-regret.sqlite" and
    # "-operational-regret.sqlite" both end in "-regret.sqlite", so both are tested before
    # the bare "-regret.sqlite". Each yields just (termination, objective).
    regret_kind = None
    if basename.endswith("-invest-regret.sqlite"):
        regret_kind = 'invest_regret'
    elif basename.endswith("-operational-regret.sqlite"):
        regret_kind = 'operational_regret'
    elif basename.endswith("-regret.sqlite"):
        regret_kind = 'regret'
    if regret_kind is not None:
        try:
            conn = sqlite3.connect(sqlite_file)
            obj = _read_objective(conn)
            term = _read_termination(conn)
            conn.close()
        except Exception:
            obj, term = None, None
        return (regret_kind, sqlite_file, term, obj)

    kind = 'operational' if basename.endswith("-operational.sqlite") else 'main'
    try:
        conn = sqlite3.connect(sqlite_file)
        meta = _load_metadata_from_conn(conn, basename)
        vshut = _weighted_var_sum(conn, 'vShutdown')
        # Objective is read for operational too (it is the Truth-operational reference for
        # the operational-regret plots; main runs reference it for the regret plots).
        obj = _read_objective(conn)
        conn.close()
    except Exception:
        return ('skip', sqlite_file, None, None)

    return (kind, sqlite_file, meta, {'vShutdown': vshut, 'Objective': obj})


def load_all(folder: str, include_nonoptimal: bool = False):
    """Discover and load all MK-*.sqlite files in *folder* concurrently.

    Returns (main_entries, operational_entries, invest_regret_obj, regret_obj,
    operational_regret_obj). Each *_obj map is keyed by the base file it references:
      invest_regret_obj / regret_obj  -> base MAIN file        -> objective
      operational_regret_obj          -> base OPERATIONAL file -> objective
    """
    pattern = os.path.join(folder, "MK-*.sqlite")
    files = sorted(f for f in glob.glob(pattern) if os.path.basename(f).startswith("MK-"))
    max_workers = min(len(files), (os.cpu_count() or 4), 60)

    printer.information(f"Found {len(files)} MK-*.sqlite files in '{folder}', loading with up to {max_workers} threads...")

    main_entries: list[dict] = []
    operational_entries: list[dict] = []
    # Each regret map is paired with a termination map (same key) for the optimal-only filter.
    invest_regret_obj: dict[str, float] = {}
    invest_regret_term: dict[str, str | None] = {}
    regret_obj: dict[str, float] = {}
    regret_term: dict[str, str | None] = {}
    operational_regret_obj: dict[str, float] = {}
    operational_regret_term: dict[str, str | None] = {}

    if not files:
        return main_entries, operational_entries, invest_regret_obj, regret_obj, operational_regret_obj

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_load_compare_file, f) for f in files]
        for future in as_completed(futures):
            kind, f, meta, payload = future.result()  # for regret kinds: meta=termination, payload=objective
            if kind == 'skip':
                continue
            if kind == 'invest_regret':
                base = f.replace("-invest-regret.sqlite", ".sqlite")
                invest_regret_obj[base] = payload
                invest_regret_term[base] = meta
            elif kind == 'regret':
                base = f.replace("-regret.sqlite", ".sqlite")
                regret_obj[base] = payload
                regret_term[base] = meta
            elif kind == 'operational_regret':
                base = f.replace("-operational-regret.sqlite", "-operational.sqlite")
                operational_regret_obj[base] = payload
                operational_regret_term[base] = meta
            else:
                entry = {**meta, **payload, 'file': f}
                (operational_entries if kind == 'operational' else main_entries).append(entry)

    if not include_nonoptimal:
        main_entries = [e for e in main_entries if e.get('termination_condition') == 'optimal']
        operational_entries = [e for e in operational_entries if e.get('termination_condition') == 'optimal']
        invest_regret_obj = {b: o for b, o in invest_regret_obj.items() if invest_regret_term.get(b) == 'optimal'}
        regret_obj = {b: o for b, o in regret_obj.items() if regret_term.get(b) == 'optimal'}
        operational_regret_obj = {b: o for b, o in operational_regret_obj.items()
                                  if operational_regret_term.get(b) == 'optimal'}

    return main_entries, operational_entries, invest_regret_obj, regret_obj, operational_regret_obj


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def _tm_value(v) -> float | None:
    """Normalize a stored shift_tm/perturb_tm value to float, or None when unset."""
    if v in (None, 'None'):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _tm_key(entry: dict) -> tuple:
    return (entry.get('shift_tm'), entry.get('perturb_tm'))


def _subcase_key(entry: dict) -> tuple:
    return tuple(entry.get(k) for k in SUB_CASE_KEYS)


def _tm_sort(tm_key: tuple):
    """Sort TM combinations by shift first, then perturb; None (unset) sorts first.

    Gives: base, perturb0.2, shift1, shift1+perturb0.2, shift2, shift2+perturb0.2, ...
    """
    shift, perturb = (_tm_value(v) for v in tm_key)
    return (shift if shift is not None else float('-inf'),
            perturb if perturb is not None else float('-inf'))


def _tm_label(shift_tm, perturb_tm, base_label: str = '(base)') -> str:
    parts = []
    shift, perturb = _tm_value(shift_tm), _tm_value(perturb_tm)
    if shift is not None:
        parts.append(f"Transition Matrix\nshifted by {shift:g}")
    if perturb is not None:
        parts.append(f"perturbTM={perturb:g}")
    # The unperturbed ("base") subplot is labelled with the dataset name instead.
    return ', '.join(parts) if parts else base_label


def _data_label(entries: list[dict]) -> str:
    """Dataset name(s) behind *entries*, from their `case_study_directory` basename.

    Used as the subplot title for the unperturbed ('base') TM combination — e.g.
    'RTS-GMLC' when the runs came from data/RTS-GMLC. Multiple distinct datasets
    are joined with ' / '; falls back to '(base)' when none is recorded.
    """
    names = set()
    for e in entries:
        d = e.get('case_study_directory')
        if d:
            names.add(str(d).replace('\\', '/').rstrip('/').rsplit('/', 1)[-1])
    return ' / '.join(sorted(names)) if names else '(base)'


# ---------------------------------------------------------------------------
# Entry filters (--nrOfClusters / --tm)
# ---------------------------------------------------------------------------

def parse_tm_spec(spec: str) -> tuple:
    """Parse one --tm spec into a (shift, perturb) pair.

    Each side is a float, None ('none'/'-' = parameter unset) or '*' (any value);
    'base' is shorthand for none:none. Raises ValueError on malformed input.
    """
    if spec.strip().lower() == 'base':
        return (None, None)
    parts = spec.split(':')
    if len(parts) != 2:
        raise ValueError(f"invalid TM spec '{spec}' (expected SHIFT:PERTURB or 'base')")
    sides = []
    for side in parts:
        side = side.strip().lower()
        if side == '*':
            sides.append('*')
        elif side in ('none', '-', ''):
            sides.append(None)
        else:
            try:
                sides.append(float(side))
            except ValueError:
                raise ValueError(f"invalid TM spec '{spec}': '{side}' is not a number, 'none' or '*'")
    return tuple(sides)


def _matches_tm(entry: dict, tm_specs: list[tuple]) -> bool:
    """True if the entry's (shift_tm, perturb_tm) matches any spec (or no specs given)."""
    if not tm_specs:
        return True
    shift, perturb = _tm_value(entry.get('shift_tm')), _tm_value(entry.get('perturb_tm'))
    return any((s == '*' or s == shift) and (p == '*' or p == perturb)
               for s, p in tm_specs)


def filter_entries(entries: list[dict], cluster_filter: set[int] | None,
                   tm_specs: list[tuple]) -> list[dict]:
    return [e for e in entries
            if (cluster_filter is None or e.get('clusters') in cluster_filter)
            and _matches_tm(e, tm_specs)]


# ---------------------------------------------------------------------------
# Box builders — each returns {tm_key: {edge: [values]}}
# ---------------------------------------------------------------------------

def _iter_vs_truth(entries: list[dict], edges: list[str], edge_value, truth_value=None):
    """Group entries by (tm_key, sub-case) and pair each edge with its Truth reference.

    Yields (tm_key, edge, truth_ref, value) for every edge entry with a non-None
    *edge_value* in a sub-case that has a Truth entry. *truth_value* extracts the
    reference from the Truth entry (defaults to *edge_value*); callers decide
    which truth_ref values to skip.
    """
    truth_value = truth_value or edge_value
    grouped: dict = defaultdict(lambda: defaultdict(dict))
    for e in entries:
        edge = e.get('edge_handling')
        if edge is not None:
            grouped[_tm_key(e)][_subcase_key(e)][edge] = e
    for tm_key, subcases in grouped.items():
        for eh_map in subcases.values():
            truth = eh_map.get('Truth')
            if truth is None:
                continue
            truth_ref = truth_value(truth)
            for edge in edges:
                ent = eh_map.get(edge)
                if ent is None:
                    continue
                val = edge_value(ent)
                if val is not None:
                    yield tm_key, edge, truth_ref, val


def _diff_boxes(pairs, mode: str, truth_skip: tuple, magnitude: bool = False) -> dict:
    """Difference-to-Truth boxes from an _iter_vs_truth stream.

    mode='relative': (val - truth) / |truth| * 100  [%]   (skips truth==0)
    mode='absolute': val - truth                          [native units]
    Sub-cases whose truth_ref is in *truth_skip* are dropped in both modes.
    *magnitude*, if True, stores abs(diff) instead of the signed diff — used for
    the "|deviation|" plots, so Cyclic/Markov's typical deviation size can be
    compared without over/under-shooting sign obscuring the comparison.
    """
    boxes: dict = defaultdict(lambda: defaultdict(list))
    for tm_key, edge, truth, val in pairs:
        if truth in truth_skip:
            continue
        diff = float(val) - float(truth)
        if mode == 'relative':
            if truth == 0:
                continue
            diff = diff / abs(float(truth)) * 100
        boxes[tm_key][edge].append(abs(diff) if magnitude else diff)
    return boxes


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
    boxes: dict = defaultdict(lambda: defaultdict(list))
    for tm_key, edge, truth, val in _iter_vs_truth(entries, edges,
                                                   lambda e: e.get('work_units')):
        if truth in (None, 0):
            continue
        boxes[tm_key][edge].append(float(val) / float(truth) * 100)
    return boxes


def build_deviation_boxes(entries: list[dict], mode: str, edges: list[str],
                          magnitude: bool = False) -> dict:
    """vShutdown deviation from the Truth entry of the same (tm, sub-case)."""
    return _diff_boxes(
        _iter_vs_truth(entries, edges, lambda e: e.get('vShutdown')),
        mode, truth_skip=(None,), magnitude=magnitude)


def build_regret_boxes(base_entries: list[dict], regret_obj: dict,
                       mode: str, edges: list[str]) -> dict:
    """Regret cost increase over the same-kind Truth objective, per (tm, sub-case).

    Generic over the three regret variants — the caller picks the pair:
      invest-regret / regret -> base_entries = main runs (vs Truth-main objective)
      operational-regret      -> base_entries = operational runs (vs Truth-operational objective)
    `regret_obj` is keyed by each base entry's own `file`, so the reference Truth and the
    regret value always come from the same run kind.

    Sub-cases whose Truth objective is missing / 0 / -1 (the EvaluateMarkov "no
    objective" sentinel) are skipped in both modes, so the relative and absolute
    plots aggregate over the same sub-cases.
    """
    return _diff_boxes(
        _iter_vs_truth(base_entries, edges,
                       edge_value=lambda e: regret_obj.get(e['file']),
                       truth_value=lambda e: e.get('Objective')),
        mode, truth_skip=(None, 0, -1))


def invest_regret_gap_band(main_entries: list[dict], invest_regret_obj: dict,
                           mode: str, edges: list[str]) -> tuple[float, float] | None:
    """Magnitude range of the MIP-gap "noise band" for the invest-regret plots.

    Every plotted invest-regret point is only known to within the solver's
    (requested) MIP gap, so a regret smaller than `mip_gap * objective` could be
    attributable to solver tolerance alone. We collect that magnitude across the
    *same* points the boxes are built from (same sub-case skips as
    `build_regret_boxes`, restricted to the drawn *edges*) and return `(lo, hi)`:

      absolute: lo/hi = min/max of `mip_gap * |invest-regret objective|` — a band,
                since the plotted results' objectives differ between sub-cases.
      relative: lo/hi = min/max of `mip_gap * 100` — collapses to a single line
                when the requested mip_gap is uniform (it is in percent already).

    The caller draws an upper band/line at `[lo, hi]` and a mirrored lower one at
    `[-hi, -lo]`. Returns None when no plotted point carries a usable mip_gap.
    """
    grouped: dict = defaultdict(lambda: defaultdict(dict))
    for e in main_entries:
        edge = e.get('edge_handling')
        if edge is not None:
            grouped[_tm_key(e)][_subcase_key(e)][edge] = e

    mags: list[float] = []
    for subcases in grouped.values():
        for eh_map in subcases.values():
            truth = eh_map.get('Truth')
            if truth is None or truth.get('Objective') in (None, 0, -1):
                continue
            for edge in edges:
                ent = eh_map.get(edge)
                if ent is None:
                    continue
                obj = invest_regret_obj.get(ent['file'])
                gap = ent.get('mip_gap')
                if obj is None or gap is None:
                    continue
                mags.append(gap * 100.0 if mode == 'relative' else gap * abs(float(obj)))

    return (min(mags), max(mags)) if mags else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _has_data(boxes: dict, edges: list[str]) -> bool:
    return any(boxes[t].get(e) for t in boxes for e in edges)


def make_boxplot_figure(boxes: dict, title: str, ylabel: str,
                        output: str | None, no_show: bool, edges: list[str],
                        logy: bool = False, ref_line: float | None = None,
                        symmetric_y: bool = False, nonneg_y: bool = False,
                        tight_y: bool = False,
                        gap_band: tuple[float, float] | None = None,
                        data_label: str = '(base)'):
    """Render one figure: a subplot per TM combination, one edge box per subplot.

    *edges* selects which strategies to draw (and their left-to-right order).
    *ref_line*, if given, draws a horizontal reference line at that y value.
    *symmetric_y*, if True, centres the (shared) y-axis on 0 so the limits are
    equidistant from 0 in both directions — used for plots whose values can be
    negative or positive (deviation / regret).
    *nonneg_y*, if True, clamps the (shared) y-axis to [0, max*1.05] — used for
    magnitude (absolute-value) plots, whose values are never negative, so no
    vertical space should be wasted below 0.
    *tight_y*, if True, sets the (shared) y-axis to [min(0, min*1.05), max*1.05]
    — used for regret plots, which are expected to be non-negative but can dip
    slightly negative from solver/MIP-gap noise; this shows just enough of the
    negative side to capture those outliers instead of mirroring the full
    positive range (as *symmetric_y* would).
    At most one of *symmetric_y* / *nonneg_y* / *tight_y* should be set.
    *gap_band* `(lo, hi)`, if given, draws a light-red MIP-gap noise band on every
    subplot: a filled band at `[lo, hi]` and a mirrored one at `[-hi, -lo]`. When
    `lo == hi` (e.g. the relative invest-regret plot, where the band is in percent)
    it degenerates to a pair of horizontal lines at `±lo` instead of filled bands.
    *data_label* titles the unperturbed ('base') TM subplot (the dataset name).

    The figure has no overall (sup)title — the per-subplot titles carry context.
    The figure is closed before returning (relevant for --separateClusters,
    which renders many figures in one process).
    """
    if not _has_data(boxes, edges):
        printer.information(f"  [skip] {title}: no data")
        return

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
        ax.set_xticklabels([EDGE_LABELS.get(e, e) for e in edges], fontsize=8)
        ax.set_xlim(0.5, len(edges) + 0.5)
        ax.set_title(_tm_label(*tm_key, base_label=data_label), fontsize=9, fontweight='bold')
        if use_log:
            ax.set_yscale('log')
        if ref_line is not None:
            ax.axhline(ref_line, color='black', linewidth=0.9, alpha=0.6)
        if gap_band is not None and not use_log:
            lo, hi = gap_band
            if hi - lo < 1e-9:  # degenerate band -> a pair of lines (relative plot)
                for y in (lo, -lo):
                    ax.axhline(y, color=GAP_COLOR, linewidth=1.1, alpha=0.9, zorder=0.5)
            else:
                ax.axhspan(lo, hi, color=GAP_COLOR, alpha=0.25, lw=0, zorder=0)
                ax.axhspan(-hi, -lo, color=GAP_COLOR, alpha=0.25, lw=0, zorder=0)
        ax.grid(axis='y', alpha=0.3, linestyle=':')

    # Centre the shared y-axis on 0 for signed plots so equal magnitudes above
    # and below the reference read the same. (Log scale is always all-positive.)
    if symmetric_y and not use_log:
        all_vals = [v for tm_key in tm_keys for edge in edges
                    for v in boxes[tm_key].get(edge, [])]
        ymax = max((abs(v) for v in all_vals), default=0.0)
        if gap_band is not None:  # keep the band/line in view even if regrets are tiny
            ymax = max(ymax, abs(gap_band[1]))
        if ymax > 0:
            axes[0].set_ylim(-ymax * 1.05, ymax * 1.05)  # sharey propagates to all
    elif nonneg_y and not use_log:
        all_vals = [v for tm_key in tm_keys for edge in edges
                    for v in boxes[tm_key].get(edge, [])]
        ymax = max((abs(v) for v in all_vals), default=0.0)
        if gap_band is not None:
            ymax = max(ymax, abs(gap_band[1]))
        if ymax > 0:
            axes[0].set_ylim(0, ymax * 1.05)  # sharey propagates to all
    elif tight_y and not use_log:
        all_vals = [v for tm_key in tm_keys for edge in edges
                    for v in boxes[tm_key].get(edge, [])]
        vmin = min((v for v in all_vals), default=0.0)
        vmax = max((v for v in all_vals), default=0.0)
        lo = vmin * 1.05 if vmin < 0 else 0.0
        hi = vmax * 1.05 if vmax > 0 else 0.0
        if gap_band is not None:  # keep the band/line in view even if regrets are tiny
            lo = min(lo, -gap_band[1])
            hi = max(hi, gap_band[1])
        if lo < 0 or hi > 0:
            axes[0].set_ylim(lo, hi)  # sharey propagates to all

    axes[0].set_ylabel(ylabel)

    # No legend: each subplot's x-axis already labels the boxes and the colors
    # are consistent across all subplots. No figure title either — the subplot
    # titles (dataset name / TM perturbation) carry the context.
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        printer.information(f"  Saved {output}")
    if not no_show:
        plt.show()
    plt.close(fig)


def render_plots(main_entries: list[dict], operational_entries: list[dict],
                 invest_regret_obj: dict, regret_obj: dict, operational_regret_obj: dict,
                 edges: list[str], out_dir: str, args,
                 fname_suffix: str = "", title_suffix: str = ""):
    """Emit the full set of 14 logical plots (each with its '_noNoEnf' twin).

    *fname_suffix* / *title_suffix* tag per-cluster outputs under --separateClusters.
    """
    # The 'base' (unperturbed TM) subplot is titled with the dataset name.
    data_label = _data_label(main_entries + operational_entries)

    def emit(boxes, title, ylabel, name_key, logy=False, ref_line=None,
             symmetric_y=False, nonneg_y=False, tight_y=False, gap_fn=None):
        """Render the full figure and a NoEnf-excluded twin ('_noNoEnf' suffix).

        *gap_fn*, if given, is called with the edge list actually drawn (full vs
        twin) and returns the `(lo, hi)` MIP-gap band for that figure (or None).
        """
        stem, ext = os.path.splitext(OUTPUT_NAMES[name_key])
        stem += fname_suffix
        title += title_suffix
        make_boxplot_figure(boxes, title, ylabel, os.path.join(out_dir, stem + ext),
                            args.no_show, edges, logy=logy, ref_line=ref_line,
                            symmetric_y=symmetric_y, nonneg_y=nonneg_y, tight_y=tight_y,
                            gap_band=gap_fn(edges) if gap_fn else None,
                            data_label=data_label)
        sub_edges = [e for e in edges if e != 'NoEnf']
        make_boxplot_figure(boxes, f"{title} (excl. NoEnf)", ylabel,
                            os.path.join(out_dir, f"{stem}_noNoEnf{ext}"),
                            args.no_show, sub_edges, logy=logy, ref_line=ref_line,
                            symmetric_y=symmetric_y, nonneg_y=nonneg_y, tight_y=tight_y,
                            gap_band=gap_fn(sub_edges) if gap_fn else None,
                            data_label=data_label)

    # --- Operational runs ---
    emit(build_runtime_boxes(operational_entries, edges),
         "Work units — Operational runs", "Work units",
         'workunits_operational_abs', logy=args.logscale)
    emit(build_runtime_relative_boxes(operational_entries, edges),
         "Work units (% of Truth) — Operational runs", "Work units [% of Truth]",
         'workunits_operational_rel', logy=args.logscale)
    emit(build_deviation_boxes(operational_entries, 'absolute', edges),
         "vShutdown deviation vs Truth — Operational runs",
         "Absolute deviation from Truth",
         'vshutdown_operational_abs', ref_line=0, symmetric_y=True)
    emit(build_deviation_boxes(operational_entries, 'relative', edges),
         "vShutdown deviation vs Truth — Operational runs",
         "Relative deviation from Truth [%]",
         'vshutdown_operational_rel', ref_line=0, symmetric_y=True)
    emit(build_deviation_boxes(operational_entries, 'absolute', edges, magnitude=True),
         "vShutdown |deviation| vs Truth — Operational runs",
         "Absolute value of deviation from Truth",
         'vshutdown_operational_abs_mag', nonneg_y=True)
    emit(build_deviation_boxes(operational_entries, 'relative', edges, magnitude=True),
         "vShutdown |deviation| vs Truth — Operational runs",
         "Absolute value of relative deviation from Truth [%]",
         'vshutdown_operational_rel_mag', nonneg_y=True)

    # --- Investment (main) runs ---
    emit(build_runtime_boxes(main_entries, edges),
         "Work units — Investment runs", "Work units",
         'workunits_investment_abs', logy=args.logscale)
    emit(build_runtime_relative_boxes(main_entries, edges),
         "Work units (% of Truth) — Investment runs", "Work units [% of Truth]",
         'workunits_investment_rel', logy=args.logscale)
    emit(build_deviation_boxes(main_entries, 'absolute', edges),
         "vShutdown deviation vs Truth — Investment runs",
         "Absolute deviation from Truth",
         'vshutdown_investment_abs', ref_line=0, symmetric_y=True)
    emit(build_deviation_boxes(main_entries, 'relative', edges),
         "vShutdown deviation vs Truth — Investment runs",
         "Relative deviation from Truth [%]",
         'vshutdown_investment_rel', ref_line=0, symmetric_y=True)
    emit(build_deviation_boxes(main_entries, 'absolute', edges, magnitude=True),
         "vShutdown |deviation| vs Truth — Investment runs",
         "Absolute value of deviation from Truth",
         'vshutdown_investment_abs_mag', nonneg_y=True)
    emit(build_deviation_boxes(main_entries, 'relative', edges, magnitude=True),
         "vShutdown |deviation| vs Truth — Investment runs",
         "Absolute value of relative deviation from Truth [%]",
         'vshutdown_investment_rel_mag', nonneg_y=True)

    # --- Invest-regret (investment runs): vGenInvest fixed from the main run, vCommit free ---
    emit(build_regret_boxes(main_entries, invest_regret_obj, 'absolute', edges),
         "Invest-regret vs Truth — Investment runs",
         "Absolute invest-regret over Truth objective",
         'invest_regret_abs', ref_line=0, tight_y=True,
         gap_fn=lambda eds: invest_regret_gap_band(main_entries, invest_regret_obj, 'absolute', eds))
    emit(build_regret_boxes(main_entries, invest_regret_obj, 'relative', edges),
         "Invest-regret vs Truth — Investment runs",
         "Invest-regret over Truth objective [%]",
         'invest_regret_rel', ref_line=0, tight_y=True,
         gap_fn=lambda eds: invest_regret_gap_band(main_entries, invest_regret_obj, 'relative', eds))

    # --- Regret (investment runs): vGenInvest + vCommit fixed from the main run; vs Truth-main ---
    emit(build_regret_boxes(main_entries, regret_obj, 'absolute', edges),
         "Regret vs Truth — Investment runs",
         "Absolute regret over Truth objective",
         'regret_abs', ref_line=0, tight_y=True)
    emit(build_regret_boxes(main_entries, regret_obj, 'relative', edges),
         "Regret vs Truth — Investment runs",
         "Regret over Truth objective [%]",
         'regret_rel', ref_line=0, tight_y=True)

    # --- Operational-regret (operational runs): fleet = Truth, vCommit fixed from the
    #     operational run; referenced against the Truth-operational objective ---
    emit(build_regret_boxes(operational_entries, operational_regret_obj, 'absolute', edges),
         "Operational-regret vs Truth — Operational runs",
         "Absolute operational-regret over Truth-operational objective",
         'operational_regret_abs', ref_line=0, tight_y=True)
    emit(build_regret_boxes(operational_entries, operational_regret_obj, 'relative', edges),
         "Operational-regret vs Truth — Operational runs",
         "Operational-regret over Truth-operational objective [%]",
         'operational_regret_rel', ref_line=0, tight_y=True)


# ---------------------------------------------------------------------------
# Aggregated results table — the mean/median/min/max *behind* each boxplot,
# grouped by (TM_variant, method), for dropping into a paper results table.
# Printed to the terminal and saved to a .txt (and .csv) alongside the PNGs.
# ---------------------------------------------------------------------------

# Method (edge handling) order within each TM_variant block of the table.
EDGE_DISPLAY_ORDER = ['NoEnf', 'Cyclic', 'Markov', 'Markov-Strict']


def tm_variant_label(tm_key: tuple) -> str:
    """(shift_tm, perturb_tm) -> paper TM-variant name ('Original', 'Shift1', ...)."""
    shift, perturb = (_tm_value(v) for v in tm_key)
    base = "Original" if shift is None else f"Shift{shift:g}"
    if perturb is not None:
        base += f"+perturb{perturb:g}"
    return base


def _agg(vals: list[float] | None) -> dict | None:
    """mean / median / min / max / n of a box's values (None when empty)."""
    if not vals:
        return None
    return {
        'mean': statistics.mean(vals),
        'median': statistics.median(vals),
        'min': min(vals),
        'max': max(vals),
        'n': len(vals),
    }


def _aggregate_boxes(boxes: dict, edges: list[str]) -> dict:
    """{tm_key: {edge: [vals]}} -> {(tm_key, edge): agg-dict}."""
    out: dict = {}
    for tm_key, edge_map in boxes.items():
        for edge in edges:
            agg = _agg(edge_map.get(edge))
            if agg is not None:
                out[(tm_key, edge)] = agg
    return out


def augment_startup(entries: list[dict]) -> None:
    """Attach an annual-weighted 'vStartup' sum to each entry, in place.

    The plots only need vShutdown; the table additionally reports NoEnf's
    start-up deviation (which differs from its shut-down deviation, whereas
    Cyclic/Markov have start-ups == shut-downs). Idempotent — entries already
    carrying 'vStartup' are skipped, so per-cluster re-aggregation is cheap.
    """
    todo = [e for e in entries if 'vStartup' not in e]
    if not todo:
        return

    def load_one(e: dict) -> None:
        try:
            conn = sqlite3.connect(e['file'])
            e['vStartup'] = _weighted_var_sum(conn, 'vStartup')
            conn.close()
        except Exception:
            e['vStartup'] = None

    max_workers = min(len(todo), (os.cpu_count() or 4), 60)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(load_one, todo))


def build_startup_deviation_boxes(entries: list[dict], mode: str, edges: list[str]) -> dict:
    """vStartup deviation from the Truth entry of the same (tm, sub-case).

    Same shape as build_deviation_boxes, but on vStartup (needs augment_startup).
    """
    return _diff_boxes(
        _iter_vs_truth(entries, edges, lambda e: e.get('vStartup')),
        mode, truth_skip=(None,))


def truth_objective_by_tm(entries: list[dict]) -> dict:
    """{tm_key: agg-dict} of the Truth-entry Objective per TM combination [M EUR]."""
    boxes: dict = defaultdict(list)
    for e in entries:
        if e.get('edge_handling') != 'Truth':
            continue
        obj = e.get('Objective')
        if obj not in (None, 0, -1):
            boxes[_tm_key(e)].append(float(obj))
    return {tm: _agg(vals) for tm, vals in boxes.items()}


def build_table_records(main_entries, operational_entries, invest_regret_obj, edges):
    """Aggregate every box into per-(TM_variant, method) records.

    Each metric is the *same* CompareMarkov box builder used for the figures, so
    the numbers match the plots exactly. Returns (records, truth_main, truth_oper).
    """
    augment_startup(operational_entries)
    augment_startup(main_entries)

    dev_op = _aggregate_boxes(build_deviation_boxes(operational_entries, 'relative', edges), edges)
    startup_op = _aggregate_boxes(build_startup_deviation_boxes(operational_entries, 'relative', edges), edges)
    regret = _aggregate_boxes(build_regret_boxes(main_entries, invest_regret_obj, 'absolute', edges), edges)
    wu_op = _aggregate_boxes(build_runtime_relative_boxes(operational_entries, edges), edges)
    wu_inv = _aggregate_boxes(build_runtime_relative_boxes(main_entries, edges), edges)
    dev_inv = _aggregate_boxes(build_deviation_boxes(main_entries, 'relative', edges), edges)

    tm_keys = set()
    for d in (dev_op, startup_op, regret, wu_op, wu_inv, dev_inv):
        tm_keys |= {k for (k, _e) in d}

    records = []
    for tm_key in sorted(tm_keys, key=_tm_sort):
        for edge in sorted(edges, key=lambda e: EDGE_DISPLAY_ORDER.index(e)
                           if e in EDGE_DISPLAY_ORDER else 99):
            key = (tm_key, edge)
            rec = {
                'TM_variant': tm_variant_label(tm_key),
                'method': edge,
                'oper_dev': dev_op.get(key),
                'startup_dev': startup_op.get(key),
                'invest_regret': regret.get(key),
                'wu_oper': wu_op.get(key),
                'wu_invest': wu_inv.get(key),
                'invest_dev': dev_inv.get(key),
            }
            if any(rec[k] for k in ('oper_dev', 'startup_dev', 'invest_regret',
                                    'wu_oper', 'wu_invest', 'invest_dev')):
                records.append(rec)

    return records, truth_objective_by_tm(main_entries), truth_objective_by_tm(operational_entries)


def _f(x, fmt="%.1f") -> str:
    return "n/a" if x is None else fmt % x


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def render_results_markdown(records, truth_main, truth_oper, title_suffix="") -> str:
    """Three Markdown tables: paper results, diagnostics, reference Truth objective."""
    out = [f"# Markov results table{title_suffix}\n"]

    out.append("## Mean over sub-cases per TM_variant x method\n")
    headers = ["TM_variant", "method",
               "oper_dev_mean_pct", "oper_dev_median_pct",
               "oper_dev_min_pct", "oper_dev_max_pct",
               "invest_regret_mean_MEUR", "invest_regret_median_MEUR",
               "wu_oper_mean_pct", "wu_invest_mean_pct", "n_runs"]
    rows = []
    for r in records:
        od, ir = r['oper_dev'], r['invest_regret']
        rows.append([
            r['TM_variant'], r['method'],
            _f(od and od['mean']), _f(od and od['median']),
            _f(od and od['min']), _f(od and od['max']),
            _f(ir and ir['mean']), _f(ir and ir['median']),
            _f(r['wu_oper'] and r['wu_oper']['mean']),
            _f(r['wu_invest'] and r['wu_invest']['mean']),
            str(od['n'] if od else (ir['n'] if ir else 0)),
        ])
    out.append(_md_table(headers, rows))
    out.append("\n`n_runs` is the operational-deviation count (the primary quantity); "
               "per-metric counts are in the diagnostics table below. invest_regret is in "
               "M EUR (LEGO objective unit). An 'n/a' cell means no sub-case had the data "
               "(e.g. no `--operational` runs, no Gurobi work units, or no optimal Truth "
               "reference).\n")

    out.append("## Diagnostics (start-up deviation, per-metric run counts)\n")
    dheaders = ["TM_variant", "method",
                "startup_dev_mean_pct", "startup_dev_median_pct",
                "invest_run_shutdown_dev_mean_pct",
                "n_oper_dev", "n_startup", "n_regret", "n_wu_oper", "n_wu_invest"]
    drows = []
    for r in records:
        sd, idv = r['startup_dev'], r['invest_dev']
        drows.append([
            r['TM_variant'], r['method'],
            _f(sd and sd['mean']), _f(sd and sd['median']),
            _f(idv and idv['mean']),
            str(r['oper_dev']['n'] if r['oper_dev'] else 0),
            str(sd['n'] if sd else 0),
            str(r['invest_regret']['n'] if r['invest_regret'] else 0),
            str(r['wu_oper']['n'] if r['wu_oper'] else 0),
            str(r['wu_invest']['n'] if r['wu_invest'] else 0),
        ])
    out.append(_md_table(dheaders, drows))
    out.append("\nStart-ups equal shut-downs for Cyclic/Markov and differ for NoEnf. "
               "`invest_run_shutdown_dev_mean_pct` is the deviation on the regular "
               "(investment) runs - a fallback view when no `--operational` runs are present.\n")

    out.append("## Reference Truth objective per TM_variant [M EUR]\n")
    rheaders = ["TM_variant", "truth_main_obj_mean_MEUR", "n_main",
                "truth_oper_obj_mean_MEUR", "n_oper"]
    rrows = []
    for tm_key in sorted(set(truth_main) | set(truth_oper), key=_tm_sort):
        tm, to = truth_main.get(tm_key), truth_oper.get(tm_key)
        rrows.append([
            tm_variant_label(tm_key),
            _f(tm and tm['mean']), str(tm['n'] if tm else 0),
            _f(to and to['mean']), str(to['n'] if to else 0),
        ])
    out.append(_md_table(rheaders, rrows))
    return "\n".join(out) + "\n"


def write_results_csv(path, records):
    cols = ["TM_variant", "method",
            "oper_dev_mean_pct", "oper_dev_median_pct",
            "oper_dev_min_pct", "oper_dev_max_pct",
            "startup_dev_mean_pct", "startup_dev_median_pct",
            "invest_regret_mean_MEUR", "invest_regret_median_MEUR",
            "invest_regret_min_MEUR", "invest_regret_max_MEUR",
            "wu_oper_mean_pct", "wu_invest_mean_pct",
            "invest_run_shutdown_dev_mean_pct",
            "n_oper_dev", "n_startup", "n_regret", "n_wu_oper", "n_wu_invest"]

    def g(agg, field):
        return "" if not agg else round(agg[field], 3)

    def n(agg):
        return agg['n'] if agg else 0

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            od, sd, ir = r['oper_dev'], r['startup_dev'], r['invest_regret']
            w.writerow([
                r['TM_variant'], r['method'],
                g(od, 'mean'), g(od, 'median'), g(od, 'min'), g(od, 'max'),
                g(sd, 'mean'), g(sd, 'median'),
                g(ir, 'mean'), g(ir, 'median'), g(ir, 'min'), g(ir, 'max'),
                g(r['wu_oper'], 'mean'), g(r['wu_invest'], 'mean'),
                g(r['invest_dev'], 'mean'),
                n(od), n(sd), n(ir), n(r['wu_oper']), n(r['wu_invest']),
            ])


def report_results_table(main_entries, operational_entries, invest_regret_obj, edges,
                         out_dir, fname_suffix="", title_suffix=""):
    """Build, print (terminal) and save (.txt + .csv) the aggregated results table.

    *fname_suffix* / *title_suffix* tag per-cluster outputs under --separateClusters.
    """
    records, truth_main, truth_oper = build_table_records(
        main_entries, operational_entries, invest_regret_obj, edges)
    if not records:
        printer.warning(f"  Results table{title_suffix}: no (TM_variant, method) cells had data.")
        return

    report = render_results_markdown(records, truth_main, truth_oper, title_suffix)
    # Terminal (plain print avoids any rich re-wrapping of the table). Guard against
    # a strict-ASCII console choking on a non-ASCII title_suffix (the file below is
    # always written UTF-8, so only the live stream needs the fallback).
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(("\n" + report + "\n").encode("utf-8", "replace"))

    stem = "compare_markov_results" + fname_suffix
    txt_path = os.path.join(out_dir, stem + ".txt")
    csv_path = os.path.join(out_dir, stem + ".csv")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    write_results_csv(csv_path, records)
    printer.information(f"  Saved {txt_path}")
    printer.information(f"  Saved {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=("Boxplots of edge-handling strategies (NoEnf, Cyclic, Markov) vs. the "
                                                  "Truth model across shift-tm / perturb-tm combinations.\n\n"
                                                  "For both operational and investment runs: work-units (absolute + as %% "
                                                  "of Truth) and vShutdown-deviation (absolute + relative), plus "
                                                  "invest-regret, regret (vs Truth-main) and operational-regret (vs Truth-operational), "
                                                  "each absolute + relative. Each plot is emitted twice: with all strategies and with "
                                                  "NoEnf excluded (suffix '_noNoEnf')."),
                                     formatter_class=RichHelpFormatter)
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing MK-*.sqlite files (default: current directory)")
    parser.add_argument("--output-dir", default=None, help="Directory to save the PNG files in (default: the input folder)")
    parser.add_argument("--no-show", action="store_true", help="Don't display the figures (useful for headless/batch runs)")
    parser.add_argument("--include-nonoptimal", action="store_true", help="Include runs whose termination_condition is not 'optimal' (default: optimal only)")
    parser.add_argument("--logscale", action="store_true", help="Use a log-scale y-axis for the work-units plots (default: linear). Has no effect on the deviation/regret plots (those can be negative).")
    parser.add_argument("--markov-strict", action="store_true", help="Also include the Markov-Strict strategy as a box in every plot (only meaningful if runs were produced with --enable-strict-markov).")
    parser.add_argument("--nrOfClusters", default=None, metavar="N,N,...",
                        help="Only include runs whose 'clusters' run-parameter is in this comma-separated list (e.g. 3,5,7).")
    parser.add_argument("--separateClusters", action="store_true",
                        help="Emit the full set of plots once per cluster count found in the (filtered) data, with a '_clusters{N}' filename suffix, instead of aggregating all cluster counts together.")
    parser.add_argument("--tm", action="append", default=None, metavar="SHIFT:PERTURB",
                        help="Only show the selected (shift_tm, perturb_tm) subplot combinations. Repeatable and/or comma-separated. Each side is a number, 'none' (parameter unset) or '*' (any); 'base' is shorthand for none:none. Examples: --tm none:0.2 --tm 1:none --tm base ; --tm \"1:*,2:*\". Default: show all.")
    parser.add_argument("--no-results-table", action="store_true",
                        help="Skip the aggregated numeric results table (the mean/median/min/max behind each boxplot). By default the table is printed to the terminal and saved as 'compare_markov_results.txt' (+ '.csv') in the output dir.")
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

    out_dir = args.output_dir or args.folder
    os.makedirs(out_dir, exist_ok=True)

    edges = list(EDGE_ORDER) + ([EDGE_STRICT] if args.markov_strict else [])

    printer.information(f"Loading MK-*.sqlite files from '{args.folder}' ...")
    main_entries, operational_entries, invest_regret_obj, regret_obj, operational_regret_obj = load_all(args.folder, include_nonoptimal=args.include_nonoptimal)

    # Entry filters; the regret maps need no filtering — they are only ever looked
    # up via the (filtered) entries' own files.
    main_entries = filter_entries(main_entries, cluster_filter, tm_specs)
    operational_entries = filter_entries(operational_entries, cluster_filter, tm_specs)

    if not main_entries and not operational_entries:
        reasons = [] if args.include_nonoptimal else ["filtering to optimal runs"]
        if cluster_filter is not None:
            reasons.append(f"--nrOfClusters {sorted(cluster_filter)}")
        if tm_specs:
            reasons.append("--tm filter")
        printer.warning(f"  No usable MK-*.sqlite files found in '{args.folder}'"
                        + (f" (after {', '.join(reasons)})" if reasons else ""))
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
                        f"invest-regret={len(invest_regret_obj)}; "
                        f"regret={len(regret_obj)}; "
                        f"operational-regret={len(operational_regret_obj)}")

    if args.separateClusters:
        cluster_values = sorted({e.get('clusters') for e in main_entries + operational_entries},
                                key=lambda v: (v is None, v))
        for value in cluster_values:
            label = f"{value} clusters" if value is not None else "no clustering"
            printer.information(f"Rendering plots for {label} ...")
            sub_main = [e for e in main_entries if e.get('clusters') == value]
            sub_oper = [e for e in operational_entries if e.get('clusters') == value]
            render_plots(sub_main, sub_oper,
                         invest_regret_obj, regret_obj, operational_regret_obj,
                         edges, out_dir, args,
                         fname_suffix=f"_clusters{value}", title_suffix=f" — {label}")
            if not args.no_results_table:
                report_results_table(sub_main, sub_oper, invest_regret_obj, edges, out_dir,
                                     fname_suffix=f"_clusters{value}", title_suffix=f" — {label}")
    else:
        render_plots(main_entries, operational_entries,
                     invest_regret_obj, regret_obj, operational_regret_obj,
                     edges, out_dir, args)
        if not args.no_results_table:
            report_results_table(main_entries, operational_entries, invest_regret_obj,
                                 edges, out_dir)


if __name__ == "__main__":
    main()
