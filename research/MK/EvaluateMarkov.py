"""
EvaluateMarkov.py - Evaluation of the Markov (MK) experiment results (MK-*.sqlite files written by Markov.py).

Subcommands (all share the loading, run-kind classification, filters and regret pairing below):
    tables   per-group comparison tables of all runs (objective, solver statistics, UC sums, investments);
             optionally plots the unit commitment (--plot)
    plots    boxplots of the edge handlings vs Truth across (shift_tm, perturb_tm) combinations, plus the
             aggregated numeric results table behind them (compare_markov_results.txt/.csv)
    summary  CSV tables (markov_runs.csv, markov_summary.csv) and plots of the per-run analysis tables
             (mk_metrics, mk_nonbinarity, mk_feasibility - written by Markov.py after every solve)
    all      tables + plots + summary from a single load

Usage
-----
python research/MK/EvaluateMarkov.py tables [folder] [--plot]
python research/MK/EvaluateMarkov.py plots [folder] --no-show --separateClusters
python research/MK/EvaluateMarkov.py summary [folder] --output-dir summary/
python research/MK/EvaluateMarkov.py all [folder] --no-show
See README.md for all options.
"""
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import functools
import glob
import math
import os
import re
import sqlite3
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing

import numpy as np
import pandas as pd
from rich_argparse import RichHelpFormatter

from InOutModule import ExcelReader
from InOutModule.printer import Printer

printer = Printer.getInstance()
printer.set_width(300)

########################################################################################################################
# Constants
########################################################################################################################

# Edge handlings in display order (the exact strings stored as run_parameters.edge_handling). 'TruthOriginal' is the
# original full-chronological reference (--original-reference-only).
EDGE_DISPLAY_ORDER = ['Truth', 'NoEnf', 'Cyclic', 'Markov', 'Markov-Strict', 'TruthOriginal']
# Edge handlings drawn as boxes in the plots (Truth is the reference, never a box); Markov-Strict only with --markov-strict
EDGE_BOXES = ['NoEnf', 'Cyclic', 'Markov']
EDGE_STRICT = 'Markov-Strict'
EDGE_COLORS = {
    'NoEnf': '#9467bd',  # purple
    'Cyclic': '#ff7f0e',  # orange
    'Markov': '#1f77b4',  # blue
    'Markov-Strict': '#2ca02c',  # green
    'Truth': '#52514e',
    'TruthOriginal': '#8c8c8c',
}
# Pretty labels (display only; the internal edge keys stay as-is for lookups/colors)
EDGE_LABELS = {'NoEnf': 'No Enf.'}
# Titles of the unit-commitment plot (tables --plot)
CASE_DISPLAY_TITLES = {
    "Markov": "Markov Transition",
    "Markov-Strict": "Markov Transition (Strict)",
    "NoEnf": "No Enforcement",
    "Cyclic": "Cyclic Connection",
    "Truth": "Reference Case",
}

RUN_KINDS = ['main', 'operational', 'regret', 'invest_regret', 'operational_regret']
REGRET_KINDS = ['regret', 'invest_regret', 'operational_regret']
KIND_TITLES = {
    'operational': "Operational runs (vGenInvest fixed to Truth's investment)",
    'regret': "Regret runs (truth re-solved with vGenInvest + vCommit fixed from the edge handling's main run)",
    'invest_regret': "Invest-regret runs (truth re-solved with vGenInvest fixed from the edge handling's main run)",
    'operational_regret': "Operational-regret runs (truth re-solved with vGenInvest fixed to Truth's, vCommit fixed from the operational run)",
}

# File suffix -> (run kind, suffix of the base file whose decisions the run evaluates). Order matters: the more
# specific suffixes ('-operational-regret', '-original-...', '-invest-regret') all end in '-regret.sqlite'.
_KIND_SUFFIXES = [
    ('-operational-regret.sqlite', 'operational_regret', '-operational.sqlite'),
    ('-original-invest-regret.sqlite', 'invest_regret', '.sqlite'),
    ('-original-regret.sqlite', 'regret', '.sqlite'),
    ('-invest-regret.sqlite', 'invest_regret', '.sqlite'),
    ('-regret.sqlite', 'regret', '.sqlite'),
    ('-operational.sqlite', 'operational', None),
]

# Run parameters that define a "sub-case" - everything except the TM perturbation params (shift_tm / perturb_tm),
# which form the per-subplot axis. Note: 'shift' is the time-series shift, not the subplot axis 'shift_tm'.
SUB_CASE_KEYS = [
    'case_study_directory', 'filter_zone', 'limit_k', 'clusters', 'shift',
    'stretch_demand', 'scale_vres', 'scale_invest_cost', 'thermal_invest_only',
    'merge_generators', 'relax_count', 'no_investment', 'rmip', 'no_crossover',
    'force_barrier', 'mip_gap', 'network', 'commit_consumption', 'startup_consumption',
]
# Grouping of the 'tables' subcommand: all sub-case keys + TM perturbation + reference (keeps -original-* runs apart)
GROUP_KEYS = SUB_CASE_KEYS[:10] + ['shift_tm', 'perturb_tm'] + SUB_CASE_KEYS[10:] + ['reference']

ANALYSIS_TABLES = ['mk_metrics', 'mk_nonbinarity', 'mk_feasibility']

# Trailing path components that Markov.py appends to the dataset folder for each preprocessing step. They are
# stripped from `case_study_directory` to recover the dataset name (e.g. 'RTS-GMLC').
_PREPROCESSING_SUBFOLDER = re.compile(r'^(data|filterZone.+|limitK.+|shift\d+|stretchDemand.+|mergeGenerators|\d+ clusters)$')
_CLUSTER_SUFFIX = re.compile(r'[/\\]?\d+ clusters[/\\]?$')


########################################################################################################################
# Loading
########################################################################################################################

def _parse_metadata_from_filename(basename):
    """Fallback for files without run_parameters: extract edge_handling and case_study_directory from the filename.

    Expected format: MK-{identifier}-{edgeHandling}.sqlite, where identifier starts with 'data{path}'.
    """
    meta = {}
    stem = basename
    if stem.startswith("MK-"):
        stem = stem[3:]
    if stem.endswith(".sqlite"):
        stem = stem[:-7]
    for eh in ["Markov-Strict", "Markov", "Cyclic", "NoEnf", "Truth"]:
        if stem.endswith(f"-{eh}"):
            meta['edge_handling'] = eh
            stem = stem[:-len(eh) - 1]
            break
    # The identifier is path.replace('/', '_'), so "data/markov/" -> "datadata_markov"; restore the first segment
    m = re.match(r'^data(.+?)(?:-relaxed\d+)?$', stem)
    if m:
        meta['case_study_directory'] = m.group(1).replace('_', '/', 1)
        if not meta['case_study_directory'].endswith('/'):
            meta['case_study_directory'] += '/'
    return meta


def _load_metadata_from_conn(conn, basename):
    """Load run parameters and solver statistics from an open SQLite connection."""
    meta = {key: None for key in GROUP_KEYS + ['edge_handling', 'run_type', 'work_units', 'solver_time', 'solver_status',
                                               'termination_condition', 'achieved_mip_gap']}
    has_run_parameters = False
    try:
        df = pd.read_sql_query('SELECT * FROM run_parameters', conn)
        if len(df) > 0:
            has_run_parameters = True
            row = df.iloc[0]
            converters = {str: ['case_study_directory', 'edge_handling', 'run_type', 'limit_k', 'network', 'filter_zone', 'shift_tm', 'reference'],
                          int: ['clusters', 'relax_count', 'shift'],
                          float: ['stretch_demand', 'scale_vres', 'scale_invest_cost', 'mip_gap', 'perturb_tm', 'commit_consumption', 'startup_consumption'],
                          bool: ['no_investment', 'rmip', 'no_crossover', 'force_barrier', 'thermal_invest_only', 'merge_generators']}
            for conv, keys in converters.items():
                for key in keys:
                    if key not in row or row[key] in (None, 'None'):
                        continue
                    val = row[key]
                    if conv is int:
                        meta[key] = int(float(val))
                    elif conv is bool:
                        meta[key] = val if isinstance(val, bool) else (val != 0 if isinstance(val, (int, float)) else str(val).lower() == 'true')
                    else:
                        meta[key] = conv(val)
    except Exception:
        pass
    try:
        df_stats = pd.read_sql_query('SELECT * FROM solver_statistics', conn)
        if len(df_stats) > 0:
            row = df_stats.iloc[0]
            for key, conv in [('work_units', float), ('solver_time', float), ('solver_status', str), ('termination_condition', str)]:
                if key in row and row[key] is not None:
                    meta[key] = conv(row[key])
            try:
                lb = float(row.get('lower_bound')) if row.get('lower_bound') is not None else None
                ub = float(row.get('upper_bound')) if row.get('upper_bound') is not None else None
            except (TypeError, ValueError):
                lb = ub = None
            if lb is not None and ub is not None and math.isfinite(lb) and math.isfinite(ub) and ub != 0:
                meta['achieved_mip_gap'] = abs(ub - lb) / abs(ub)
            else:
                # Fallback (e.g. OOM runs without bounds): the mip_gap column written from Gurobi's MIPGap
                try:
                    gap = float(row.get('mip_gap'))
                    if gap == gap:  # not NaN
                        meta['achieved_mip_gap'] = gap
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass
    if not has_run_parameters:
        for key, val in _parse_metadata_from_filename(basename).items():
            if meta[key] is None:
                meta[key] = val
    return meta


def _weighted_sums(conn, var_names: list[str]) -> dict:
    """Annual-weighted sums of (rp, k)-indexed variables, aggregated in SQL via temporary weight tables (the large
    variable tables are never transferred to Python). None for variables whose table is missing."""
    sums = {var_name: None for var_name in var_names}
    try:
        wrp_rows = conn.execute('SELECT * FROM pWeight_rp').fetchall()
        wk_rows = conn.execute('SELECT * FROM pWeight_k').fetchall()
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _wrp (rp TEXT PRIMARY KEY, w REAL)")
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _wk (k TEXT PRIMARY KEY, w REAL)")
        conn.executemany("INSERT OR IGNORE INTO _wrp VALUES (?,?)", [(str(r[0]), float(r[1])) for r in wrp_rows])
        conn.executemany("INSERT OR IGNORE INTO _wk VALUES (?,?)", [(str(r[0]), float(r[1])) for r in wk_rows])
    except Exception:
        return sums
    for var_name in var_names:
        try:
            val = conn.execute(f'SELECT SUM(v."values" * wrp.w * wk.w) FROM "{var_name}" v '
                               f'JOIN _wrp wrp ON CAST(v.rp AS TEXT) = wrp.rp JOIN _wk wk ON CAST(v.k AS TEXT) = wk.k').fetchone()[0]
            sums[var_name] = float(val) if val is not None else 0.0
        except Exception:
            pass
    return sums


def _investment_results(conn) -> dict:
    """First-stage objective, vGenInvest and invested capacity (vGenInvest * pMaxProd), in total and per technology."""
    results = {}
    try:
        val = conn.execute('SELECT SUM(var_times_coefficient) FROM objective_terms WHERE var_name IN ("vGenInvest", "vLineInvest")').fetchone()[0]
        if val is None:
            raise ValueError
        results['first_stage_objective'] = float(val)
    except Exception:
        # Fallback for old files without var_times_coefficient: compute by hand
        total = 0.0
        for var_table, cost_table in [('vGenInvest', 'pInvestCost'), ('vLineInvest', 'pFixedCost')]:
            try:
                df_var = pd.read_sql_query(f'SELECT * FROM {var_table}', conn)
                df_cost = pd.read_sql_query(f'SELECT * FROM {cost_table}', conn)
                var_idx = [c for c in df_var.columns if c not in ('values', 'index')]
                cost_idx = [c for c in df_cost.columns if c not in ('values', 'index')]
                merged = df_var.merge(df_cost, left_on=var_idx, right_on=cost_idx, suffixes=('', '_cost'))
                total += (merged['values'] * merged['values_cost']).sum()
            except Exception:
                pass
        results['first_stage_objective'] = total

    try:
        df_inv = pd.read_sql_query('SELECT * FROM vGenInvest', conn)
    except Exception:
        results['vGenInvest'] = 0
        return results
    results['vGenInvest'] = df_inv['values'].sum()
    try:
        df_gtec = pd.read_sql_query('SELECT * FROM gtec', conn)
        if 'index' in df_gtec.columns:
            df_gtec = df_gtec.drop(columns=['index'])
        df_gtec.columns = ['g', 'tec']
        g_col = df_inv.columns[0]
        df_merged = df_inv.merge(df_gtec, left_on=g_col, right_on='g', how='left')
        for tec, group in df_merged.groupby('tec'):
            results[f'vGenInvest[{tec}]'] = group['values'].sum()
        df_pmax = pd.read_sql_query('SELECT * FROM pMaxProd', conn)
        df_pmax = df_pmax.rename(columns={df_pmax.columns[0]: g_col, 'values': 'pmax'})
        df_merged = df_merged.merge(df_pmax[[g_col, 'pmax']], on=g_col, how='left')
        df_merged['cap'] = df_merged['values'] * df_merged['pmax'].fillna(0)
        results['vCapInvest'] = df_merged['cap'].sum()
        for tec, group in df_merged.groupby('tec'):
            results[f'vCapInvest[{tec}]'] = group['cap'].sum()
    except Exception:
        pass
    return results


def _read_table(conn, table: str) -> pd.DataFrame | None:
    try:
        return pd.read_sql(f"SELECT * FROM {table}", conn)
    except Exception:
        return None


def classify_file(path: str) -> tuple[str, str | None]:
    """(run kind, base file) of a result file, from its suffix. The base file is the run whose decisions a regret
    run evaluates (the main run, or the operational run for operational-regret); None for main/operational."""
    for suffix, kind, base_suffix in _KIND_SUFFIXES:
        if path.endswith(suffix):
            return kind, (path[:-len(suffix)] + base_suffix if base_suffix else None)
    return 'main', None


def _load_file(path: str, full: bool, analysis: bool) -> tuple[dict, pd.DataFrame | None]:
    """Load one result file into an entry dict (module-level so it can be pickled for ProcessPoolExecutor).

    Always: run kind, run parameters, solver statistics, Objective and the weighted vStartup/vShutdown sums.
    full: additionally vGenP/vCommit/vPNS/vEPS sums and the investment results ('tables' subcommand).
    analysis: additionally the mk_* analysis tables ('summary' subcommand); returns mk_nonbinary_values as 2nd value.
    """
    kind, base_file = classify_file(path)
    entry = {'file': path, 'basename': os.path.basename(path), 'kind': kind, 'base_file': base_file}
    values = None
    try:
        with closing(sqlite3.connect(path)) as conn:
            entry.update(_load_metadata_from_conn(conn, entry['basename']))
            try:
                row = conn.execute('SELECT "values" FROM objective LIMIT 1').fetchone()
                entry['Objective'] = float(row[0]) if row and row[0] is not None else None
            except Exception:
                entry['Objective'] = None
            entry.update(_weighted_sums(conn, ['vGenP', 'vCommit', 'vStartup', 'vShutdown', 'vPNS', 'vEPS'] if full else ['vStartup', 'vShutdown']))
            if full:
                entry.update(_investment_results(conn))
            if analysis:
                for table in ANALYSIS_TABLES:
                    df = _read_table(conn, table)
                    if df is not None and len(df) > 0:
                        for key, value in df.iloc[0].items():
                            entry.setdefault(key, value)  # Run parameters / solver statistics win (e.g. work_units)
                values = _read_table(conn, 'mk_nonbinary_values')
                if values is not None:
                    values = values.assign(file=path)
    except Exception as e:
        printer.warning(f"Could not load '{path}': {e}")
    return entry, values


def load_entries(folder: str, recursive: bool = False, full: bool = False, analysis: bool = False) -> tuple[list[dict], pd.DataFrame]:
    """Load all MK-*.sqlite files of a folder in parallel. Returns (entries, fractional UC values of all files)."""
    pattern = os.path.join(folder, '**', 'MK-*.sqlite') if recursive else os.path.join(folder, 'MK-*.sqlite')
    files = sorted(glob.glob(pattern, recursive=recursive))
    if not files:
        return [], pd.DataFrame()
    max_workers = min(len(files), os.cpu_count() or 4, 60)
    printer.information(f"Loading {len(files)} MK-*.sqlite file(s) from '{folder}' with up to {max_workers} processes ...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        loaded = list(executor.map(functools.partial(_load_file, full=full, analysis=analysis), files))
    entries = [entry for entry, _ in loaded]
    value_frames = [values for _, values in loaded if values is not None and len(values) > 0]
    return entries, (pd.concat(value_frames, ignore_index=True) if value_frames else pd.DataFrame())


def _is_optimal(entry: dict) -> bool:
    return entry.get('termination_condition') == 'optimal'


def edge_sort_key(edge) -> int:
    return EDGE_DISPLAY_ORDER.index(edge) if edge in EDGE_DISPLAY_ORDER else 99


########################################################################################################################
# Dataset, TM-variant and sub-case helpers
########################################################################################################################

def _dataset_name(case_study_directory) -> str:
    """Dataset name from a `case_study_directory`, ignoring preprocessing subfolders, e.g. 'data/RTS-GMLC/7 clusters/'
    -> 'RTS-GMLC'."""
    parts = [p for p in str(case_study_directory).replace('\\', '/').split('/') if p]
    while parts and _PREPROCESSING_SUBFOLDER.match(parts[-1]):
        parts.pop()
    return parts[-1] if parts else ''


def _base_directory(case_study_directory) -> str | None:
    """Folder before the '{N} clusters' subfolder - the folder of the original reference (TruthOriginal)."""
    return _CLUSTER_SUFFIX.sub('', str(case_study_directory).rstrip('/\\')) if case_study_directory else None


def _data_label(entries: list[dict]) -> str:
    """Dataset name(s) behind entries (' / '-joined if several), '(base)' if none is recorded."""
    names = {_dataset_name(e['case_study_directory']) for e in entries if e.get('case_study_directory')} - {''}
    return ' / '.join(sorted(names)) if names else '(base)'


def _tm_value(v) -> float | None:
    """Normalize a stored shift_tm (str) / perturb_tm (float) value to float, or None when unset."""
    if v in (None, 'None'):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def tm_key(entry: dict) -> tuple:
    return entry.get('shift_tm'), entry.get('perturb_tm')


def subcase_key(entry: dict) -> tuple:
    return tuple(entry.get(k) for k in SUB_CASE_KEYS)


def _tm_sort(key: tuple):
    """Sort TM combinations by shift first, then perturb; None (unset) sorts first."""
    shift, perturb = (_tm_value(v) for v in key)
    return (shift if shift is not None else float('-inf'),
            perturb if perturb is not None else float('-inf'))


def _tm_label(shift_tm, perturb_tm, base_label: str = '(base)') -> str:
    parts = []
    shift, perturb = _tm_value(shift_tm), _tm_value(perturb_tm)
    if shift is not None:
        parts.append(f"Transition Matrix\nshifted by {shift:g}")
    if perturb is not None:
        parts.append(f"perturbTM={perturb:g}")
    return ', '.join(parts) if parts else base_label  # The unperturbed subplot is labelled with the dataset name


def tm_variant_label(key: tuple) -> str:
    """(shift_tm, perturb_tm) -> paper TM-variant name ('Original', 'Shift1', ...)."""
    shift, perturb = (_tm_value(v) for v in key)
    label = "Original" if shift is None else f"Shift{shift:g}"
    if perturb is not None:
        label += f"+perturb{perturb:g}"
    return label


########################################################################################################################
# Filters and regret pairing
########################################################################################################################

def parse_tm_spec(spec: str) -> tuple:
    """Parse one --tm spec into a (shift, perturb) pair. Each side is a float, None ('none'/'-' = parameter unset)
    or '*' (any value); 'base' is shorthand for none:none. Raises ValueError on malformed input."""
    if spec.strip().lower() == 'base':
        return None, None
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


def _matches_filters(entry: dict, cluster_filter: set[int] | None, tm_specs: list[tuple]) -> bool:
    """--nrOfClusters (entries without clusters are dropped when given) and --tm."""
    if cluster_filter is not None and entry.get('clusters') not in cluster_filter:
        return False
    if not tm_specs:
        return True
    shift, perturb = _tm_value(entry.get('shift_tm')), _tm_value(entry.get('perturb_tm'))
    return any((s == '*' or s == shift) and (p == '*' or p == perturb) for s, p in tm_specs)


def select_entries(entries: list[dict], include_nonoptimal: bool, cluster_filter: set[int] | None, tm_specs: list[tuple]) -> list[dict]:
    """Entries used for the plots and the summary: optimal (unless include_nonoptimal) main/operational runs that match
    the filters, plus the optimal regret runs whose base run (the run whose decisions they evaluate) was selected."""
    selected = [e for e in entries if e['kind'] in ('main', 'operational') and (include_nonoptimal or _is_optimal(e))
                and _matches_filters(e, cluster_filter, tm_specs)]
    selected_files = {e['file'] for e in selected}
    return selected + [e for e in entries if e['kind'] in REGRET_KINDS and (include_nonoptimal or _is_optimal(e))
                       and e['base_file'] in selected_files]


def _original_key(entry: dict) -> tuple:
    """Key that pairs a run with its original reference (TruthOriginal): the folder before clustering + all other
    sub-case keys (TruthOriginal does not depend on the number of RPs)."""
    return (_base_directory(entry.get('case_study_directory')),) + tuple(entry.get(k) for k in SUB_CASE_KEYS if k not in ('case_study_directory', 'clusters'))


def attach_regret(entries: list[dict], include_nonoptimal: bool = False) -> None:
    """Add 'reference_objective', 'regret' (objective - reference objective) and 'regret_pct' to every regret entry,
    in place. References: regret/invest-regret -> Truth (main run of the same TM variant and sub-case),
    operational-regret -> Truth-operational, -original-* runs -> TruthOriginal. Only optimal references are used,
    unless include_nonoptimal."""
    references = {}
    for e in entries:
        if e.get('Objective') is None or not (include_nonoptimal or _is_optimal(e)):
            continue
        if e.get('edge_handling') == 'Truth' and e['kind'] in ('main', 'operational') and e.get('reference') is None:
            references[(e['kind'], tm_key(e), subcase_key(e))] = e['Objective']
        elif e.get('edge_handling') == 'TruthOriginal':
            references[('original', _original_key(e))] = e['Objective']
    for e in entries:
        if e['kind'] not in REGRET_KINDS:
            continue
        if e.get('reference') == 'original':
            ref = references.get(('original', _original_key(e)))
        else:
            ref = references.get(('operational' if e['kind'] == 'operational_regret' else 'main', tm_key(e), subcase_key(e)))
        obj = e.get('Objective')
        valid = ref not in (None, 0) and obj is not None
        e['reference_objective'] = ref
        e['regret'] = obj - ref if valid else None
        e['regret_pct'] = (obj - ref) / abs(ref) * 100 if valid else None


########################################################################################################################
# 'tables': per-group comparison tables
########################################################################################################################

def _build_table_row(entry: dict) -> dict:
    return {**entry,
            'Case': entry.get('edge_handling') or entry['basename'],
            'Work Units': entry.get('work_units'),
            'Status': entry.get('solver_status'),
            'Term. Cond.': entry.get('termination_condition'),
            'MIP-Gap': entry.get('achieved_mip_gap')}


def _group_header(group_entry: dict) -> str:
    g = group_entry
    parts = []
    for key, label in [('case_study_directory', 'data'), ('filter_zone', 'zone'), ('limit_k', 'limitK')]:
        if g.get(key):
            parts.append(f"{label}={g[key]}")
    if g.get('clusters') and g['clusters'] > 1:
        parts.append(f"clusters={g['clusters']}")
    if g.get('shift'):
        parts.append(f"shift={g['shift']}")
    for key in ['stretch_demand', 'scale_vres', 'scale_invest_cost']:
        if g.get(key) and g[key] != 1.0:
            parts.append(f"{key}={g[key]}")
    for key, label in [('thermal_invest_only', 'thermal-invest-only'), ('merge_generators', 'merge-generators')]:
        if g.get(key):
            parts.append(label)
    if g.get('relax_count'):
        parts.append(f"relaxed={g['relax_count']}")
    for key, label in [('no_investment', 'no-investment'), ('rmip', 'rMIP'), ('no_crossover', 'no-crossover'), ('force_barrier', 'force-barrier')]:
        if g.get(key):
            parts.append(label)
    if g.get('mip_gap') is not None:
        parts.append(f"mip-gap={g['mip_gap']}")
    if g.get('network') is not None:
        parts.append(f"network={g['network']}")
    for key in ['commit_consumption', 'startup_consumption', 'perturb_tm']:
        if g.get(key) is not None:
            parts.append(f"{key}={g[key]:g}")
    for key in ['shift_tm', 'reference']:
        if g.get(key) is not None:
            parts.append(f"{key}={g[key]}")
    return ', '.join(parts) if parts else '(default)'


def print_comparison_table(group_entries):
    """Print the comparison tables (results, vGenInvest, invested capacity) for a group of runs of one kind."""
    group_entries.sort(key=lambda e: edge_sort_key(e['edge_handling']))

    # Reference for the % columns: Truth; regret tables have no Truth row (Truth is never a regret run) -> Markov
    truth_entry = next((e for e in group_entries if e['edge_handling'] == 'Truth'), None)
    markov_entry = next((e for e in group_entries if e['edge_handling'] == 'Markov'), None)
    ref_entry = truth_entry if truth_entry is not None else markov_entry
    using_markov_ref = truth_entry is None and markov_entry is not None

    def _pct(entry, key, ref_val):
        val = entry.get(key)
        return (val - ref_val) / abs(ref_val) * 100 if ref_val not in (None, 0) and val is not None else None

    for entry in group_entries:
        for key in ["Objective", "vStartup", "vShutdown"]:
            entry[f"{key} %"] = _pct(entry, key, ref_entry.get(key) if ref_entry else None)
        obj, fso = entry.get('Objective'), entry.get('first_stage_objective')
        entry['1st-Stage [%]'] = fso / abs(obj) * 100 if obj not in (0, None) and fso is not None else None

    columns = ["Case", "Objective", "Objective %", "first_stage_objective", "1st-Stage [%]",
               "Work Units", "Status", "Term. Cond.", "MIP-Gap",
               "vGenP", "vCommit", "vStartup", "vStartup %",
               "vShutdown", "vShutdown %", "vPNS", "vEPS"]
    _print_table(columns, group_entries, highlight_pct_yellow=using_markov_ref)
    if using_markov_ref:
        printer.console.print("[yellow]  Differences compared to Markov result[/yellow]")

    # Investment tables (vGenInvest, and invested capacity vGenInvest * pMaxProd if pMaxProd is available)
    for prefix, title in [("vGenInvest", "vGenInvest"), ("vCapInvest", "vCapInvest  (vGenInvest * pMaxProd — invested capacity)")]:
        if not any(e.get(prefix) not in (None, "") for e in group_entries):
            continue
        tec_columns = sorted(set(k for e in group_entries for k in e if k.startswith(f"{prefix}[")))
        display = {prefix: "Total", f"{prefix} %": "[%]"}
        columns = ["Case", prefix, f"{prefix} %"]
        for key in [prefix] + tec_columns:
            for entry in group_entries:
                entry[f"{key} %"] = _pct(entry, key, ref_entry.get(key) if ref_entry else None)
        for tec in tec_columns:
            display[tec] = tec[len(prefix) + 1:-1]
            display[f"{tec} %"] = "[%]"
            columns += [tec, f"{tec} %"]
        printer.information("")
        printer.information(f"  {title}")
        _print_table(columns, group_entries, display, highlight_pct_yellow=using_markov_ref)
        if using_markov_ref:
            printer.console.print("[yellow]  Differences compared to Markov result[/yellow]")


def _print_table(columns, group_entries, display_names=None, highlight_pct_yellow=False):
    """Print a formatted table with given columns and entries."""
    table = []
    highlights = []  # parallel to table: None, 'red', or 'yellow'
    for col in columns:
        if display_names and col in display_names:
            header = display_names[col]
        elif col.endswith(" %"):
            header = "[%]"
        else:
            header = col
        column_data = [header]
        highlight_data = [None]
        for entry in group_entries:
            value = entry.get(col, "")
            if value is None:
                value = ""
            elif col == "1st-Stage [%]":
                value = f"{value:.1f}%"
            elif col.endswith(" %"):
                value = f"{value:+.0f}%"
            elif col == "MIP-Gap":
                value = f"{value * 100:.4f}%"
            elif isinstance(value, float):
                value = f"{value:.2f}"
            elif isinstance(value, int):
                value = f"{value:d}"
            else:
                value = f"{value}"
            column_data.append(value)

            color = None
            if col in ("Status", "Term. Cond."):
                status = entry.get('solver_status')
                if status is not None and status != 'ok':
                    color = 'red'
            elif col == "MIP-Gap":
                gap = entry.get('achieved_mip_gap')
                if gap is not None:
                    run_gap = entry.get('mip_gap')
                    if gap > (run_gap if run_gap is not None else 0.0001):
                        color = 'red'
            elif col.endswith(" %") and highlight_pct_yellow:
                color = 'yellow'
            highlight_data.append(color)

        table.append(column_data)
        highlights.append(highlight_data)

    col_widths = [max(len(table[j][i]) for i in range(len(table[j]))) for j in range(len(table))]

    for i in range(len(table[0])):
        parts = []
        needs_console = False
        for j in range(len(table)):
            cell = f"{table[j][i]:{'>' if i != 0 else ''}{col_widths[j]}}"
            color = highlights[j][i]
            if color:
                cell = f"[{color}]{cell}[/{color}]"
                needs_console = True
            parts.append(cell)
        line = " | ".join(parts)
        if needs_console:
            printer.console.print(line)
        else:
            printer.information(line)


def plot_unit_commitment(sqlite_files, case_labels, case_study_folder=None, number_of_hours=24 * 7, start_hour=1, no_show=False):
    """Plot unit commitment from sqlite result files."""
    import matplotlib.pyplot as plt

    plt.rcParams['figure.dpi'] = 300

    frames = [_load_unit_commitment_from_sqlite(f, label) for f, label in zip(sqlite_files, case_labels)]
    df = pd.concat(frames)

    # Load vGenInvest from each file: track which generators were invested in at least one
    # model, and also record the per-(case, generator) investment value for background shading.
    invested_generators = set()
    per_case_investment: dict[tuple[str, str], float] = {}
    for sqlite_file, label in zip(sqlite_files, case_labels):
        try:
            with closing(sqlite3.connect(sqlite_file)) as cnx:
                df_inv = pd.read_sql("SELECT * FROM vGenInvest", cnx)
            for _, row in df_inv.iterrows():
                g_name = row[df_inv.columns[0]]
                val = float(row['values'])
                per_case_investment[(label, g_name)] = val
                if val > 0:
                    invested_generators.add(g_name)
        except Exception:
            pass

    # Filter to invested generators only (if we found investment data)
    all_generators = df.index.get_level_values("g").unique()
    generators = [g for g in all_generators if g in invested_generators] if invested_generators else list(all_generators)

    # Load hindex mapping from a non-Truth sqlite file (Truth has a different time structure)
    # Fall back to Excel if not available
    hindex = None
    hindex_source = next((f for f, l in zip(sqlite_files, case_labels) if l.strip() not in ("Truth", "Truth ")), sqlite_files[0])
    try:
        with closing(sqlite3.connect(hindex_source)) as cnx:
            hindex = pd.read_sql("SELECT * FROM hindex", cnx)
        if hindex.empty:
            hindex = None
        else:
            if 'index' in hindex.columns:
                hindex = hindex.drop(columns=['index'])
            # hindex columns from pyo.Set(dimen=3) are 0, 1, 2 -> rename to p, rp, k
            if set(hindex.columns) != {'p', 'rp', 'k'}:
                hindex = hindex.rename(columns={hindex.columns[0]: 'p', hindex.columns[1]: 'rp', hindex.columns[2]: 'k'})
    except Exception:
        pass

    if hindex is None:
        if case_study_folder is None:
            printer.error("No hindex table found in sqlite files and no case-study-folder provided in .sqlite or with --case-study-folder for fallback")
            return
        printer.warning("No hindex table in sqlite, falling back to Excel file")
        cs_folder = case_study_folder if case_study_folder.endswith("/") else case_study_folder + "/"
        hindex = ExcelReader.get_Power_Hindex(cs_folder + "Power_Hindex.xlsx")
        hindex = hindex.reset_index()

    hindex["p_int"] = hindex["p"].str.extract(r'(\d+)').astype(int)
    hindex["rp_int"] = hindex["rp"].str.extract(r'(\d+)').astype(int)
    hindex["k_int"] = hindex["k"].str.extract(r'(\d+)').astype(int)

    hindex = hindex.loc[(hindex["p_int"] >= start_hour) & (hindex["p_int"] <= start_hour + number_of_hours - 1)]

    index = [i + 1 for i in range(len(hindex))]
    nr_cases = len(df.index.get_level_values("case").unique())
    nr_generators = len(generators)

    fig, axs = plt.subplots(nr_cases, nr_generators, figsize=(6 * nr_generators, 2 * nr_cases), squeeze=False)

    for i, case in enumerate(df.index.get_level_values("case").unique()):
        is_truth = case.strip() in ("Truth", "Truth ")
        for j, g in enumerate(generators):

            data_vGenP = {}
            data_bar_startup = {}
            data_bar_shutdown = {}
            data_bar_min_uptime_height = {}
            data_bar_min_downtime_bottom = {}
            data_demand = {}
            data_vPNS = {}
            data_vEPS = {}
            data_vCommit = {}

            for counter, (_, row) in enumerate(hindex.iterrows()):
                counter += 1
                rp = "rp01" if is_truth else row["rp"]
                k = row["p"].replace("h", "k") if is_truth else row["k"]
                data_vGenP[counter] = df.loc[case, rp, k, g]["vGenP"]
                data_vCommit[counter] = df.loc[case, rp, k, g]["vCommit"]
                data_bar_startup[counter] = df.loc[case, rp, k, g]["vStartup"]
                data_bar_shutdown[counter] = df.loc[case, rp, k, g]["vShutdown"]
                data_demand[counter] = df.loc[case, rp, k, g]["pDemandP"]
                data_vPNS[counter] = df.loc[case, rp, k, g]["vPNS"]
                data_vEPS[counter] = df.loc[case, rp, k, g]["vEPS"]

            for counter, (_, row) in enumerate(hindex.iterrows()):
                counter += 1
                rp = "rp01" if is_truth else row["rp"]
                k = row["p"].replace("h", "k") if is_truth else row["k"]
                data_bar_min_uptime_height[counter] = sum(
                    [data_bar_startup[a] for a in
                     [counter - b for b in range(0, int(df.loc[case, rp, k, g]["pMinUpTime"] - 1)) if counter - b > 0]])
                data_bar_min_downtime_bottom[counter] = 1 - sum(
                    [data_bar_shutdown[a] for a in
                     [counter - b for b in range(0, int(df.loc[case, rp, k, g]["pMinDownTime"] - 1)) if counter - b > 0]])

            axs2 = axs[i, j].twinx()
            display_name = CASE_DISPLAY_TITLES.get(case.strip(), case.strip())
            if i == 0 and nr_generators > 1:
                axs2.set_title(f"{g}\n{display_name}")
            else:
                axs2.set_title(display_name)
            axs2.set_ylim(0, 3)

            # Grey hatched background when this strategy did not invest in this generator
            # (skipped when only one unit is available — shading a single column is pointless)
            inv_val = per_case_investment.get((case, g))
            if nr_generators > 1 and inv_val is not None and inv_val <= 0:
                x_fill = [min(index) - 0.5, max(index) + 0.5]
                hatch_kw = dict(color='grey', alpha=0.12, hatch='///', edgecolor='#999999', linewidth=0, zorder=0)
                axs[i, j].fill_between(x_fill, -1, 1, **hatch_kw)
                axs2.fill_between(x_fill, 0, 3, **hatch_kw)

            axs2.bar(index, data_bar_startup.values(), color="green", alpha=0.5,
                     bottom=[list(data_vCommit.values())[-1]] + list(data_vCommit.values())[:-1], width=1,
                     label="Start-up")
            axs2.bar(index, data_bar_shutdown.values(), color="red", alpha=0.5, bottom=data_vCommit.values(), width=1,
                     label="Shutd.")
            axs2.plot(index, data_vCommit.values(), color="black", alpha=0.5, label="Commit", linewidth=1.5)
            axs2.set_ylabel("Start-up & Shutdown", color="black")

            axs2.bar(index, data_bar_min_uptime_height.values(), color="green", alpha=0.2, width=1)
            axs2.bar(index, bottom=data_bar_min_downtime_bottom.values(),
                     height=[1 - x for x in data_bar_min_downtime_bottom.values()], color="red", alpha=0.2, width=1)

            axs2.hlines(y=1, xmin=0, xmax=len(data_bar_shutdown.values()), color="gray", linestyle=(0, (1, 1)),
                        alpha=0.5)
            axs2.set_yticks([0, 1], ["0", "1"])
            axs2.legend(loc='lower right', fontsize='x-small')

            # Plot demand on second y-axis, add PNS and EPS
            axs[i, j].set_ylim(-1, 1)
            axs[i, j].plot(index, data_demand.values(), color="blue", alpha=0.3, label="Demand")
            axs[i, j].plot(index, data_vGenP.values(), color="black", alpha=0.3, label="Gen.")

            axs[i, j].bar(index, data_vPNS.values(), color="orange", alpha=0.3, label="PNS", bottom=data_vGenP.values())
            axs[i, j].bar(index, data_vEPS.values(), color="purple", alpha=0.3, label="EPS", bottom=data_demand.values())
            axs[i, j].legend(loc='upper right', fontsize='x-small')

            axs[i, j].hlines(y=0, xmin=0, xmax=len(data_bar_shutdown.values()), color="gray", linestyle=(0, (1, 1)),
                             alpha=0.5)
            axs[i, j].set_ylabel("Demand & Generation", color="black")
            axs[i, j].set_yticks([0, 0.5, 1], ["0.0", "0.5", "1.0"])

            # Set ticks and vertical lines
            index_labels = []
            index_positions = []
            axvline_thick_positions = []
            axvline_thin_positions = []
            for x in index:
                if x == index[0] or x == index[-1]:
                    index_labels.append(x + start_hour - 1)
                    index_positions.append(x)
                    if (x + start_hour - 2) % 24 == 0:
                        axvline_thick_positions.append(x)
                    else:
                        axvline_thin_positions.append(x)
                elif (x + start_hour - 2) % 24 == 0:
                    axvline_thick_positions.append(x)
                    if abs(x - index[0]) > 2 and abs(x - index[-1]) > 2:
                        index_labels.append(x + start_hour - 1)
                        index_positions.append(x)

            axs[i, j].set_xticks(index_positions)
            axs[i, j].set_xticklabels(index_labels)
            if i == nr_cases - 1:
                axs[i, j].set_xlabel("Individual time steps (hours 1 to 144)")
            for x in axvline_thick_positions:
                axs[i, j].axvline(x=x, color="gray", linestyle="--", alpha=0.5)
            for x in axvline_thin_positions:
                axs[i, j].axvline(x=x, color="gray", linestyle="-", alpha=0.2)

    plt.tight_layout()

    # Save plot using a naming scheme matching the sqlite files
    base = os.path.splitext(sqlite_files[0])[0]
    label_suffix = f"-{case_labels[0].strip().replace('.', '').replace(' ', '')}"
    plot_prefix = base[:-len(label_suffix)] if base.endswith(label_suffix) else base
    plot_file = f"{plot_prefix}-unit_commitment.png"
    plt.savefig(plot_file)
    printer.information(f"Saved unit commitment plot to '{plot_file}'")

    if not no_show:
        plt.show()


def _load_unit_commitment_from_sqlite(sqlite_file, case_label):
    """Load unit commitment data from a sqlite file and return a DataFrame with case/rp/k/g index."""
    g_rename = {"thermalGenerators": "g"}
    with closing(sqlite3.connect(sqlite_file)) as cnx:
        vCommit = pd.read_sql("SELECT * FROM vCommit", cnx).rename(columns={"values": "vCommit", **g_rename})
        vStartup = pd.read_sql("SELECT * FROM vStartup", cnx).rename(columns={"values": "vStartup", **g_rename})
        vShutdown = pd.read_sql("SELECT * FROM vShutdown", cnx).rename(columns={"values": "vShutdown", **g_rename})
        vGenP = pd.read_sql("SELECT * FROM vGenP", cnx).rename(columns={"values": "vGenP"})
        vPNS = pd.read_sql("SELECT * FROM vPNS", cnx).rename(columns={"values": "vPNS"})
        vEPS = pd.read_sql("SELECT * FROM vEPS", cnx).rename(columns={"values": "vEPS"})
        pDemandP = pd.read_sql("SELECT * FROM pDemandP", cnx).rename(columns={"values": "pDemandP"})
        pMinUpTime = pd.read_sql("SELECT * FROM pMinUpTime", cnx).rename(columns={"values": "pMinUpTime", **g_rename})
        pMinDownTime = pd.read_sql("SELECT * FROM pMinDownTime", cnx).rename(columns={"values": "pMinDownTime", **g_rename})

    idx = ["rp", "k", "g"]
    df = vCommit.set_index(idx)
    df = df.join(vStartup.set_index(idx)["vStartup"])
    df = df.join(vShutdown.set_index(idx)["vShutdown"])
    df = df.join(vGenP.set_index(idx)["vGenP"])

    df = df.join(pDemandP.groupby(["rp", "k"])["pDemandP"].sum(), on=["rp", "k"])
    df = df.join(vPNS.groupby(["rp", "k"])["vPNS"].sum(), on=["rp", "k"])
    df = df.join(vEPS.groupby(["rp", "k"])["vEPS"].sum(), on=["rp", "k"])

    df = df.join(pMinUpTime.set_index("g"), on="g")
    df = df.join(pMinDownTime.set_index("g"), on="g")

    df["case"] = case_label
    return df.reset_index().set_index(["case", "rp", "k", "g"])


def run_tables(entries: list[dict], args) -> None:
    """Print one block per run-parameter group: the main table, then a table per further run kind."""
    groups = defaultdict(lambda: defaultdict(list))  # group key -> kind -> rows
    for entry in entries:
        groups[tuple(entry.get(k) for k in GROUP_KEYS)][entry['kind']].append(_build_table_row(entry))

    for kinds in groups.values():
        any_row = next(rows[0] for rows in kinds.values() if rows)
        printer.information(f"\n{'=' * 80}")
        printer.information(f"Group: {_group_header(any_row)}")
        printer.information(f"{'=' * 80}")

        if kinds.get('main'):
            print_comparison_table(kinds['main'])
        for kind in RUN_KINDS[1:]:
            if kinds.get(kind):
                printer.information("")
                printer.information(f"  {KIND_TITLES[kind]}")
                print_comparison_table(kinds[kind])

        if args.plot and kinds.get('main'):
            sorted_entries = sorted(kinds['main'], key=lambda e: edge_sort_key(e.get('edge_handling')))
            try:
                plot_unit_commitment([e['file'] for e in sorted_entries], [e.get('edge_handling', '') for e in sorted_entries],
                                     args.case_study_folder or any_row.get('case_study_directory'), args.number_of_hours, args.start_hour, args.no_show)
            except Exception as e:
                printer.error(f"Failed to plot: {e}")


########################################################################################################################
# 'plots': boxplots vs Truth across TM variants, and the results table behind them
########################################################################################################################

GAP_COLOR = '#ff6666'  # MIP-gap noise band on the invest-regret plots

# Base filename per logical plot (the "(excl. NoEnf)" twin inserts "_noNoEnf")
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


def _of_kind(entries: list[dict], kind: str) -> list[dict]:
    return [e for e in entries if e['kind'] == kind]


def _iter_vs_truth(entries: list[dict], edges: list[str], edge_value):
    """Group entries by (TM variant, sub-case) and pair each edge with the Truth entry of its group.
    Yields (tm_key, edge, truth_value, value) for every edge entry with a non-None value in a group with Truth."""
    grouped = defaultdict(lambda: defaultdict(dict))
    for e in entries:
        if e.get('edge_handling') is not None:
            grouped[tm_key(e)][subcase_key(e)][e['edge_handling']] = e
    for key, subcases in grouped.items():
        for eh_map in subcases.values():
            truth = eh_map.get('Truth')
            if truth is None:
                continue
            for edge in edges:
                ent = eh_map.get(edge)
                if ent is not None and edge_value(ent) is not None:
                    yield key, edge, edge_value(truth), edge_value(ent)


def build_runtime_boxes(entries: list[dict], edges: list[str]) -> dict:
    """Absolute work units per (tm_key, edge), aggregated over sub-cases."""
    boxes = defaultdict(lambda: defaultdict(list))
    for e in entries:
        if e.get('edge_handling') in edges and e.get('work_units') is not None:
            boxes[tm_key(e)][e['edge_handling']].append(float(e['work_units']))
    return boxes


def build_runtime_relative_boxes(entries: list[dict], edges: list[str]) -> dict:
    """Work units as % of the Truth run of the same (tm, sub-case) (needs Gurobi work units on both)."""
    boxes = defaultdict(lambda: defaultdict(list))
    for key, edge, truth, val in _iter_vs_truth(entries, edges, lambda e: e.get('work_units')):
        if truth not in (None, 0):
            boxes[key][edge].append(float(val) / float(truth) * 100)
    return boxes


def build_deviation_boxes(entries: list[dict], mode: str, edges: list[str], var: str = 'vShutdown', magnitude: bool = False) -> dict:
    """Deviation of a weighted UC sum (vShutdown / vStartup) from the Truth entry of the same (tm, sub-case).
    mode='relative': (val - truth) / |truth| * 100 (skips truth == 0); 'absolute': val - truth.
    magnitude: store |deviation| instead of the signed value."""
    boxes = defaultdict(lambda: defaultdict(list))
    for key, edge, truth, val in _iter_vs_truth(entries, edges, lambda e: e.get(var)):
        if truth is None or (mode == 'relative' and truth == 0):
            continue
        diff = float(val) - float(truth)
        if mode == 'relative':
            diff = diff / abs(float(truth)) * 100
        boxes[key][edge].append(abs(diff) if magnitude else diff)
    return boxes


def build_regret_boxes(entries: list[dict], kind: str, mode: str, edges: list[str]) -> dict:
    """Regret (see attach_regret) of the regret runs of one kind (RP-copies reference only):
    'absolute' -> objective - reference objective (M EUR), 'relative' -> in % of the reference objective."""
    boxes = defaultdict(lambda: defaultdict(list))
    for e in entries:
        if e['kind'] == kind and e.get('reference') is None and e.get('edge_handling') in edges:
            value = e.get('regret_pct' if mode == 'relative' else 'regret')
            if value is not None:
                boxes[tm_key(e)][e['edge_handling']].append(value)
    return boxes


def invest_regret_gap_band(entries: list[dict], mode: str, edges: list[str]) -> tuple[float, float] | None:
    """(min, max) of the MIP-gap noise magnitude over the plotted invest-regret points: a regret smaller than
    mip_gap * objective could be solver tolerance alone. 'relative': mip_gap * 100 (a line for a uniform gap),
    'absolute': mip_gap * |invest-regret objective|. None if no plotted point carries a requested mip_gap."""
    mags = [e['mip_gap'] * 100.0 if mode == 'relative' else e['mip_gap'] * abs(e['Objective'])
            for e in entries if e['kind'] == 'invest_regret' and e.get('reference') is None and e.get('edge_handling') in edges
            and e.get('regret') is not None and e.get('mip_gap') is not None]
    return (min(mags), max(mags)) if mags else None


def make_boxplot_figure(boxes: dict, title: str, ylabel: str, output: str | None, no_show: bool, edges: list[str],
                        logy: bool = False, ref_line: float | None = None, symmetric_y: bool = False,
                        nonneg_y: bool = False, tight_y: bool = False, gap_band: tuple[float, float] | None = None,
                        data_label: str = '(base)'):
    """Render one figure: a subplot per TM combination (shared y-axis), one box per edge handling.

    ref_line: horizontal reference line. At most one y-axis mode: symmetric_y (centred on 0, for signed values),
    nonneg_y ([0, max], for magnitudes), tight_y ([min(0, min), max], for regrets, which are expected >= 0 but can
    dip below from solver noise). gap_band (lo, hi): MIP-gap noise band at [lo, hi] and [-hi, -lo] (a pair of lines
    if lo == hi). data_label titles the unperturbed ('base') TM subplot. The figure is closed before returning.
    """
    import matplotlib.pyplot as plt

    if not any(boxes[t].get(e) for t in boxes for e in edges):
        printer.information(f"  [skip] {title}: no data")
        return

    tm_keys = sorted(boxes.keys(), key=_tm_sort)
    fig, axes = plt.subplots(1, len(tm_keys), figsize=(max(2.6 * len(edges) * len(tm_keys) / 3, 4.0), 5.0), sharey=True, squeeze=False)
    axes = axes[0]
    positions = {edge: i + 1 for i, edge in enumerate(edges)}
    all_vals = [v for key in tm_keys for edge in edges for v in boxes[key].get(edge, [])]
    use_log = logy and all(v > 0 for v in all_vals)  # Log scale only if every drawn value is strictly positive

    for ax, key in zip(axes, tm_keys):
        for edge in edges:
            data = boxes[key].get(edge, [])
            if not data:
                continue
            bp = ax.boxplot([data], positions=[positions[edge]], widths=0.6, patch_artist=True, showfliers=True)
            for patch in bp['boxes']:
                patch.set_facecolor(EDGE_COLORS[edge])
                patch.set_alpha(0.75)
            for med in bp['medians']:
                med.set_color('black')
            ax.annotate(f"n={len(data)}", xy=(positions[edge], 0.99), xycoords=('data', 'axes fraction'),
                        ha='center', va='top', fontsize=7, color='dimgray')

        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels([EDGE_LABELS.get(e, e) for e in edges], fontsize=8)
        ax.set_xlim(0.5, len(edges) + 0.5)
        ax.set_title(_tm_label(*key, base_label=data_label), fontsize=9, fontweight='bold')
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

    band_hi = abs(gap_band[1]) if gap_band is not None else 0.0  # Keep the band in view even if the values are tiny
    if not use_log and (symmetric_y or nonneg_y):
        ymax = max(max((abs(v) for v in all_vals), default=0.0), band_hi)
        if ymax > 0:
            axes[0].set_ylim(-ymax * 1.05 if symmetric_y else 0, ymax * 1.05)  # sharey propagates to all
    elif not use_log and tight_y:
        lo = min(min(all_vals, default=0.0) * 1.05, 0.0, -band_hi)
        hi = max(max(all_vals, default=0.0) * 1.05, 0.0, band_hi)
        if lo < 0 or hi > 0:
            axes[0].set_ylim(lo, hi)

    axes[0].set_ylabel(ylabel)
    fig.tight_layout()  # No legend / suptitle: x-tick labels, consistent colors and subplot titles carry the context
    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        printer.information(f"  Saved {output}")
    if not no_show:
        plt.show()
    plt.close(fig)


def render_plots(entries: list[dict], edges: list[str], out_dir: str, args, fname_suffix: str = "", title_suffix: str = ""):
    """Emit all logical plots, each with a NoEnf-excluded '_noNoEnf' twin."""
    main_entries, operational_entries = _of_kind(entries, 'main'), _of_kind(entries, 'operational')
    data_label = _data_label(main_entries + operational_entries)

    def emit(boxes, title, ylabel, name_key, gap_fn=None, **kwargs):
        stem, ext = os.path.splitext(OUTPUT_NAMES[name_key])
        stem += fname_suffix
        title += title_suffix
        for drawn_edges, name, twin_title in [(edges, stem, title), ([e for e in edges if e != 'NoEnf'], f"{stem}_noNoEnf", f"{title} (excl. NoEnf)")]:
            make_boxplot_figure(boxes, twin_title, ylabel, os.path.join(out_dir, name + ext), args.no_show, drawn_edges,
                                gap_band=gap_fn(drawn_edges) if gap_fn else None, data_label=data_label, **kwargs)

    for run_entries, run_label, key in [(operational_entries, "Operational runs", "operational"), (main_entries, "Investment runs", "investment")]:
        emit(build_runtime_boxes(run_entries, edges), f"Work units — {run_label}", "Work units",
             f'workunits_{key}_abs', logy=args.logscale)
        emit(build_runtime_relative_boxes(run_entries, edges), f"Work units (% of Truth) — {run_label}", "Work units [% of Reference]",
             f'workunits_{key}_rel', logy=args.logscale)
        emit(build_deviation_boxes(run_entries, 'absolute', edges), f"vShutdown deviation vs Truth — {run_label}",
             "Absolute deviation from Truth", f'vshutdown_{key}_abs', ref_line=0, symmetric_y=True)
        emit(build_deviation_boxes(run_entries, 'relative', edges), f"vShutdown deviation vs Truth — {run_label}",
             "Deviation of shutdowns relative to the Reference [%]" if key == "operational" else "Relative deviation from Truth [%]",
             f'vshutdown_{key}_rel', ref_line=0, symmetric_y=True)
        emit(build_deviation_boxes(run_entries, 'absolute', edges, magnitude=True), f"vShutdown |deviation| vs Truth — {run_label}",
             "Absolute value of deviation from Truth", f'vshutdown_{key}_abs_mag', nonneg_y=True)
        emit(build_deviation_boxes(run_entries, 'relative', edges, magnitude=True), f"vShutdown |deviation| vs Truth — {run_label}",
             "Unsigned relative deviation of shutdowns [%]" if key == "operational" else "Absolute value of relative deviation from Truth [%]",
             f'vshutdown_{key}_rel_mag', nonneg_y=True)

    regret_plots = [('invest_regret', "Invest-regret vs Truth — Investment runs", "Investment regret [million EUR]", "Invest-regret over Truth objective [%]"),
                    ('regret', "Regret vs Truth — Investment runs", "Absolute regret over Truth objective", "Regret over Truth objective [%]"),
                    ('operational_regret', "Operational-regret vs Truth — Operational runs", "Absolute operational-regret over Truth-operational objective",
                     "Operational-regret over Truth-operational objective [%]")]
    for kind, title, ylabel_abs, ylabel_rel in regret_plots:
        for mode, ylabel, suffix in [('absolute', ylabel_abs, 'abs'), ('relative', ylabel_rel, 'rel')]:
            gap_fn = (lambda eds, m=mode: invest_regret_gap_band(entries, m, eds)) if kind == 'invest_regret' else None
            emit(build_regret_boxes(entries, kind, mode, edges), title, ylabel, f'{kind}_{suffix}', gap_fn=gap_fn, ref_line=0, tight_y=True)


def _agg(vals: list[float] | None) -> dict | None:
    """mean / median / min / max / n of a box's values (None when empty)."""
    if not vals:
        return None
    return {'mean': statistics.mean(vals), 'median': statistics.median(vals), 'min': min(vals), 'max': max(vals), 'n': len(vals)}


def _aggregate_boxes(boxes: dict, edges: list[str]) -> dict:
    """{tm_key: {edge: [vals]}} -> {(tm_key, edge): agg-dict}."""
    return {(key, edge): _agg(edge_map.get(edge)) for key, edge_map in boxes.items() for edge in edges if edge_map.get(edge)}


def truth_objective_by_tm(entries: list[dict]) -> dict:
    """{tm_key: agg-dict} of the Truth-entry Objective per TM combination [M EUR]."""
    boxes = defaultdict(list)
    for e in entries:
        if e.get('edge_handling') == 'Truth' and e.get('Objective') not in (None, 0):
            boxes[tm_key(e)].append(float(e['Objective']))
    return {key: _agg(vals) for key, vals in boxes.items()}


def build_table_records(entries: list[dict], edges: list[str]):
    """Aggregate the boxes behind the plots into per-(TM_variant, method) records - the same builders the figures use,
    so the numbers match the plots exactly. Returns (records, truth_main, truth_oper)."""
    main_entries, operational_entries = _of_kind(entries, 'main'), _of_kind(entries, 'operational')
    metrics = {
        'oper_dev': _aggregate_boxes(build_deviation_boxes(operational_entries, 'relative', edges), edges),
        'startup_dev': _aggregate_boxes(build_deviation_boxes(operational_entries, 'relative', edges, var='vStartup'), edges),
        'invest_regret': _aggregate_boxes(build_regret_boxes(entries, 'invest_regret', 'absolute', edges), edges),
        'wu_oper': _aggregate_boxes(build_runtime_relative_boxes(operational_entries, edges), edges),
        'wu_invest': _aggregate_boxes(build_runtime_relative_boxes(main_entries, edges), edges),
        'invest_dev': _aggregate_boxes(build_deviation_boxes(main_entries, 'relative', edges), edges),
    }
    tm_keys = {key for values in metrics.values() for key, _ in values}
    records = []
    for key in sorted(tm_keys, key=_tm_sort):
        for edge in sorted(edges, key=edge_sort_key):
            rec = {'TM_variant': tm_variant_label(key), 'method': edge, **{name: values.get((key, edge)) for name, values in metrics.items()}}
            if any(rec[name] for name in metrics):
                records.append(rec)
    return records, truth_objective_by_tm(main_entries), truth_objective_by_tm(operational_entries)


def _f(x, fmt="%.1f") -> str:
    return "n/a" if x is None else fmt % x


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def render_results_markdown(records, truth_main, truth_oper, title_suffix="") -> str:
    """Three Markdown tables: paper results, diagnostics, reference Truth objective."""
    out = [f"# Markov results table{title_suffix}\n", "## Mean over sub-cases per TM_variant x method\n"]
    headers = ["TM_variant", "method", "oper_dev_mean_pct", "oper_dev_median_pct", "oper_dev_min_pct", "oper_dev_max_pct",
               "invest_regret_mean_MEUR", "invest_regret_median_MEUR", "wu_oper_mean_pct", "wu_invest_mean_pct", "n_runs"]
    rows = []
    for r in records:
        od, ir = r['oper_dev'], r['invest_regret']
        rows.append([r['TM_variant'], r['method'],
                     _f(od and od['mean']), _f(od and od['median']), _f(od and od['min']), _f(od and od['max']),
                     _f(ir and ir['mean']), _f(ir and ir['median']),
                     _f(r['wu_oper'] and r['wu_oper']['mean']), _f(r['wu_invest'] and r['wu_invest']['mean']),
                     str(od['n'] if od else (ir['n'] if ir else 0))])
    out.append(_md_table(headers, rows))
    out.append("\n`n_runs` is the operational-deviation count (the primary quantity); per-metric counts are in the "
               "diagnostics table below. invest_regret is in M EUR (LEGO objective unit). An 'n/a' cell means no sub-case "
               "had the data (e.g. no `--operational` runs, no Gurobi work units, or no optimal Truth reference).\n")

    out.append("## Diagnostics (start-up deviation, per-metric run counts)\n")
    dheaders = ["TM_variant", "method", "startup_dev_mean_pct", "startup_dev_median_pct", "invest_run_shutdown_dev_mean_pct",
                "n_oper_dev", "n_startup", "n_regret", "n_wu_oper", "n_wu_invest"]
    drows = []
    for r in records:
        sd, idv = r['startup_dev'], r['invest_dev']
        drows.append([r['TM_variant'], r['method'], _f(sd and sd['mean']), _f(sd and sd['median']), _f(idv and idv['mean'])]
                     + [str(r[k]['n'] if r[k] else 0) for k in ['oper_dev', 'startup_dev', 'invest_regret', 'wu_oper', 'wu_invest']])
    out.append(_md_table(dheaders, drows))
    out.append("\nStart-ups equal shut-downs for Cyclic/Markov and differ for NoEnf. `invest_run_shutdown_dev_mean_pct` is the "
               "deviation on the regular (investment) runs - a fallback view when no `--operational` runs are present.\n")

    out.append("## Reference Truth objective per TM_variant [M EUR]\n")
    rrows = []
    for key in sorted(set(truth_main) | set(truth_oper), key=_tm_sort):
        tm, to = truth_main.get(key), truth_oper.get(key)
        rrows.append([tm_variant_label(key), _f(tm and tm['mean']), str(tm['n'] if tm else 0), _f(to and to['mean']), str(to['n'] if to else 0)])
    out.append(_md_table(["TM_variant", "truth_main_obj_mean_MEUR", "n_main", "truth_oper_obj_mean_MEUR", "n_oper"], rrows))
    return "\n".join(out) + "\n"


def write_results_csv(path, records):
    cols = ["TM_variant", "method", "oper_dev_mean_pct", "oper_dev_median_pct", "oper_dev_min_pct", "oper_dev_max_pct",
            "startup_dev_mean_pct", "startup_dev_median_pct", "invest_regret_mean_MEUR", "invest_regret_median_MEUR",
            "invest_regret_min_MEUR", "invest_regret_max_MEUR", "wu_oper_mean_pct", "wu_invest_mean_pct",
            "invest_run_shutdown_dev_mean_pct", "n_oper_dev", "n_startup", "n_regret", "n_wu_oper", "n_wu_invest"]

    def g(agg, field):
        return "" if not agg else round(agg[field], 3)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            od, sd, ir = r['oper_dev'], r['startup_dev'], r['invest_regret']
            w.writerow([r['TM_variant'], r['method'],
                        g(od, 'mean'), g(od, 'median'), g(od, 'min'), g(od, 'max'),
                        g(sd, 'mean'), g(sd, 'median'),
                        g(ir, 'mean'), g(ir, 'median'), g(ir, 'min'), g(ir, 'max'),
                        g(r['wu_oper'], 'mean'), g(r['wu_invest'], 'mean'), g(r['invest_dev'], 'mean')]
                       + [r[k]['n'] if r[k] else 0 for k in ['oper_dev', 'startup_dev', 'invest_regret', 'wu_oper', 'wu_invest']])


def report_results_table(entries: list[dict], edges: list[str], out_dir: str, fname_suffix="", title_suffix=""):
    """Build, print (terminal) and save (.txt + .csv) the aggregated results table."""
    records, truth_main, truth_oper = build_table_records(entries, edges)
    if not records:
        printer.warning(f"  Results table{title_suffix}: no (TM_variant, method) cells had data.")
        return
    report = render_results_markdown(records, truth_main, truth_oper, title_suffix)
    try:  # Plain print avoids rich re-wrapping the table; fall back for a strict-ASCII console
        print("\n" + report)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(("\n" + report + "\n").encode("utf-8", "replace"))
    stem = os.path.join(out_dir, "compare_markov_results" + fname_suffix)
    with open(stem + ".txt", "w", encoding="utf-8") as fh:
        fh.write(report)
    write_results_csv(stem + ".csv", records)
    printer.information(f"  Saved {stem}.txt")
    printer.information(f"  Saved {stem}.csv")


def run_plots(entries: list[dict], args, out_dir: str) -> None:
    edges = EDGE_BOXES + ([EDGE_STRICT] if args.markov_strict else [])
    counts = {kind: len(_of_kind(entries, kind)) for kind in RUN_KINDS}
    printer.information(f"  Using [{'all' if args.include_nonoptimal else 'optimal-only'}]: " + ", ".join(f"{kind}={n}" for kind, n in counts.items()))
    if args.separateClusters:
        for value in sorted({e.get('clusters') for e in entries}, key=lambda v: (v is None, v)):
            label = f"{value} clusters" if value is not None else "no clustering"
            printer.information(f"Rendering plots for {label} ...")
            subset = [e for e in entries if e.get('clusters') == value]
            render_plots(subset, edges, out_dir, args, fname_suffix=f"_clusters{value}", title_suffix=f" — {label}")
            if not args.no_results_table:
                report_results_table(subset, edges, out_dir, fname_suffix=f"_clusters{value}", title_suffix=f" — {label}")
    else:
        render_plots(entries, edges, out_dir, args)
        if not args.no_results_table:
            report_results_table(entries, edges, out_dir)


########################################################################################################################
# 'summary': CSVs and plots of the per-run analysis tables
########################################################################################################################

TEXT_SECONDARY = '#52514e'
SUMMARY_KEYS = ['dataset', 'clusters', 'edge_handling', 'run_kind']
SUMMARY_METRICS = ['num_vars', 'num_bin_vars', 'num_constrs', 'num_nonzeros', 'tm_nonzero', 'tm_entries', 'solver_max_mem_gb',
                   'process_peak_rss_gb', 'solver_runtime_s', 'work_units', 'Objective', 'operating_cost', 'investment_cost',
                   'curtailment_pct', 'curtailment_mwh', 'load_shedding_pct', 'load_shedding_mwh', 'uc_corrections',
                   'regret', 'regret_pct',
                   'nb_transitions_fractional_pct', 'nb_transitions_max_delta', 'nb_transitions_mean_delta_fractional',
                   'nb_all_window_fractional_pct', 'nb_all_max_delta',
                   'feas_infeasible_pct', 'feas_infeasible_rounded_pct']


def runs_dataframe(entries: list[dict]) -> pd.DataFrame:
    """One row per run, with dataset and run kind ('@original' appended for the original-reference runs)."""
    runs = pd.DataFrame(entries)
    runs['dataset'] = runs['case_study_directory'].map(lambda d: _dataset_name(d) if d else None)
    runs['run_kind'] = runs['kind'] + np.where(runs['reference'] == 'original', '@original', '')
    return runs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean over the variants (stretch demand, TM shift/perturbation, ...) per dataset x number of RPs x edge handling
    x run kind, plus the maximum of the max-type metrics."""
    metrics = [m for m in SUMMARY_METRICS if m in runs.columns]
    numeric = runs[SUMMARY_KEYS + metrics].copy()
    for m in metrics:
        numeric[m] = pd.to_numeric(numeric[m], errors='coerce')
    grouped = numeric.groupby(SUMMARY_KEYS, dropna=False)
    summary = grouped[metrics].mean()
    summary.insert(0, 'n_runs', grouped.size())
    for m in ['nb_transitions_max_delta', 'nb_all_max_delta', 'solver_max_mem_gb', 'process_peak_rss_gb']:
        if m in metrics:
            summary[f"{m}_max"] = grouped[m].max()
    return summary.reset_index()


def _style_axis(ax):
    ax.grid(axis='y', color='#e5e4e0', linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)


def plot_nonbinarity(runs: pd.DataFrame, values: pd.DataFrame, output_dir: str) -> None:
    """Per dataset: histogram of the fractionality of each RP transition instance (max distance to 0/1 of the UC
    variables in the relaxed window), weighted by how often the transition occurs in the chronology. Bars are in % of
    all transition instances, so the bars of a panel sum up to the share of fractional transitions."""
    import matplotlib.pyplot as plt
    if values.empty:
        printer.information("[skip] nonbinarity plots: no fractional UC values found")
        return
    main = runs[(runs['run_kind'] == 'main') & runs['edge_handling'].isin(['Markov', 'Markov-Strict'])]
    values = values[values['in_window'].astype(bool) & values['file'].isin(main['file'])]
    bins = np.linspace(0, 0.5, 11)
    for dataset, ds_runs in main.groupby('dataset'):
        rps = sorted(ds_runs['clusters'].dropna().unique())
        if not rps:
            continue
        edges = [e for e in EDGE_DISPLAY_ORDER if e in set(ds_runs['edge_handling'])]
        fig, axes = plt.subplots(1, len(rps), figsize=(2.6 * len(rps) + 1, 3.2), sharey=True, squeeze=False)
        for ax, n_rp in zip(axes[0], rps):
            for offset, edge in enumerate(edges):
                files = ds_runs[(ds_runs['clusters'] == n_rp) & (ds_runs['edge_handling'] == edge)]
                total = pd.to_numeric(files.get('nb_transition_instances'), errors='coerce').sum()
                if total <= 0:
                    continue
                inst = values[values['file'].isin(files['file'])].groupby(['file', 'rp', 'g']).agg(delta=('delta', 'max'), weight=('transitions_into_rp', 'first'))
                hist, _ = np.histogram(inst['delta'], bins=bins, weights=inst['weight'])
                width = (bins[1] - bins[0]) / len(edges)
                ax.bar(bins[:-1] + offset * width, hist / total * 100, width=width * 0.9, align='edge',
                       color=EDGE_COLORS.get(edge, '#2a78d6'), label=EDGE_LABELS.get(edge, edge))
            ax.set_title(f"{int(n_rp)} RPs", fontsize=10)
            ax.set_xlim(0, 0.5)
            ax.set_xlabel("distance to 0/1", fontsize=9, color=TEXT_SECONDARY)
            _style_axis(ax)
        axes[0][0].set_ylabel("% of RP transitions", fontsize=9, color=TEXT_SECONDARY)
        if len(edges) > 1:
            axes[0][-1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = os.path.join(output_dir, f"nonbinarity_{dataset}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        printer.information(f"Saved {path}")


def plot_feasibility(runs: pd.DataFrame, output_dir: str) -> None:
    """Per dataset: % of chronological transitions (boundary x unit) infeasible in the full chronological model, per
    number of RPs. Solid: strict (fractional values count as violation), dashed: after rounding the commitment."""
    import matplotlib.pyplot as plt
    main = runs[(runs['run_kind'] == 'main') & runs['feas_infeasible_pct'].notna()] if 'feas_infeasible_pct' in runs else pd.DataFrame()
    if main.empty:
        printer.information("[skip] feasibility plots: no feasibility results found")
        return
    for dataset, ds_runs in main.groupby('dataset'):
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        for edge in [e for e in EDGE_DISPLAY_ORDER if e in set(ds_runs['edge_handling'])]:
            agg = ds_runs[ds_runs['edge_handling'] == edge].groupby('clusters')[['feas_infeasible_pct', 'feas_infeasible_rounded_pct']].agg(
                lambda s: pd.to_numeric(s, errors='coerce').mean())
            if agg.empty:
                continue
            color = EDGE_COLORS.get(edge, '#2a78d6')
            label = EDGE_LABELS.get(edge, edge)
            ax.plot(agg.index, agg['feas_infeasible_pct'], color=color, linewidth=2, marker='o', markersize=5, label=label)
            ax.plot(agg.index, agg['feas_infeasible_rounded_pct'], color=color, linewidth=2, linestyle='--', marker='o', markersize=5,
                    markerfacecolor='white', label=f"{label} (rounded)")
        ax.set_xticks(sorted(ds_runs['clusters'].dropna().unique()))
        ax.set_xlabel("number of RPs", fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylabel("% infeasible RP transitions", fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylim(bottom=0)
        _style_axis(ax)
        ax.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        path = os.path.join(output_dir, f"feasibility_{dataset}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        printer.information(f"Saved {path}")


def run_summary(all_entries: list[dict], selected: list[dict], values: pd.DataFrame, args, out_dir: str) -> None:
    """markov_runs.csv: every run (incl. non-optimal and filtered-out ones); markov_summary.csv and plots: the
    selected runs (optimal unless --include-nonoptimal, --nrOfClusters / --tm)."""
    runs_path = os.path.join(out_dir, "markov_runs.csv")
    runs_dataframe(all_entries).to_csv(runs_path, index=False)
    printer.information(f"Saved {runs_path} ({len(all_entries)} runs)")
    if not selected:
        printer.warning("No runs selected for markov_summary.csv and the summary plots")
        return
    runs = runs_dataframe(selected)
    summary_path = os.path.join(out_dir, "markov_summary.csv")
    summarize(runs).to_csv(summary_path, index=False)
    printer.information(f"Saved {summary_path} (from {len(selected)} runs)")
    if not args.no_plots:
        plot_nonbinarity(runs, values, out_dir)
        plot_feasibility(runs, out_dir)


########################################################################################################################
# CLI
########################################################################################################################

def _add_tables_args(p):
    g = p.add_argument_group("tables")
    g.add_argument("--plot", action="store_true", help="Plot the unit commitment of each group's main runs")
    g.add_argument("--case-study-folder", type=str, default=None, help="Case study folder for the unit-commitment plot (fallback if neither hindex nor case_study_directory are in the .sqlite)")
    g.add_argument("--number-of-hours", type=int, default=6 * 24, help="Number of hours in the unit-commitment plot (default: 144)")
    g.add_argument("--start-hour", type=int, default=1, help="Start hour of the unit-commitment plot (default: 1)")


def _add_plots_args(p):
    g = p.add_argument_group("plots")
    g.add_argument("--logscale", action="store_true", help="Log-scale y-axis for the work-units plots (default: linear)")
    g.add_argument("--markov-strict", action="store_true", help="Also draw a Markov-Strict box (runs with --enable-strict-markov)")
    g.add_argument("--separateClusters", action="store_true", help="Emit the full plot set (and results table) once per cluster count, with a '_clusters{N}' filename suffix")
    g.add_argument("--no-results-table", action="store_true", help="Skip the aggregated results table (compare_markov_results.txt/.csv)")


def _add_summary_args(p):
    g = p.add_argument_group("summary")
    g.add_argument("--no-plots", action="store_true", help="Only write the CSV tables")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("folder", nargs="?", default=".", help="Folder containing MK-*.sqlite files (default: current directory)")
    common.add_argument("--recursive", action="store_true", help="Also search subfolders for MK-*.sqlite files")
    common.add_argument("--output-dir", default=None, help="Directory for PNGs, CSVs and the results table (default: the input folder)")
    common.add_argument("--include-nonoptimal", action="store_true", help="plots/summary: also use runs that did not solve to optimality (tables always show all runs)")
    common.add_argument("--nrOfClusters", default=None, metavar="N,N,...", help="Only runs whose 'clusters' run parameter is in this comma-separated list (e.g. 3,5,7)")
    common.add_argument("--tm", action="append", default=None, metavar="SHIFT:PERTURB",
                        help="Only the selected (shift_tm, perturb_tm) combinations. Repeatable and/or comma-separated; each side a number, "
                             "'none' (unset) or '*' (any); 'base' = none:none. Examples: --tm none:0.2 --tm 1:none ; --tm \"base,1:*\"")
    common.add_argument("--no-show", action="store_true", help="Don't display figures (only save them)")

    parser = argparse.ArgumentParser(description="Evaluate the Markov edge-handling results (MK-*.sqlite files)", formatter_class=RichHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text, adders in [("tables", "Per-group comparison tables of all runs (optionally with unit-commitment plots)", [_add_tables_args]),
                                    ("plots", "Boxplots vs Truth across TM variants and the results table behind them", [_add_plots_args]),
                                    ("summary", "CSVs and plots of the per-run analysis tables (metrics, non-binarity, feasibility)", [_add_summary_args]),
                                    ("all", "tables + plots + summary from a single load", [_add_tables_args, _add_plots_args, _add_summary_args])]:
        sub = subparsers.add_parser(name, parents=[common], help=help_text, description=help_text, formatter_class=RichHelpFormatter)
        for adder in adders:
            adder(sub)
    args = parser.parse_args()

    try:
        cluster_filter = {int(t) for t in args.nrOfClusters.split(',') if t.strip()} if args.nrOfClusters else None
    except ValueError:
        parser.error(f"--nrOfClusters expects a comma-separated list of integers, got '{args.nrOfClusters}'")
    try:
        tm_specs = [parse_tm_spec(s) for raw in (args.tm or []) for s in raw.split(',') if s.strip()]
    except ValueError as exc:
        parser.error(f"--tm: {exc}")
    out_dir = args.output_dir or args.folder
    os.makedirs(out_dir, exist_ok=True)

    do_tables, do_plots, do_summary = (args.command in (c, "all") for c in ("tables", "plots", "summary"))
    entries, values = load_entries(args.folder, recursive=args.recursive, full=do_tables, analysis=do_summary)
    if not entries:
        printer.warning(f"No MK-*.sqlite files found in '{args.folder}'")
        return
    attach_regret(entries, include_nonoptimal=args.include_nonoptimal)

    if do_tables:
        run_tables([e for e in entries if _matches_filters(e, cluster_filter, tm_specs)], args)
    if do_plots or do_summary:
        selected = select_entries(entries, args.include_nonoptimal, cluster_filter, tm_specs)
        if do_plots:
            if not _of_kind(selected, 'main') and not _of_kind(selected, 'operational'):
                printer.warning("plots: no usable runs" + ("" if args.include_nonoptimal else " (optimal only)") + " after filtering")
            else:
                run_plots(selected, args, out_dir)
        if do_summary:
            run_summary(entries, selected, values, args, out_dir)


if __name__ == "__main__":
    main()
