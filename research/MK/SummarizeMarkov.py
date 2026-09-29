#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SummarizeMarkov.py - Collect the per-run analysis tables that Markov.py writes into every MK-*.sqlite
(mk_metrics, mk_nonbinarity, mk_nonbinary_values, mk_feasibility - see MarkovAnalysis.py) into CSV tables and
plots for the paper.

Outputs (in --output-dir, default: the input folder):
    markov_runs.csv               one row per sqlite file: run parameters, solver statistics, all metrics,
                                  non-binarity and feasibility values, plus the objective of the matching
                                  reference (Truth / Truth-operational / TruthOriginal) and the regret against it
    markov_summary.csv            mean over the variants (stretch demand, TM shift/perturbation) per
                                  dataset x number of RPs x edge handling x run kind
    nonbinarity_{dataset}.png     distribution of how non-binary the UC variables are at RP transitions, one panel
                                  per number of RPs (Markov edge handlings, main runs)
    feasibility_{dataset}.png     share of chronological transitions that are infeasible (strict / rounded) per
                                  number of RPs and edge handling (main runs)

Usage
-----
python research/MK/SummarizeMarkov.py [folder] [--output-dir DIR] [--include-nonoptimal] [--no-plots]
"""

import argparse
import glob
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rich_argparse import RichHelpFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from CompareMarkov import EDGE_COLORS, EDGE_LABELS, _dataset_name  # noqa: E402

TEXT_SECONDARY = '#52514e'
EDGE_DISPLAY_ORDER = ['Truth', 'NoEnf', 'Cyclic', 'Markov', 'Markov-Strict', 'TruthOriginal']
_CLUSTER_SUFFIX = re.compile(r'[/\\]?\d+ clusters[/\\]?$')

# Run parameters that identify a variant; the reference Truth of a run shares all of them (except the edge handling)
VARIANT_KEYS = ['dataset', 'base_directory', 'clusters', 'shift_tm', 'perturb_tm', 'filter_zone', 'limit_k', 'shift', 'stretch_demand',
                'scale_vres', 'scale_invest_cost', 'thermal_invest_only', 'merge_generators', 'relax_count', 'no_investment', 'rmip',
                'mip_gap', 'network', 'commit_consumption', 'startup_consumption']
SUMMARY_KEYS = ['dataset', 'clusters', 'edge_handling', 'run_kind']
SUMMARY_METRICS = ['num_vars', 'num_bin_vars', 'num_constrs', 'num_nonzeros', 'tm_nonzero', 'tm_entries', 'solver_max_mem_gb',
                   'process_peak_rss_gb', 'solver_runtime_s', 'work_units', 'objective', 'operating_cost', 'investment_cost',
                   'curtailment_pct', 'curtailment_mwh', 'load_shedding_pct', 'load_shedding_mwh', 'uc_corrections',
                   'regret', 'regret_pct',
                   'nb_transitions_fractional_pct', 'nb_transitions_max_delta', 'nb_transitions_mean_delta_fractional',
                   'nb_all_window_fractional_pct', 'nb_all_max_delta',
                   'feas_infeasible_pct', 'feas_infeasible_rounded_pct']


def _read_table(cnx, table):
    try:
        return pd.read_sql(f"SELECT * FROM {table}", cnx)
    except Exception:
        return None


def _load_file(path: str) -> tuple[dict, pd.DataFrame | None]:
    """One row of run parameters + statistics + analysis values, and the fractional UC values."""
    with sqlite3.connect(path) as cnx:
        row = {'file': os.path.basename(path)}
        for table in ['run_parameters', 'solver_statistics', 'mk_metrics', 'mk_nonbinarity', 'mk_feasibility']:
            df = _read_table(cnx, table)
            if df is not None and len(df) > 0:
                for key, value in df.iloc[0].items():
                    row.setdefault(key, value)  # run_parameters first: keeps e.g. its work_limit
        values = _read_table(cnx, 'mk_nonbinary_values')
    return row, values


def _clean(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) or str(value) in ('None', 'nan', '') else value


def load_runs(folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(glob.glob(os.path.join(folder, '**', 'MK-*.sqlite'), recursive=True))
    with ThreadPoolExecutor() as pool:
        loaded = list(pool.map(_load_file, files))
    runs = pd.DataFrame([row for row, _ in loaded])
    if runs.empty:
        return runs, pd.DataFrame()
    runs = pd.DataFrame({col: runs[col].map(_clean) for col in runs.columns})
    runs['run_type'] = runs.get('run_type', pd.Series(index=runs.index, dtype=object)).fillna('main')
    runs['reference'] = runs.get('reference', pd.Series(index=runs.index, dtype=object)).fillna('rpcopies')
    runs['run_kind'] = runs['run_type'] + np.where(runs['reference'] == 'original', '@original', '')
    runs['dataset'] = runs['case_study_directory'].map(lambda d: _dataset_name(d) if d else None)
    runs['base_directory'] = runs['case_study_directory'].map(lambda d: _CLUSTER_SUFFIX.sub('', str(d).rstrip('/\\')) if d else None)
    runs['clusters'] = pd.to_numeric(runs.get('clusters'), errors='coerce')

    value_frames = []
    for (row, values), file in zip(loaded, runs['file']):
        if values is not None and len(values) > 0:
            value_frames.append(values.assign(file=file))
    values = pd.concat(value_frames, ignore_index=True) if value_frames else pd.DataFrame()
    return runs, values


def add_regret(runs: pd.DataFrame) -> pd.DataFrame:
    """Objective of the matching reference and regret = objective - reference objective.
    invest-regret/regret -> Truth (main); operational-regret -> Truth (operational); *@original -> TruthOriginal."""
    runs = runs.copy()
    keys = [k for k in VARIANT_KEYS if k in runs.columns]
    key_str = runs[keys].astype(str).agg('|'.join, axis=1)
    orig_keys = [k for k in keys if k not in ('clusters', 'shift_tm', 'perturb_tm', 'base_directory')]
    orig_key_str = runs[orig_keys].astype(str).agg('|'.join, axis=1)

    truth_main = dict(zip(key_str[(runs['edge_handling'] == 'Truth') & (runs['run_kind'] == 'main')],
                          runs.loc[(runs['edge_handling'] == 'Truth') & (runs['run_kind'] == 'main'), 'objective']))
    truth_op = dict(zip(key_str[(runs['edge_handling'] == 'Truth') & (runs['run_kind'] == 'operational')],
                        runs.loc[(runs['edge_handling'] == 'Truth') & (runs['run_kind'] == 'operational'), 'objective']))
    is_orig = runs['edge_handling'] == 'TruthOriginal'
    # TruthOriginal is keyed by its own folder, which is the base folder of the RP runs
    truth_orig = dict(zip(runs.loc[is_orig, 'case_study_directory'].astype(str).str.rstrip('/\\') + '#' + orig_key_str[is_orig],
                          runs.loc[is_orig, 'objective']))

    def _reference(i):
        kind = runs.at[i, 'run_kind']
        if kind in ('invest-regret', 'regret'):
            return truth_main.get(key_str[i])
        if kind == 'operational-regret':
            return truth_op.get(key_str[i])
        if kind.endswith('@original') and kind != 'main@original':
            return truth_orig.get(f"{runs.at[i, 'base_directory']}#{orig_key_str[i]}")
        return None

    runs['reference_objective'] = [_reference(i) for i in runs.index]
    obj = pd.to_numeric(runs['objective'], errors='coerce')
    ref = pd.to_numeric(runs['reference_objective'], errors='coerce')
    runs['regret'] = obj - ref
    runs['regret_pct'] = (obj - ref) / ref.abs() * 100
    return runs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in SUMMARY_METRICS if m in runs.columns]
    numeric = runs[SUMMARY_KEYS + metrics].copy()
    for m in metrics:
        numeric[m] = pd.to_numeric(numeric[m], errors='coerce')
    grouped = numeric.groupby(SUMMARY_KEYS, dropna=False)
    summary = grouped[metrics].mean()
    summary.insert(0, 'n_runs', grouped.size())
    for m in ['nb_transitions_max_delta', 'nb_all_max_delta', 'solver_max_mem_gb', 'process_peak_rss_gb']:
        if m in metrics:
            summary[f"{m}_max"] = grouped[m].max()  # Maximum over variants in addition to the mean
    return summary.reset_index()


########################################################################################################################
# Plots
########################################################################################################################

def _edges_present(df):
    return [e for e in EDGE_DISPLAY_ORDER if e in set(df['edge_handling'])]


def plot_nonbinarity(runs: pd.DataFrame, values: pd.DataFrame, output_dir: str) -> None:
    """Per dataset: histogram of the fractionality of each RP transition instance (max distance to 0/1 of the UC
    variables in the relaxed window), weighted by how often the transition occurs in the chronology. Bars are in % of
    all transition instances, so the bars of a panel sum up to the share of fractional transitions."""
    import matplotlib.pyplot as plt
    if values.empty:
        print("[skip] nonbinarity plots: no fractional UC values found")
        return
    main = runs[(runs['run_kind'] == 'main') & runs['edge_handling'].isin(['Markov', 'Markov-Strict'])]
    values = values[values['in_window'].astype(bool) & values['file'].isin(main['file'])]
    bins = np.linspace(0, 0.5, 11)
    for dataset, ds_runs in main.groupby('dataset'):
        rps = sorted(ds_runs['clusters'].dropna().unique())
        if not rps:
            continue
        edges = _edges_present(ds_runs)
        fig, axes = plt.subplots(1, len(rps), figsize=(2.6 * len(rps) + 1, 3.2), sharey=True, squeeze=False)
        for ax, n_rp in zip(axes[0], rps):
            for offset, edge in enumerate(edges):
                files = ds_runs[(ds_runs['clusters'] == n_rp) & (ds_runs['edge_handling'] == edge)]
                total = pd.to_numeric(files['nb_transition_instances'], errors='coerce').sum()
                v = values[values['file'].isin(files['file'])]
                if total <= 0:
                    continue
                inst = v.groupby(['file', 'rp', 'g']).agg(delta=('delta', 'max'), weight=('transitions_into_rp', 'first'))
                hist, _ = np.histogram(inst['delta'], bins=bins, weights=inst['weight'])
                width = (bins[1] - bins[0]) / len(edges)
                ax.bar(bins[:-1] + offset * width, hist / total * 100, width=width * 0.9, align='edge',
                       color=EDGE_COLORS.get(edge, '#2a78d6'), label=EDGE_LABELS.get(edge, edge))
            ax.set_title(f"{int(n_rp)} RPs", fontsize=10)
            ax.set_xlim(0, 0.5)
            ax.set_xlabel("distance to 0/1", fontsize=9, color=TEXT_SECONDARY)
            ax.grid(axis='y', color='#e5e4e0', linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ['top', 'right']:
                ax.spines[side].set_visible(False)
        axes[0][0].set_ylabel("% of RP transitions", fontsize=9, color=TEXT_SECONDARY)
        if len(edges) > 1:
            axes[0][-1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = os.path.join(output_dir, f"nonbinarity_{dataset}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")


def plot_feasibility(runs: pd.DataFrame, output_dir: str) -> None:
    """Per dataset: % of chronological transitions (boundary x unit) infeasible in the full chronological model, per
    number of RPs. Solid: strict (fractional values count as violation), dashed: after rounding the commitment."""
    import matplotlib.pyplot as plt
    main = runs[(runs['run_kind'] == 'main') & runs['feas_infeasible_pct'].notna()] if 'feas_infeasible_pct' in runs else pd.DataFrame()
    if main.empty:
        print("[skip] feasibility plots: no feasibility results found")
        return
    for dataset, ds_runs in main.groupby('dataset'):
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        for edge in _edges_present(ds_runs):
            e = ds_runs[ds_runs['edge_handling'] == edge]
            agg = e.groupby('clusters')[['feas_infeasible_pct', 'feas_infeasible_rounded_pct']].agg(lambda s: pd.to_numeric(s, errors='coerce').mean())
            if agg.empty:
                continue
            color = EDGE_COLORS.get(edge, '#2a78d6')
            label = EDGE_LABELS.get(edge, edge)
            ax.plot(agg.index, agg['feas_infeasible_pct'], color=color, linewidth=2, marker='o', markersize=5, label=f"{label}")
            ax.plot(agg.index, agg['feas_infeasible_rounded_pct'], color=color, linewidth=2, linestyle='--', marker='o', markersize=5,
                    markerfacecolor='white', label=f"{label} (rounded)")
        ax.set_xticks(sorted(ds_runs['clusters'].dropna().unique()))
        ax.set_xlabel("number of RPs", fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylabel("% infeasible RP transitions", fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', color='#e5e4e0', linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        path = os.path.join(output_dir, f"feasibility_{dataset}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize the analysis tables of MK-*.sqlite files into CSVs and plots", formatter_class=RichHelpFormatter)
    parser.add_argument("folder", nargs="?", default=".", help="Folder with MK-*.sqlite files (searched recursively; default: current directory)")
    parser.add_argument("--output-dir", default=None, help="Directory for CSVs and plots (default: the input folder)")
    parser.add_argument("--include-nonoptimal", action="store_true", help="Also use runs whose termination condition is not 'optimal' in the summary and plots")
    parser.add_argument("--no-plots", action="store_true", help="Only write the CSV tables")
    args = parser.parse_args()
    output_dir = args.output_dir or args.folder
    os.makedirs(output_dir, exist_ok=True)

    runs, values = load_runs(args.folder)
    if runs.empty:
        print(f"No MK-*.sqlite files found in '{args.folder}'")
        return
    runs = add_regret(runs)
    runs.to_csv(os.path.join(output_dir, "markov_runs.csv"), index=False)
    print(f"Saved {os.path.join(output_dir, 'markov_runs.csv')} ({len(runs)} runs)")

    used = runs if args.include_nonoptimal else runs[runs['termination_condition'] == 'optimal']
    print(f"Using {len(used)} of {len(runs)} runs for summary and plots" + ("" if args.include_nonoptimal else " (optimal only)"))
    summary = summarize(used)
    summary.to_csv(os.path.join(output_dir, "markov_summary.csv"), index=False)
    print(f"Saved {os.path.join(output_dir, 'markov_summary.csv')}")

    if not args.no_plots:
        plot_nonbinarity(used, values, output_dir)
        plot_feasibility(used, output_dir)


if __name__ == "__main__":
    main()
