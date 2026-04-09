import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import glob
import os
import re
import sqlite3
from collections import defaultdict

import pandas as pd

from InOutModule import ExcelReader
from InOutModule.printer import Printer
from rich_argparse import RichHelpFormatter

printer = Printer.getInstance()
printer.set_width(300)

EDGE_HANDLING_SORT = {"Truth": 0, "NoEnf": 1, "Cyclic": 2, "Markov": 3, "Markov-Strict": 4}


def _parse_metadata_from_filename(basename):
    """Fallback: extract edge_handling and case_study_directory from MK filename.

    Expected format: MK-{identifier}-{edgeHandling}.sqlite
    where identifier may contain parts like datadata_markov, relaxed58, etc.
    """
    meta = {}
    # Strip MK- prefix and .sqlite suffix
    stem = basename
    if stem.startswith("MK-"):
        stem = stem[3:]
    if stem.endswith(".sqlite"):
        stem = stem[:-7]

    # Edge handling is the last dash-separated segment
    # Known edge handling names (match from the end)
    for eh in ["Markov-Strict", "Markov", "Cyclic", "NoEnf", "Truth"]:
        if stem.endswith(f"-{eh}"):
            meta['edge_handling'] = eh
            stem = stem[:-len(eh) - 1]
            break

    # Try to extract case_study_directory from "data{path}" prefix
    # e.g. "datadata_markov" -> "data/markov"
    # e.g. "datadata_NREL-118_limitKk0001-k0336_..." -> "data/NREL-118" with limitK etc.
    m = re.match(r'^data(.+?)(?:-relaxed\d+)?$', stem)
    if m:
        raw = m.group(1)
        # Convert underscores back to slashes for the path (first underscore only for simple cases)
        # The identifier format is: path.replace('/', '_') so "data/markov/" -> "data_markov_"
        # We restore the first segment as "data/" prefix
        meta['case_study_directory'] = raw.replace('_', '/', 1)
        if not meta['case_study_directory'].endswith('/'):
            meta['case_study_directory'] += '/'

    return meta


def load_file_metadata(sqlite_file):
    """Load run parameters and solver statistics from a MK sqlite file."""
    meta = {
        'case_study_directory': None,
        'limit_k': None,
        'clusters': None,
        'shift': None,
        'stretch_demand': None,
        'relax_count': None,
        'no_investment': None,
        'rmip': None,
        'no_crossover': None,
        'force_barrier': None,
        'edge_handling': None,
        'run_type': None,
        'work_units': None,
        'solver_time': None,
    }

    has_run_parameters = False

    try:
        conn = sqlite3.connect(sqlite_file)

        # --- run_parameters ---
        try:
            df = pd.read_sql_query('SELECT * FROM run_parameters', conn)
            if len(df) > 0:
                has_run_parameters = True
                row = df.iloc[0]
                for key in ['case_study_directory', 'edge_handling', 'run_type']:
                    if key in row and row[key] not in (None, 'None'):
                        meta[key] = str(row[key])
                for key in ['limit_k']:
                    if key in row and row[key] not in (None, 'None'):
                        meta[key] = str(row[key])
                for key in ['clusters', 'relax_count', 'shift']:
                    if key in row and row[key] not in (None, 'None'):
                        meta[key] = int(float(row[key]))
                for key in ['stretch_demand']:
                    if key in row and row[key] not in (None, 'None'):
                        meta[key] = float(row[key])
                for key in ['no_investment', 'rmip', 'no_crossover', 'force_barrier']:
                    if key in row and row[key] not in (None, 'None'):
                        val = row[key]
                        meta[key] = val if isinstance(val, bool) else str(val).lower() == 'true'
        except Exception:
            pass

        # --- solver_statistics ---
        try:
            df_stats = pd.read_sql_query('SELECT * FROM solver_statistics', conn)
            if len(df_stats) > 0:
                row = df_stats.iloc[0]
                if 'work_units' in row and row['work_units'] is not None:
                    meta['work_units'] = float(row['work_units'])
                if 'solver_time' in row and row['solver_time'] is not None:
                    meta['solver_time'] = float(row['solver_time'])
        except Exception:
            pass

        conn.close()
    except Exception:
        pass

    # Fallback: parse metadata from filename if run_parameters table is missing
    if not has_run_parameters:
        basename = os.path.basename(sqlite_file)
        parsed = _parse_metadata_from_filename(basename)
        for key, val in parsed.items():
            if meta[key] is None:
                meta[key] = val

    return meta


def load_results_from_sqlite(sqlite_file):
    """Load objective and weighted variable sums from a MK sqlite file."""
    results = {}
    try:
        conn = sqlite3.connect(sqlite_file)

        # Objective
        try:
            df_obj = pd.read_sql_query('SELECT * FROM objective', conn)
            results['Objective'] = float(df_obj.iloc[0]['values'])
        except Exception:
            results['Objective'] = -1

        # Weights
        try:
            df_wrp = pd.read_sql_query('SELECT * FROM pWeight_rp', conn)
            wrp = dict(zip(df_wrp.iloc[:, 0], df_wrp['values']))
        except Exception:
            wrp = {}
        try:
            df_wk = pd.read_sql_query('SELECT * FROM pWeight_k', conn)
            wk = dict(zip(df_wk.iloc[:, 0], df_wk['values']))
        except Exception:
            wk = {}

        # Weighted sums for indexed variables
        for var_name in ['vGenP', 'vCommit', 'vStartup', 'vShutdown', 'vPNS', 'vEPS']:
            try:
                df_var = pd.read_sql_query(f'SELECT * FROM {var_name}', conn)
                df_var['weight'] = df_var['rp'].map(wrp) * df_var['k'].map(wk)
                results[var_name] = (df_var['values'] * df_var['weight']).sum()
            except Exception:
                results[var_name] = 0

        # vGenInvest (not time-indexed)
        try:
            df_inv = pd.read_sql_query('SELECT * FROM vGenInvest', conn)
            results['vGenInvest'] = df_inv['values'].sum()
        except Exception:
            results['vGenInvest'] = 0

        # vGenInvest by technology
        try:
            df_inv = pd.read_sql_query('SELECT * FROM vGenInvest', conn)
            df_gtec = pd.read_sql_query('SELECT * FROM gtec', conn)
            # gtec has columns: index, 0, 1 (generator, technology)
            if 'index' in df_gtec.columns:
                df_gtec = df_gtec.drop(columns=['index'])
            df_gtec.columns = ['g', 'tec']
            df_merged = df_inv.merge(df_gtec, left_on=df_inv.columns[0], right_on='g', how='left')
            for tec, group in df_merged.groupby('tec'):
                results[f'vGenInvest[{tec}]'] = group['values'].sum()
        except Exception:
            pass

        conn.close()
    except Exception as e:
        printer.error(f"Failed to load results from '{sqlite_file}': {e}")

    return results


def print_comparison_table(group_entries):
    """Print a comparison table for a group of MK results, similar to the original Markov.py output."""

    # Sort by edge handling type
    group_entries.sort(key=lambda e: EDGE_HANDLING_SORT.get(e['edge_handling'] or '', 99))

    # Find truth entry for relative calculations
    truth_entry = next((e for e in group_entries if e['edge_handling'] == 'Truth'), None)

    # Compute relative change columns
    for entry in group_entries:
        for key in ["Objective", "vStartup", "vShutdown"]:
            val = entry.get(key)
            if truth_entry is not None and truth_entry.get(key) not in (-1, 0, None) and val not in (-1, None):
                entry[f"{key} %"] = (val - truth_entry[key]) / abs(truth_entry[key]) * 100
            else:
                entry[f"{key} %"] = None

    columns = ["Case", "Objective", "Objective %",
               "Work Units", "vGenP", "vCommit", "vStartup", "vStartup %",
               "vShutdown", "vShutdown %", "vPNS", "vEPS"]

    _print_table(columns, group_entries)

    # Print investment table if any entry has vGenInvest data
    invest_columns = ["Case", "vGenInvest"]
    tec_columns = sorted(set(
        k for e in group_entries for k in e if k.startswith("vGenInvest[")
    ))
    invest_columns.extend(tec_columns)

    has_invest = any(e.get("vGenInvest") not in (None, "") for e in group_entries)
    if has_invest:
        printer.information("")
        _print_table(invest_columns, group_entries)


def _print_table(columns, group_entries):
    """Print a formatted table with given columns and entries."""
    table = []
    for col in columns:
        column_data = [col]
        for entry in group_entries:
            value = entry.get(col, "")
            if value is None:
                value = ""
            elif col.endswith(" %"):
                value = f"{value:+.0f}%"
            elif isinstance(value, float):
                value = f"{value:.2f}"
            elif isinstance(value, int):
                value = f"{value:d}"
            else:
                value = f"{value}"
            column_data.append(value)
        table.append(column_data)

    for i in range(len(table[0])):
        printer.information(" | ".join(
            f"{table[j][i]:{'>' if i != 0 else ''}{max(len(table[j][i2]) for i2 in range(len(table[j])))}}"
            for j in range(len(table))
        ))


def plot_unit_commitment(sqlite_files, case_labels, case_study_folder=None, number_of_hours=24 * 7, start_hour=1, no_show=False):
    """Plot unit commitment from sqlite result files."""
    import matplotlib.pyplot as plt

    plt.rcParams['figure.dpi'] = 300

    frames = [_load_unit_commitment_from_sqlite(f, label) for f, label in zip(sqlite_files, case_labels)]
    df = pd.concat(frames)

    # Load vGenInvest from each file and find generators where at least one model invested
    invested_generators = set()
    for sqlite_file in sqlite_files:
        try:
            cnx = sqlite3.connect(sqlite_file)
            df_inv = pd.read_sql("SELECT * FROM vGenInvest", cnx)
            cnx.close()
            for _, row in df_inv.iterrows():
                if row['values'] > 0:
                    invested_generators.add(row[df_inv.columns[0]])
        except Exception:
            pass

    # Filter to invested generators only (if we found investment data)
    all_generators = df.index.get_level_values("g").unique()
    if invested_generators:
        generators = [g for g in all_generators if g in invested_generators]
    else:
        generators = list(all_generators)

    # Load hindex mapping from a non-Truth sqlite file (Truth has a different time structure)
    # Fall back to Excel if not available
    hindex = None
    hindex_source = next((f for f, l in zip(sqlite_files, case_labels) if l.strip() not in ("Truth", "Truth ")), sqlite_files[0])
    try:
        cnx = sqlite3.connect(hindex_source)
        hindex = pd.read_sql("SELECT * FROM hindex", cnx)
        cnx.close()
        if hindex.empty:
            hindex = None
        else:
            # Drop the sqlite index column if present
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

    fig, axs = plt.subplots(nr_cases, nr_generators,
                            figsize=(6 * nr_generators, 2 * nr_cases),
                            squeeze=False)

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
            if i == 0:
                axs2.set_title(f"{g}\n{case}")
            else:
                axs2.set_title(f"{case}")
            axs2.set_ylim(0, 3)
            axs2.bar(index, data_bar_startup.values(), color="green", alpha=0.5,
                     bottom=[list(data_vCommit.values())[-1]] + list(data_vCommit.values())[:-1], width=1,
                     label="Startup")
            axs2.bar(index, data_bar_shutdown.values(), color="red", alpha=0.5, bottom=data_vCommit.values(), width=1,
                     label="Shutd.")
            axs2.plot(index, data_vCommit.values(), color="gray", alpha=0.5, label="Commit", linewidth=1.5)
            axs2.set_ylabel("Startup / Shutdown", color="black")

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
            axs[i, j].plot(index, data_vGenP.values(), color="black", alpha=0.3, label="Prod.")

            axs[i, j].bar(index, data_vPNS.values(), color="orange", alpha=0.3, label="PNS", bottom=data_vGenP.values())
            axs[i, j].bar(index, data_vEPS.values(), color="purple", alpha=0.3, label="EPS", bottom=data_demand.values())
            axs[i, j].legend(loc='upper right', fontsize='x-small')

            axs[i, j].hlines(y=0, xmin=0, xmax=len(data_bar_shutdown.values()), color="gray", linestyle=(0, (1, 1)),
                             alpha=0.5)
            axs[i, j].set_ylabel("Generation / Demand", color="black")
            axs[i, j].set_yticks([0, 0.5, 1], ["0.0", "0.5", "1.0"])

            # Set ticks and vertical lines
            index_labels = []
            index_positions = []
            axvline_thick_positions = []
            axvline_thin_positions = []
            for x in index:
                if x == index[0]:
                    index_labels.append(x + start_hour - 1)
                    index_positions.append(x)
                    if (x + start_hour - 2) % 24 == 0:
                        axvline_thick_positions.append(x)
                    else:
                        axvline_thin_positions.append(x)
                elif x == index[-1]:
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
    cnx = sqlite3.connect(sqlite_file)
    g_rename = {"thermalGenerators": "g"}
    vCommit = pd.read_sql("SELECT * FROM vCommit", cnx).rename(columns={"values": "vCommit", **g_rename})
    vStartup = pd.read_sql("SELECT * FROM vStartup", cnx).rename(columns={"values": "vStartup", **g_rename})
    vShutdown = pd.read_sql("SELECT * FROM vShutdown", cnx).rename(columns={"values": "vShutdown", **g_rename})
    vGenP = pd.read_sql("SELECT * FROM vGenP", cnx).rename(columns={"values": "vGenP"})
    vPNS = pd.read_sql("SELECT * FROM vPNS", cnx).rename(columns={"values": "vPNS"})
    vEPS = pd.read_sql("SELECT * FROM vEPS", cnx).rename(columns={"values": "vEPS"})
    pDemandP = pd.read_sql("SELECT * FROM pDemandP", cnx).rename(columns={"values": "pDemandP"})
    pMinUpTime = pd.read_sql("SELECT * FROM pMinUpTime", cnx).rename(columns={"values": "pMinUpTime", **g_rename})
    pMinDownTime = pd.read_sql("SELECT * FROM pMinDownTime", cnx).rename(columns={"values": "pMinDownTime", **g_rename})
    cnx.close()

    idx = ["rp", "k", "g"]
    df = vCommit.set_index(idx)
    df = df.join(vStartup.set_index(idx)["vStartup"])
    df = df.join(vShutdown.set_index(idx)["vShutdown"])
    df = df.join(vGenP.set_index(idx)["vGenP"])

    pDemandP_agg = pDemandP.groupby(["rp", "k"])["pDemandP"].sum()
    vPNS_agg = vPNS.groupby(["rp", "k"])["vPNS"].sum()
    vEPS_agg = vEPS.groupby(["rp", "k"])["vEPS"].sum()
    df = df.join(pDemandP_agg, on=["rp", "k"])
    df = df.join(vPNS_agg, on=["rp", "k"])
    df = df.join(vEPS_agg, on=["rp", "k"])

    df = df.join(pMinUpTime.set_index("g"), on="g")
    df = df.join(pMinDownTime.set_index("g"), on="g")

    df["case"] = case_label
    df = df.reset_index().set_index(["case", "rp", "k", "g"])
    return df


def main(folder=".", plot=False, case_study_folder=None, number_of_hours=6 * 24, start_hour=1, no_show=False):
    # Find all MK-*.sqlite files (excluding regret and invest-regret)
    all_sqlite = sorted(f for f in glob.glob(os.path.join(folder, "*.sqlite"))
                        if os.path.basename(f).startswith("MK-"))

    if not all_sqlite:
        printer.warning(f"No MK-*.sqlite files found in '{folder}'")
        return

    printer.information(f"Found {len(all_sqlite)} MK sqlite file(s) in '{folder}'")

    # Load metadata and results for each file
    entries = []
    for sqlite_file in all_sqlite:
        basename = os.path.basename(sqlite_file)

        # Skip regret/invest-regret files for the main table
        if basename.endswith("-regret.sqlite") or basename.endswith("-invest-regret.sqlite"):
            continue

        meta = load_file_metadata(sqlite_file)
        results = load_results_from_sqlite(sqlite_file)

        entry = {
            'file': sqlite_file,
            'basename': basename,
            **meta,
            **results,
            'Case': meta.get('edge_handling') or basename,
            'Work Units': meta.get('work_units'),
        }
        entries.append(entry)

    if not entries:
        printer.warning("No (non-regret) MK result files found")
        return

    # Group by run parameters (case_study_directory, limit_k, clusters, shift, stretch_demand, relax_count, no_investment)
    groups = defaultdict(list)
    for entry in entries:
        key = (
            entry.get('case_study_directory'),
            entry.get('limit_k'),
            entry.get('clusters'),
            entry.get('shift'),
            entry.get('stretch_demand'),
            entry.get('relax_count'),
            entry.get('no_investment'),
            entry.get('rmip'),
            entry.get('no_crossover'),
        )
        groups[key].append(entry)

    for group_key, group_entries in groups.items():
        case_dir, limit_k, clusters, shift, stretch_demand, relax_count, no_investment, rmip, no_crossover = group_key

        # Print group header
        parts = []
        if case_dir:
            parts.append(f"data={case_dir}")
        if limit_k:
            parts.append(f"limitK={limit_k}")
        if clusters and clusters > 1:
            parts.append(f"clusters={clusters}")
        if shift and shift != 0:
            parts.append(f"shift={shift}")
        if stretch_demand and stretch_demand != 1.0:
            parts.append(f"stretch_demand={stretch_demand}")
        if relax_count and relax_count > 0:
            parts.append(f"relaxed={relax_count}")
        if no_investment:
            parts.append("no-investment")
        if rmip:
            parts.append("rMIP")
        if no_crossover:
            parts.append("no-crossover")

        printer.information(f"\n{'=' * 80}")
        printer.information(f"Group: {', '.join(parts) if parts else '(default)'}")
        printer.information(f"{'=' * 80}")

        print_comparison_table(group_entries)

        # Plot if requested
        if plot:
            sorted_entries = sorted(group_entries, key=lambda e: EDGE_HANDLING_SORT.get(e.get('edge_handling', ''), 99))
            sqlite_files = [e['file'] for e in sorted_entries]
            case_labels = [e.get('edge_handling', '') for e in sorted_entries]
            try:
                plot_unit_commitment(sqlite_files, case_labels, case_study_folder or case_dir, number_of_hours, start_hour, no_show)
            except Exception as e:
                printer.error(f"Failed to plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Markov edge-handling results from SQLite files",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing MK-*.sqlite files (default: current directory)")
    parser.add_argument("--plot", action="store_true", help="Plot unit commitment results")
    parser.add_argument("--case-study-folder", type=str, default=None, help="Path to case study folder (fallback for plotting if neither hindex nor case study folder are present in .sqlite)")
    parser.add_argument("--number-of-hours", type=int, default=6 * 24, help="Number of hours to plot (default: 144)")
    parser.add_argument("--start-hour", type=int, default=1, help="Start hour for plot (default: 1)")
    parser.add_argument("--no-show", action="store_true", help="Don't show the plot after creation (only save it)")
    args = parser.parse_args()

    main(args.folder, args.plot, args.case_study_folder, args.number_of_hours, args.start_hour, args.no_show)
