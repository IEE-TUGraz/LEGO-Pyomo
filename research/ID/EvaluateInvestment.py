"""
EvaluateInvestment.py
=====================
Compares generator investment decisions (vGenInvest × pMaxProd × 1000 → MW)
across inflow aggregation levels against the hourly baseline.

Reads all ID-*.sqlite files (non-regret only) in the target folder and prints,
per comparison group and inflow multiplier, a table:

  Technology | Hourly MW | Daily              | Weekly             | Monthly            | Yearly
             |           | Abs Diff   | %      | Abs Diff   | %      | ...

Results are grouped by all run parameters other than scale_inflows,
inflow_aggregation, and is_regret.  Within each group, one sub-table is printed
per inflow multiplier.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import glob
import os
import sqlite3

import pandas as pd

from InOutModule.printer import Printer

printer = Printer.getInstance()
printer.set_width(200)

AGGREGATION_LEVELS = ["daily", "weekly", "monthly", "yearly"]

# Default values matching InflowDetails.py
DEFAULTS = {
    "scale_demand": 1.0,
    "scale_inflows": 1.0,
    "scale_vres_max_prod": 1.0,
    "scale_pmax": 1.0,
    "cluster_on_original_data": False,
    "scale_ror_to_inflow_scaling": False,
    "rmip": False,
    "single_node": False,
    "number_of_rps": 0,
    "length_of_rps": 24,
    "limit_k": None,
}

GROUP_KEYS = [
    "case_study_directory",
    "scale_demand",
    "scale_vres_max_prod",
    "scale_pmax",
    "cluster_on_original_data",
    "scale_ror_to_inflow_scaling",
    "rmip",
    "single_node",
    "number_of_rps",
    "length_of_rps",
    "limit_k",
]


# ─── SQLite helpers ───────────────────────────────────────────────────────────

def _read_run_parameters(conn):
    """Return run_parameters as a dict with defaults filled in for missing keys."""
    df = pd.read_sql_query("SELECT * FROM run_parameters", conn)
    row = df.iloc[0].to_dict()

    params = {}
    for key, default in DEFAULTS.items():
        raw = row.get(key)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            params[key] = default
        elif isinstance(default, bool):
            params[key] = bool(int(raw))
        elif isinstance(default, int):
            params[key] = int(raw)
        elif isinstance(default, float):
            params[key] = float(raw)
        else:
            params[key] = raw

    params["case_study_directory"] = row.get("case_study_directory")
    params["inflow_aggregation"] = str(row.get("inflow_aggregation", ""))
    params["is_regret"] = bool(int(row.get("is_regret", 0)))
    return params


def _read_investments_by_tec(conn):
    """
    Return a dict {technology: total_MW} for this model run.

    MW = vGenInvest (fractional units) × pMaxProd (p.u.) × 1000
    Technologies with zero total investment are included (as 0.0) so that
    comparisons can show when a technology disappears entirely.
    """
    inv = pd.read_sql_query('SELECT g, "values" FROM vGenInvest', conn)
    gtec = pd.read_sql_query('SELECT "0" as g, "1" as tec FROM gtec', conn)
    pmax = pd.read_sql_query('SELECT g, "values" as pmax FROM pMaxProd', conn)

    merged = inv.merge(gtec, on="g").merge(pmax, on="g")
    merged["MW"] = merged["values"] * merged["pmax"] * 1000

    return merged.groupby("tec")["MW"].sum().to_dict()


def _load_file(sqlite_file):
    """Load run parameters and investments from one sqlite file."""
    try:
        conn = sqlite3.connect(sqlite_file)
        try:
            params = _read_run_parameters(conn)
            if params["is_regret"]:
                return None  # skip regret files
            investments = _read_investments_by_tec(conn)
            return {
                "file": sqlite_file,
                "params": params,
                "investments": investments,
            }
        finally:
            conn.close()
    except Exception as e:
        printer.error(f"  Failed to load '{sqlite_file}': {e}")
        return None


# ─── Grouping helpers ─────────────────────────────────────────────────────────

def _group_key(params):
    return tuple(params[k] for k in GROUP_KEYS)


def _group_label(params):
    parts = [f"case={params['case_study_directory']}"]
    if params["limit_k"]:
        parts.append(f"limitK={params['limit_k']}")
    if params["scale_demand"] != 1.0:
        parts.append(f"demand={params['scale_demand']:.2g}")
    if params["scale_vres_max_prod"] != 1.0:
        parts.append(f"vresMaxProd={params['scale_vres_max_prod']:.2g}")
    if params["scale_pmax"] != 1.0:
        parts.append(f"pmax={params['scale_pmax']:.2g}")
    if params["cluster_on_original_data"]:
        parts.append("clustOriginal")
    if params["scale_ror_to_inflow_scaling"]:
        parts.append("rorScaled")
    if params["rmip"]:
        parts.append("rMIP")
    if params["single_node"]:
        parts.append("SN")
    if params["number_of_rps"] != 0:
        parts.append(f"rps={params['number_of_rps']}")
    if params["length_of_rps"] != 24:
        parts.append(f"ks={params['length_of_rps']}")
    return ", ".join(parts)


# ─── Printing ─────────────────────────────────────────────────────────────────

def _print_mult_table(mult, technologies, hourly_inv, agg_invs, total_width, sep_row,
                      W_TEC, W_HOURLY, W_ABS, W_PCT):
    """Print one sub-table for a given inflow multiplier."""
    printer.information(f"\n  Inflow Multiplier: {mult:.4g}")
    printer.information(sep_row)

    for tec in technologies:
        hourly_mw = hourly_inv.get(tec, 0.0)
        hourly_str = f"{hourly_mw:>{W_HOURLY},.1f}"

        line = f"  {tec:<{W_TEC}s}  {hourly_str}"
        for agg in AGGREGATION_LEVELS:
            inv = agg_invs.get(agg)
            if inv is None:
                line += f" | {'N/A':>{W_ABS}s}  {'N/A':>{W_PCT}s}"
            else:
                agg_mw = inv.get(tec, 0.0)
                diff = agg_mw - hourly_mw
                pct = diff / hourly_mw * 100 if hourly_mw != 0 else (None if agg_mw == 0 else float("inf"))
                diff_str = f"{diff:>+{W_ABS},.1f}"
                pct_str = f"{pct:>+{W_PCT - 1}.1f}%" if pct is not None and pct != float("inf") else (
                    "    0.0%" if agg_mw == 0 else f"{'new':>{W_PCT}s}"
                )
                line += f" | {diff_str}  {pct_str}"

        printer.information(line + " |")

    printer.information(sep_row)


def _print_group(group_label, mult_data):
    """
    Print all sub-tables for one comparison group.

    mult_data: dict { scale_inflows: { 'hourly': {tec: MW}, 'daily': ..., ... } }
    """
    # Fixed column widths
    W_TEC    = 14
    W_HOURLY = 12   # "Hourly MW" value
    W_ABS    = 12   # abs diff
    W_PCT    =  8   # rel%

    # Build header
    def agg_header(agg):
        inner = W_ABS + 2 + W_PCT
        return agg.capitalize().center(inner)

    def agg_subheader():
        return f"{'Abs Diff':>{W_ABS}s}  {'Rel%':>{W_PCT}s}"

    def agg_sep():
        return "-" * (W_ABS + 2 + W_PCT)

    prefix_width = 2 + W_TEC + 2 + W_HOURLY
    header1 = (
        f"  {'Technology':<{W_TEC}s}  {'Hourly MW':>{W_HOURLY}s}"
        + "".join(f" | {agg_header(agg)}" for agg in AGGREGATION_LEVELS)
        + " |"
    )
    header2 = (
        f"  {'':>{W_TEC}s}  {'':>{W_HOURLY}s}"
        + "".join(f" | {agg_subheader()}" for _ in AGGREGATION_LEVELS)
        + " |"
    )
    sep_fixed = "-" * prefix_width
    sep_row = sep_fixed + "".join(f"-+-{agg_sep()}" for _ in AGGREGATION_LEVELS) + "-+"
    total_width = len(header1)

    printer.information("\n" + "=" * total_width)
    printer.information(f"GROUP: {group_label}")
    printer.information("=" * total_width)
    printer.information(header1)
    printer.information(header2)

    for mult in sorted(mult_data.keys()):
        by_agg = mult_data[mult]
        hourly_inv = by_agg.get("hourly")
        if hourly_inv is None:
            printer.warning(f"  No hourly baseline for mult={mult}, skipping")
            continue

        # Collect all technologies that appear in any model (non-zero in at least one)
        all_tecs = set(hourly_inv.keys())
        for agg in AGGREGATION_LEVELS:
            inv = by_agg.get(agg)
            if inv:
                all_tecs.update(inv.keys())
        # Only show technologies with non-zero MW in at least one run
        technologies = sorted(
            t for t in all_tecs
            if hourly_inv.get(t, 0) > 0
            or any((by_agg.get(agg) or {}).get(t, 0) > 0 for agg in AGGREGATION_LEVELS)
        )

        _print_mult_table(
            mult, technologies, hourly_inv, by_agg,
            total_width, sep_row, W_TEC, W_HOURLY, W_ABS, W_PCT,
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(folder="."):
    pattern = os.path.join(folder, "ID-*.sqlite")
    sqlite_files = sorted(glob.glob(pattern))

    if not sqlite_files:
        printer.warning(f"No ID-*.sqlite files found in '{folder}'")
        return

    printer.information(f"Found {len(sqlite_files)} ID-*.sqlite file(s) in '{folder}'")

    all_data = []
    for f in sqlite_files:
        printer.information(f"Loading '{os.path.basename(f)}'...")
        entry = _load_file(f)
        if entry is not None:
            all_data.append(entry)

    if not all_data:
        printer.error("No usable files loaded.")
        return

    # Group by comparison group key
    groups = {}
    for entry in all_data:
        key = _group_key(entry["params"])
        groups.setdefault(key, []).append(entry)

    for gkey in sorted(groups.keys(), key=str):
        entries = groups[gkey]
        label = _group_label(entries[0]["params"])

        # Nest: scale_inflows → inflow_aggregation → investment dict
        mult_data = {}
        for entry in entries:
            mult = entry["params"]["scale_inflows"]
            agg  = entry["params"]["inflow_aggregation"]
            mult_data.setdefault(mult, {})[agg] = entry["investments"]

        _print_group(label, mult_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare generator investments across inflow aggregation levels (ID-*.sqlite)"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder containing ID-*.sqlite files (default: current directory)",
    )
    args = parser.parse_args()
    main(args.folder)
