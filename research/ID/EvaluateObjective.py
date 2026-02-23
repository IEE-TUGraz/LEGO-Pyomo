"""
EvaluateObjective.py
====================
Evaluates the impact of inflow aggregation on optimization objectives.

Reads all ID-*.sqlite files in the target folder and prints a table:

  Inflow Mult | Hydro Share | Daily              | Weekly             | Monthly            | Yearly
              |             | abs diff   | %      | abs diff   | %      | abs diff   | %      | ...

Where:
  - Inflow Multiplier : scale_inflows value from run_parameters
  - Hydro Share       : weighted generation of Hydro+RoR generators / total weighted generation
                        (from the hourly baseline model)
  - Daily/Weekly/Monthly/Yearly : relative regret = regret_obj − hourly_obj
                                  shown as both absolute difference and percentage

Results are grouped by all run parameters other than scale_inflows, inflow_aggregation, is_regret.
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

# Keys that form the comparison group (everything except scale_inflows / aggregation / is_regret)
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
            params[key] = raw  # str or None

    params["case_study_directory"] = row.get("case_study_directory")
    params["inflow_aggregation"] = str(row.get("inflow_aggregation", ""))
    params["is_regret"] = bool(int(row.get("is_regret", 0)))
    return params


def _read_objective(conn):
    """Return the scalar objective value."""
    df = pd.read_sql_query("SELECT * FROM objective", conn)
    return float(df.iloc[0]["values"])


def _read_hydro_share(conn):
    """
    Compute the share of weighted generation from Hydro+RoR generators.

    Hydro generators = union of:
      - hydroStorageUnits table (hydro storage, tec='Hydro')
      - generators from gtec where tec is 'RoR' (run-of-river)

    Returns hydro_weighted_gen / total_weighted_gen, or None if data unavailable.
    """
    try:
        # Identify hydro generators
        hydro_gens = set()

        try:
            hsu = pd.read_sql_query("SELECT \"0\" as g FROM hydroStorageUnits", conn)
            hydro_gens.update(hsu["g"].tolist())
        except Exception:
            pass

        gtec = pd.read_sql_query("SELECT \"0\" as g, \"1\" as tec FROM gtec", conn)
        ror_gens = gtec[gtec["tec"].str.lower() == "ror"]["g"].tolist()
        hydro_gens.update(ror_gens)

        if not hydro_gens:
            return None

        # Load vGenP and weights
        vgenp = pd.read_sql_query('SELECT rp, k, g, "values" FROM vGenP', conn)
        wk = pd.read_sql_query('SELECT k, "values" as wk FROM pWeight_k', conn)
        wrp = pd.read_sql_query('SELECT rp, "values" as wrp FROM pWeight_rp', conn)

        vgenp = vgenp.merge(wk, on="k").merge(wrp, on="rp")
        vgenp["weighted"] = vgenp["values"] * vgenp["wk"] * vgenp["wrp"]

        total = vgenp["weighted"].sum()
        if total == 0:
            return None

        hydro = vgenp[vgenp["g"].isin(hydro_gens)]["weighted"].sum()
        return hydro / total

    except Exception as e:
        printer.warning(f"  Could not compute hydro share: {e}")
        return None


def _load_file(sqlite_file):
    """Load all needed data from a single sqlite file. Returns dict or None on error."""
    try:
        conn = sqlite3.connect(sqlite_file)
        try:
            params = _read_run_parameters(conn)
            objective = _read_objective(conn)
            # Compute hydro share for hourly baseline and all regret files
            hydro_share = None
            if not params["is_regret"] and params["inflow_aggregation"] == "hourly":
                hydro_share = _read_hydro_share(conn)
            elif params["is_regret"]:
                hydro_share = _read_hydro_share(conn)
            return {
                "file": sqlite_file,
                "params": params,
                "objective": objective,
                "hydro_share": hydro_share,
            }
        finally:
            conn.close()
    except Exception as e:
        printer.error(f"  Failed to load '{sqlite_file}': {e}")
        return None


# ─── Grouping ─────────────────────────────────────────────────────────────────

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

def _fmt_regret(abs_diff, pct):
    if abs_diff is None:
        return f"{'N/A':>12s}  {'N/A':>8s}"
    return f"{abs_diff:>12,.0f}  {pct:>+7.2f}%"


def _print_table(group_label, rows):
    """
    rows: list of dicts with keys:
      inflow_mult, hydro_share, daily, weekly, monthly, yearly
      Each aggregation key maps to (abs_diff, pct) or None.
    """
    # Fixed column widths
    W_MULT = 14
    W_HYDRO = 15
    W_ABS = 12  # "Abs Diff" value
    W_PCT = 8  # "Rel%" value (includes the % sign)
    # Prefix before the aggregation blocks
    prefix_width = 2 + W_MULT + 2 + W_HYDRO  # "  " + mult + "  " + hydro

    def agg_header(agg):
        """Aggregation name centred over the block (excluding the leading ' | ')."""
        inner = W_ABS + 2 + W_PCT  # 22 chars
        return agg.capitalize().center(inner)

    def agg_subheader():
        return f"{'Abs Diff':>{W_ABS}s}  {'Rel%':>{W_PCT}s}"

    def agg_separator():
        return "-" * (W_ABS + 2 + W_PCT)  # 22 dashes

    header1 = (
            f"  {'Inflow Mult':>{W_MULT}s}  {'Hydro Share':>{W_HYDRO}s}"
            + "".join(f" | {agg_header(agg)}" for agg in AGGREGATION_LEVELS)
    )
    header2 = (
            f"  {'':>{W_MULT}s}  {'':>{W_HYDRO}s}"
            + "".join(f" | {agg_subheader()}" for _ in AGGREGATION_LEVELS)
    )
    # Separator row: plain dashes for the fixed columns, agg-dashes separated by " | "
    sep_fixed = "-" * (prefix_width)
    sep_row = sep_fixed + "".join(f"-+-{agg_separator()}" for _ in AGGREGATION_LEVELS)

    total_width = len(header1)

    printer.information("\n" + "=" * total_width)
    printer.information(f"GROUP: {group_label}")
    printer.information("=" * total_width)
    printer.information(header1)
    printer.information(header2)
    printer.information(sep_row)

    for row in rows:
        mult_str = f"{row['inflow_mult']:.4g}"
        hydro_str = row["hydro_share"] if row["hydro_share"] is not None else "N/A"

        line = f"  {mult_str:>{W_MULT}s}  {hydro_str:>{W_HYDRO}s}"
        for agg in AGGREGATION_LEVELS:
            val = row.get(agg)
            if val is None:
                line += f" | {'N/A':>{W_ABS}s}  {'N/A':>{W_PCT}s}"
            else:
                abs_diff, pct = val
                line += f" | {abs_diff:>{W_ABS},.0f}  {pct:>+{W_PCT - 1}.2f}%"

        printer.information(line)

    printer.information(sep_row)

    printer.information("-" * total_width)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(folder="."):
    pattern = os.path.join(folder, "ID-*.sqlite")
    sqlite_files = sorted(glob.glob(pattern))

    if not sqlite_files:
        printer.warning(f"No ID-*.sqlite files found in '{folder}'")
        return

    printer.information(f"Found {len(sqlite_files)} ID-*.sqlite file(s) in '{folder}'")

    # Load all files
    all_data = []
    for f in sqlite_files:
        printer.information(f"Loading '{os.path.basename(f)}'...")
        entry = _load_file(f)
        if entry is not None:
            all_data.append(entry)

    if not all_data:
        printer.error("No files could be loaded.")
        return

    # Group by comparison group key
    groups = {}
    for entry in all_data:
        key = _group_key(entry["params"])
        groups.setdefault(key, []).append(entry)

    for gkey in sorted(groups.keys(), key=lambda k: str(k)):
        entries = groups[gkey]
        label = _group_label(entries[0]["params"])

        # Sub-group by scale_inflows
        by_mult = {}
        for entry in entries:
            mult = entry["params"]["scale_inflows"]
            by_mult.setdefault(mult, []).append(entry)

        rows = []
        for mult in sorted(by_mult.keys()):
            mult_entries = by_mult[mult]

            # Hourly baseline (non-regret)
            hourly = next(
                (e for e in mult_entries
                 if e["params"]["inflow_aggregation"] == "hourly" and not e["params"]["is_regret"]),
                None
            )
            if hourly is None:
                printer.warning(f"  No hourly baseline for mult={mult} in group '{label}', skipping row")
                continue

            hourly_obj = hourly["objective"]

            # Collect hydro shares from hourly + all regret files for range display
            hydro_shares = []
            if hourly["hydro_share"] is not None:
                hydro_shares.append(hourly["hydro_share"])

            row = {
                "inflow_mult": mult,
            }

            for agg in AGGREGATION_LEVELS:
                regret = next(
                    (e for e in mult_entries
                     if e["params"]["inflow_aggregation"] == agg and e["params"]["is_regret"]),
                    None
                )
                if regret is None:
                    row[agg] = None
                else:
                    if regret["hydro_share"] is not None:
                        hydro_shares.append(regret["hydro_share"])
                    abs_diff = regret["objective"] - hourly_obj
                    pct = abs_diff / hourly_obj * 100 if hourly_obj != 0 else None
                    row[agg] = (abs_diff, pct)

            # Build hydro share range string
            if hydro_shares:
                lo, hi = min(hydro_shares), max(hydro_shares)
                if abs(hi - lo) < 0.0001:
                    row["hydro_share"] = f"{lo:.1%}"
                else:
                    row["hydro_share"] = f"{lo:.1%}–{hi:.1%}"
            else:
                row["hydro_share"] = None

            rows.append(row)

        if rows:
            _print_table(label, rows)
        else:
            printer.warning(f"  No valid rows for group '{label}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate inflow aggregation impact on objectives from ID-*.sqlite files"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder containing ID-*.sqlite files (default: current directory)",
    )
    args = parser.parse_args()
    main(args.folder)
