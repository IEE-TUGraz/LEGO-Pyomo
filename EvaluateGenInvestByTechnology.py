import argparse
import glob
import os
import re
import sqlite3
from collections import defaultdict

import pandas as pd

from InOutModule.printer import Printer

printer = Printer.getInstance()
printer.set_width(180)


def extract_parameters(filename):
    """
    Extract dcBuffer, tpBuffer, zone, demand, and pmax values from filename or SQLite database.
    First tries to read from run_parameters table in SQLite, falls back to filename parsing.
    """
    # Try to read from SQLite database first
    try:
        conn = sqlite3.connect(filename)
        df = pd.read_sql_query('SELECT * FROM run_parameters', conn)
        conn.close()

        if len(df) > 0:
            row = df.iloc[0]
            dc_buffer = int(row['dc_buffer']) if 'dc_buffer' in row and row['dc_buffer'] != 'None' else None
            tp_buffer = int(row['tp_buffer']) if 'tp_buffer' in row and row['tp_buffer'] != 'None' else None
            zone = str(row['zoi']) if 'zoi' in row and row['zoi'] != 'None' else None
            demand = float(row['scale_demand']) if 'scale_demand' in row and row['scale_demand'] != 'None' else 1.0
            pmax = float(row['scale_pmax']) if 'scale_pmax' in row and row['scale_pmax'] != 'None' else 1.0
            return dc_buffer, tp_buffer, zone, demand, pmax
    except Exception:
        pass  # Fall back to filename parsing

    # Fall back to parsing filename
    match = re.search(r'-zoi(?P<zone>[^-]+)-.*?dcBuffer(?P<dc>\d+)-tpBuffer(?P<tp>\d+)(?:-demand(?P<demand>\d+(?:\.\d+)?))?(?:-pmax(?P<pmax>\d+(?:\.\d+)?))?', filename)
    if match:
        demand = float(match.group('demand')) if match.group('demand') else 1.0
        pmax = float(match.group('pmax')) if match.group('pmax') else 1.0
        return int(match.group('dc')), int(match.group('tp')), match.group('zone'), demand, pmax
    return None, None, None, 1.0, 1.0


def print_run_parameters_from_sqlite(sqlite_file):
    """
    Print run parameters from SQLite file if available.
    :param sqlite_file: Path to the SQLite database file
    """
    try:
        conn = sqlite3.connect(sqlite_file)
        df = pd.read_sql_query('SELECT * FROM run_parameters', conn)
        conn.close()

        if len(df) > 0:
            row = df.iloc[0]
            params_str = ", ".join([f"{col}={row[col]}" for col in df.columns])
            printer.information(f"  Run parameters: {params_str}")
    except Exception:
        pass  # Table doesn't exist or other error - silently ignore


def evaluate_gen_investment_by_technology_from_sqlite(sqlite_file, filter_zoi=True):
    """
    Evaluate generator investment capacity by technology from a SQLite file.

    Calculates invested capacity as vGenInvest × pMaxProd and aggregates by technology.

    :param sqlite_file: Path to the SQLite database file
    :param filter_zoi: If True, only include generators in zone of interest (zoi_i).
                       If False, include all generators.
    :return: Dictionary mapping technology -> total invested capacity
    """
    conn = sqlite3.connect(sqlite_file)

    try:
        # Load required tables
        vGenInvest_df = pd.read_sql_query('SELECT * FROM vGenInvest', conn)
        pMaxProd_df = pd.read_sql_query('SELECT * FROM pMaxProd', conn)
        gtec_df = pd.read_sql_query('SELECT * FROM gtec', conn)

        # Rename columns for gtec (index, generator, technology)
        gtec_df = gtec_df.rename(columns={'0': 'g', '1': 'tec'})

        # Get ZOI filtering data if needed
        if filter_zoi:
            zoi_i_df = pd.read_sql_query('SELECT * FROM zoi_i', conn)

            # Check if zoi_i is empty (happens when --zoi None creates 0 ZOI buses)
            if len(zoi_i_df) == 0:
                # Empty ZOI means no filtering (treat all generators as in ZOI)
                zoi_generators = None
            else:
                gi_df = pd.read_sql_query('SELECT * FROM gi', conn)

                # Rename columns for gi and zoi_i
                gi_df = gi_df.rename(columns={'0': 'g', '1': 'i'})
                zoi_i_df = zoi_i_df.rename(columns={'0': 'i'})

                # Get set of ZOI buses
                zoi_buses = set(zoi_i_df['i'])

                # Get generators in ZOI
                zoi_generators = set(gi_df[gi_df['i'].isin(zoi_buses)]['g'])
        else:
            zoi_generators = None

        # Merge investment and max production data
        invest_data = vGenInvest_df.merge(pMaxProd_df, on='g', suffixes=('_invest', '_maxprod'))

        # Merge with technology mapping
        invest_data = invest_data.merge(gtec_df[['g', 'tec']], on='g')

        # Filter by ZOI if requested
        if zoi_generators is not None:
            invest_data = invest_data[invest_data['g'].isin(zoi_generators)]

        # Calculate capacity: vGenInvest × pMaxProd (convert GW to MW)
        invest_data['capacity_MW'] = invest_data['values_invest'] * invest_data['values_maxprod'] * 1000

        # Aggregate by technology
        tech_capacity = invest_data.groupby('tec')['capacity_MW'].sum().to_dict()

        return tech_capacity

    finally:
        conn.close()


def print_summary_table(results):
    """Print summary table of all files."""
    # Calculate the maximum filename length for proper alignment
    max_filename_len = max(len(sqlite_file) for sqlite_file, _, _, _, _, _, _, _ in results)
    # Ensure minimum width for readability
    filename_width = max(max_filename_len, len("Filename"))
    # Calculate total table width
    table_width = filename_width + 2 + 8 + 8 + 15 + 8 + 8 + 15 + 15  # 2 for spacing, rest for columns

    printer.information("\n" + "=" * table_width)
    printer.information("Summary of Generator Investment Capacity by Zone")
    printer.information("=" * table_width)
    printer.information(
        f"  {'Filename':<{filename_width}s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Zone':>15s} {'Demand':>8s} {'PMax':>8s} "
        f"{'Total Cap (MW)':>15s} {'ZOI Cap (MW)':>15s}"
    )
    printer.information("-" * table_width)

    for sqlite_file, dc_buffer, tp_buffer, zone, demand, pmax, total_invest, zoi_invest in results:
        dc_str = str(dc_buffer) if dc_buffer is not None else "N/A"
        tp_str = str(tp_buffer) if tp_buffer is not None else "N/A"
        zone_str = zone if zone is not None else "N/A"
        demand_str = f"{demand:.1f}" if demand is not None else "N/A"
        pmax_str = f"{pmax:.1f}" if pmax is not None else "N/A"
        total_cap = sum(total_invest.values())
        zoi_cap = sum(zoi_invest.values())

        printer.information(
            f"  {sqlite_file:<{filename_width}s} {dc_str:>8s} {tp_str:>8s} {zone_str:>15s} {demand_str:>8s} {pmax_str:>8s} "
            f"{total_cap:>15.2f} {zoi_cap:>15.2f}"
        )


def evaluate_gen_investment_with_custom_zoi(sqlite_file, zoi_sqlite_file):
    """
    Evaluate generator investment using ZOI definition from another SQLite file.

    This allows comparing investments across different models using a consistent
    zone definition. For example, comparing zoiNone with zoneR1 using R1's zoi_i.

    :param sqlite_file: Path to the SQLite file containing investment data
    :param zoi_sqlite_file: Path to the SQLite file containing zoi_i definition
    :return: Dictionary mapping technology -> total invested capacity
    """
    # Load investment data from target file
    invest_conn = sqlite3.connect(sqlite_file)
    vGenInvest_df = pd.read_sql_query('SELECT * FROM vGenInvest', invest_conn)
    pMaxProd_df = pd.read_sql_query('SELECT * FROM pMaxProd', invest_conn)
    gtec_df = pd.read_sql_query('SELECT * FROM gtec', invest_conn).rename(columns={'0': 'g', '1': 'tec'})
    gi_df = pd.read_sql_query('SELECT * FROM gi', invest_conn).rename(columns={'0': 'g', '1': 'i'})
    invest_conn.close()

    # Load ZOI definition from reference file
    zoi_conn = sqlite3.connect(zoi_sqlite_file)
    zoi_i_df = pd.read_sql_query('SELECT * FROM zoi_i', zoi_conn).rename(columns={'0': 'i'})
    zoi_conn.close()

    # Get set of ZOI buses
    zoi_buses = set(zoi_i_df['i'])

    # Merge investment data
    invest_data = vGenInvest_df.merge(pMaxProd_df, on='g', suffixes=('_invest', '_maxprod'))
    invest_data = invest_data.merge(gtec_df[['g', 'tec']], on='g')
    invest_data = invest_data.merge(gi_df[['g', 'i']], on='g')
    invest_data['capacity_MW'] = invest_data['values_invest'] * invest_data['values_maxprod'] * 1000

    # Filter by ZOI buses
    zone_data = invest_data[invest_data['i'].isin(zoi_buses)]
    tech_capacity = zone_data.groupby('tec')['capacity_MW'].sum().to_dict()

    return tech_capacity


def evaluate_zone_investments_from_none_model(none_sqlite_file, zone_sqlite_files):
    """
    Evaluate investments in the zoiNone model partitioned by zone definitions.

    Takes the zoiNone model and calculates investments within each zone
    (R1, R2, R3) by using the zone definitions from the zone-specific models.

    :param none_sqlite_file: Path to zoiNone SQLite file
    :param zone_sqlite_files: Dict mapping zone name -> SQLite file path
    :return: Dict mapping zone name -> technology investments
    """
    zone_investments = {}

    # For each zone, evaluate zoiNone using that zone's zoi_i
    for zone_name, zone_file in zone_sqlite_files.items():
        tech_capacity = evaluate_gen_investment_with_custom_zoi(none_sqlite_file, zone_file)
        zone_investments[zone_name] = tech_capacity

    return zone_investments


def print_safety_checks(group, zoi_none_entry):
    """
    Validate that zone investments sum to zoiNone investment.

    Checks that investments in zoiNone model, when partitioned by zones R1/R2/R3,
    sum to the total zoiNone investment. This verifies our zone filtering works correctly.
    """
    none_sqlite_file, _, _, _, _, _, none_total, none_zoi = zoi_none_entry

    # Get zone SQLite files (exclude both Python None and string "None")
    zone_files = {}
    for sqlite_file, dc_buffer, tp_buffer, zone, demand, pmax, zone_total, zone_zoi in group:
        if zone != "None" and zone is not None:
            zone_files[zone] = sqlite_file

    if not zone_files:
        printer.warning("No zone files found for safety check")
        return

    printer.information("\n" + "-" * 160)
    printer.information("SAFETY CHECK: Zone Partition Validation")
    printer.information("-" * 160)
    printer.information("Checking that zoiNone investments partitioned by zones sum to zoiNone total")
    printer.information("")

    # Evaluate zoiNone model with each zone's bus membership
    zone_investments = evaluate_zone_investments_from_none_model(none_sqlite_file, zone_files)

    # Calculate sum across all zones
    sum_by_tech = defaultdict(float)
    for zone_name, tech_dict in zone_investments.items():
        for tec, capacity in tech_dict.items():
            sum_by_tech[tec] += capacity

    sum_all = sum(sum_by_tech.values())
    none_total_all = sum(none_total.values())
    diff = sum_all - none_total_all

    printer.information(f"  zoiNone total investment:              {none_total_all:>15.2f} MW")
    printer.information(f"  zoiNone partitioned by zones:")
    for zone_name in sorted(zone_investments.keys()):
        zone_sum = sum(zone_investments[zone_name].values())
        printer.information(f"    {zone_name:>4s}: {zone_sum:>15.2f} MW")
    printer.information(f"  Sum of partitioned zones:              {sum_all:>15.2f} MW")
    printer.information(f"  Difference:                            {diff:>15.2f} MW")

    tolerance = 0.01  # 0.01 MW tolerance
    if abs(diff) < tolerance:
        printer.success(f"  Status:                                [PASSED]")
    else:
        printer.error(f"  Status:                                [FAILED] - Check zone definitions")


def main(folder="."):
    # Find all SQLite files in the specified folder
    search_pattern = os.path.join(folder, "*.sqlite")
    sqlite_files = glob.glob(search_pattern)

    if not sqlite_files:
        printer.warning(f"No SQLite files found in '{folder}'")
        return

    printer.information(f"Found {len(sqlite_files)} SQLite file(s) in '{folder}'")

    # Process each SQLite file
    results = []
    file_groups = {}
    all_technologies = set()
    uniform_representation_results = []  # Special handling for TP and SN

    for sqlite_file in sorted(sqlite_files):
        printer.information(f"\nProcessing '{sqlite_file}'...")
        print_run_parameters_from_sqlite(sqlite_file)

        try:
            # Calculate investments (both total and ZOI)
            total_invest = evaluate_gen_investment_by_technology_from_sqlite(sqlite_file, filter_zoi=False)
            zoi_invest = evaluate_gen_investment_by_technology_from_sqlite(sqlite_file, filter_zoi=True)

            # Track all technologies
            all_technologies.update(total_invest.keys())
            all_technologies.update(zoi_invest.keys())

            # Extract parameters
            dc_buffer, tp_buffer, zone, demand, pmax = extract_parameters(sqlite_file)

            # Create base identifier
            base_identifier = re.sub(r'-zoi[^-]+', '-zoi', sqlite_file)

            # Store results
            total_cap = sum(total_invest.values())
            zoi_cap = sum(zoi_invest.values())
            printer.success(f"  Total Investment: {total_cap:.2f} MW")
            printer.success(f"  ZOI Investment: {zoi_cap:.2f} MW")

            results.append((sqlite_file, dc_buffer, tp_buffer, zone, demand, pmax, total_invest, zoi_invest))

            # Check if this is a uniform representation (TP or SN)
            if zone in ['TP', 'SN']:
                uniform_representation_results.append((sqlite_file, dc_buffer, tp_buffer, zone, demand, pmax, total_invest, zoi_invest))
            else:
                # Group files for comparison (normal zones)
                if base_identifier not in file_groups:
                    file_groups[base_identifier] = []
                file_groups[base_identifier].append(
                    (sqlite_file, dc_buffer, tp_buffer, zone, demand, pmax, total_invest, zoi_invest)
                )

        except Exception as e:
            printer.error(f"  Failed to process '{sqlite_file}': {e}")
            import traceback
            traceback.print_exc()

    # Print summary table
    if results:
        print_summary_table(results)

    # Compare zones for each group
    for base_identifier, group in file_groups.items():
        # Find zoiNone entry (check both Python None and string "None")
        zoi_none_entry = next((entry for entry in group if (entry[3] == "None" or entry[3] is None)), None)

        if zoi_none_entry is None:
            continue

        # Get other zones (exclude both Python None and string "None")
        other_zones = [entry for entry in group if entry[3] != "None" and entry[3] is not None]

        if not other_zones:
            continue

        none_sqlite_file, _, _, _, _, _, none_total, _ = zoi_none_entry

        printer.information("\n" + "=" * 160)
        printer.information(f"Technology Investment Comparison for: {base_identifier}")
        printer.information("=" * 160)

        # For each zone
        for sqlite_file, dc_buffer, tp_buffer, zone, demand, pmax, zone_total, zone_zoi in other_zones:
            # IMPORTANT: Evaluate zoiNone using the same zoi_i as the zone-specific file
            # This ensures we compare the same geographic area in both models
            none_zoi = evaluate_gen_investment_with_custom_zoi(none_sqlite_file, sqlite_file)

            printer.information(f"\nComparing Zone: {zone}")
            printer.information("-" * 160)
            printer.information(
                f"  {'Technology':<20s} "
                f"{'Total (None)':>15s} {'Total (Zone)':>15s} {'Diff':>12s} {'Rel%':>10s} "
                f"{'ZOI (None)':>15s} {'ZOI (Zone)':>15s} {'Diff':>12s} {'Rel%':>10s}"
            )
            printer.information("-" * 160)

            # Print each technology
            for tec in sorted(all_technologies):
                none_total_val = none_total.get(tec, 0.0)
                zone_total_val = zone_total.get(tec, 0.0)
                diff_total = zone_total_val - none_total_val
                rel_total = (diff_total / none_total_val * 100) if none_total_val != 0 else 0.0

                none_zoi_val = none_zoi.get(tec, 0.0)
                zone_zoi_val = zone_zoi.get(tec, 0.0)
                diff_zoi = zone_zoi_val - none_zoi_val
                rel_zoi = (diff_zoi / none_zoi_val * 100) if none_zoi_val != 0 else 0.0

                printer.information(
                    f"  {tec:<20s} "
                    f"{none_total_val:>15.2f} {zone_total_val:>15.2f} {diff_total:>12.2f} {rel_total:>9.1f}% "
                    f"{none_zoi_val:>15.2f} {zone_zoi_val:>15.2f} {diff_zoi:>12.2f} {rel_zoi:>9.1f}%"
                )

        # Safety checks
        print_safety_checks(group, zoi_none_entry)

    # Display uniform representation results if any
    if uniform_representation_results:
        # Group by base identifier (without zone)
        uniform_groups = {}
        for entry in uniform_representation_results:
            sqlite_file, dc_buffer, tp_buffer, zone, demand, pmax, total_invest, zoi_invest = entry
            base_id = re.sub(r'-zoi[^-]+', '-zoi', sqlite_file)
            if base_id not in uniform_groups:
                uniform_groups[base_id] = []
            uniform_groups[base_id].append(entry)

        for base_id, group in uniform_groups.items():
            # Find corresponding zoiNone entry from regular results
            zoi_none_entry = None
            for sqlite_file, dc_buffer, tp_buffer, zone, demand, pmax, total_invest, zoi_invest in results:
                # Check for both Python None and string "None" to handle SQLite vs filename parsing
                if (zone == "None" or zone is None) and re.sub(r'-zoi[^-]+', '-zoi', sqlite_file) == base_id:
                    zoi_none_entry = (sqlite_file, dc_buffer, tp_buffer, zone, demand, pmax, total_invest, zoi_invest)
                    break

            if not zoi_none_entry:
                printer.warning(f"No DC-OPF baseline found for {base_id}")
                continue

            none_sqlite_file, _, _, _, _, _, none_total, none_zoi = zoi_none_entry
            none_total_cap = sum(none_total.values())

            printer.information("\n" + "=" * 140)
            printer.information(f"Uniform Technical Representations for: {base_id}")
            printer.information("=" * 140)
            printer.information(f"  {'Type':<10s} {'Description':<50s} {'Total Cap (MW)':>25s} {'Abs. Diff (MW)':>20s} {'Rel. Diff (%)':>15s}")
            printer.information("-" * 140)

            # Collect entries and sort: DC-OPF, TP, SN
            entries = []

            # Add DC-OPF as first entry
            entries.append(('DC-OPF', 'DC Optimal Power Flow (all lines as DC-OPF)', none_total_cap, 0.0, 0.0))

            # Add TP and SN
            for sqlite_file, dc_buffer, tp_buffer, zone, demand, pmax, total_invest, zoi_invest in group:
                if zone == 'TP':
                    desc = "Transport Model (all lines as TP)"
                    sort_order = 1
                elif zone == 'SN':
                    desc = "Single Node / Copper Plate (all lines as SN)"
                    sort_order = 2
                else:
                    desc = f"Uniform {zone}"
                    sort_order = 3

                total_cap = sum(total_invest.values())
                abs_diff = total_cap - none_total_cap
                rel_diff_pct = (abs_diff / none_total_cap * 100) if none_total_cap != 0 else 0.0
                entries.append((zone, desc, total_cap, abs_diff, rel_diff_pct, sort_order))

            # Sort entries: DC-OPF (0), TP (1), SN (2)
            entries_sorted = sorted([e for e in entries if len(e) > 5], key=lambda x: x[5])

            # Print all entries
            for entry in [entries[0]] + entries_sorted:
                if len(entry) == 5:
                    zone_type, desc, total_cap, abs_diff, rel_diff_pct = entry
                    printer.information(f"  {zone_type:<10s} {desc:<50s} {total_cap:>25.2f} {abs_diff:>20.2f} {rel_diff_pct:>14.1f}%")
                else:
                    zone_type, desc, total_cap, abs_diff, rel_diff_pct, _ = entry
                    printer.information(f"  {zone_type:<10s} {desc:<50s} {total_cap:>25.2f} {abs_diff:>20.2f} {rel_diff_pct:>14.1f}%")

            printer.information("-" * 140)
            printer.information("Note: DC-OPF is the baseline model with full network representation.")

            # Add detailed technology comparison for TP and SN
            for sqlite_file, dc_buffer, tp_buffer, zone, demand, pmax, total_invest, zoi_invest in sorted(group, key=lambda x: (0 if x[3] == 'TP' else 1)):
                if zone == 'TP':
                    desc = "Transport Model (all lines as TP)"
                elif zone == 'SN':
                    desc = "Single Node / Copper Plate (all lines as SN)"
                else:
                    continue

                printer.information(f"\nComparing: {desc} ({zone})")
                printer.information("-" * 140)
                printer.information(
                    f"  {'Technology':<20s} "
                    f"{'Total (DC-OPF)':>15s} {'Total (' + zone + ')':>15s} {'Diff':>12s} {'Rel%':>10s}"
                )
                printer.information("-" * 140)

                # Print each technology
                for tec in sorted(all_technologies):
                    none_total_val = none_total.get(tec, 0.0)
                    zone_total_val = total_invest.get(tec, 0.0)
                    diff_total = zone_total_val - none_total_val
                    rel_total = (diff_total / none_total_val * 100) if none_total_val != 0 else 0.0

                    printer.information(
                        f"  {tec:<20s} "
                        f"{none_total_val:>15.2f} {zone_total_val:>15.2f} {diff_total:>12.2f} {rel_total:>9.1f}%"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze generator investment capacity by technology from SQLite files", fromfile_prefix_chars='@')
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing SQLite files (default: current directory)")
    args = parser.parse_args()
    main(args.folder)
