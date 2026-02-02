import glob
import os
import re
import sqlite3
import time

import cloudpickle
import pandas as pd

from InOutModule.printer import Printer
from LEGO import LEGOUtilities

printer = Printer.getInstance()


def extract_parameters(filename):
    """
    Extract dcBuffer, tpBuffer, zone, demand, and pmax values from filename or SQLite database.
    First tries to read from run_parameters table in corresponding SQLite, falls back to filename parsing.
    """
    # Try to read from corresponding SQLite database first
    sqlite_file = filename.replace('.pkl', '.sqlite')
    if os.path.exists(sqlite_file):
        try:
            conn = sqlite3.connect(sqlite_file)
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


def print_run_parameters_from_pkl(pkl_file):
    """
    Print run parameters from corresponding SQLite file if available.
    :param pkl_file: Path to the pickle file
    """
    sqlite_file = pkl_file.replace('.pkl', '.sqlite')
    if os.path.exists(sqlite_file):
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


def main():
    # Find all pickle files in the current directory
    pickle_files = glob.glob("*.pkl")

    if not pickle_files:
        printer.warning("No pickle files found in current directory")
        return

    printer.information(f"Found {len(pickle_files)} pickle file(s)")

    # Process each pickle file and group by base identifier (everything except zone)
    results = []
    file_groups = {}  # Key: base_identifier, Value: list of (pkl_file, dc_buffer, tp_buffer, zone, zoi_value, model)
    uniform_representation_results = []  # Special handling for TP and SN

    for pkl_file in sorted(pickle_files):
        printer.information(f"\nProcessing '{pkl_file}'...")
        print_run_parameters_from_pkl(pkl_file)

        try:
            # Load the ZOI data (lightweight structure instead of full model)
            start_time = time.time()
            with open(pkl_file, mode='rb') as file:
                zoi_data = cloudpickle.load(file)
            load_time = time.time() - start_time
            printer.information(f"  Loaded in {load_time:.2f} seconds")

            # Check if this is old format (full model) or new format (zoi_data dict)
            if isinstance(zoi_data, dict) and 'zoi_objective_value' in zoi_data:
                # New format: extract pre-calculated value
                zoi_value = zoi_data['zoi_objective_value']
                printer.information(f"  Using pre-calculated ZOI objective value")
            else:
                # Old format: full model - calculate ZOI objective
                printer.warning(f"  Old format detected (full model), calculating ZOI objective...")
                calc_start_time = time.time()
                _, zoi_value = LEGOUtilities.evaluate_zoi_objective(zoi_data, line_filter="both")
                calc_time = time.time() - calc_start_time
                printer.information(f"  ZOI objective calculated in {calc_time:.2f} seconds")

            # Extract parameters from filename
            dc_buffer, tp_buffer, zone, demand, pmax = extract_parameters(pkl_file)

            # Create base identifier by removing zone from filename
            base_identifier = re.sub(r'-zoi[^-]+', '-zoi', pkl_file)

            printer.success(f"  ZOI Objective: {zoi_value:.2f}")
            results.append((pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value))

            # Check if this is a uniform representation (TP or SN)
            if zone in ['TP', 'SN']:
                uniform_representation_results.append((pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, zoi_data))
            else:
                # Group files for comparison (normal zones)
                if base_identifier not in file_groups:
                    file_groups[base_identifier] = []
                file_groups[base_identifier].append((pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, zoi_data))

        except Exception as e:
            printer.error(f"  Failed to process '{pkl_file}': {e}")

    # Print summary
    if results:
        # Calculate the maximum filename length for proper alignment
        max_filename_len = max(len(pkl_file) for pkl_file, _, _, _, _, _, _ in results)
        # Ensure minimum width for readability
        filename_width = max(max_filename_len, len("Filename"))
        # Calculate total table width
        table_width = filename_width + 2 + 8 + 8 + 8 + 8 + 14  # 2 for spacing, rest for columns

        printer.information("\n" + "=" * table_width)
        printer.information("Summary of ZOI Objective Values:")
        printer.information("=" * table_width)
        printer.information(f"  {'Filename':<{filename_width}s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Demand':>8s} {'PMax':>8s} {'ZOI Objective':>14s}")
        printer.information("-" * table_width)
        for pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value in results:
            dc_str = str(dc_buffer) if dc_buffer is not None else "N/A"
            tp_str = str(tp_buffer) if tp_buffer is not None else "N/A"
            demand_str = f"{demand:.1f}" if demand is not None else "N/A"
            pmax_str = f"{pmax:.1f}" if pmax is not None else "N/A"
            printer.information(f"  {pkl_file:<{filename_width}s} {dc_str:>8s} {tp_str:>8s} {demand_str:>8s} {pmax_str:>8s} {zoi_value:>14.2f}")

    # Compare zoiNone models with other zones in their groups
    for base_identifier, group in file_groups.items():
        # Find the zoiNone model in this group (check both Python None and string "None")
        zoi_none_entry = next((entry for entry in group if (entry[3] is None or entry[3] == "None")), None)

        if zoi_none_entry is None:
            continue

        zoi_none_file, _, _, _, _, _, zoi_none_original_value, zoi_none_data = zoi_none_entry

        # Get other zones in this group (exclude both Python None and string "None")
        other_zones = [entry for entry in group if entry[3] is not None and entry[3] != "None"]

        if not other_zones:
            continue

        printer.information("\n" + "=" * 120)
        printer.information(f"ZOI-None Model Comparisons for: {base_identifier}")
        printer.information("=" * 120)
        printer.information(f"  {'Zone':<20s} {'ZOI Objective (zoiNone model)':>30s} {'ZOI Objective (original)':>30s} {'Difference':>15s} {'Rel. Diff (%)':>15s}")
        printer.information("-" * 120)

        sum_of_original_zone_objectives = 0.0

        for pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, original_zoi_value, zone_data in other_zones:
            try:
                # Check if data is in new format (dict) or old format (model)
                if isinstance(zone_data, dict) and 'sets' in zone_data:
                    # New format: extract zoi_i from zone_data
                    zone_zoi_i = zone_data['sets']['zoi_i']
                else:
                    # Old format: extract from model
                    zone_zoi_i = list(zone_data.zoi_i)

                # Recalculate ZOI objective for zoiNone data with zone's ZOI definition
                if isinstance(zoi_none_data, dict) and 'zoi_objective_value' in zoi_none_data:
                    # New format: use lightweight recalculation
                    zoi_none_value = LEGOUtilities.evaluate_zoi_objective_from_data(
                        zoi_none_data, new_zoi_i=zone_zoi_i, line_filter="both"
                    )
                else:
                    # Old format: modify model and recalculate
                    zoi_none_data.zoi_i.clear()
                    zoi_none_data.zoi_i.construct()
                    for bus in zone_zoi_i:
                        zoi_none_data.zoi_i.add(bus)
                    _, zoi_none_value = LEGOUtilities.evaluate_zoi_objective(zoi_none_data, line_filter="both")

                difference = zoi_none_value - original_zoi_value
                rel_diff_pct = (difference / zoi_none_value * 100) if zoi_none_value != 0 else 0.0
                sum_of_original_zone_objectives += original_zoi_value
                printer.information(f"  {zone:<20s} {zoi_none_value:>30.2f} {original_zoi_value:>30.2f} {difference:>15.2f} {rel_diff_pct:>14.1f}%")

            except Exception as e:
                printer.error(f"  Failed to compare zone {zone}: {e}")
                import traceback
                traceback.print_exc()

        # Safety check: sum of original zone objectives should equal original zoiNone objective
        printer.information("-" * 120)
        printer.information(f"  {'SAFETY CHECK':<20s} {'Sum of original zone objectives':>30s} {'Original zoiNone objective':>30s} {'Difference':>15s} {'Rel. Diff (%)':>15s}")
        safety_difference = sum_of_original_zone_objectives - zoi_none_original_value
        safety_rel_diff_pct = (safety_difference / sum_of_original_zone_objectives * 100) if sum_of_original_zone_objectives != 0 else 0.0
        tolerance_pct = 0.01  # 0.01% relative tolerance
        if abs(safety_rel_diff_pct) < tolerance_pct:
            printer.success(f"  {'[PASSED]':<20s} {sum_of_original_zone_objectives:>30.2f} {zoi_none_original_value:>30.2f} {safety_difference:>15.2f} {safety_rel_diff_pct:>14.1f}%")
        else:
            printer.error(f"  {'[FAILED]':<20s} {sum_of_original_zone_objectives:>30.2f} {zoi_none_original_value:>30.2f} {safety_difference:>15.2f} {safety_rel_diff_pct:>14.1f}%")

    # Display uniform representation results if any
    if uniform_representation_results:
        # Group by base identifier (without zone)
        uniform_groups = {}
        for entry in uniform_representation_results:
            pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, zoi_data = entry
            base_id = re.sub(r'-zoi[^-]+', '-zoi', pkl_file)
            if base_id not in uniform_groups:
                uniform_groups[base_id] = []
            uniform_groups[base_id].append(entry)

        for base_id, group in uniform_groups.items():
            # Find corresponding zoiNone entry from regular results
            none_entry = None
            for pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value in results:
                if (zone is None or zone == "None") and re.sub(r'-zoi[^-]+', '-zoi', pkl_file) == base_id:
                    none_entry = (pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value)
                    break

            if not none_entry:
                printer.warning(f"No DC-OPF baseline found for {base_id}")
                continue

            none_pkl_file, _, _, _, _, _, none_zoi_value = none_entry
            none_sqlite_file = none_pkl_file.replace('.pkl', '.sqlite')
            none_total_obj = None
            if os.path.exists(none_sqlite_file):
                try:
                    import sqlite3
                    conn = sqlite3.connect(none_sqlite_file)
                    cursor = conn.execute('SELECT `values` FROM objective')
                    none_total_obj = cursor.fetchone()[0]
                    conn.close()
                except Exception:
                    pass

            if none_total_obj is None:
                printer.warning(f"Could not read DC-OPF objective for {base_id}")
                continue

            printer.information("\n" + "=" * 140)
            printer.information(f"Uniform Technical Representations for: {base_id}")
            printer.information("=" * 140)
            printer.information(f"  {'Type':<10s} {'Description':<50s} {'Total Objective':>20s} {'Abs. Diff':>15s} {'Rel. Diff (%)':>15s}")
            printer.information("-" * 140)

            # Collect entries and sort: DC-OPF, TP, SN
            entries = []

            # Add DC-OPF as first entry
            entries.append(('DC-OPF', 'DC Optimal Power Flow (all lines as DC-OPF)', none_total_obj, 0.0, 0.0))

            # Add TP and SN
            for pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, zoi_data in group:
                if zone == 'TP':
                    desc = "Transport Model (all lines as TP)"
                    sort_order = 1
                elif zone == 'SN':
                    desc = "Single Node / Copper Plate (all lines as SN)"
                    sort_order = 2
                else:
                    desc = f"Uniform {zone}"
                    sort_order = 3

                # Get total objective from SQLite if available
                sqlite_file = pkl_file.replace('.pkl', '.sqlite')
                total_obj = None
                if os.path.exists(sqlite_file):
                    try:
                        import sqlite3
                        conn = sqlite3.connect(sqlite_file)
                        cursor = conn.execute('SELECT `values` FROM objective')
                        total_obj = cursor.fetchone()[0]
                        conn.close()
                    except Exception:
                        pass

                if total_obj is not None:
                    abs_diff = total_obj - none_total_obj
                    rel_diff_pct = (abs_diff / none_total_obj * 100) if none_total_obj != 0 else 0.0
                    entries.append((zone, desc, total_obj, abs_diff, rel_diff_pct, sort_order))

            # Sort entries: DC-OPF (0), TP (1), SN (2)
            entries_sorted = sorted([e for e in entries if len(e) > 5], key=lambda x: x[5])

            # Print all entries
            for entry in [entries[0]] + entries_sorted:
                if len(entry) == 5:
                    zone_type, desc, total_obj, abs_diff, rel_diff_pct = entry
                    printer.information(f"  {zone_type:<10s} {desc:<50s} {total_obj:>20.2f} {abs_diff:>15.2f} {rel_diff_pct:>14.1f}%")
                else:
                    zone_type, desc, total_obj, abs_diff, rel_diff_pct, _ = entry
                    printer.information(f"  {zone_type:<10s} {desc:<50s} {total_obj:>20.2f} {abs_diff:>15.2f} {rel_diff_pct:>14.1f}%")

            printer.information("-" * 140)
            printer.information("Note: DC-OPF is the baseline model with full network representation.")


if __name__ == "__main__":
    main()
