import argparse
import glob
import os
import re
import sqlite3
import time

import pandas as pd

from InOutModule.printer import Printer
from LEGO import LEGOUtilities

printer = Printer.getInstance()

# Default values from TechnicalRepresentation.py
DEFAULT_DC_BUFFER = 1
DEFAULT_TP_BUFFER = 1
DEFAULT_SCALE_DEMAND = 1.0
DEFAULT_SCALE_PMAX = 1.0


def load_zoi_data_from_sqlite(sqlite_file):
    """
    Load ZOI objective data from SQLite file.

    :param sqlite_file: Path to SQLite database file
    :return: Dictionary with ZOI data in the same format as extract_zoi_objective_data
    """
    conn = sqlite3.connect(sqlite_file)

    try:
        # Load sets
        sets_data = {}
        for set_name in ['i', 'g', 'gi', 'la', 'zoi_i']:
            try:
                df = pd.read_sql_query(f'SELECT * FROM {set_name}', conn)
                # Drop the 'index' column added by pandas
                if 'index' in df.columns:
                    df = df.drop(columns=['index'])

                if set_name in ['gi', 'la']:
                    # These are tuple sets - convert rows to tuples
                    # Columns are named '0', '1', ('2' for la)
                    sets_data[set_name] = [tuple(row) for row in df.values]
                else:
                    # Scalar sets - column is named '0'
                    if '0' in df.columns and len(df) > 0:
                        sets_data[set_name] = df['0'].tolist()
                    else:
                        sets_data[set_name] = []
            except Exception as e:
                printer.warning(f"Could not load set {set_name}: {e}")
                sets_data[set_name] = []

        # Check for hubConnections (optional)
        try:
            df = pd.read_sql_query('SELECT * FROM hubConnections', conn)
            sets_data['hubConnections'] = [tuple(row) for row in df.values]
        except Exception:
            sets_data['hubConnections'] = []

        # Load objective decomposition
        try:
            df_constant = pd.read_sql_query('SELECT * FROM objective_constant', conn)
            constant = df_constant.iloc[0]['constant']
        except Exception as e:
            printer.error(f"Could not load objective_constant: {e}")
            raise

        try:
            df_terms = pd.read_sql_query('SELECT * FROM objective_terms', conn)
            # Parse string indices back to their original types (tuples, strings, etc.)
            linear_vars_info = []  # List of (var_name, var_index) tuples
            import ast

            # Check if var_name column exists (new format) or not (old format)
            if 'var_name' in df_terms.columns:
                # New format: includes variable names
                for _, row in df_terms.iterrows():
                    var_name = row['var_name']
                    idx_str = row['var_index']
                    try:
                        idx = ast.literal_eval(idx_str)
                    except (ValueError, SyntaxError):
                        idx = idx_str
                    linear_vars_info.append((var_name, idx))
            else:
                # Old format: only indices (backward compatibility)
                for idx_str in df_terms['var_index']:
                    try:
                        idx = ast.literal_eval(idx_str)
                    except (ValueError, SyntaxError):
                        idx = idx_str
                    linear_vars_info.append((None, idx))  # No var_name available

            linear_coefs = df_terms['coefficient'].tolist()
        except Exception as e:
            printer.error(f"Could not load objective_terms: {e}")
            raise

        objective_data = {
            'constant': constant,
            'linear_vars_info': linear_vars_info,  # Now includes (var_name, index) pairs
            'linear_coefs': linear_coefs,
        }

        # Load variable values
        # Get all variable names from the database
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'v%'")
        var_tables = [row[0] for row in cursor.fetchall()]

        var_values = {}
        for var_name in var_tables:
            try:
                df = pd.read_sql_query(f'SELECT * FROM {var_name}', conn)
                # Reconstruct variable indices and values
                if 'values' in df.columns:
                    # Get index columns (all columns except 'values')
                    index_cols = [col for col in df.columns if col != 'values']

                    if len(index_cols) == 0:
                        # Scalar variable - store with (var_name, var_name) as key
                        var_values[(var_name, var_name)] = df['values'].iloc[0]
                    else:
                        # Indexed variable - store with (var_name, index) as composite key
                        for _, row in df.iterrows():
                            if len(index_cols) == 1:
                                idx = row[index_cols[0]]
                            else:
                                idx = tuple(row[col] for col in index_cols)
                            var_values[(var_name, idx)] = row['values']
            except Exception as e:
                printer.warning(f"Could not load variable {var_name}: {e}")

        # Load total objective
        try:
            df_obj = pd.read_sql_query('SELECT * FROM objective', conn)
            total_objective = df_obj.iloc[0]['values']
        except Exception:
            total_objective = None

        # Load work_units from solver_statistics (may not exist)
        work_units = None
        try:
            df_stats = pd.read_sql_query('SELECT * FROM solver_statistics', conn)
            if 'work_units' in df_stats.columns:
                val = df_stats.iloc[0]['work_units']
                if val is not None:
                    work_units = float(val)
        except Exception:
            pass

        # Create ZOI data structure
        zoi_data = {
            'var_values': var_values,
            'sets': sets_data,
            'objective': objective_data,
            'total_objective': total_objective,
            'work_units': work_units,
        }

        return zoi_data

    finally:
        conn.close()


def extract_parameters(filename):
    """
    Extract all parameters from filename.
    Uses default values if parameters are missing from the filename.

    Handles both old and new filename formats:
    - Old: TR-datadata_<name>-zoi<zone>-limitK<k>-dcBuffer<dc>-tpBuffer<tp>-demand<d>-pmax<p>
    - New: TR-datadata_<name>-limitK<k>-demand<d>-pmax<p>-dcBuffer<dc>-tpBuffer<tp>-zoi<zone>

    Returns:
        tuple: (input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, params_dict)
               where params_dict contains info about which values are defaults
    """
    params_dict = {}

    # Remove file extension first
    filename_no_ext = filename.replace('.sqlite', '')

    # Extract input directory name (everything after "TR-data" until first parameter)
    input_dir = None
    input_dir_match = re.search(r'TR-data(.+?)(?:-(?:limitK|demand|pmax|dcBuffer|tpBuffer|zoi))', filename_no_ext)
    if input_dir_match:
        input_dir = input_dir_match.group(1)
        params_dict['input_dir'] = {'value': input_dir, 'is_default': False}
    else:
        params_dict['input_dir'] = {'value': None, 'is_default': True}

    # Extract limitK (optional) - format: -limitKk0001-k0048
    limit_k = None
    limit_k_match = re.search(r'-limitK(k\d+-k\d+)', filename_no_ext)
    if limit_k_match:
        limit_k = limit_k_match.group(1)
        params_dict['limit_k'] = {'value': limit_k, 'is_default': False}
    else:
        params_dict['limit_k'] = {'value': None, 'is_default': True}

    # Extract zone - can appear anywhere in the filename
    zone_match = re.search(r'-zoi([^-]+?)(?:-|$)', filename_no_ext)
    if not zone_match:
        # No zone found - return all defaults
        return (input_dir, limit_k, None, None, None, DEFAULT_SCALE_DEMAND, DEFAULT_SCALE_PMAX, {
            'input_dir': params_dict.get('input_dir', {'value': None, 'is_default': True}),
            'limit_k': params_dict.get('limit_k', {'value': None, 'is_default': True}),
            'zone': {'value': None, 'is_default': True},
            'dc_buffer': {'value': None, 'is_default': True},
            'tp_buffer': {'value': None, 'is_default': True},
            'scale_demand': {'value': DEFAULT_SCALE_DEMAND, 'is_default': True},
            'scale_pmax': {'value': DEFAULT_SCALE_PMAX, 'is_default': True}
        })

    zone = zone_match.group(1)
    params_dict['zone'] = {'value': zone, 'is_default': False}

    # Extract dcBuffer (optional)
    dc_match = re.search(r'-dcBuffer(\d+)', filename_no_ext)
    if dc_match:
        dc_buffer = int(dc_match.group(1))
        params_dict['dc_buffer'] = {'value': dc_buffer, 'is_default': False}
    else:
        dc_buffer = DEFAULT_DC_BUFFER
        params_dict['dc_buffer'] = {'value': dc_buffer, 'is_default': True}

    # Extract tpBuffer (optional)
    tp_match = re.search(r'-tpBuffer(\d+)', filename_no_ext)
    if tp_match:
        tp_buffer = int(tp_match.group(1))
        params_dict['tp_buffer'] = {'value': tp_buffer, 'is_default': False}
    else:
        tp_buffer = DEFAULT_TP_BUFFER
        params_dict['tp_buffer'] = {'value': tp_buffer, 'is_default': True}

    # Extract demand (optional)
    demand_match = re.search(r'-demand(\d+(?:\.\d+)?)', filename_no_ext)
    if demand_match:
        demand = float(demand_match.group(1))
        params_dict['scale_demand'] = {'value': demand, 'is_default': False}
    else:
        demand = DEFAULT_SCALE_DEMAND
        params_dict['scale_demand'] = {'value': demand, 'is_default': True}

    # Extract pmax (optional)
    pmax_match = re.search(r'-pmax(\d+(?:\.\d+)?)', filename_no_ext)
    if pmax_match:
        pmax = float(pmax_match.group(1))
        params_dict['scale_pmax'] = {'value': pmax, 'is_default': False}
    else:
        pmax = DEFAULT_SCALE_PMAX
        params_dict['scale_pmax'] = {'value': pmax, 'is_default': True}

    return input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, params_dict


def print_run_parameters(params_dict):
    """
    Print run parameters extracted from filename.
    :param params_dict: Dictionary of parameters with default flags
    """
    if not params_dict:
        printer.warning("  Could not extract parameters from filename")
        return

    # Check if zone is a uniform representation (None, DC, TP, SN)
    zone_info = params_dict.get('zone', {})
    zone = zone_info.get('value')
    is_uniform_repr = zone in ['DC', 'TP', 'SN'] or zone is None or zone == "None"

    param_strs = []
    for key, info in params_dict.items():
        value = info['value']
        is_default = info['is_default']

        # Mark dc_buffer and tp_buffer as unused for uniform representations
        if is_uniform_repr and key in ['dc_buffer', 'tp_buffer']:
            param_strs.append(f"{key}= (unused)")
        elif is_default:
            param_strs.append(f"{key}={value} (default)")
        else:
            param_strs.append(f"{key}={value}")

    printer.information(f"  Run parameters: {', '.join(param_strs)}")


def main(folder="."):
    # Find all SQLite files in the specified folder
    sqlite_pattern = os.path.join(folder, "*.sqlite")
    sqlite_files = sorted(glob.glob(sqlite_pattern))

    if not sqlite_files:
        printer.warning(f"No SQLite files found in '{folder}'")
        return

    printer.information(f"Found {len(sqlite_files)} SQLite file(s) in '{folder}'")

    # Process each file and group by base identifier
    results = []
    all_files_data = []  # Store all file data for flexible grouping
    uniform_files = []  # Store uniform representation files (DC, TP, SN)

    for sqlite_file in sqlite_files:
        printer.information(f"\nProcessing '{sqlite_file}'...")

        try:
            # Extract parameters from filename
            input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, params_dict = extract_parameters(sqlite_file)
            print_run_parameters(params_dict)

            # Load the ZOI data from SQLite
            start_time = time.time()
            printer.information(f"  Loading from SQLite database...")
            zoi_data = load_zoi_data_from_sqlite(sqlite_file)
            load_time = time.time() - start_time
            printer.information(f"  Loaded in {load_time:.2f} seconds")

            # Check if this is a uniform representation
            is_uniform_repr = zone in ['DC', 'TP', 'SN'] or zone is None or zone == "None"

            # Calculate objectives based on type
            zoi_value = None
            if is_uniform_repr:
                # Uniform representation: skip ZOI objective, only calculate total objective
                printer.information(f"  Uniform representation detected - skipping ZOI objective calculation")
            else:
                # Zone-specific: calculate ZOI objective
                calc_start_time = time.time()
                zoi_i = zoi_data['sets']['zoi_i']
                zoi_value = LEGOUtilities.evaluate_zoi_objective_from_data(zoi_data, new_zoi_i=zoi_i, line_filter="both")
                calc_time = time.time() - calc_start_time
                printer.information(f"  ZOI objective calculated in {calc_time:.2f} seconds")
                printer.success(f"  ZOI Objective: {zoi_value:.2f}")

            # Calculate total objective from data (for all cases)
            calc_total_obj = None
            stored_total_obj = None
            try:
                # Get stored objective if available
                if 'total_objective' in zoi_data:
                    stored_total_obj = zoi_data['total_objective']

                # Calculate total objective from components
                obj_data = zoi_data['objective']
                var_data = zoi_data['var_values']
                calc_total_obj = obj_data['constant']

                # Use (var_name, index) composite keys
                for (var_name, idx), coef in zip(obj_data['linear_vars_info'], obj_data['linear_coefs']):
                    key = (var_name, idx)
                    if key in var_data:
                        calc_total_obj += coef * var_data[key]
            except Exception as e:
                printer.warning(f"  Could not calculate total objective: {e}")

            if calc_total_obj is not None:
                printer.success(f"  Calculated Total Objective: {calc_total_obj:.2f}")

                # Compare with stored objective
                if stored_total_obj is not None:
                    diff = abs(calc_total_obj - stored_total_obj)
                    rel_diff_pct = (diff / stored_total_obj * 100) if stored_total_obj != 0 else 0.0
                    if diff > 0.01:  # Tolerance of 0.01
                        printer.warning(f"  Stored Total Objective: {stored_total_obj:.2f} (DIFFERENCE: {diff:.4f}, {rel_diff_pct:.4f}%)")
                    else:
                        printer.information(f"  Stored Total Objective: {stored_total_obj:.2f} (matches calculated)")
                else:
                    printer.warning(f"  Stored Total Objective: Not available in data")

            results.append((sqlite_file, input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, calc_total_obj))

            # Store file data
            file_data = {
                'sqlite_file': sqlite_file,
                'input_dir': input_dir,
                'limit_k': limit_k,
                'dc_buffer': dc_buffer,
                'tp_buffer': tp_buffer,
                'zone': zone,
                'demand': demand,
                'pmax': pmax,
                'zoi_value': zoi_value,
                'zoi_data': zoi_data,
                'total_obj': calc_total_obj,
                'work_units': zoi_data.get('work_units')
            }

            # Separate uniform representations from zone-specific runs
            if zone in ['DC', 'TP', 'SN'] or zone is None or zone == "None":
                uniform_files.append(file_data)
            else:
                all_files_data.append(file_data)

        except Exception as e:
            printer.error(f"  Failed to process '{sqlite_file}': {e}")

    # Group zone-specific files by their parameters
    # Key: (input_dir, limit_k, dc_buffer, tp_buffer, demand, pmax)
    file_groups = {}

    for file_data in all_files_data:
        # Group by all parameters
        group_key = (file_data['input_dir'], file_data['limit_k'], file_data['dc_buffer'],
                     file_data['tp_buffer'], file_data['demand'], file_data['pmax'])

        if group_key not in file_groups:
            file_groups[group_key] = {
                'zone_files': [],
                'uniform_files': []
            }

        file_groups[group_key]['zone_files'].append(file_data)

    # Match uniform files to groups (ignoring dcBuffer/tpBuffer)
    for uniform_file in uniform_files:
        # Match based on input_dir, limit_k, demand, and pmax
        for group_key, group_data in file_groups.items():
            input_dir, limit_k, dc_buf, tp_buf, demand, pmax = group_key
            if (uniform_file['input_dir'] == input_dir and uniform_file['limit_k'] == limit_k and
                    uniform_file['demand'] == demand and uniform_file['pmax'] == pmax):
                group_data['uniform_files'].append(uniform_file)

    # Print summary
    if results:
        # Sort results by groups: input_dir, limit_k, demand, pmax, then by dcBuffer/tpBuffer (with uniform reps first), then by zone
        def sort_key(result):
            sqlite_file, input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, total_obj = result
            # For uniform representations (None, TP, SN), put them first in each group
            is_uniform = zone in ['DC', 'TP', 'SN'] or zone is None or zone == "None"
            # Sort order: input_dir, limit_k, demand, pmax, uniform flag, dcBuffer, tpBuffer, zone
            zone_str = str(zone) if zone is not None else ""
            input_dir_str = str(input_dir) if input_dir is not None else ""
            limit_k_str = str(limit_k) if limit_k is not None else ""
            dc_sort = -1 if is_uniform else (dc_buffer if dc_buffer is not None else 999)
            tp_sort = -1 if is_uniform else (tp_buffer if tp_buffer is not None else 999)
            return (input_dir_str, limit_k_str, demand, pmax, 0 if is_uniform else 1, dc_sort, tp_sort, zone_str)

        sorted_results = sorted(results, key=sort_key)

        # Calculate the maximum filename length for proper alignment
        max_filename_len = max(len(sqlite_file) for sqlite_file, _, _, _, _, _, _, _, _, _ in sorted_results)
        # Ensure minimum width for readability
        filename_width = max(max_filename_len, len("Filename"))
        # Calculate total table width
        table_width = filename_width + 2 + 8 + 8 + 8 + 8 + 14 + 16

        printer.information("\n" + "=" * table_width)
        printer.information("Summary of Objective Values (grouped by parameters):")
        printer.information("=" * table_width)
        printer.information(f"  {'Filename':<{filename_width}s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Demand':>8s} {'PMax':>8s} {'ZOI Objective':>14s} {'Total Objective':>16s}")
        printer.information("-" * table_width)

        # Track previous group to insert separators
        prev_group = None

        for sqlite_file, input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, total_obj in sorted_results:
            # Check if we're starting a new group (input_dir, limit_k, demand, pmax)
            current_group = (input_dir, limit_k, demand, pmax)
            if prev_group is not None and current_group != prev_group:
                # Insert separator line between groups
                printer.information("-" * table_width)
            prev_group = current_group

            # Show '-' for dcBuffer and tpBuffer for uniform representations
            is_uniform = zone in ['DC', 'TP', 'SN'] or zone is None or zone == "None"
            dc_str = "-" if is_uniform else (str(dc_buffer) if dc_buffer is not None else "N/A")
            tp_str = "-" if is_uniform else (str(tp_buffer) if tp_buffer is not None else "N/A")
            demand_str = f"{demand:.1f}" if demand is not None else "N/A"
            pmax_str = f"{pmax:.1f}" if pmax is not None else "N/A"
            # Show N/A for ZOI objective if it's a uniform representation
            zoi_str = "N/A" if (is_uniform or zoi_value is None) else f"{zoi_value:.2f}"
            # Show total objective if available
            total_str = f"{total_obj:.2f}" if total_obj is not None else "N/A"
            printer.information(f"  {sqlite_file:<{filename_width}s} {dc_str:>8s} {tp_str:>8s} {demand_str:>8s} {pmax_str:>8s} {zoi_str:>14s} {total_str:>16s}")

    # Group uniform files by all grouping parameters for easier access
    uniform_groups = {}
    for uniform_file in uniform_files:
        group_key = (uniform_file['input_dir'], uniform_file['limit_k'],
                     uniform_file['demand'], uniform_file['pmax'])
        if group_key not in uniform_groups:
            uniform_groups[group_key] = []
        uniform_groups[group_key].append(uniform_file)

    # Organize groups by (input_dir, limit_k, demand, pmax) to show zone-specific and uniform representations together
    main_groups = {}
    for (input_dir, limit_k, dc_buf, tp_buf, demand, pmax), group_data in file_groups.items():
        key = (input_dir, limit_k, demand, pmax)
        if key not in main_groups:
            main_groups[key] = []
        main_groups[key].append((dc_buf, tp_buf, group_data))

    # Process each (input_dir, limit_k, demand, pmax) group
    for (input_dir, limit_k, demand, pmax), configs in sorted(main_groups.items(), key=lambda x: (x[0][0] or '', x[0][1] or '', x[0][2], x[0][3])):
        # Build group description
        group_desc_parts = []
        if input_dir:
            group_desc_parts.append(f"input={input_dir}")
        if limit_k:
            group_desc_parts.append(f"limitK={limit_k}")
        group_desc_parts.append(f"demand={demand}")
        group_desc_parts.append(f"pmax={pmax}")
        group_desc = ", ".join(group_desc_parts)

        printer.information("\n" + "=" * 140)
        printer.information(f"COMPARISON GROUP: {group_desc}")
        printer.information("=" * 140)

        # --- 1) Uniform Technical Representation Comparisons (shown first) ---
        uniform_group_key = (input_dir, limit_k, demand, pmax)
        if uniform_group_key in uniform_groups:
            group_uniform_files = uniform_groups[uniform_group_key]

            # Find baseline: prefer zoiNone, fall back to zoiDC
            dc_opf_file = next(
                (f for f in group_uniform_files if (f['zone'] is None or f['zone'] == "None")),
                None
            )
            if dc_opf_file is None:
                dc_opf_file = next(
                    (f for f in group_uniform_files if f['zone'] == 'DC'),
                    None
                )

            if dc_opf_file is not None:
                dc_opf_total_obj = dc_opf_file.get('total_obj')
                baseline_zone = dc_opf_file.get('zone')
                other_files = [f for f in group_uniform_files if f['zone'] != baseline_zone]

                if dc_opf_total_obj is not None and other_files:
                    # Build label/description for baseline
                    zone_labels = {
                        'DC': ('DC-OPF', 'DC Optimal Power Flow (all lines as DC-OPF)'),
                        'TP': ('TP', 'Transport Model (all lines as TP)'),
                        'SN': ('SN', 'Single Node / Copper Plate (all lines as SN)'),
                    }
                    if baseline_zone is None or baseline_zone == "None":
                        baseline_label, baseline_desc = 'None', 'No ZOI adjustment (original Excel settings)'
                    else:
                        baseline_label, baseline_desc = zone_labels.get(baseline_zone, (baseline_zone, f'Uniform {baseline_zone}'))

                    # Collect (label, desc, total_obj, abs_diff, rel_diff_pct, sort_order, work_units)
                    entries = []
                    entries.append((baseline_label, baseline_desc + ' (BASELINE)',
                                    dc_opf_total_obj, 0.0, 0.0, -99, dc_opf_file.get('work_units')))

                    sort_order_map = {'DC': 0, 'TP': 1, 'SN': 2}
                    for uf in other_files:
                        zone = uf['zone']
                        if zone is None or zone == "None":
                            label, desc, so = 'None', 'No ZOI adjustment (original Excel settings)', -1
                        else:
                            label, desc = zone_labels.get(zone, (zone, f'Uniform {zone}'))
                            so = sort_order_map.get(zone, 3)
                        total_obj = uf.get('total_obj')
                        if total_obj is not None:
                            abs_diff = total_obj - dc_opf_total_obj
                            rel_diff_pct = (abs_diff / dc_opf_total_obj * 100) if dc_opf_total_obj != 0 else 0.0
                            entries.append((label, desc, total_obj, abs_diff, rel_diff_pct, so, uf.get('work_units')))

                    # Sort by sort_order (baseline first via -99, then others)
                    entries.sort(key=lambda e: e[5])

                    printer.information(f"\nUniform Technical Representation Comparisons")
                    printer.information("-" * 140)
                    printer.information(f"  {'Type':<10s} {'Description':<60s} {'Total Objective':>18s} {'Abs. Diff':>15s} {'Rel. Diff (%)':>15s} {'Work Units':>12s}")
                    printer.information("-" * 140)
                    for label, desc, total_obj, abs_diff, rel_diff_pct, _, wu in entries:
                        wu_str = f"{wu:.2f}" if wu is not None else "N/A"
                        printer.information(f"  {label:<10s} {desc:<60s} {total_obj:>18.2f} {abs_diff:>15.2f} {rel_diff_pct:>14.1f}% {wu_str:>12s}")
                    printer.information("-" * 140)

        # --- 2) Zone-Specific Comparisons (one per dc_buf / tp_buf config) ---
        for dc_buf, tp_buf, group_data in sorted(configs):
            # Find baseline model: prefer zoiNone, fall back to zoiDC
            baseline_file_data = next(
                (f for f in group_data['uniform_files']
                 if (f['zone'] is None or f['zone'] == "None")),
                None
            )
            if baseline_file_data is None:
                baseline_file_data = next(
                    (f for f in group_data['uniform_files'] if f['zone'] == 'DC'),
                    None
                )

            if baseline_file_data is None:
                continue

            zone_files = group_data['zone_files']
            if not zone_files:
                continue

            baseline_total_obj = baseline_file_data.get('total_obj')
            baseline_zone = baseline_file_data.get('zone')
            baseline_label = "zoiNone" if (baseline_zone is None or baseline_zone == "None") else f"zoi{baseline_zone}"

            printer.information(f"\nZone-Specific Comparisons (Baseline: {baseline_label})")
            printer.information(f"Parameters: {group_desc}, dcBuffer={dc_buf}, tpBuffer={tp_buf}")
            printer.information("-" * 140)
            printer.information(f"  {'Zone':<12s} {'Baseline for Zone':>18s} {'Zone-Specific Run':>18s} {'Difference':>15s} {'Rel. Diff (%)':>15s} {'Work Units':>12s}")
            printer.information("-" * 140)

            sum_of_baseline_zone_objectives = 0.0

            for zone_file_data in zone_files:
                try:
                    zone = zone_file_data['zone']
                    original_zoi_value = zone_file_data['zoi_value']
                    baseline_data = baseline_file_data['zoi_data']
                    zone_zoi_i = zone_file_data['zoi_data']['sets']['zoi_i']

                    # Recalculate ZOI objective for baseline data with zone's ZOI definition
                    baseline_zone_objective = LEGOUtilities.evaluate_zoi_objective_from_data(
                        baseline_data, new_zoi_i=zone_zoi_i, line_filter="both"
                    )

                    difference = baseline_zone_objective - original_zoi_value
                    rel_diff_pct = (difference / baseline_zone_objective * 100) if baseline_zone_objective != 0 else 0.0
                    sum_of_baseline_zone_objectives += baseline_zone_objective
                    wu = zone_file_data.get('work_units')
                    wu_str = f"{wu:.2f}" if wu is not None else "N/A"
                    printer.information(f"  {zone:<12s} {baseline_zone_objective:>18.2f} {original_zoi_value:>18.2f} {difference:>15.2f} {rel_diff_pct:>14.1f}% {wu_str:>12s}")

                except Exception as e:
                    printer.error(f"  Failed to compare zone {zone}: {e}")
                    import traceback
                    traceback.print_exc()

            # Safety check: sum of baseline's zone objectives should equal baseline's overall objective
            printer.information("-" * 140)
            if baseline_total_obj is not None:
                printer.information(f"  {'SAFETY CHECK':<12s} {'Sum Baseline':>18s} {'Baseline Total':>18s} {'Difference':>15s} {'Rel. Diff (%)':>15s}")
                safety_difference = sum_of_baseline_zone_objectives - baseline_total_obj
                safety_rel_diff_pct = (safety_difference / baseline_total_obj * 100) if baseline_total_obj != 0 else 0.0
                tolerance_pct = 0.01  # 0.01% relative tolerance
                if abs(safety_rel_diff_pct) < tolerance_pct:
                    printer.success(f"  {'[PASSED]':<12s} {sum_of_baseline_zone_objectives:>18.2f} {baseline_total_obj:>18.2f} {safety_difference:>15.2f} {safety_rel_diff_pct:>14.1f}%")
                else:
                    printer.error(f"  {'[FAILED]':<12s} {sum_of_baseline_zone_objectives:>18.2f} {baseline_total_obj:>18.2f} {safety_difference:>15.2f} {safety_rel_diff_pct:>14.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ZOI (Zone of Interest) objectives from SQLite files", fromfile_prefix_chars='@')
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing SQLite files (default: current directory)")
    args = parser.parse_args()
    main(args.folder)
