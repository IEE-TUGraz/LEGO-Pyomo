import argparse
import glob
import os
import re
import sqlite3
import time

import cloudpickle
import pandas as pd
import pyomo.environ as pe

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

        # Create ZOI data structure
        zoi_data = {
            'var_values': var_values,
            'sets': sets_data,
            'objective': objective_data,
            'total_objective': total_objective,
        }

        return zoi_data

    finally:
        conn.close()


def extract_parameters(filename):
    """
    Extract dcBuffer, tpBuffer, zone, demand, and pmax values from filename.
    Uses default values if parameters are missing from the filename.

    Returns:
        tuple: (dc_buffer, tp_buffer, zone, demand, pmax, params_dict)
               where params_dict contains info about which values are defaults
    """
    params_dict = {}

    # Parse filename - all parameters except zone are optional
    # Remove file extension first
    filename_no_ext = filename.replace('.sqlite', '').replace('.pkl', '')
    match = re.search(r'-zoi(?P<zone>[^-]+)(?:.*?-dcBuffer(?P<dc>\d+))?(?:.*?-tpBuffer(?P<tp>\d+))?(?:.*?-demand(?P<demand>\d+(?:\.\d+)?))?(?:.*?-pmax(?P<pmax>\d+(?:\.\d+)?))?', filename_no_ext)

    if match:
        # Zone is always required in filename
        zone = match.group('zone')
        params_dict['zone'] = {'value': zone, 'is_default': False}

        # dcBuffer and tpBuffer are optional (defaults to 1)
        if match.group('dc'):
            dc_buffer = int(match.group('dc'))
            params_dict['dc_buffer'] = {'value': dc_buffer, 'is_default': False}
        else:
            dc_buffer = DEFAULT_DC_BUFFER
            params_dict['dc_buffer'] = {'value': dc_buffer, 'is_default': True}

        if match.group('tp'):
            tp_buffer = int(match.group('tp'))
            params_dict['tp_buffer'] = {'value': tp_buffer, 'is_default': False}
        else:
            tp_buffer = DEFAULT_TP_BUFFER
            params_dict['tp_buffer'] = {'value': tp_buffer, 'is_default': True}

        # demand and pmax are optional (defaults to 1.0)
        if match.group('demand'):
            demand = float(match.group('demand'))
            params_dict['scale_demand'] = {'value': demand, 'is_default': False}
        else:
            demand = DEFAULT_SCALE_DEMAND
            params_dict['scale_demand'] = {'value': demand, 'is_default': True}

        if match.group('pmax'):
            pmax = float(match.group('pmax'))
            params_dict['scale_pmax'] = {'value': pmax, 'is_default': False}
        else:
            pmax = DEFAULT_SCALE_PMAX
            params_dict['scale_pmax'] = {'value': pmax, 'is_default': True}

        return dc_buffer, tp_buffer, zone, demand, pmax, params_dict

    # No match - return all defaults
    return (None, None, None, DEFAULT_SCALE_DEMAND, DEFAULT_SCALE_PMAX, {
        'zone': {'value': None, 'is_default': True},
        'dc_buffer': {'value': None, 'is_default': True},
        'tp_buffer': {'value': None, 'is_default': True},
        'scale_demand': {'value': DEFAULT_SCALE_DEMAND, 'is_default': True},
        'scale_pmax': {'value': DEFAULT_SCALE_PMAX, 'is_default': True}
    })


def print_run_parameters(params_dict):
    """
    Print run parameters extracted from filename.
    :param params_dict: Dictionary of parameters with default flags
    """
    if not params_dict:
        printer.warning("  Could not extract parameters from filename")
        return

    # Check if zone is a uniform representation (None, TP, SN)
    zone_info = params_dict.get('zone', {})
    zone = zone_info.get('value')
    is_uniform_repr = zone in ['TP', 'SN'] or zone is None or zone == "None"

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
    # Find all data files (both .sqlite and .pkl) in the specified folder
    sqlite_pattern = os.path.join(folder, "*.sqlite")
    pkl_pattern = os.path.join(folder, "*.pkl")

    sqlite_files = set(glob.glob(sqlite_pattern))
    pkl_files = set(glob.glob(pkl_pattern))

    # Create a unified list of base filenames (without extension)
    all_base_files = set()
    for f in sqlite_files:
        all_base_files.add(f.replace('.sqlite', ''))
    for f in pkl_files:
        all_base_files.add(f.replace('.pkl', ''))

    if not all_base_files:
        printer.warning(f"No data files (.sqlite or .pkl) found in '{folder}'")
        return

    printer.information(f"Found {len(all_base_files)} data file(s) in '{folder}'")

    # Process each file and group by base identifier
    results = []
    all_files_data = []  # Store all file data for flexible grouping
    uniform_files = []  # Store uniform representation files (None, TP, SN)

    for base_file in sorted(all_base_files):
        # Prefer SQLite over pickle
        sqlite_file = base_file + '.sqlite'
        pkl_file = base_file + '.pkl'

        if os.path.exists(sqlite_file):
            file_to_process = sqlite_file
            use_sqlite = True
        elif os.path.exists(pkl_file):
            file_to_process = pkl_file
            use_sqlite = False
        else:
            continue

        printer.information(f"\nProcessing '{file_to_process}'...")

        try:
            # Extract parameters from filename
            dc_buffer, tp_buffer, zone, demand, pmax, params_dict = extract_parameters(file_to_process)
            print_run_parameters(params_dict)

            # Load the ZOI data
            start_time = time.time()
            if use_sqlite:
                printer.information(f"  Loading from SQLite database...")
                try:
                    zoi_data = load_zoi_data_from_sqlite(sqlite_file)
                    load_time = time.time() - start_time
                    printer.information(f"  Loaded in {load_time:.2f} seconds")
                except Exception as e:
                    # SQLite loading failed (probably missing objective decomposition tables)
                    # Fall back to pickle file if available
                    if os.path.exists(pkl_file):
                        printer.warning(f"  SQLite loading failed ({e}), falling back to pickle file...")
                        start_time = time.time()
                        with open(pkl_file, mode='rb') as file:
                            zoi_data = cloudpickle.load(file)
                        load_time = time.time() - start_time
                        printer.information(f"  Loaded from pickle in {load_time:.2f} seconds")
                    else:
                        raise
            else:
                printer.information(f"  Loading from pickle file...")
                with open(pkl_file, mode='rb') as file:
                    zoi_data = cloudpickle.load(file)
                load_time = time.time() - start_time
                printer.information(f"  Loaded in {load_time:.2f} seconds")

            # Check if this is a uniform representation
            is_uniform_repr = zone in ['TP', 'SN'] or zone is None or zone == "None"

            # Calculate objectives based on type
            zoi_value = None
            if is_uniform_repr:
                # Uniform representation: skip ZOI objective, only calculate total objective
                printer.information(f"  Uniform representation detected - skipping ZOI objective calculation")
            else:
                # Zone-specific: calculate ZOI objective (always calculate freshly, never use pre-calculated)
                calc_start_time = time.time()
                if isinstance(zoi_data, dict) and 'sets' in zoi_data:
                    # Dict format (from SQLite or new pickle format): use lightweight recalculation from data
                    zoi_i = zoi_data['sets']['zoi_i']
                    zoi_value = LEGOUtilities.evaluate_zoi_objective_from_data(zoi_data, new_zoi_i=zoi_i, line_filter="both")
                else:
                    # Old format: full model - calculate ZOI objective
                    _, zoi_value = LEGOUtilities.evaluate_zoi_objective(zoi_data, line_filter="both")
                calc_time = time.time() - calc_start_time
                printer.information(f"  ZOI objective calculated in {calc_time:.2f} seconds")
                printer.success(f"  ZOI Objective: {zoi_value:.2f}")

            # Calculate total objective from data (for all cases)
            calc_total_obj = None
            stored_total_obj = None
            try:
                if isinstance(zoi_data, dict):
                    # Get stored objective if available
                    if 'total_objective' in zoi_data:
                        stored_total_obj = zoi_data['total_objective']

                    # Always calculate total objective from components
                    if 'objective' in zoi_data and 'var_values' in zoi_data:
                        obj_data = zoi_data['objective']
                        var_data = zoi_data['var_values']
                        calc_total_obj = obj_data['constant']

                        # Check if we have new format (var_name, index) or old format (index only)
                        if 'linear_vars_info' in obj_data:
                            # New format: use (var_name, index) composite keys
                            for (var_name, idx), coef in zip(obj_data['linear_vars_info'], obj_data['linear_coefs']):
                                key = (var_name, idx)
                                if key in var_data:
                                    calc_total_obj += coef * var_data[key]
                        elif 'linear_vars_indices' in obj_data:
                            # Old format: use index only (backward compatibility)
                            for idx, coef in zip(obj_data['linear_vars_indices'], obj_data['linear_coefs']):
                                if idx in var_data:
                                    calc_total_obj += coef * var_data[idx]
                else:
                    # Old format: get objective from model
                    obj_list = list(zoi_data.component_objects(ctype=pe.Objective, active=True))
                    if obj_list:
                        calc_total_obj = pe.value(obj_list[0])
                        stored_total_obj = calc_total_obj  # In old format, they're the same
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

            results.append((file_to_process, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, calc_total_obj))

            # Store file data
            file_data = {
                'pkl_file': file_to_process,  # Store the actual file used
                'dc_buffer': dc_buffer,
                'tp_buffer': tp_buffer,
                'zone': zone,
                'demand': demand,
                'pmax': pmax,
                'zoi_value': zoi_value,
                'zoi_data': zoi_data,
                'total_obj': calc_total_obj  # Use calculated total objective
            }

            # Separate uniform representations from zone-specific runs
            if zone in ['TP', 'SN'] or zone is None or zone == "None":
                uniform_files.append(file_data)
            else:
                all_files_data.append(file_data)

        except Exception as e:
            printer.error(f"  Failed to process '{file_to_process}': {e}")

    # Group zone-specific files by their parameters (including dcBuffer/tpBuffer)
    # Key: (case_study_base, dc_buffer, tp_buffer, demand, pmax)
    file_groups = {}

    for file_data in all_files_data:
        # Extract case study base name (remove path and specific parameters from filename)
        # Group by dcBuffer, tpBuffer, demand, pmax
        group_key = (file_data['dc_buffer'], file_data['tp_buffer'],
                     file_data['demand'], file_data['pmax'])

        if group_key not in file_groups:
            file_groups[group_key] = {
                'zone_files': [],
                'uniform_files': []
            }

        file_groups[group_key]['zone_files'].append(file_data)

    # Match uniform files to groups (ignoring dcBuffer/tpBuffer)
    for uniform_file in uniform_files:
        # Match based on demand and pmax only
        for group_key, group_data in file_groups.items():
            dc_buf, tp_buf, demand, pmax = group_key
            if uniform_file['demand'] == demand and uniform_file['pmax'] == pmax:
                group_data['uniform_files'].append(uniform_file)

    # Print summary
    if results:
        # Sort results by groups: demand, pmax, then by dcBuffer/tpBuffer (with uniform reps first), then by zone
        def sort_key(result):
            pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, total_obj = result
            # For uniform representations (None, TP, SN), put them first in each demand/pmax group
            is_uniform = zone in ['TP', 'SN'] or zone is None or zone == "None"
            # Sort order: demand, pmax, uniform flag, dcBuffer, tpBuffer, zone
            zone_str = str(zone) if zone is not None else ""
            dc_sort = -1 if is_uniform else (dc_buffer if dc_buffer is not None else 999)
            tp_sort = -1 if is_uniform else (tp_buffer if tp_buffer is not None else 999)
            return (demand, pmax, 0 if is_uniform else 1, dc_sort, tp_sort, zone_str)

        sorted_results = sorted(results, key=sort_key)

        # Calculate the maximum filename length for proper alignment
        max_filename_len = max(len(pkl_file) for pkl_file, _, _, _, _, _, _, _ in sorted_results)
        # Ensure minimum width for readability
        filename_width = max(max_filename_len, len("Filename"))
        # Calculate total table width (added 16 for Total Objective column)
        table_width = filename_width + 2 + 8 + 8 + 8 + 8 + 14 + 16

        printer.information("\n" + "=" * table_width)
        printer.information("Summary of Objective Values (grouped by parameters):")
        printer.information("=" * table_width)
        printer.information(f"  {'Filename':<{filename_width}s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Demand':>8s} {'PMax':>8s} {'ZOI Objective':>14s} {'Total Objective':>16s}")
        printer.information("-" * table_width)

        # Track previous group to insert separators
        prev_group = None

        for pkl_file, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, total_obj in sorted_results:
            # Check if we're starting a new group (demand, pmax)
            current_group = (demand, pmax)
            if prev_group is not None and current_group != prev_group:
                # Insert separator line between groups
                printer.information("-" * table_width)
            prev_group = current_group

            # Show '-' for dcBuffer and tpBuffer for uniform representations
            is_uniform = zone in ['TP', 'SN'] or zone is None or zone == "None"
            dc_str = "-" if is_uniform else (str(dc_buffer) if dc_buffer is not None else "N/A")
            tp_str = "-" if is_uniform else (str(tp_buffer) if tp_buffer is not None else "N/A")
            demand_str = f"{demand:.1f}" if demand is not None else "N/A"
            pmax_str = f"{pmax:.1f}" if pmax is not None else "N/A"
            # Show N/A for ZOI objective if it's a uniform representation
            zoi_str = "N/A" if (is_uniform or zoi_value is None) else f"{zoi_value:.2f}"
            # Show total objective if available
            total_str = f"{total_obj:.2f}" if total_obj is not None else "N/A"
            printer.information(f"  {pkl_file:<{filename_width}s} {dc_str:>8s} {tp_str:>8s} {demand_str:>8s} {pmax_str:>8s} {zoi_str:>14s} {total_str:>16s}")

    # Group uniform files by demand and pmax for easier access
    uniform_groups = {}
    for uniform_file in uniform_files:
        group_key = (uniform_file['demand'], uniform_file['pmax'])
        if group_key not in uniform_groups:
            uniform_groups[group_key] = []
        uniform_groups[group_key].append(uniform_file)

    # Organize groups by (demand, pmax) to show zone-specific and uniform representations together
    demand_pmax_groups = {}
    for (dc_buf, tp_buf, demand, pmax), group_data in file_groups.items():
        key = (demand, pmax)
        if key not in demand_pmax_groups:
            demand_pmax_groups[key] = []
        demand_pmax_groups[key].append((dc_buf, tp_buf, group_data))

    # Process each (demand, pmax) group
    for (demand, pmax), configs in sorted(demand_pmax_groups.items()):
        # Get gold standard file for this group (from first config)
        first_config = configs[0] if configs else None
        if first_config:
            _, _, group_data = first_config
            zoi_none_for_header = next(
                (f for f in group_data['uniform_files']
                 if (f['zone'] is None or f['zone'] == "None")),
                None
            )
        else:
            zoi_none_for_header = None

        # Extract case study name from filename
        case_study_name = "Unknown"
        if zoi_none_for_header:
            pkl_file = zoi_none_for_header['pkl_file']
            # Extract from filename like "TR-datadata_NREL-118-zoi..."
            # Use non-greedy match to capture everything between "TR-data" and "-zoi"
            match = re.search(r'TR-data(.+?)-zoi', pkl_file)
            if match:
                case_study_name = match.group(1)

        printer.information("\n" + "=" * 140)
        printer.information(f"COMPARISON GROUP: {case_study_name}, demand={demand}, pmax={pmax}")
        if zoi_none_for_header:
            printer.information(f"Gold Standard: DC-OPF (zoi=None) - {zoi_none_for_header['pkl_file']}")
        printer.information("=" * 140)

        # Show zone-specific comparisons for each (dc_buffer, tp_buffer) configuration
        for dc_buf, tp_buf, group_data in sorted(configs):
            # Find the zoiNone model in uniform files for this group
            zoi_none_file_data = next(
                (f for f in group_data['uniform_files']
                 if (f['zone'] is None or f['zone'] == "None")),
                None
            )

            if zoi_none_file_data is None:
                continue

            zone_files = group_data['zone_files']
            if not zone_files:
                continue

            # Get total objective from zoiNone pkl file for safety check
            zoi_none_total_obj = zoi_none_file_data.get('total_obj')

            printer.information(f"Zone-Specific Comparisons")
            printer.information(f"Parameters: dcBuffer={dc_buf}, tpBuffer={tp_buf}, demand={demand}, pmax={pmax}")
            printer.information("-" * 140)
            printer.information(f"  {'Zone':<20s} {'Gold Std for Zone':>30s} {'Zone-Specific Run':>30s} {'Difference':>15s} {'Rel. Diff (%)':>15s}")
            printer.information("-" * 140)

            sum_of_gold_standard_zone_objectives = 0.0

            for zone_file_data in zone_files:
                try:
                    zone = zone_file_data['zone']
                    original_zoi_value = zone_file_data['zoi_value']
                    zone_data = zone_file_data['zoi_data']
                    zoi_none_data = zoi_none_file_data['zoi_data']

                    # Check if data is in new format (dict) or old format (model)
                    if isinstance(zone_data, dict) and 'sets' in zone_data:
                        # New format: extract zoi_i from zone_data
                        zone_zoi_i = zone_data['sets']['zoi_i']
                    else:
                        # Old format: extract from model
                        zone_zoi_i = list(zone_data.zoi_i)

                    # Recalculate ZOI objective for zoiNone (gold standard) data with zone's ZOI definition
                    if isinstance(zoi_none_data, dict) and 'zoi_objective_value' in zoi_none_data:
                        # New format: use lightweight recalculation
                        gold_standard_zone_objective = LEGOUtilities.evaluate_zoi_objective_from_data(
                            zoi_none_data, new_zoi_i=zone_zoi_i, line_filter="both"
                        )
                    else:
                        # Old format: modify model and recalculate
                        zoi_none_data.zoi_i.clear()
                        zoi_none_data.zoi_i.construct()
                        for bus in zone_zoi_i:
                            zoi_none_data.zoi_i.add(bus)
                        _, gold_standard_zone_objective = LEGOUtilities.evaluate_zoi_objective(zoi_none_data, line_filter="both")

                    difference = gold_standard_zone_objective - original_zoi_value
                    rel_diff_pct = (difference / gold_standard_zone_objective * 100) if gold_standard_zone_objective != 0 else 0.0
                    sum_of_gold_standard_zone_objectives += gold_standard_zone_objective
                    printer.information(f"  {zone:<20s} {gold_standard_zone_objective:>30.2f} {original_zoi_value:>30.2f} {difference:>15.2f} {rel_diff_pct:>14.1f}%")

                except Exception as e:
                    printer.error(f"  Failed to compare zone {zone}: {e}")
                    import traceback
                    traceback.print_exc()

            # Safety check: sum of gold standard's zone objectives should equal gold standard's overall objective
            printer.information("-" * 140)
            if zoi_none_total_obj is not None:
                printer.information(f"  {'SAFETY CHECK':<20s} {'Sum of Gold Std Zones':>30s} {'Gold Std Total Obj':>30s} {'Difference':>15s} {'Rel. Diff (%)':>15s}")
                safety_difference = sum_of_gold_standard_zone_objectives - zoi_none_total_obj
                safety_rel_diff_pct = (safety_difference / zoi_none_total_obj * 100) if zoi_none_total_obj != 0 else 0.0
                tolerance_pct = 0.01  # 0.01% relative tolerance
                if abs(safety_rel_diff_pct) < tolerance_pct:
                    printer.success(f"  {'[PASSED]':<20s} {sum_of_gold_standard_zone_objectives:>30.2f} {zoi_none_total_obj:>30.2f} {safety_difference:>15.2f} {safety_rel_diff_pct:>14.1f}%")
                else:
                    printer.error(f"  {'[FAILED]':<20s} {sum_of_gold_standard_zone_objectives:>30.2f} {zoi_none_total_obj:>30.2f} {safety_difference:>15.2f} {safety_rel_diff_pct:>14.1f}%")
            else:
                printer.warning(f"  {'SAFETY CHECK':<20s} Could not calculate total objective from pkl file - skipping safety check")
                printer.warning(f"  {'N/A':<20s} {sum_of_gold_standard_zone_objectives:>30.2f} {'N/A':>30s}")

        # Show uniform technical representation comparisons for this (demand, pmax) group
        if (demand, pmax) in uniform_groups:
            group_uniform_files = uniform_groups[(demand, pmax)]

            # Find DC-OPF baseline (zoi=None)
            dc_opf_file = next(
                (f for f in group_uniform_files if (f['zone'] is None or f['zone'] == "None")),
                None
            )

            if dc_opf_file is not None:
                # Get total objective from DC-OPF pkl file
                dc_opf_total_obj = dc_opf_file.get('total_obj')

                # Find TP and SN files
                tp_sn_files = [f for f in group_uniform_files if f['zone'] in ['TP', 'SN']]

                if dc_opf_total_obj is not None and tp_sn_files:
                    printer.information("\n" + "-" * 140)
                    printer.information(f"Uniform Technical Representation Comparisons")
                    printer.information(f"Parameters: demand={demand}, pmax={pmax}")
                    printer.information("-" * 140)
                    printer.information(f"  {'Type':<10s} {'Description':<50s} {'Total Objective':>20s} {'Abs. Diff':>15s} {'Rel. Diff (%)':>15s}")
                    printer.information("-" * 140)

                    # Collect entries and sort: DC-OPF, TP, SN
                    entries = []

                    # Add DC-OPF as first entry
                    entries.append(('DC-OPF', 'DC Optimal Power Flow (all lines as DC-OPF)', dc_opf_total_obj, 0.0, 0.0))

                    # Add TP and SN
                    for uniform_file in tp_sn_files:
                        zone = uniform_file['zone']

                        if zone == 'TP':
                            desc = "Transport Model (all lines as TP)"
                            sort_order = 1
                        elif zone == 'SN':
                            desc = "Single Node / Copper Plate (all lines as SN)"
                            sort_order = 2
                        else:
                            desc = f"Uniform {zone}"
                            sort_order = 3

                        # Get total objective from pkl file
                        total_obj = uniform_file.get('total_obj')

                        if total_obj is not None:
                            abs_diff = total_obj - dc_opf_total_obj
                            rel_diff_pct = (abs_diff / dc_opf_total_obj * 100) if dc_opf_total_obj != 0 else 0.0
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ZOI (Zone of Interest) objectives from pickle files", fromfile_prefix_chars='@')
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing pickle files (default: current directory)")
    args = parser.parse_args()
    main(args.folder)
