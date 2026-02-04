import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import ast
import glob
import os
import sqlite3
import time
from collections import defaultdict

import pandas as pd

from InOutModule.printer import Printer
from LEGO import LEGOUtilities
from TechnicalRepresentation import is_uniform_representation, ZONE_LABELS, load_file_metadata, print_run_parameters, make_run_sort_key

printer = Printer.getInstance()


def _safe_literal_eval(s):
    """Parse a string as a Python literal, returning the string unchanged on failure."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


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
            linear_vars_info = [
                (var_name, _safe_literal_eval(idx_str))
                for var_name, idx_str in zip(df_terms['var_name'], df_terms['var_index'])
            ]
            linear_coefs = df_terms['coefficient'].tolist()
        except Exception as e:
            printer.error(f"Could not load objective_terms: {e}")
            raise

        objective_data = {
            'constant': constant,
            'linear_vars_info': linear_vars_info,  # Now includes (var_name, index) pairs
            'linear_coefs': linear_coefs,
        }

        # Load only variable tables actually referenced in the objective
        needed_var_names = set(info[0] for info in linear_vars_info)

        var_values = {}
        for var_name in needed_var_names:
            try:
                df = pd.read_sql_query(f'SELECT * FROM {var_name}', conn)
                if 'values' not in df.columns:
                    continue
                index_cols = [col for col in df.columns if col != 'values']

                if len(index_cols) == 0:
                    var_values[(var_name, var_name)] = df['values'].iloc[0]
                elif len(index_cols) == 1:
                    var_values.update({
                        (var_name, idx): val
                        for idx, val in zip(df[index_cols[0]], df['values'])
                    })
                else:
                    var_values.update({
                        (var_name, idx): val
                        for idx, val in zip(zip(*(df[col] for col in index_cols)), df['values'])
                    })
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
    regret_files = []  # Store regret files

    for sqlite_file in sqlite_files:
        printer.information(f"\nProcessing '{sqlite_file}'...")

        try:
            # Load metadata from SQLite
            meta = load_file_metadata(sqlite_file)
            print_run_parameters(meta)
            input_dir, limit_k = meta['input_dir'], meta['limit_k']
            dc_buffer, tp_buffer = meta['dc_buffer'], meta['tp_buffer']
            zone, demand, pmax = meta['zone'], meta['demand'], meta['pmax']
            is_regret_file = os.path.basename(sqlite_file).endswith('-regret.sqlite')

            # Load the ZOI data from SQLite
            start_time = time.time()
            printer.information(f"  Loading from SQLite database...")
            zoi_data = load_zoi_data_from_sqlite(sqlite_file)
            load_time = time.time() - start_time
            printer.information(f"  Loaded in {load_time:.2f} seconds")

            # Calculate objectives based on type
            zoi_value = None
            if is_uniform_representation(zone) or is_regret_file:
                printer.information(f"  {'Regret' if is_regret_file else 'Uniform'} run — skipping ZOI objective calculation")
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

            results.append((sqlite_file, input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, calc_total_obj, is_regret_file))

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

            # Separate into uniform, zone-specific, and regret
            if is_regret_file:
                regret_files.append(file_data)
            elif is_uniform_representation(zone):
                uniform_files.append(file_data)
            else:
                all_files_data.append(file_data)

        except Exception as e:
            printer.error(f"  Failed to process '{sqlite_file}': {e}")

    # Print summary
    if results:
        def sort_key(result):
            sqlite_file, input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, total_obj, is_regret = result
            return make_run_sort_key(input_dir, limit_k, demand, pmax, dc_buffer, tp_buffer, zone) + (1 if is_regret else 0,)

        sorted_results = sorted(results, key=sort_key)

        # Calculate the maximum filename length for proper alignment
        max_filename_len = max(len(sqlite_file) for sqlite_file, _, _, _, _, _, _, _, _, _, _ in sorted_results)
        # Ensure minimum width for readability
        filename_width = max(max_filename_len, len("Filename"))
        # Calculate total table width
        table_width = filename_width + 2 + 16 + 8 + 8 + 8 + 8 + 14 + 16

        printer.information("\n" + "=" * table_width)
        printer.information("Summary of Objective Values (grouped by parameters):")
        printer.information("=" * table_width)
        printer.information(f"  {'Filename':<{filename_width}s} {'LimitK':>16s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Demand':>8s} {'PMax':>8s} {'ZOI Objective':>14s} {'Total Objective':>16s}")
        printer.information("-" * table_width)

        # Track previous group to insert separators
        prev_group = None

        for sqlite_file, input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, zoi_value, total_obj, _ in sorted_results:
            # Check if we're starting a new group (input_dir, limit_k, demand, pmax)
            current_group = (input_dir, limit_k, demand, pmax)
            if prev_group is not None and current_group != prev_group:
                # Insert separator line between groups
                printer.information("-" * table_width)
            prev_group = current_group

            # Show '-' for dcBuffer and tpBuffer for uniform representations
            is_uniform = is_uniform_representation(zone)
            dc_str = "-" if is_uniform else (str(dc_buffer) if dc_buffer is not None else "N/A")
            tp_str = "-" if is_uniform else (str(tp_buffer) if tp_buffer is not None else "N/A")
            demand_str = f"{demand:.1f}" if demand is not None else "N/A"
            pmax_str = f"{pmax:.1f}" if pmax is not None else "N/A"
            # Show N/A for ZOI objective if it's a uniform representation
            zoi_str = "N/A" if (is_uniform or zoi_value is None) else f"{zoi_value:.2f}"
            # Show total objective if available
            total_str = f"{total_obj:.2f}" if total_obj is not None else "N/A"
            limit_k_str = limit_k if limit_k else "N/A"
            printer.information(f"  {sqlite_file:<{filename_width}s} {limit_k_str:>16s} {dc_str:>8s} {tp_str:>8s} {demand_str:>8s} {pmax_str:>8s} {zoi_str:>14s} {total_str:>16s}")

    # Group uniform files by all grouping parameters for easier access
    uniform_groups = {}
    for uniform_file in uniform_files:
        group_key = (uniform_file['input_dir'], uniform_file['limit_k'],
                     uniform_file['demand'], uniform_file['pmax'])
        if group_key not in uniform_groups:
            uniform_groups[group_key] = []
        uniform_groups[group_key].append(uniform_file)

    # Group regret files by (input_dir, limit_k, demand, pmax)
    regret_groups = {}
    for regret_file in regret_files:
        group_key = (regret_file['input_dir'], regret_file['limit_k'],
                     regret_file['demand'], regret_file['pmax'])
        if group_key not in regret_groups:
            regret_groups[group_key] = []
        regret_groups[group_key].append(regret_file)

    # Lookup maps for non-regret source work units
    uniform_source_map = {}  # (group_key, zone) -> file_data
    for gk, files in uniform_groups.items():
        for f in files:
            uniform_source_map[(gk, f['zone'])] = f

    zone_source_map = {}  # (group_key, zone, dc_buf, tp_buf) -> file_data
    zone_files_by_group = defaultdict(lambda: defaultdict(list))  # group_key -> buf_key -> [file_data]
    for fd in all_files_data:
        gk = (fd['input_dir'], fd['limit_k'], fd['demand'], fd['pmax'])
        zone_source_map[(gk, fd['zone'], fd['dc_buffer'], fd['tp_buffer'])] = fd
        zone_files_by_group[gk][(fd['dc_buffer'], fd['tp_buffer'])].append(fd)

    # Process each comparison group
    all_group_keys = sorted(
        set(uniform_groups.keys()) | set(regret_groups.keys()) | set(zone_files_by_group.keys()),
        key=lambda k: (k[0] or '', k[1] or '', k[2], k[3])
    )

    for group_key in all_group_keys:
        input_dir, limit_k, demand, pmax = group_key

        group_desc_parts = []
        if input_dir:
            group_desc_parts.append(f"input={input_dir}")
        if limit_k:
            group_desc_parts.append(f"limitK={limit_k}")
        group_desc_parts.append(f"demand={demand}")
        group_desc_parts.append(f"pmax={pmax}")
        group_desc = ", ".join(group_desc_parts)

        printer.information("\n" + "=" * 155)
        printer.information(f"COMPARISON GROUP: {group_desc}")
        printer.information("=" * 155)

        # DC baseline (prefer zoiDC, fall back to zoiNone)
        uniforms = uniform_groups.get(group_key, [])
        dc_baseline = next((f for f in uniforms if f['zone'] == 'DC'), None)
        if dc_baseline is None:
            dc_baseline = next((f for f in uniforms if (f['zone'] is None or f['zone'] == "None")), None)

        regrets = regret_groups.get(group_key, [])
        if dc_baseline is None or not regrets:
            continue

        dc_total_obj = dc_baseline['total_obj']
        dc_wu = dc_baseline.get('work_units')

        # Build rows: DC baseline first, then uniform regret (SN/TP), then zone regret sorted by buffers
        rows = []
        rows.append({
            'source': 'DC',
            'is_uniform': True,
            'dc_buf': None, 'tp_buf': None,
            'total_obj': dc_total_obj,
            'abs_regret': 0.0, 'rel_regret': 0.0,
            'wu': dc_wu, 'wu_rel': None,
            'sort': (-2, 0, 0, ''),
        })

        for rf in regrets:
            zone = rf['zone']
            total_obj = rf.get('total_obj')
            if total_obj is None:
                continue

            is_uniform_source = is_uniform_representation(zone)

            # Work units come from the non-regret source run
            if is_uniform_source:
                source = uniform_source_map.get((group_key, zone))
            else:
                source = zone_source_map.get((group_key, zone, rf.get('dc_buffer'), rf.get('tp_buffer')))
            source_wu = source.get('work_units') if source else None

            abs_regret = total_obj - dc_total_obj
            rel_regret = (abs_regret / dc_total_obj * 100) if dc_total_obj != 0 else 0.0
            wu_rel = ((source_wu - dc_wu) / dc_wu * 100) if (source_wu is not None and dc_wu is not None and dc_wu != 0) else None

            rows.append({
                'source': zone if is_uniform_source else f"zoi({zone})",
                'is_uniform': is_uniform_source,
                'dc_buf': rf.get('dc_buffer'), 'tp_buf': rf.get('tp_buffer'),
                'total_obj': total_obj,
                'abs_regret': abs_regret, 'rel_regret': rel_regret,
                'wu': source_wu, 'wu_rel': wu_rel,
                'sort': (-1, 0, 0, zone) if is_uniform_source else (0, rf.get('dc_buffer') or 0, rf.get('tp_buffer') or 0, zone),
            })

        rows.sort(key=lambda r: r['sort'])

        # Print merged regret comparison table
        printer.information(f"\nRegret Comparisons (baseline: DC, total obj = {dc_total_obj:.2f})")
        printer.information("-" * 155)
        printer.information(f"  {'Source':<12s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Total Objective':>18s} {'Abs. Regret':>15s} {'Rel. Regret (%)':>15s} {'Work Units':>12s} {'WU Rel. (%)':>12s}")
        printer.information("-" * 155)

        for row in rows:
            dc_str = "-" if row['is_uniform'] else (str(row['dc_buf']) if row['dc_buf'] is not None else "N/A")
            tp_str = "-" if row['is_uniform'] else (str(row['tp_buf']) if row['tp_buf'] is not None else "N/A")
            wu_str = f"{row['wu']:.2f}" if row['wu'] is not None else "N/A"
            wu_rel_str = f"{row['wu_rel']:.1f}%" if row['wu_rel'] is not None else "-"
            printer.information(f"  {row['source']:<12s} {dc_str:>8s} {tp_str:>8s} {row['total_obj']:>18.2f} {row['abs_regret']:>15.2f} {row['rel_regret']:>14.1f}% {wu_str:>12s} {wu_rel_str:>12s}")

        # Safety check: sum of baseline's per-zone ZOI objectives == baseline total
        buf_configs = zone_files_by_group.get(group_key, {})
        if buf_configs and dc_baseline.get('zoi_data'):
            first_buf_key = next(iter(sorted(buf_configs.keys())))
            zone_files_for_check = buf_configs[first_buf_key]
            baseline_data = dc_baseline['zoi_data']
            sum_baseline_zones = 0.0
            for zf in sorted(zone_files_for_check, key=lambda z: z['zone']):
                zone_zoi_i = zf['zoi_data']['sets']['zoi_i']
                sum_baseline_zones += LEGOUtilities.evaluate_zoi_objective_from_data(
                    baseline_data, new_zoi_i=zone_zoi_i, line_filter="both"
                )

            printer.information("-" * 155)
            printer.information(f"  {'SAFETY CHECK':<12s} {'Sum Baseline':>18s} {'Baseline Total':>18s} {'Difference':>15s} {'Rel. Diff (%)':>15s}")
            safety_diff = sum_baseline_zones - dc_total_obj
            safety_rel = (safety_diff / dc_total_obj * 100) if dc_total_obj != 0 else 0.0
            if abs(safety_rel) < 0.01:
                printer.success(f"  {'[PASSED]':<12s} {sum_baseline_zones:>18.2f} {dc_total_obj:>18.2f} {safety_diff:>15.2f} {safety_rel:>14.1f}%")
            else:
                printer.error(f"  {'[FAILED]':<12s} {sum_baseline_zones:>18.2f} {dc_total_obj:>18.2f} {safety_diff:>15.2f} {safety_rel:>14.1f}%")
        printer.information("-" * 155)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ZOI (Zone of Interest) objectives from SQLite files", fromfile_prefix_chars='@')
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing SQLite files (default: current directory)")
    args = parser.parse_args()
    main(args.folder)
