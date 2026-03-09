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
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from InOutModule.printer import Printer
from TechnicalRepresentation import is_uniform_representation, is_utilization_run, load_file_metadata, make_run_sort_key

printer = Printer.getInstance()


def _safe_literal_eval(s):
    """Parse a string as a Python literal, returning the string unchanged on failure."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _fast_parse_index(s):
    """Fast tuple/scalar parser for objective term indices.

    Handles the common formats stored by SQLiteWriter:
    - "(val1, val2, ...)" -> tuple
    - "scalar" -> scalar string
    - "123" or "1.5" -> number

    Much faster than ast.literal_eval for the simple cases we encounter.
    """
    if not isinstance(s, str):
        return s
    stripped = s.strip()
    if stripped.startswith('(') and stripped.endswith(')'):
        # Tuple: strip parens, split by comma, strip quotes from each element
        inner = stripped[1:-1]
        if not inner:
            return ()
        parts = []
        for part in inner.split(','):
            part = part.strip()
            if not part:
                continue
            # Strip surrounding quotes
            if (part.startswith("'") and part.endswith("'")) or (part.startswith('"') and part.endswith('"')):
                parts.append(part[1:-1])
            else:
                # Try numeric
                try:
                    parts.append(int(part))
                except ValueError:
                    try:
                        parts.append(float(part))
                    except ValueError:
                        parts.append(part)
        return tuple(parts) if len(parts) != 1 else parts[0]
    # Scalar
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return stripped


def load_zoi_data_from_sqlite(sqlite_file, conn=None):
    """
    Load ZOI objective data from SQLite file.

    :param sqlite_file: Path to SQLite database file
    :param conn: Optional existing SQLite connection to reuse
    :return: Dictionary with ZOI data in the same format as extract_zoi_objective_data
    """
    own_conn = conn is None
    if own_conn:
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
                printer.warning(f"'{os.path.basename(sqlite_file)}': Could not load set {set_name}: {e}")
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
            printer.error(f"'{os.path.basename(sqlite_file)}': Could not load objective_constant: {e}")
            raise

        try:
            df_terms = pd.read_sql_query('SELECT * FROM objective_terms', conn)
            linear_vars_info = [
                (var_name, _fast_parse_index(idx_str))
                for var_name, idx_str in zip(df_terms['var_name'], df_terms['var_index'])
            ]
            linear_coefs = df_terms['coefficient'].tolist()
        except Exception as e:
            printer.error(f"'{os.path.basename(sqlite_file)}': Could not load objective_terms: {e}")
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
                printer.warning(f"'{os.path.basename(sqlite_file)}': Could not load variable {var_name}: {e}")

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
        if own_conn:
            conn.close()


def load_ens_pns_from_sqlite(sqlite_file, conn=None):
    """
    Compute weighted total ENS and PNS from SQLite file.

    Total ENS = sum over (rp, k, i) of vEPS[rp,k,i] * pWeight_rp[rp] * pWeight_k[k]
    Total PNS = sum over (rp, k, i) of vPNS[rp,k,i] * pWeight_rp[rp] * pWeight_k[k]

    :param conn: Optional existing SQLite connection to reuse
    :return: (ens, pns) tuple of floats
    """
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(sqlite_file)
    try:
        df_wk = pd.read_sql_query('SELECT * FROM pWeight_k', conn)
        df_wrp = pd.read_sql_query('SELECT * FROM pWeight_rp', conn)
        wk = dict(zip(df_wk.iloc[:, 0], df_wk['values']))
        wrp = dict(zip(df_wrp.iloc[:, 0], df_wrp['values']))

        totals = {}
        for var_name in ['vEPS', 'vPNS']:
            df = pd.read_sql_query(f'SELECT * FROM {var_name}', conn)
            df['_w'] = df['rp'].map(wrp) * df['k'].map(wk)
            totals[var_name] = (df['values'] * df['_w']).sum()
        return totals['vEPS'], totals['vPNS']
    finally:
        if own_conn:
            conn.close()


def load_bus_zones_from_sqlite(sqlite_file, conn=None):
    """
    Load i_zone set table from SQLite file. The set contains (i, zone) tuples,
    stored as columns '0' (bus) and '1' (zone).

    :param sqlite_file: Path to SQLite database file
    :param conn: Optional existing SQLite connection to reuse
    :return: dict mapping bus name to zone name {bus: zone}
    """
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(sqlite_file)
    try:
        df = pd.read_sql_query('SELECT * FROM i_zone', conn)
        # Drop the 'index' column added by pandas
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        # Set stored as columns '0' and '1'
        if '0' in df.columns and '1' in df.columns:
            return dict(zip(df['0'], df['1']))
        else:
            # Fallback: first two columns
            return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    except Exception:
        return {}
    finally:
        if own_conn:
            conn.close()


def load_ens_pns_per_zone_from_sqlite(sqlite_file, bus_zones, conn=None):
    """
    Compute weighted ENS and PNS per zone and total from SQLite file.

    :param sqlite_file: Path to SQLite database file
    :param bus_zones: dict mapping bus name to zone name
    :param conn: Optional existing SQLite connection to reuse
    :return: dict {'_total': (ens, pns), 'Zone1': (ens, pns), ...}
    """
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(sqlite_file)
    try:
        df_wk = pd.read_sql_query('SELECT * FROM pWeight_k', conn)
        df_wrp = pd.read_sql_query('SELECT * FROM pWeight_rp', conn)
        wk = dict(zip(df_wk.iloc[:, 0], df_wk['values']))
        wrp = dict(zip(df_wrp.iloc[:, 0], df_wrp['values']))

        result = {}
        for var_name in ['vEPS', 'vPNS']:
            df = pd.read_sql_query(f'SELECT * FROM {var_name}', conn)
            df['_w'] = df['rp'].map(wrp) * df['k'].map(wk)
            df['_weighted'] = df['values'] * df['_w']
            df['_zone'] = df['i'].map(bus_zones)

            # Total
            result[('_total', var_name)] = df['_weighted'].sum()

            # Per zone
            for zone, grp in df.groupby('_zone'):
                result[(zone, var_name)] = grp['_weighted'].sum()

        # Build output dict
        zones = sorted(set(bus_zones.values()))
        output = {
            '_total': (result.get(('_total', 'vEPS'), 0), result.get(('_total', 'vPNS'), 0))
        }
        for zone in zones:
            output[zone] = (result.get((zone, 'vEPS'), 0), result.get((zone, 'vPNS'), 0))
        return output
    finally:
        if own_conn:
            conn.close()


def load_all_file_data_from_sqlite(sqlite_file):
    """
    Load all needed data from a single SQLite file using one connection.
    Consolidates load_zoi_data_from_sqlite, load_bus_zones_from_sqlite,
    and load_ens_pns_per_zone_from_sqlite into a single connection.

    :return: (zoi_data, bus_zones, ens_pns_per_zone, ens, pns) tuple
    """
    conn = sqlite3.connect(sqlite_file)
    try:
        zoi_data = load_zoi_data_from_sqlite(sqlite_file, conn=conn)
        bus_zones = load_bus_zones_from_sqlite(sqlite_file, conn=conn)

        ens, pns = None, None
        ens_pns_per_zone = {}
        try:
            if bus_zones:
                ens_pns_per_zone = load_ens_pns_per_zone_from_sqlite(sqlite_file, bus_zones, conn=conn)
                ens = ens_pns_per_zone.get('_total', (0, 0))[0]
                pns = ens_pns_per_zone.get('_total', (0, 0))[1]
            else:
                ens, pns = load_ens_pns_from_sqlite(sqlite_file, conn=conn)
        except Exception as e:
            printer.warning(f"  '{os.path.basename(sqlite_file)}': Could not load ENS/PNS: {e}")

        return zoi_data, bus_zones, ens_pns_per_zone, ens, pns
    finally:
        conn.close()


def compute_per_zone_objectives(zoi_data, bus_zones):
    """
    Compute objective value per zone in a single pass over all objective terms.

    Instead of calling evaluate_zoi_objective_from_data N times (once per zone),
    this determines each variable's zone assignment once and accumulates per-zone
    contributions in a single loop over all objective terms.

    :param zoi_data: ZOI data loaded from sqlite
    :param bus_zones: dict mapping bus name to zone name
    :return: dict {'Zone1': objective, 'Zone2': objective, ...}
    """
    zones = sorted(set(bus_zones.values()))
    all_i = set(zoi_data['sets']['i'])
    all_g = set(zoi_data['sets']['g'])
    gi_set = set(zoi_data['sets']['gi'])
    all_lines = set(zoi_data['sets']['la'])
    hub_connections = set(zoi_data['sets']['hubConnections']) if zoi_data['sets']['hubConnections'] else set()

    # Pre-compute: bus -> zone, generator -> zone (via gi mapping)
    g_to_zone = {}
    for g, i in gi_set:
        if i in bus_zones:
            g_to_zone[g] = bus_zones[i]

    # Pre-compute: line -> (zone_i, zone_j) for cross-zone splitting
    line_zones = {}
    for i, j, c in all_lines:
        line_zones[(i, j, c)] = (bus_zones.get(i), bus_zones.get(j))

    constant = zoi_data['objective']['constant']
    var_values = zoi_data['var_values']
    obj_data = zoi_data['objective']

    # Accumulate per-zone contributions in single pass
    raw = {zone: 0.0 for zone in zones}
    zone_cache = {}  # idx -> (zone, weight) or list of (zone, weight) for cross-zone

    if 'linear_vars_info' in obj_data:
        for (var_name, idx), coef in zip(obj_data['linear_vars_info'], obj_data['linear_coefs']):
            var_key = (var_name, idx)
            val = var_values.get(var_key, 0.0)
            if val == 0.0 or coef == 0.0:
                continue
            contribution = coef * val

            # Determine zone assignment for this index (cached)
            if idx not in zone_cache:
                zone_cache[idx] = _determine_variable_zone(idx, all_i, all_g, g_to_zone, bus_zones, line_zones, hub_connections)
            zone_assignment = zone_cache[idx]

            if zone_assignment is None:
                continue  # Variable outside all zones
            elif isinstance(zone_assignment, list):
                # Cross-zone (e.g. line between zones): split proportionally
                for zone, weight in zone_assignment:
                    if zone in raw:
                        raw[zone] += weight * contribution
            else:
                if zone_assignment in raw:
                    raw[zone_assignment] += contribution

    # Distribute the constant proportionally across zones
    raw_total = sum(raw.values())
    result = {}
    if raw_total != 0:
        for zone in zones:
            result[zone] = raw[zone] + constant * (raw[zone] / raw_total)
    else:
        share = constant / len(zones) if zones else 0
        for zone in zones:
            result[zone] = share

    return result


def _determine_variable_zone(idx, all_i, all_g, g_to_zone, bus_zones, line_zones, hub_connections):
    """
    Determine which zone a variable belongs to based on its index.

    :return: zone name (str), list of (zone, weight) for cross-zone, or None if outside all zones
    """
    if idx is None:
        return None  # Constant term, handled separately

    if not isinstance(idx, tuple):
        idx = (idx,)

    # Check for line indices (i, j, c) embedded in the index
    if len(idx) >= 3:
        for start in range(len(idx) - 2):
            potential_line = (idx[start], idx[start + 1], idx[start + 2])
            if potential_line in line_zones:
                zone_i, zone_j = line_zones[potential_line]
                if zone_i == zone_j:
                    return zone_i
                # Cross-zone line: split 50/50
                result = []
                if zone_i is not None:
                    result.append((zone_i, 0.5))
                if zone_j is not None:
                    result.append((zone_j, 0.5))
                return result if result else None

    # Check for hub connections
    if hub_connections:
        for start in range(len(idx) - 1):
            potential_hub = (idx[start], idx[start + 1])
            if potential_hub in hub_connections:
                bus = potential_hub[1]
                return bus_zones.get(bus)

    # Check bus membership
    for elem in idx:
        if elem in all_i:
            return bus_zones.get(elem)

    # Check generator membership
    for elem in idx:
        if elem in all_g:
            return g_to_zone.get(elem)

    return None  # Can't determine zone


def print_metric_table(title, rows, zones, dc_row_value, metric_key, wu_key=None):
    """
    Print a generic metric comparison table with per-zone breakdowns.

    :param title: Table title
    :param rows: list of dicts, each with 'source', metric_key (total value), 'per_zone' (dict zone->value),
                 optionally wu_key, 'is_uniform', 'dc_buf', 'tp_buf'
    :param zones: sorted list of zone names
    :param dc_row_value: the DC baseline value for this metric (for regret calculation)
    :param metric_key: key in row dict for the total metric value
    :param wu_key: optional key for work units column
    """
    # Calculate column widths
    zone_cols = len(zones)
    # Each zone gets: value col (14) + rel% col (10)
    zone_width = zone_cols * 24

    # Base columns: Source(14) + DC-Buf(8) + TP-Buf(8) + Total(16) + SumZones(16) + Check(10) + Rel%(10) + WU(12) + WU%(10)
    base_width = 14 + 8 + 8 + 16 + 16 + 10 + 10
    if wu_key:
        base_width += 12 + 10
    total_width = base_width + zone_width + 2

    printer.information(f"\n{title}")
    printer.information("-" * total_width)

    # Header
    header = f"  {'Source':<12s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Total':>14s} {'Sum(zones)':>14s} {'Check':>8s} {'Rel%':>8s}"
    if wu_key:
        header += f" {'WU':>10s} {'WU%':>8s}"
    for zone in zones:
        header += f" {zone:>12s} {zone + '%':>10s}"
    printer.information(header)
    printer.information("-" * total_width)

    for row in rows:
        total_val = row.get(metric_key)
        if total_val is None:
            continue

        per_zone = row.get('per_zone', {})
        sum_zones = sum(per_zone.get(z, 0) for z in zones)
        check_diff = total_val - sum_zones if total_val != 0 else 0

        # Regret relative to DC baseline
        if dc_row_value is not None and dc_row_value != 0:
            rel_pct = ((total_val - dc_row_value) / dc_row_value * 100)
        else:
            rel_pct = None

        no_buf = row.get('is_uniform') or (row.get('dc_buf') is None and row.get('tp_buf') is None)
        dc_str = "-" if no_buf else str(row.get('dc_buf'))
        tp_str = "-" if no_buf else str(row.get('tp_buf'))

        check_str = f"{check_diff:.2f}" if abs(check_diff) > 0.01 else "OK"
        rel_str = f"{rel_pct:.1f}%" if (row['source'] != 'DC' and rel_pct is not None) else ("-" if row['source'] == 'DC' else "N/A")

        line = f"  {row['source']:<12s} {dc_str:>8s} {tp_str:>8s} {total_val:>14.2f} {sum_zones:>14.2f} {check_str:>8s} {rel_str:>8s}"

        if wu_key:
            wu_val = row.get(wu_key)
            wu_str = f"{wu_val:.2f}" if wu_val is not None else "N/A"
            dc_wu = rows[0].get(wu_key) if rows else None  # First row is DC
            if wu_val is not None and dc_wu is not None and dc_wu != 0 and row['source'] != 'DC':
                wu_rel = (wu_val - dc_wu) / dc_wu * 100
                wu_rel_str = f"{wu_rel:.1f}%"
            else:
                wu_rel_str = "-"
            line += f" {wu_str:>10s} {wu_rel_str:>8s}"

        for zone in zones:
            zone_val = per_zone.get(zone, 0)
            # Per-zone relative to DC baseline's per-zone value
            dc_zone_val = rows[0].get('per_zone', {}).get(zone, 0) if rows else 0
            if row['source'] == 'DC':
                zone_rel_str = "-"
            elif dc_zone_val != 0:
                zone_rel = (zone_val - dc_zone_val) / dc_zone_val * 100
                zone_rel_str = f"{zone_rel:.1f}%"
            else:
                zone_rel_str = "N/A"
            line += f" {zone_val:>12.2f} {zone_rel_str:>10s}"

        printer.information(line)

    printer.information("-" * total_width)


def _process_single_file(sqlite_file):
    """
    Process a single SQLite file: load all data, compute per-zone objectives, and return results.
    Designed to be called in parallel via ProcessPoolExecutor.

    :return: (file_data dict, result tuple, log messages list) or None on failure
    """
    logs = []  # Collect log messages for deferred printing
    try:
        meta = load_file_metadata(sqlite_file)
        input_dir, limit_k = meta['input_dir'], meta['limit_k']
        dc_buffer, tp_buffer = meta['dc_buffer'], meta['tp_buffer']
        zone, demand, pmax = meta['zone'], meta['demand'], meta['pmax']
        basename = os.path.basename(sqlite_file)
        is_regret_file = basename.endswith('-regret.sqlite')

        start_time = time.time()
        zoi_data, bus_zones, ens_pns_per_zone, ens, pns = load_all_file_data_from_sqlite(sqlite_file)
        load_time = time.time() - start_time
        logs.append(('info', f"  '{basename}': loaded in {load_time:.2f}s"))

        per_zone_objectives = {}
        if bus_zones:
            per_zone_objectives = compute_per_zone_objectives(zoi_data, bus_zones)

        calc_total_obj = None
        stored_total_obj = None
        try:
            if 'total_objective' in zoi_data:
                stored_total_obj = zoi_data['total_objective']

            obj_data = zoi_data['objective']
            var_data = zoi_data['var_values']
            calc_total_obj = obj_data['constant']

            for (var_name, idx), coef in zip(obj_data['linear_vars_info'], obj_data['linear_coefs']):
                key = (var_name, idx)
                if key in var_data:
                    calc_total_obj += coef * var_data[key]
        except Exception as e:
            logs.append(('warn', f"  '{basename}': could not calculate total objective: {e}"))

        if calc_total_obj is not None:
            if stored_total_obj is not None:
                diff = abs(calc_total_obj - stored_total_obj)
                if diff > 0.01:
                    rel_diff_pct = (diff / stored_total_obj * 100) if stored_total_obj != 0 else 0.0
                    logs.append(('warn', f"  '{basename}': obj={calc_total_obj:.2f}, stored={stored_total_obj:.2f} (DIFF: {diff:.4f}, {rel_diff_pct:.4f}%)"))
                else:
                    logs.append(('info', f"  '{basename}': obj={calc_total_obj:.2f} (matches stored)"))

        result_tuple = (sqlite_file, input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, calc_total_obj, is_regret_file, ens, pns)

        file_data = {
            'sqlite_file': sqlite_file,
            'input_dir': input_dir,
            'limit_k': limit_k,
            'dc_buffer': dc_buffer,
            'tp_buffer': tp_buffer,
            'zone': zone,
            'demand': demand,
            'pmax': pmax,
            'zoi_data': zoi_data,
            'total_obj': calc_total_obj,
            'work_units': zoi_data.get('work_units'),
            'ens': ens,
            'pns': pns,
            'bus_zones': bus_zones,
            'per_zone_objectives': per_zone_objectives,
            'ens_pns_per_zone': ens_pns_per_zone,
            'util_source_csv': meta.get('util_source_csv'),
            'util_dc_threshold': meta.get('util_dc_threshold'),
            'util_tp_threshold': meta.get('util_tp_threshold'),
        }

        return file_data, result_tuple, is_regret_file, meta, logs

    except Exception as e:
        return None, None, None, None, [('error', f"  Failed to process '{sqlite_file}': {e}")]


def main(folder="."):
    # Find all SQLite files in the specified folder
    sqlite_files = sorted(f for f in glob.glob(os.path.join(folder, "*.sqlite")) if os.path.basename(f).startswith("TR-"))

    if not sqlite_files:
        printer.warning(f"No SQLite files found in '{folder}'")
        return

    printer.information(f"Found {len(sqlite_files)} SQLite file(s) in '{folder}'")

    # Process each file
    results = []
    all_files_data = []
    uniform_files = []
    regret_files = []
    utilization_files = []

    start_time = time.time()
    n_workers = min(len(sqlite_files), os.cpu_count() or 1, 61)  # Windows limits ProcessPoolExecutor to 61 workers

    if n_workers > 1:
        printer.information(f"Processing files with {n_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            processed = list(executor.map(_process_single_file, sqlite_files))
    else:
        processed = [_process_single_file(f) for f in sqlite_files]

    for file_data, result_tuple, is_regret_file, meta, logs in processed:
        # Print log messages
        for level, msg in logs:
            if level == 'error':
                printer.error(msg)
            elif level == 'warn':
                printer.warning(msg)
            elif level == 'success':
                printer.success(msg)
            else:
                printer.information(msg)

        if file_data is None:
            continue

        results.append(result_tuple)
        zone = file_data['zone']

        if is_regret_file:
            regret_files.append(file_data)
        elif is_utilization_run(meta):
            utilization_files.append(file_data)
        elif is_uniform_representation(zone):
            uniform_files.append(file_data)
        else:
            all_files_data.append(file_data)

    printer.information(f"\nProcessed {len(sqlite_files)} files in {time.time() - start_time:.2f}s")

    # Print summary
    if results:
        def sort_key(result):
            sqlite_file, input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, total_obj, is_regret, _, _ = result
            return make_run_sort_key(input_dir, limit_k, demand, pmax, dc_buffer, tp_buffer, zone) + (1 if is_regret else 0,)

        sorted_results = sorted(results, key=sort_key)

        # Calculate the maximum filename length for proper alignment
        max_filename_len = max(len(sqlite_file) for sqlite_file, _, _, _, _, _, _, _, _, _, _, _ in sorted_results)
        # Ensure minimum width for readability
        filename_width = max(max_filename_len, len("Filename"))
        # Calculate total table width
        table_width = filename_width + 2 + 16 + 8 + 8 + 8 + 8 + 16 + 14 + 14

        printer.information("\n" + "=" * table_width)
        printer.information("Summary of Objective Values (grouped by parameters):")
        printer.information("=" * table_width)
        printer.information(f"  {'Filename':<{filename_width}s} {'LimitK':>16s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Demand':>8s} {'PMax':>8s} {'Total Objective':>16s} {'ENS':>14s} {'PNS':>14s}")
        printer.information("-" * table_width)

        # Track previous group to insert separators
        prev_group = None

        for sqlite_file, input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, total_obj, _, ens, pns in sorted_results:
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
            # Show total objective if available
            total_str = f"{total_obj:.2f}" if total_obj is not None else "N/A"
            limit_k_str = limit_k if limit_k else "N/A"
            ens_str = f"{ens:.2f}" if ens is not None else "N/A"
            pns_str = f"{pns:.2f}" if pns is not None else "N/A"
            printer.information(f"  {sqlite_file:<{filename_width}s} {limit_k_str:>16s} {dc_str:>8s} {tp_str:>8s} {demand_str:>8s} {pmax_str:>8s} {total_str:>16s} {ens_str:>14s} {pns_str:>14s}")

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

    # Group utilization files by (input_dir, limit_k, demand, pmax)
    utilization_groups = {}
    for uf in utilization_files:
        group_key = (uf['input_dir'], uf['limit_k'], uf['demand'], uf['pmax'])
        if group_key not in utilization_groups:
            utilization_groups[group_key] = []
        utilization_groups[group_key].append(uf)

    # Lookup map for utilization source work units: (group_key, dc_th, tp_th) -> file_data
    utilization_source_map = {}
    for uf in utilization_files:
        gk = (uf['input_dir'], uf['limit_k'], uf['demand'], uf['pmax'])
        utilization_source_map[(gk, uf['util_dc_threshold'], uf['util_tp_threshold'])] = uf

    # Process each comparison group
    all_group_keys = sorted(
        set(uniform_groups.keys()) | set(regret_groups.keys()) | set(zone_files_by_group.keys()) | set(utilization_groups.keys()),
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
        utils = utilization_groups.get(group_key, [])

        dc_total_obj = dc_baseline['total_obj']
        dc_wu = dc_baseline.get('work_units')

        # Determine available zones from DC baseline's bus_zones
        dc_bus_zones = dc_baseline.get('bus_zones', {})
        zones = sorted(set(dc_bus_zones.values())) if dc_bus_zones else []

        # Build rows for per-zone tables
        table_rows = []

        # DC baseline row
        dc_per_zone_obj = dc_baseline.get('per_zone_objectives', {})
        dc_ens_pns_pz = dc_baseline.get('ens_pns_per_zone', {})
        dc_per_zone_ens = {z: dc_ens_pns_pz.get(z, (0, 0))[0] for z in zones}
        dc_per_zone_pns = {z: dc_ens_pns_pz.get(z, (0, 0))[1] for z in zones}

        table_rows.append({
            'source': 'DC',
            'is_uniform': True,
            'dc_buf': None, 'tp_buf': None,
            'total_obj': dc_total_obj,
            'per_zone': dc_per_zone_obj,
            'wu': dc_wu,
            'ens': dc_baseline.get('ens'),
            'pns': dc_baseline.get('pns'),
            'per_zone_ens': dc_per_zone_ens,
            'per_zone_pns': dc_per_zone_pns,
            'sort': (-2, '', 0, 0, 0),
        })

        for rf in regrets:
            zone = rf['zone']
            total_obj = rf.get('total_obj')
            if total_obj is None:
                continue

            # Look up source model for work units
            rf_is_util = is_utilization_run(rf)
            if rf_is_util:
                source = utilization_source_map.get((group_key, rf.get('util_dc_threshold'), rf.get('util_tp_threshold')))
            elif is_uniform_representation(zone):
                source = uniform_source_map.get((group_key, zone))
            else:
                source = zone_source_map.get((group_key, zone, rf.get('dc_buffer'), rf.get('tp_buffer')))
            source_wu = source.get('work_units') if source else None

            # Per-zone objectives for regret file
            rf_per_zone_obj = rf.get('per_zone_objectives', {})
            rf_ens_pns_pz = rf.get('ens_pns_per_zone', {})
            rf_per_zone_ens = {z: rf_ens_pns_pz.get(z, (0, 0))[0] for z in zones}
            rf_per_zone_pns = {z: rf_ens_pns_pz.get(z, (0, 0))[1] for z in zones}

            if rf_is_util:
                dc_th = rf.get('util_dc_threshold')
                tp_th = rf.get('util_tp_threshold')
                source_label = f"DC{dc_th}/TP{tp_th}"
            else:
                source_label = (str(zone) if zone is not None else "None") if is_uniform_representation(zone) else f"zoi({zone})"

            table_rows.append({
                'source': source_label,
                'is_uniform': is_uniform_representation(zone) and not rf_is_util,
                'dc_buf': rf.get('dc_buffer'), 'tp_buf': rf.get('tp_buffer'),
                'total_obj': total_obj,
                'per_zone': rf_per_zone_obj,
                'wu': source_wu,
                'ens': rf.get('ens'),
                'pns': rf.get('pns'),
                'per_zone_ens': rf_per_zone_ens,
                'per_zone_pns': rf_per_zone_pns,
                'sort': (1 if rf_is_util else -1 if is_uniform_representation(zone) else 0,
                         str(zone), 0,
                         rf.get('dc_buffer') or 0, rf.get('tp_buffer') or 0),
            })

        table_rows.sort(key=lambda r: r['sort'])

        # Print Objective table with per-zone breakdown
        print_metric_table(
            f"Objectives + WU (baseline: DC, total obj = {dc_total_obj:.2f})",
            table_rows, zones, dc_total_obj, 'total_obj', wu_key='wu'
        )

        # Print PNS table with per-zone breakdown
        pns_rows = []
        for row in table_rows:
            pns_rows.append({
                'source': row['source'],
                'is_uniform': row['is_uniform'],
                'dc_buf': row['dc_buf'], 'tp_buf': row['tp_buf'],
                'pns_total': row.get('pns', 0) or 0,
                'per_zone': row.get('per_zone_pns', {}),
                'sort': row['sort'],
            })
        dc_pns = dc_baseline.get('pns', 0) or 0
        print_metric_table(
            f"PNS (Power Not Served)",
            pns_rows, zones, dc_pns, 'pns_total'
        )

        # Print ENS table with per-zone breakdown
        ens_rows = []
        for row in table_rows:
            ens_rows.append({
                'source': row['source'],
                'is_uniform': row['is_uniform'],
                'dc_buf': row['dc_buf'], 'tp_buf': row['tp_buf'],
                'ens_total': row.get('ens', 0) or 0,
                'per_zone': row.get('per_zone_ens', {}),
                'sort': row['sort'],
            })
        dc_ens = dc_baseline.get('ens', 0) or 0
        print_metric_table(
            f"ENS (Excess Not Served / Excess Power Served)",
            ens_rows, zones, dc_ens, 'ens_total'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ZOI (Zone of Interest) objectives from SQLite files", fromfile_prefix_chars='@')
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing SQLite files (default: current directory)")
    args = parser.parse_args()
    main(args.folder)
