import argparse
import glob
import os
import sqlite3
from collections import defaultdict

import pandas as pd

from InOutModule.printer import Printer
from TechnicalRepresentation import is_uniform_representation, ZONE_LABELS

printer = Printer.getInstance()
printer.set_width(180)


def load_file_metadata(sqlite_file):
    """
    Load all metadata for a file from run_parameters and solver_statistics.

    Returns:
        dict with keys: input_dir, limit_k, dc_buffer, tp_buffer, zone, demand, pmax, work_units
    """
    meta = {
        'input_dir': None,
        'limit_k': None,
        'dc_buffer': None,
        'tp_buffer': None,
        'zone': None,
        'demand': 1.0,
        'pmax': 1.0,
        'work_units': None,
    }

    try:
        conn = sqlite3.connect(sqlite_file)

        # --- run_parameters ---
        try:
            df = pd.read_sql_query('SELECT * FROM run_parameters', conn)
            if len(df) > 0:
                row = df.iloc[0]
                # input_dir from case_study_directory (e.g. "data/NREL-118")
                if 'case_study_directory' in row and row['case_study_directory'] not in (None, 'None'):
                    meta['input_dir'] = str(row['case_study_directory'])
                # limit_k
                if 'limit_k' in row and row['limit_k'] not in (None, 'None'):
                    meta['limit_k'] = str(row['limit_k'])
                # zone / zoi
                if 'zoi' in row and row['zoi'] not in (None, 'None'):
                    meta['zone'] = str(row['zoi'])
                # buffers
                if 'dc_buffer' in row and row['dc_buffer'] not in (None, 'None'):
                    meta['dc_buffer'] = int(float(row['dc_buffer']))
                if 'tp_buffer' in row and row['tp_buffer'] not in (None, 'None'):
                    meta['tp_buffer'] = int(float(row['tp_buffer']))
                # scales
                if 'scale_demand' in row and row['scale_demand'] not in (None, 'None'):
                    meta['demand'] = float(row['scale_demand'])
                if 'scale_pmax' in row and row['scale_pmax'] not in (None, 'None'):
                    meta['pmax'] = float(row['scale_pmax'])
        except Exception:
            pass

        # --- solver_statistics (work_units) ---
        try:
            df_stats = pd.read_sql_query('SELECT * FROM solver_statistics', conn)
            if 'work_units' in df_stats.columns:
                val = df_stats.iloc[0]['work_units']
                if val is not None:
                    meta['work_units'] = float(val)
        except Exception:
            pass

        conn.close()
    except Exception:
        pass

    return meta


def print_run_parameters(meta):
    """Print run parameters extracted from metadata."""
    parts = []
    if meta['input_dir']:
        parts.append(f"input_dir={meta['input_dir']}")
    if meta['limit_k']:
        parts.append(f"limit_k={meta['limit_k']}")
    parts.append(f"zone={meta['zone']}")
    if is_uniform_representation(meta['zone']):
        parts.append(f"dc_buffer= (unused)")
        parts.append(f"tp_buffer= (unused)")
    else:
        parts.append(f"dc_buffer={meta['dc_buffer']}" + (" (default)" if meta['dc_buffer'] == 1 else ""))
        parts.append(f"tp_buffer={meta['tp_buffer']}" + (" (default)" if meta['tp_buffer'] == 1 else ""))
    parts.append(f"scale_demand={meta['demand']}")
    parts.append(f"scale_pmax={meta['pmax']}")
    printer.information(f"  Run parameters: {', '.join(parts)}")


def evaluate_gen_investment_by_technology_from_sqlite(sqlite_file, filter_zoi=True):
    """
    Evaluate generator investment capacity by technology from a SQLite file.

    Calculates invested capacity as vGenInvest x pMaxProd and aggregates by technology.

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

        # Calculate capacity: vGenInvest x pMaxProd (convert GW to MW)
        invest_data['capacity_MW'] = invest_data['values_invest'] * invest_data['values_maxprod'] * 1000

        # Aggregate by technology
        tech_capacity = invest_data.groupby('tec')['capacity_MW'].sum().to_dict()

        return tech_capacity

    finally:
        conn.close()


def evaluate_gen_investment_with_custom_zoi(sqlite_file, zoi_sqlite_file):
    """
    Evaluate generator investment using ZOI definition from another SQLite file.

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


def main(folder="."):
    # Find all SQLite files in the specified folder
    sqlite_files = sorted(glob.glob(os.path.join(folder, "*.sqlite")))

    if not sqlite_files:
        printer.warning(f"No SQLite files found in '{folder}'")
        return

    printer.information(f"Found {len(sqlite_files)} SQLite file(s) in '{folder}'")

    # Process each SQLite file
    # Each entry: { meta, sqlite_file, total_invest, zoi_invest }
    all_entries = []
    all_technologies = set()

    for sqlite_file in sqlite_files:
        printer.information(f"\nProcessing '{sqlite_file}'...")

        try:
            meta = load_file_metadata(sqlite_file)
            print_run_parameters(meta)

            # Calculate investments (both total and ZOI)
            total_invest = evaluate_gen_investment_by_technology_from_sqlite(sqlite_file, filter_zoi=False)
            zoi_invest = evaluate_gen_investment_by_technology_from_sqlite(sqlite_file, filter_zoi=True)

            # Track all technologies
            all_technologies.update(total_invest.keys())
            all_technologies.update(zoi_invest.keys())

            total_cap = sum(total_invest.values())
            zoi_cap = sum(zoi_invest.values())
            printer.success(f"  Total Investment: {total_cap:.2f} MW")
            printer.success(f"  ZOI Investment: {zoi_cap:.2f} MW")

            all_entries.append({
                'sqlite_file': sqlite_file,
                'meta': meta,
                'total_invest': total_invest,
                'zoi_invest': zoi_invest,
            })

        except Exception as e:
            printer.error(f"  Failed to process '{sqlite_file}': {e}")
            import traceback
            traceback.print_exc()

    if not all_entries:
        return

    # --- Summary table ---
    print_summary_table(all_entries)

    # --- Split into uniform (DC/TP/SN/None) and zone-specific entries ---
    uniform_entries = []
    zone_entries = []
    for entry in all_entries:
        zone = entry['meta']['zone']
        if is_uniform_representation(zone):
            uniform_entries.append(entry)
        else:
            zone_entries.append(entry)

    # --- Group everything by (input_dir, limit_k, demand, pmax) ---
    # uniform_groups: key -> list of uniform entries
    # zone_groups:    key -> { (dc_buf, tp_buf) -> list of zone entries }
    uniform_groups = defaultdict(list)
    zone_groups = defaultdict(lambda: defaultdict(list))

    for entry in uniform_entries:
        m = entry['meta']
        key = (m['input_dir'], m['limit_k'], m['demand'], m['pmax'])
        uniform_groups[key].append(entry)

    for entry in zone_entries:
        m = entry['meta']
        key = (m['input_dir'], m['limit_k'], m['demand'], m['pmax'])
        buf_key = (m['dc_buffer'], m['tp_buffer'])
        zone_groups[key][buf_key].append(entry)

    # Collect all top-level group keys
    all_group_keys = sorted(set(uniform_groups.keys()) | set(zone_groups.keys()),
                            key=lambda k: (k[0] or '', k[1] or '', k[2], k[3]))

    # --- Per-group comparisons ---
    for (input_dir, limit_k, demand, pmax) in all_group_keys:
        # Build group description
        group_desc_parts = []
        if input_dir:
            group_desc_parts.append(f"input={input_dir}")
        if limit_k:
            group_desc_parts.append(f"limitK={limit_k}")
        group_desc_parts.append(f"demand={demand}")
        group_desc_parts.append(f"pmax={pmax}")
        group_desc = ", ".join(group_desc_parts)

        printer.information("\n" + "=" * 160)
        printer.information(f"COMPARISON GROUP: {group_desc}")
        printer.information("=" * 160)

        group_key = (input_dir, limit_k, demand, pmax)
        uniforms = uniform_groups.get(group_key, [])

        # Find baseline among uniforms: prefer zoiNone, fall back to zoiDC
        baseline_entry = next((e for e in uniforms if e['meta']['zone'] in (None, "None")), None)
        if baseline_entry is None:
            baseline_entry = next((e for e in uniforms if e['meta']['zone'] == 'DC'), None)

        baseline_zone = baseline_entry['meta']['zone'] if baseline_entry else None
        baseline_label = "zoiNone" if (baseline_zone is None or baseline_zone == "None") else f"zoi{baseline_zone}"

        # --- 1) Uniform Technical Representation Comparisons ---
        if uniforms and baseline_entry is not None:
            baseline_total = baseline_entry['total_invest']
            baseline_total_cap = sum(baseline_total.values())
            other_uniforms = [e for e in uniforms if e['meta']['zone'] != baseline_zone]

            if other_uniforms:
                if baseline_zone is None or baseline_zone == "None":
                    bl_label, bl_desc = 'None', 'No ZOI adjustment (original Excel settings)'
                else:
                    bl_label, bl_desc = ZONE_LABELS.get(baseline_zone, (baseline_zone, f'Uniform {baseline_zone}'))

                # (label, desc, total_cap, abs_diff, rel_diff, sort_order, work_units)
                entries = [(bl_label, bl_desc + ' (BASELINE)', baseline_total_cap, 0.0, 0.0, -99,
                            baseline_entry['meta'].get('work_units'))]

                sort_order_map = {'DC': 0, 'TP': 1, 'SN': 2}
                for e in other_uniforms:
                    zone = e['meta']['zone']
                    if zone is None or zone == "None":
                        label, desc, so = 'None', 'No ZOI adjustment (original Excel settings)', -1
                    else:
                        label, desc = ZONE_LABELS.get(zone, (zone, f'Uniform {zone}'))
                        so = sort_order_map.get(zone, 3)
                    total_cap = sum(e['total_invest'].values())
                    abs_diff = total_cap - baseline_total_cap
                    rel_diff = (abs_diff / baseline_total_cap * 100) if baseline_total_cap != 0 else 0.0
                    entries.append((label, desc, total_cap, abs_diff, rel_diff, so, e['meta'].get('work_units')))

                entries.sort(key=lambda x: x[5])

                printer.information(f"\nUniform Technical Representation Comparisons")
                printer.information("-" * 160)
                printer.information(f"  {'Type':<10s} {'Description':<60s} {'Total Cap (MW)':>18s} {'Abs. Diff (MW)':>18s} {'Rel. Diff (%)':>15s} {'Work Units':>12s}")
                printer.information("-" * 160)
                for label, desc, total_cap, abs_diff, rel_diff, _, wu in entries:
                    wu_str = f"{wu:.2f}" if wu is not None else "N/A"
                    printer.information(f"  {label:<10s} {desc:<60s} {total_cap:>18.2f} {abs_diff:>18.2f} {rel_diff:>14.1f}% {wu_str:>12s}")
                printer.information("-" * 160)

                # Detailed technology breakdown for each non-baseline uniform type
                for e in sorted(other_uniforms, key=lambda x: sort_order_map.get(x['meta']['zone'], 99)):
                    zone = e['meta']['zone']
                    if zone is None or zone == "None":
                        continue
                    _, desc = ZONE_LABELS.get(zone, (zone, f'Uniform {zone}'))

                    printer.information(f"\nComparing: {desc} ({zone}) vs {bl_label} (Baseline)")
                    printer.information("-" * 160)
                    printer.information(
                        f"  {'Technology':<30s} "
                        f"{'Total (' + bl_label + ')':>18s} {'Total (' + zone + ')':>18s} {'Diff':>15s} {'Rel%':>10s}"
                    )
                    printer.information("-" * 160)
                    for tec in sorted(all_technologies):
                        base_val = baseline_total.get(tec, 0.0)
                        zone_val = e['total_invest'].get(tec, 0.0)
                        diff = zone_val - base_val
                        rel = (diff / base_val * 100) if base_val != 0 else 0.0
                        printer.information(
                            f"  {tec:<30s} "
                            f"{base_val:>18.2f} {zone_val:>18.2f} {diff:>15.2f} {rel:>9.1f}%"
                        )

        # --- 2) Zone-Specific Comparisons (one block per dc_buf / tp_buf) ---
        if group_key in zone_groups and baseline_entry is not None:
            baseline_total = baseline_entry['total_invest']
            baseline_total_cap = sum(baseline_total.values())
            baseline_sqlite = baseline_entry['sqlite_file']

            for (dc_buf, tp_buf), zone_file_entries in sorted(zone_groups[group_key].items()):
                if not zone_file_entries:
                    continue

                printer.information(f"\nZone-Specific Comparisons (Baseline: {baseline_label})")
                printer.information(f"Parameters: {group_desc}, dcBuffer={dc_buf}, tpBuffer={tp_buf}")
                printer.information("-" * 160)
                printer.information(
                    f"  {'Zone':<12s} "
                    f"{'Baseline for Zone':>18s} {'Zone-Specific Run':>18s} {'Diff':>15s} {'Rel. Diff (%)':>15s} {'Work Units':>12s}"
                )
                printer.information("-" * 160)

                sum_baseline_zones = 0.0

                for entry in sorted(zone_file_entries, key=lambda e: e['meta']['zone']):
                    zone = entry['meta']['zone']
                    zone_zoi_cap = sum(entry['zoi_invest'].values())

                    # Recalculate baseline investment using this zone's zoi_i
                    baseline_zone_invest = evaluate_gen_investment_with_custom_zoi(baseline_sqlite, entry['sqlite_file'])
                    baseline_zone_cap = sum(baseline_zone_invest.values())
                    sum_baseline_zones += baseline_zone_cap

                    diff = baseline_zone_cap - zone_zoi_cap
                    rel_diff = (diff / baseline_zone_cap * 100) if baseline_zone_cap != 0 else 0.0
                    wu = entry['meta'].get('work_units')
                    wu_str = f"{wu:.2f}" if wu is not None else "N/A"

                    printer.information(
                        f"  {zone:<12s} "
                        f"{baseline_zone_cap:>18.2f} {zone_zoi_cap:>18.2f} {diff:>15.2f} {rel_diff:>14.1f}% {wu_str:>12s}"
                    )

                # Safety check
                printer.information("-" * 160)
                printer.information(f"  {'SAFETY CHECK':<12s} {'Sum Baseline':>18s} {'Baseline Total':>18s} {'Difference':>15s} {'Rel. Diff (%)':>15s}")
                safety_diff = sum_baseline_zones - baseline_total_cap
                safety_rel = (safety_diff / baseline_total_cap * 100) if baseline_total_cap != 0 else 0.0
                if abs(safety_rel) < 0.01:
                    printer.success(f"  {'[PASSED]':<12s} {sum_baseline_zones:>18.2f} {baseline_total_cap:>18.2f} {safety_diff:>15.2f} {safety_rel:>14.1f}%")
                else:
                    printer.error(f"  {'[FAILED]':<12s} {sum_baseline_zones:>18.2f} {baseline_total_cap:>18.2f} {safety_diff:>15.2f} {safety_rel:>14.1f}%")

                # Detailed technology comparison per zone
                for entry in sorted(zone_file_entries, key=lambda e: e['meta']['zone']):
                    zone = entry['meta']['zone']
                    zone_zoi = entry['zoi_invest']
                    baseline_zone_invest = evaluate_gen_investment_with_custom_zoi(baseline_sqlite, entry['sqlite_file'])

                    printer.information(f"\n  Comparing Zone {zone}: {baseline_label} vs zoi{zone}")
                    printer.information("  " + "-" * 100)
                    printer.information(
                        f"  {'Technology':<30s} "
                        f"{'ZOI (' + baseline_label + ')':>18s} {'ZOI (zoi' + zone + ')':>18s} {'Diff':>15s} {'Rel%':>10s}"
                    )
                    printer.information("  " + "-" * 100)
                    for tec in sorted(all_technologies):
                        base_val = baseline_zone_invest.get(tec, 0.0)
                        zone_val = zone_zoi.get(tec, 0.0)
                        diff = zone_val - base_val
                        rel = (diff / base_val * 100) if base_val != 0 else 0.0
                        printer.information(
                            f"  {tec:<30s} "
                            f"{base_val:>18.2f} {zone_val:>18.2f} {diff:>15.2f} {rel:>9.1f}%"
                        )


def print_summary_table(entries):
    """Print summary table of all files."""
    if not entries:
        return

    max_filename_len = max(len(e['sqlite_file']) for e in entries)
    filename_width = max(max_filename_len, len("Filename"))
    table_width = filename_width + 2 + 8 + 8 + 15 + 8 + 8 + 15 + 15

    printer.information("\n" + "=" * table_width)
    printer.information("Summary of Generator Investment Capacity by Zone")
    printer.information("=" * table_width)
    printer.information(
        f"  {'Filename':<{filename_width}s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Zone':>15s} {'Demand':>8s} {'PMax':>8s} "
        f"{'Total Cap (MW)':>15s} {'ZOI Cap (MW)':>15s}"
    )
    printer.information("-" * table_width)

    # Sort by input_dir, limit_k, demand, pmax, then uniform first, then zone
    def sort_key(e):
        m = e['meta']
        zone = m['zone']
        is_uniform = is_uniform_representation(zone)
        return (m['input_dir'] or '', m['limit_k'] or '', m['demand'], m['pmax'],
                0 if is_uniform else 1,
                -1 if is_uniform else (m['dc_buffer'] or 999),
                -1 if is_uniform else (m['tp_buffer'] or 999),
                str(zone) if zone else '')

    prev_group = None
    for e in sorted(entries, key=sort_key):
        m = e['meta']
        current_group = (m['input_dir'], m['limit_k'], m['demand'], m['pmax'])
        if prev_group is not None and current_group != prev_group:
            printer.information("-" * table_width)
        prev_group = current_group

        zone = m['zone']
        is_uniform = is_uniform_representation(zone)
        dc_str = "-" if is_uniform else (str(m['dc_buffer']) if m['dc_buffer'] is not None else "N/A")
        tp_str = "-" if is_uniform else (str(m['tp_buffer']) if m['tp_buffer'] is not None else "N/A")
        zone_str = zone if zone is not None else "None"
        total_cap = sum(e['total_invest'].values())
        zoi_cap = sum(e['zoi_invest'].values())

        printer.information(
            f"  {e['sqlite_file']:<{filename_width}s} {dc_str:>8s} {tp_str:>8s} {zone_str:>15s} "
            f"{m['demand']:>8.1f} {m['pmax']:>8.1f} {total_cap:>15.2f} {zoi_cap:>15.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze generator investment capacity by technology from SQLite files", fromfile_prefix_chars='@')
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing SQLite files (default: current directory)")
    args = parser.parse_args()
    main(args.folder)
