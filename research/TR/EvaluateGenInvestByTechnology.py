import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import glob
import os
import sqlite3
from collections import defaultdict

import pandas as pd

from InOutModule.printer import Printer
from TechnicalRepresentation import is_uniform_representation, load_file_metadata, print_run_parameters, make_run_sort_key

printer = Printer.getInstance()
printer.set_width(180)


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

            is_regret = os.path.basename(sqlite_file).endswith('-regret.sqlite')

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
                'is_regret': is_regret,
            })

        except Exception as e:
            printer.error(f"  Failed to process '{sqlite_file}': {e}")
            import traceback
            traceback.print_exc()

    if not all_entries:
        return

    # --- Recalculate ZOI Cap for regret entries using source's zoi_i ---
    entry_by_file = {e['sqlite_file']: e for e in all_entries}
    for entry in all_entries:
        if entry['is_regret']:
            source_file = entry['sqlite_file'].replace('-regret.sqlite', '.sqlite')
            if source_file in entry_by_file:
                entry['zoi_invest'] = evaluate_gen_investment_with_custom_zoi(
                    entry['sqlite_file'], source_file
                )

    # Build regret lookup: source_file -> regret_entry
    regret_by_source = {}
    for entry in all_entries:
        if entry['is_regret']:
            source_file = entry['sqlite_file'].replace('-regret.sqlite', '.sqlite')
            regret_by_source[source_file] = entry

    # Non-regret entries for summary and comparisons
    non_regret_entries = [e for e in all_entries if not e['is_regret']]

    # --- Summary table (non-regret only, with Safety Check) ---
    print_summary_table(non_regret_entries, regret_by_source)

    # --- Split into uniform (DC/TP/SN/None) and zone-specific entries ---
    uniform_entries = []
    zone_entries = []
    for entry in non_regret_entries:
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

        # DC-OPF baseline
        dc_baseline = next((e for e in uniforms if e['meta']['zone'] == 'DC'), None)
        if dc_baseline is None or group_key not in zone_groups:
            continue

        dc_total_invest = dc_baseline['total_invest']
        dc_total_cap = sum(dc_total_invest.values())
        dc_wu = dc_baseline['meta'].get('work_units')
        dc_sqlite = dc_baseline['sqlite_file']

        # --- Zone-Specific Comparisons (one block per dc_buf / tp_buf) ---
        for (dc_buf, tp_buf), zone_file_entries in sorted(zone_groups[group_key].items()):
            if not zone_file_entries:
                continue

            printer.information(f"\nZone-Specific Comparisons (Baseline: DC-OPF)")
            printer.information(f"Parameters: {group_desc}, dcBuffer={dc_buf}, tpBuffer={tp_buf}")
            printer.information("-" * 160)
            printer.information(
                f"  {'Zone':<12s} "
                f"{'ZOI Cap (DC)':>15s} {'ZOI Cap (Zone)':>15s} {'ZOI Diff':>12s} {'ZOI Rel%':>10s} "
                f"{'Total (DC)':>14s} {'Total (Regret)':>15s} {'Regret Diff':>12s} {'Regret Rel%':>11s} "
                f"{'WU':>12s} {'WU Rel%':>10s}"
            )
            printer.information("-" * 160)

            sum_baseline_zones = 0.0
            baseline_zone_cache = {}  # zone -> {tec: capacity}

            for entry in sorted(zone_file_entries, key=lambda e: e['meta']['zone']):
                zone = entry['meta']['zone']
                zone_zoi_cap = sum(entry['zoi_invest'].values())

                # DC baseline's investment in this zone's ZOI
                baseline_zone_invest = evaluate_gen_investment_with_custom_zoi(dc_sqlite, entry['sqlite_file'])
                baseline_zone_cache[zone] = baseline_zone_invest
                baseline_zone_cap = sum(baseline_zone_invest.values())
                sum_baseline_zones += baseline_zone_cap

                zoi_diff = zone_zoi_cap - baseline_zone_cap
                zoi_rel = (zoi_diff / baseline_zone_cap * 100) if baseline_zone_cap != 0 else 0.0

                # Regret comparison
                regret_entry = regret_by_source.get(entry['sqlite_file'])
                if regret_entry:
                    regret_total_cap = sum(regret_entry['total_invest'].values())
                    regret_diff = regret_total_cap - dc_total_cap
                    regret_rel = (regret_diff / dc_total_cap * 100) if dc_total_cap != 0 else 0.0
                    regret_total_str = f"{regret_total_cap:.2f}"
                    regret_diff_str = f"{regret_diff:.2f}"
                    regret_rel_str = f"{regret_rel:.1f}%"
                else:
                    regret_total_str = "N/A"
                    regret_diff_str = "N/A"
                    regret_rel_str = "N/A"

                # Work units (of the zone model, relative to DC)
                wu = entry['meta'].get('work_units')
                wu_str = f"{wu:.2f}" if wu is not None else "N/A"
                wu_rel = ((wu - dc_wu) / dc_wu * 100) if (wu is not None and dc_wu is not None and dc_wu != 0) else None
                wu_rel_str = f"{wu_rel:.1f}%" if wu_rel is not None else "-"

                printer.information(
                    f"  {zone:<12s} "
                    f"{baseline_zone_cap:>15.2f} {zone_zoi_cap:>15.2f} {zoi_diff:>12.2f} {zoi_rel:>9.1f}% "
                    f"{dc_total_cap:>14.2f} {regret_total_str:>15s} {regret_diff_str:>12s} {regret_rel_str:>11s} "
                    f"{wu_str:>12s} {wu_rel_str:>10s}"
                )

            # Safety check: sum of DC per-zone investments == DC total
            printer.information("-" * 160)
            printer.information(f"  {'SAFETY CHECK':<12s} {'Sum of Zones':>15s} {'DC Total':>15s} {'Diff':>12s} {'Rel%':>10s}")
            safety_diff = sum_baseline_zones - dc_total_cap
            safety_rel = (safety_diff / dc_total_cap * 100) if dc_total_cap != 0 else 0.0
            if abs(safety_rel) < 0.01:
                printer.success(f"  {'[PASSED]':<12s} {sum_baseline_zones:>15.2f} {dc_total_cap:>15.2f} {safety_diff:>12.2f} {safety_rel:>9.1f}%")
            else:
                printer.error(f"  {'[FAILED]':<12s} {sum_baseline_zones:>15.2f} {dc_total_cap:>15.2f} {safety_diff:>12.2f} {safety_rel:>9.1f}%")

            # Detailed technology comparisons per zone
            for entry in sorted(zone_file_entries, key=lambda e: e['meta']['zone']):
                zone = entry['meta']['zone']
                zone_zoi = entry['zoi_invest']
                baseline_zone_invest = baseline_zone_cache[zone]

                # 1) ZOI investment comparison (DC-OPF vs zone model, in ZOI)
                printer.information(f"\n  Comparing Zone {zone}: DC-OPF vs zoi{zone} (ZOI investments)")
                printer.information("  " + "-" * 100)
                printer.information(
                    f"  {'Technology':<30s} "
                    f"{'ZOI (DC-OPF)':>18s} {'ZOI (zoi' + zone + ')':>18s} {'Diff':>15s} {'Rel%':>10s}"
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

                # 2) Regret total investment comparison (DC-OPF vs regret model)
                regret_entry = regret_by_source.get(entry['sqlite_file'])
                if regret_entry:
                    regret_total_invest = regret_entry['total_invest']
                    printer.information(f"\n  Comparing Zone {zone}: DC-OPF vs Regret (Total investments)")
                    printer.information("  " + "-" * 100)
                    printer.information(
                        f"  {'Technology':<30s} "
                        f"{'Total (DC-OPF)':>18s} {'Total (Regret)':>18s} {'Diff':>15s} {'Rel%':>10s}"
                    )
                    printer.information("  " + "-" * 100)
                    for tec in sorted(all_technologies):
                        base_val = dc_total_invest.get(tec, 0.0)
                        regret_val = regret_total_invest.get(tec, 0.0)
                        diff = regret_val - base_val
                        rel = (diff / base_val * 100) if base_val != 0 else 0.0
                        printer.information(
                            f"  {tec:<30s} "
                            f"{base_val:>18.2f} {regret_val:>18.2f} {diff:>15.2f} {rel:>9.1f}%"
                        )


def print_summary_table(entries, regret_by_source):
    """Print summary table of all non-regret files with Safety Check against regret models."""
    if not entries:
        return

    max_filename_len = max(len(e['sqlite_file']) for e in entries)
    filename_width = max(max_filename_len, len("Filename"))
    table_width = filename_width + 2 + 16 + 8 + 8 + 8 + 8 + 15 + 15 + 16

    printer.information("\n" + "=" * table_width)
    printer.information("Summary of Generator Investment Capacity")
    printer.information("=" * table_width)
    printer.information(
        f"  {'Filename':<{filename_width}s} {'LimitK':>16s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Demand':>8s} {'PMax':>8s} "
        f"{'Total Cap (MW)':>15s} {'ZOI Cap (MW)':>15s} {'Safety Check':>16s}"
    )
    printer.information("-" * table_width)

    def sort_key(e):
        m = e['meta']
        return make_run_sort_key(m['input_dir'], m['limit_k'], m['demand'], m['pmax'],
                                 m['dc_buffer'], m['tp_buffer'], m['zone'])

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
        limit_k_str = m['limit_k'] if m['limit_k'] else "N/A"
        total_cap = sum(e['total_invest'].values())
        zoi_cap = sum(e['zoi_invest'].values())

        # Safety Check: compare ZOI Cap with regret model's ZOI Cap
        regret_entry = regret_by_source.get(e['sqlite_file'])
        if regret_entry is None:
            # DC and None have no regret model
            safety_str = "-"
        else:
            regret_zoi_cap = sum(regret_entry['zoi_invest'].values())
            safety_diff = abs(zoi_cap - regret_zoi_cap)
            safety_rel = (safety_diff / zoi_cap * 100) if zoi_cap != 0 else 0.0
            if safety_rel < 0.01:
                safety_str = "PASS"
            else:
                safety_str = f"FAIL ({safety_rel:.2f}%)"

        printer.information(
            f"  {e['sqlite_file']:<{filename_width}s} {limit_k_str:>16s} {dc_str:>8s} {tp_str:>8s} "
            f"{m['demand']:>8.1f} {m['pmax']:>8.1f} {total_cap:>15.2f} {zoi_cap:>15.2f} {safety_str:>16s}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze generator investment capacity by technology from SQLite files", fromfile_prefix_chars='@')
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing SQLite files (default: current directory)")
    args = parser.parse_args()
    main(args.folder)
