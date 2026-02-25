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
from TechnicalRepresentation import is_uniform_representation, is_utilization_run, load_file_metadata, print_run_parameters, make_run_sort_key
from EvaluateZOIObjective import print_metric_table, load_bus_zones_from_sqlite

printer = Printer.getInstance()
printer.set_width(180)


def compute_per_zone_investments(sqlite_file, bus_zones):
    """
    Compute invested capacity (MW) per technology per zone.

    Maps each generator to a zone via gi -> bus -> bus_zones,
    then computes capacity = vGenInvest * pMaxProd * 1000.

    :param sqlite_file: Path to the SQLite database file
    :param bus_zones: dict mapping bus name to zone name
    :return: (tech_totals, tech_per_zone) where
             tech_totals = {technology: total_capacity_MW}
             tech_per_zone = {technology: {zone: capacity_MW}}
    """
    conn = sqlite3.connect(sqlite_file)
    try:
        vGenInvest_df = pd.read_sql_query('SELECT * FROM vGenInvest', conn)
        pMaxProd_df = pd.read_sql_query('SELECT * FROM pMaxProd', conn)
        gtec_df = pd.read_sql_query('SELECT * FROM gtec', conn).rename(columns={'0': 'g', '1': 'tec'})
        gi_df = pd.read_sql_query('SELECT * FROM gi', conn).rename(columns={'0': 'g', '1': 'i'})
    finally:
        conn.close()

    # Merge all data
    invest_data = vGenInvest_df.merge(pMaxProd_df, on='g', suffixes=('_invest', '_maxprod'))
    invest_data = invest_data.merge(gtec_df[['g', 'tec']], on='g')
    invest_data = invest_data.merge(gi_df[['g', 'i']], on='g')

    # Compute capacity in MW
    invest_data['capacity_MW'] = invest_data['values_invest'] * invest_data['values_maxprod'] * 1000

    # Map bus to zone
    invest_data['zone'] = invest_data['i'].map(bus_zones)

    # Aggregate by technology (total)
    tech_totals = invest_data.groupby('tec')['capacity_MW'].sum().to_dict()

    # Aggregate by technology and zone
    tech_per_zone = defaultdict(dict)
    for (tec, zone), grp in invest_data.groupby(['tec', 'zone']):
        tech_per_zone[tec][zone] = grp['capacity_MW'].sum()

    return tech_totals, dict(tech_per_zone)


def main(folder="."):
    # Find all SQLite files in the specified folder
    sqlite_files = sorted(f for f in glob.glob(os.path.join(folder, "*.sqlite")) if os.path.basename(f).startswith("TR-"))

    if not sqlite_files:
        printer.warning(f"No SQLite files found in '{folder}'")
        return

    printer.information(f"Found {len(sqlite_files)} SQLite file(s) in '{folder}'")

    # Process each SQLite file — load metadata for all, investments only from regret/DC files
    all_entries = []  # source (non-regret) entries with metadata
    regret_invest = {}  # source_file -> (tech_totals, tech_per_zone, bus_zones)

    for sqlite_file in sqlite_files:
        printer.information(f"\nProcessing '{sqlite_file}'...")

        try:
            meta = load_file_metadata(sqlite_file)
            print_run_parameters(meta)

            basename = os.path.basename(sqlite_file)
            is_regret = basename.endswith('-regret.sqlite')

            if is_regret:
                # Load investment data from regret file (has complete bus-zone mapping)
                bus_zones = load_bus_zones_from_sqlite(sqlite_file)
                tech_totals, tech_per_zone = compute_per_zone_investments(sqlite_file, bus_zones)
                source_file = sqlite_file.replace('-regret.sqlite', '.sqlite')
                regret_invest[source_file] = (tech_totals, tech_per_zone, bus_zones)
                total_cap = sum(tech_totals.values())
                printer.success(f"  Total Investment (regret): {total_cap:.2f} MW")
            else:
                all_entries.append({
                    'sqlite_file': sqlite_file,
                    'meta': meta,
                    'is_util': is_utilization_run(meta),
                })

        except Exception as e:
            printer.error(f"  Failed to process '{sqlite_file}': {e}")
            import traceback
            traceback.print_exc()

    if not all_entries:
        return

    # For DC/None baselines (no regret file), load investments directly
    for entry in all_entries:
        zone = entry['meta']['zone']
        sf = entry['sqlite_file']
        if sf not in regret_invest and not entry['is_util'] and (zone == 'DC' or zone is None or zone == "None"):
            bus_zones = load_bus_zones_from_sqlite(sf)
            tech_totals, tech_per_zone = compute_per_zone_investments(sf, bus_zones)
            regret_invest[sf] = (tech_totals, tech_per_zone, bus_zones)
            total_cap = sum(tech_totals.values())
            printer.success(f"  Total Investment (DC baseline): {total_cap:.2f} MW")

    # Check for missing regret files
    missing_regret = []
    for entry in all_entries:
        zone = entry['meta']['zone']
        sf = entry['sqlite_file']
        if sf not in regret_invest:
            missing_regret.append(entry)
            printer.error(f"  No regret file found for '{sf}' — per-zone investment data unavailable")

    # Attach investment data to entries (from regret files)
    for entry in all_entries:
        sf = entry['sqlite_file']
        if sf in regret_invest:
            tech_totals, tech_per_zone, bus_zones = regret_invest[sf]
            entry['tech_totals'] = tech_totals
            entry['tech_per_zone'] = tech_per_zone
            entry['bus_zones'] = bus_zones
        else:
            entry['tech_totals'] = {}
            entry['tech_per_zone'] = {}
            entry['bus_zones'] = {}

    # --- Summary table ---
    print_summary_table(all_entries, regret_invest)

    # --- Group entries by (input_dir, limit_k, demand, pmax) ---
    groups = defaultdict(list)
    for entry in all_entries:
        if entry['sqlite_file'] not in regret_invest:
            continue  # skip entries without investment data
        m = entry['meta']
        key = (m['input_dir'], m['limit_k'], m['demand'], m['pmax'])
        groups[key].append(entry)

    all_group_keys = sorted(groups.keys(), key=lambda k: (k[0] or '', k[1] or '', k[2], k[3]))

    # --- Per-group comparisons: one table per technology ---
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

        entries = groups[group_key]

        # DC baseline
        dc_baseline = next((e for e in entries if e['meta']['zone'] == 'DC' and not e['is_util']), None)
        if dc_baseline is None:
            dc_baseline = next((e for e in entries if (e['meta']['zone'] is None or e['meta']['zone'] == "None") and not e['is_util']), None)
        if dc_baseline is None:
            printer.warning("  No DC baseline found, skipping comparison group")
            continue

        # Determine zones from DC baseline
        dc_bus_zones = dc_baseline.get('bus_zones', {})
        zones = sorted(set(dc_bus_zones.values())) if dc_bus_zones else []

        # Sort entries: DC first, then uniform, then zoi, then util last
        def entry_sort_key(entry_):
            m_ = entry_['meta']
            base = make_run_sort_key(m_['input_dir'], m_['limit_k'], m_['demand'], m_['pmax'],
                                     m_['dc_buffer'], m_['tp_buffer'], m_['zone'])
            return (1,) + base if entry_['is_util'] else (0,) + base

        sorted_entries = sorted(entries, key=entry_sort_key)

        # Collect all technologies across this group
        group_technologies = set()
        for entry in sorted_entries:
            group_technologies.update(entry['tech_totals'].keys())

        # For each technology, build rows and print table
        for tec in sorted(group_technologies):
            dc_tec_total = dc_baseline['tech_totals'].get(tec, 0)

            rows = []
            for entry in sorted_entries:
                m = entry['meta']
                zone = m['zone']
                if entry['is_util']:
                    source_label = f"DC{m['util_dc_threshold']}/TP{m['util_tp_threshold']}"
                    is_uniform = False
                else:
                    is_uniform = is_uniform_representation(zone)
                    source_label = (str(zone) if zone is not None else "None") if is_uniform else f"zoi({zone})"

                tec_total = entry['tech_totals'].get(tec, 0)
                tec_per_zone = entry['tech_per_zone'].get(tec, {})

                rows.append({
                    'source': source_label,
                    'is_uniform': is_uniform,
                    'dc_buf': m['dc_buffer'],
                    'tp_buf': m['tp_buffer'],
                    'capacity': tec_total,
                    'per_zone': tec_per_zone,
                    'wu': m.get('work_units'),
                })

            # Only print if there's any nonzero capacity for this technology
            if dc_tec_total > 0 or any(r['capacity'] > 0 for r in rows):
                print_metric_table(
                    f"Technology: {tec} (capacity in MW)",
                    rows, zones, dc_tec_total, 'capacity', wu_key='wu'
                )


def print_summary_table(entries, regret_invest):
    """Print summary table of all source files with data source indicator."""
    if not entries:
        return

    max_filename_len = max(len(e['sqlite_file']) for e in entries)
    filename_width = max(max_filename_len, len("Filename"))
    table_width = filename_width + 2 + 16 + 8 + 8 + 8 + 8 + 15 + 16

    printer.information("\n" + "=" * table_width)
    printer.information("Summary of Generator Investment Capacity")
    printer.information("=" * table_width)
    printer.information(
        f"  {'Filename':<{filename_width}s} {'LimitK':>16s} {'DC-Buf':>8s} {'TP-Buf':>8s} {'Demand':>8s} {'PMax':>8s} "
        f"{'Total Cap (MW)':>15s} {'Data Source':>16s}"
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
        no_buf = is_uniform_representation(zone) or (m['dc_buffer'] is None and m['tp_buffer'] is None)
        dc_str = "-" if no_buf else str(m['dc_buffer'])
        tp_str = "-" if no_buf else str(m['tp_buffer'])
        limit_k_str = m['limit_k'] if m['limit_k'] else "N/A"
        total_cap = sum(e['tech_totals'].values())

        # Data source indicator
        sf = e['sqlite_file']
        if sf in regret_invest:
            if e['is_util']:
                source_str = "regret"
            elif zone == 'DC' or zone is None or zone == "None":
                source_str = "DC (self)"
            else:
                source_str = "regret"
        else:
            source_str = "MISSING"

        printer.information(
            f"  {e['sqlite_file']:<{filename_width}s} {limit_k_str:>16s} {dc_str:>8s} {tp_str:>8s} "
            f"{m['demand']:>8.1f} {m['pmax']:>8.1f} {total_cap:>15.2f} {source_str:>16s}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze generator investment capacity by technology from SQLite files", fromfile_prefix_chars='@')
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing SQLite files (default: current directory)")
    args = parser.parse_args()
    main(args.folder)
