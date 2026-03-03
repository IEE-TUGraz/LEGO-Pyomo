import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import glob
import os
import sqlite3
from collections import defaultdict

from InOutModule.printer import Printer
from TechnicalRepresentation import is_uniform_representation, is_utilization_run, load_file_metadata, print_run_parameters, make_run_sort_key

printer = Printer.getInstance()
printer.set_width(180)


def count_line_representations(sqlite_file):
    """
    Count DC-OPF, TP, and total lines in a solved model's SQLite file.

    DC-OPF lines are identified by having entries in dual_eDC_ExiLinePij
    (the DC power flow constraint is skipped for TP/SN lines).
    TP lines = total lines in model - DC-OPF lines.
    SN lines were eliminated during merge_single_node_buses() and are not in the model.

    :return: dict with keys 'dc', 'tp', 'total' (total = lines remaining in model)
    """
    conn = sqlite3.connect(sqlite_file)
    try:
        # Count total lines in model (existing + candidate)
        n_le = conn.execute('SELECT COUNT(*) FROM le').fetchone()[0]
        n_lc = conn.execute('SELECT COUNT(*) FROM lc').fetchone()[0]
        total = n_le + n_lc

        # Count DC-OPF lines from dual table (only DC-OPF lines have this constraint active)
        has_dual_table = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='dual_eDC_ExiLinePij'"
        ).fetchone()[0]

        if has_dual_table:
            n_dc = conn.execute(
                'SELECT COUNT(*) FROM (SELECT DISTINCT "2", "3", "4" FROM dual_eDC_ExiLinePij)'
            ).fetchone()[0]
        else:
            n_dc = 0

        n_tp = total - n_dc
    finally:
        conn.close()

    return {'dc': n_dc, 'tp': n_tp, 'total': total}


def main(folder="."):
    sqlite_files = sorted(
        f for f in glob.glob(os.path.join(folder, "*.sqlite"))
        if os.path.basename(f).startswith("TR-") and not os.path.basename(f).endswith("-regret.sqlite")
    )

    if not sqlite_files:
        printer.warning(f"No TR-*.sqlite files found in '{folder}'")
        return

    printer.information(f"Found {len(sqlite_files)} TR-*.sqlite file(s) in '{folder}'")

    # Process each file
    entries = []
    for sqlite_file in sqlite_files:
        try:
            meta = load_file_metadata(sqlite_file)
            print_run_parameters(meta)
            counts = count_line_representations(sqlite_file)
            printer.information(f"  Lines in model: {counts['total']} (DC-OPF: {counts['dc']}, TP: {counts['tp']})")
            entries.append({
                'sqlite_file': sqlite_file,
                'meta': meta,
                'counts': counts,
                'is_util': is_utilization_run(meta),
            })
        except Exception as e:
            printer.error(f"  Failed to process '{sqlite_file}': {e}")
            import traceback
            traceback.print_exc()

    if not entries:
        return

    # Group by (input_dir, limit_k, demand, pmax)
    groups = defaultdict(list)
    for entry in entries:
        m = entry['meta']
        key = (m['input_dir'], m['limit_k'], m['demand'], m['pmax'])
        groups[key].append(entry)

    all_group_keys = sorted(groups.keys(), key=lambda k: (k[0] or '', k[1] or '', k[2], k[3]))

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

        group_entries = groups[group_key]

        # Find DC baseline to determine total original lines
        dc_baseline = next(
            (e for e in group_entries if e['meta']['zone'] == 'DC' and not e['is_util']), None
        )
        if dc_baseline is None:
            dc_baseline = next(
                (e for e in group_entries
                 if (e['meta']['zone'] is None or e['meta']['zone'] == "None") and not e['is_util']),
                None
            )

        total_original = dc_baseline['counts']['total'] if dc_baseline else None

        # Sort entries
        def entry_sort_key(entry_):
            m_ = entry_['meta']
            base = make_run_sort_key(m_['input_dir'], m_['limit_k'], m_['demand'], m_['pmax'],
                                     m_['dc_buffer'], m_['tp_buffer'], m_['zone'])
            return (1,) + base if entry_['is_util'] else (0,) + base

        sorted_entries = sorted(group_entries, key=entry_sort_key)

        # Print table
        print_representation_table(group_desc, sorted_entries, total_original)


def print_representation_table(group_desc, entries, total_original):
    """Print a summary table of line representations for a comparison group."""
    col_source = 16
    col_num = 8

    header_parts = [
        f"{'Source':>{col_source}s}",
        f"{'DC-Buf':>{col_num}s}",
        f"{'TP-Buf':>{col_num}s}",
        f"{'DC-OPF':>{col_num}s}",
        f"{'TP':>{col_num}s}",
        f"{'In Model':>{col_num}s}",
        f"{'Elim.':>{col_num}s}",
    ]
    if total_original is not None:
        header_parts.append(f"{'Original':>{col_num}s}")

    header = "  " + " ".join(header_parts)
    table_width = len(header)

    printer.information("\n" + "=" * table_width)
    printer.information(f"  LINE REPRESENTATIONS: {group_desc}")
    printer.information("=" * table_width)
    printer.information(header)
    printer.information("-" * table_width)

    for entry in entries:
        m = entry['meta']
        c = entry['counts']
        zone = m['zone']

        if entry['is_util']:
            source_label = f"DC{m['util_dc_threshold']}/TP{m['util_tp_threshold']}"
        elif is_uniform_representation(zone):
            source_label = str(zone) if zone is not None else "None"
        else:
            source_label = f"zoi({zone})"

        no_buf = is_uniform_representation(zone) or (m['dc_buffer'] is None and m['tp_buffer'] is None)
        dc_str = "-" if no_buf else str(m['dc_buffer'])
        tp_str = "-" if no_buf else str(m['tp_buffer'])

        eliminated = total_original - c['total'] if total_original is not None else None
        elim_str = str(eliminated) if eliminated is not None else "N/A"

        row_parts = [
            f"{source_label:>{col_source}s}",
            f"{dc_str:>{col_num}s}",
            f"{tp_str:>{col_num}s}",
            f"{c['dc']:>{col_num}d}",
            f"{c['tp']:>{col_num}d}",
            f"{c['total']:>{col_num}d}",
            f"{elim_str:>{col_num}s}",
        ]
        if total_original is not None:
            row_parts.append(f"{total_original:>{col_num}d}")

        printer.information("  " + " ".join(row_parts))

    printer.information("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze line technical representation (DC-OPF/TP/eliminated) from TR SQLite files",
        fromfile_prefix_chars='@'
    )
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing SQLite files (default: current directory)")
    args = parser.parse_args()
    main(args.folder)
