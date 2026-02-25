import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import sqlite3

import pandas as pd

from InOutModule.printer import Printer
from TechnicalRepresentation import load_file_metadata, print_run_parameters

printer = Printer.getInstance()


def get_pmax(row, pmax_dict):
    key = (row['i'], row['j'], row['c'])
    if key in pmax_dict:
        return pmax_dict[key]
    return pmax_dict.get((row['j'], row['i'], row['c']), None)


def compute_line_utilization(sqlite_file):
    conn = sqlite3.connect(sqlite_file)
    try:
        df_wk = pd.read_sql_query('SELECT * FROM pWeight_k', conn)
        wk = dict(zip(df_wk.iloc[:, 0], df_wk['values']))

        df_wrp = pd.read_sql_query('SELECT * FROM pWeight_rp', conn)
        wrp = dict(zip(df_wrp.iloc[:, 0], df_wrp['values']))

        df_pmax = pd.read_sql_query('SELECT * FROM pPmax', conn)
        df_pmax.columns = ['i', 'j', 'c', 'values']
        pmax = {(row['i'], row['j'], row['c']): row['values'] for _, row in df_pmax.iterrows()}

        df_line = pd.read_sql_query('SELECT * FROM vLineP', conn)
        df_line.columns = ['rp', 'k', 'i', 'j', 'c', 'values']
    finally:
        conn.close()

    df_line['pmax'] = df_line.apply(lambda r: get_pmax(r, pmax), axis=1)
    df_line = df_line.dropna(subset=['pmax'])

    df_line['weight'] = df_line['rp'].map(wrp) * df_line['k'].map(wk)
    df_line['weighted_util'] = df_line['values'].abs() / df_line['pmax'] * df_line['weight']

    total_weight = df_line[['rp', 'k', 'weight']].drop_duplicates()['weight'].sum()

    result = df_line.groupby(['i', 'j', 'c']).agg(
        util_sum=('weighted_util', 'sum'),
        pmax=('pmax', 'first')
    ).reset_index()
    result['mean_util_pct'] = result['util_sum'] / total_weight * 100
    result = result.sort_values('mean_util_pct', ascending=False).reset_index(drop=True)

    return result, total_weight


def print_line_utilization(sqlite_file, result, meta):
    input_dir = meta.get('input_dir', 'N/A')
    zoi = meta.get('zone', 'None')
    dc_buf = meta.get('dc_buffer', 0.0)
    tp_buf = meta.get('tp_buffer', 0.0)

    printer.information(f"\n=== Line Utilization: {sqlite_file} ===")
    printer.information(f"Source: {input_dir} | ZOI: {zoi} | DC-buf: {dc_buf} | TP-buf: {tp_buf}")
    printer.information("")

    header = f"{'Rank':>5}  {'Line (i -> j, c)':<30}  {'pPmax':>12}  {'Mean Util [%]':>14}"
    printer.information(header)
    printer.information("-" * len(header))

    for rank, row in result.iterrows():
        line_str = f"{row['i']} -> {row['j']}, {row['c']}"
        printer.information(f"{rank + 1:>5}  {line_str:<30}  {row['pmax']:>12.1f}  {row['mean_util_pct']:>13.1f}%")


def build_identifier(meta):
    """Reconstruct the run identifier from metadata, following the same non-default encoding as TechnicalRepresentation."""
    from TechnicalRepresentation import is_uniform_representation, DC_BUFFER_DEFAULT, TP_BUFFER_DEFAULT, SCALE_DEFAULT
    input_dir = meta.get('input_dir', '') or ''
    name = input_dir.replace("/", "_").replace("\\", "_")
    parts = [f"data{name}"]

    if meta.get('limit_k'):
        parts.append(f"limitK{meta['limit_k']}")
    if meta.get('demand', 1.0) != SCALE_DEFAULT:
        parts.append(f"demand{meta['demand']:.1f}")
    if meta.get('pmax', 1.0) != SCALE_DEFAULT:
        parts.append(f"pmax{meta['pmax']:.1f}")

    zone = meta.get('zone')
    if not is_uniform_representation(zone):
        dc_buf = meta.get('dc_buffer')
        tp_buf = meta.get('tp_buffer')
        if dc_buf is not None and dc_buf != DC_BUFFER_DEFAULT:
            parts.append(f"dcBuffer{dc_buf}")
        if tp_buf is not None and tp_buf != TP_BUFFER_DEFAULT:
            parts.append(f"tpBuffer{tp_buf}")

    if zone and zone != 'None':
        parts.append(f"zoi{zone}")

    return "-".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Analyze transmission line utilization from SQLite result files")
    parser.add_argument("sqlite_files", nargs="+", help="SQLite result file(s) to analyze")
    args = parser.parse_args()

    for sqlite_file in args.sqlite_files:
        try:
            meta = load_file_metadata(sqlite_file)
            print_run_parameters(meta)

            result, total_weight = compute_line_utilization(sqlite_file)
            printer.information(f"  Total weight: {total_weight:.4f}, Lines analyzed: {len(result)}")

            print_line_utilization(sqlite_file, result, meta)

            # Write CSV output
            identifier = build_identifier(meta)
            csv_path = f"TR-EvaluateLines-{identifier}.csv"
            csv_df = result[['i', 'j', 'c', 'pmax', 'mean_util_pct']].copy()
            csv_df = csv_df.rename(columns={'pmax': 'pPmax'})
            csv_df['mean_util'] = csv_df['mean_util_pct'] / 100.0
            csv_df = csv_df.drop(columns=['mean_util_pct'])
            csv_df.to_csv(csv_path, index=False)
            printer.information(f"  CSV written to: {csv_path}")
        except Exception as e:
            printer.error(f"Failed to process '{sqlite_file}': {e}")
            raise


if __name__ == "__main__":
    main()
