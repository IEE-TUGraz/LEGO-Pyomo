import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
import os
import sqlite3
import time
from collections import deque

import pandas as pd
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from rich_argparse import RichHelpFormatter

from InOutModule import SQLiteWriter
from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO

printer = Printer.getInstance()

# Set up logging so that infeasible constraints are logged by pyomo
logger = logging.getLogger("pyomo")
logger.setLevel("INFO")

DC_BUFFER_DEFAULT = 1
TP_BUFFER_DEFAULT = 1

SCALE_DEFAULT = 1.0  # Scaling default should always be 1.0 (no scaling)

ZONE_LABELS = {
    'DC': ('DC-OPF', 'DC Optimal Power Flow (all lines as DC-OPF)'),
    'TP': ('TP', 'Transport Model (all lines as TP)'),
    'SN': ('SN', 'Single Node / Copper Plate (all lines as SN)'),
}

# Maps uniform-representation keys to the pTecRepr value used in dPower_Network
UNIFORM_REPR_MAP = {'DC': 'DC-OPF', 'TP': 'TP', 'SN': 'SN'}


def is_uniform_representation(zone: str | None) -> bool:
    """Check if zone represents a uniform technical representation (DC, TP, SN, or None)."""
    return zone in UNIFORM_REPR_MAP or zone is None or zone == "None"


def is_utilization_run(meta: dict) -> bool:
    """Check if a file was produced by a utilization-based run (--utilization flag)."""
    return meta.get('util_source_csv') is not None


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
        'util_source_csv': None,
        'util_dc_threshold': None,
        'util_tp_threshold': None,
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
                if 'util_source_csv' in row and row['util_source_csv'] not in (None, 'None'):
                    meta['util_source_csv'] = str(row['util_source_csv'])
                if 'util_dc_threshold' in row and row['util_dc_threshold'] not in (None, 'None'):
                    meta['util_dc_threshold'] = float(row['util_dc_threshold'])
                if 'util_tp_threshold' in row and row['util_tp_threshold'] not in (None, 'None'):
                    meta['util_tp_threshold'] = float(row['util_tp_threshold'])
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


def make_run_sort_key(input_dir, limit_k, demand, pmax, dc_buffer, tp_buffer, zone):
    """Generate a sort key for ordering runs: groups by (input_dir, limit_k, demand, pmax),
    uniform representations first, then by buffer sizes and zone."""
    is_uniform = is_uniform_representation(zone)
    return (
        input_dir or '',
        limit_k or '',
        demand,
        pmax,
        0 if is_uniform else 1,
        -1 if is_uniform else (dc_buffer if dc_buffer is not None else 999),
        -1 if is_uniform else (tp_buffer if tp_buffer is not None else 999),
        str(zone) if zone else '',
    )


def assign_technical_representation_by_layers(cs: CaseStudy, dc_buffer: int, tp_buffer: int) -> None:
    """
    Assigns technical representation (DC-OPF, TP, or SN) to network lines based on their distance
    from the zone of interest (ZOI). Modifies cs.dPower_Network in place.

    Algorithm:
    1. Lines connecting two buses within the ZOI are assigned DC-OPF
    2. Lines are then assigned to layers based on their distance from the ZOI using BFS
    3. DC-Buffer: number of layers outside ZOI that should be DC-OPF
    4. TP-Buffer: number of layers after DC-Buffer that should be TP
    5. All remaining lines are assigned SN

    :param cs: CaseStudy object
    :param dc_buffer: Number of layers outside ZOI to assign as DC-OPF
    :param tp_buffer: Number of layers after dc_buffer to assign as TP
    """
    # Get buses in the zone of interest
    zoi_buses = set(cs.dPower_BusInfo[cs.dPower_BusInfo['zoi'] == 1].index)
    printer.information(f"Found {len(zoi_buses)} buses in zone of interest")

    # Initialize all lines as unassigned
    line_layer = {}  # Maps (i, j, c) -> layer number (-1 for ZOI internal lines)

    # Step 1: Identify lines within ZOI (both ends in ZOI)
    zoi_internal_lines = []
    for idx in cs.dPower_Network.index:
        i, j, c = idx
        if i in zoi_buses and j in zoi_buses:
            line_layer[idx] = -1  # Special marker for ZOI internal lines
            zoi_internal_lines.append(idx)

    # Step 2: Build adjacency structure for BFS
    # We need to track which buses are reachable at each layer
    bus_layer = {bus: -1 for bus in zoi_buses}  # ZOI buses are at layer -1
    visited_buses = set(zoi_buses)

    # BFS to assign layers to buses and lines
    # Initialize queue with ZOI buses so the loop handles all layers uniformly
    queue = deque((bus, -1) for bus in zoi_buses)

    while queue:
        next_queue = deque()

        # Collect all buses from current layer
        buses_in_current_layer = set()
        current_layer = None
        while queue:
            bus, layer = queue.popleft()
            buses_in_current_layer.add(bus)
            current_layer = layer

        # Lines from current layer connect to buses in next layer
        next_layer = current_layer + 1

        # Process unassigned lines: discover new buses and assign cross-layer connections
        unassigned = [idx for idx in cs.dPower_Network.index if idx not in line_layer]

        for i, j, c in unassigned:
            idx = (i, j, c)
            i_in_current = i in buses_in_current_layer
            j_in_current = j in buses_in_current_layer
            i_visited = i in visited_buses
            j_visited = j in visited_buses

            # Line connects current layer to unvisited bus - discovery line
            if (i_in_current and not j_visited) or (j_in_current and not i_visited):
                new_bus = j if i_in_current else i
                line_layer[idx] = next_layer
                bus_layer[new_bus] = next_layer
                visited_buses.add(new_bus)
                next_queue.append((new_bus, next_layer))
            # Both endpoints already visited - assign to max layer + 1 if same layer, else max layer
            elif i_visited and j_visited:
                line_layer[idx] = max(bus_layer[i], bus_layer[j]) + (1 if bus_layer[i] == bus_layer[j] else 0)

        # Report lines assigned to next_layer in this iteration
        if next_layer >= 0:
            lines_in_next = len([l for l in line_layer.values() if l == next_layer])
            if lines_in_next > 0:
                printer.information(f"Layer {next_layer}: {lines_in_next} lines")

        queue = next_queue

    # Step 3: Assign technical representations based on layers
    # Categorize all lines into lists for batch assignment
    zoi_dc_opf_lines = []
    buffer_dc_opf_lines = []
    tp_lines = []
    sn_lines = []

    for idx in cs.dPower_Network.index:
        if idx in line_layer:
            layer = line_layer[idx]
            match layer:
                case -1:  # ZOI internal
                    zoi_dc_opf_lines.append(idx)
                case _ if layer < dc_buffer:  # Within DC buffer
                    buffer_dc_opf_lines.append(idx)
                case _ if layer < dc_buffer + tp_buffer:  # Within TP buffer
                    tp_lines.append(idx)
                case _:  # Beyond both buffers
                    sn_lines.append(idx)
        else:
            printer.error(f"Line {idx} is not connected to the rest of the network - this should not happen!")

    # Batch assignment to avoid performance warnings (only if lists are non-empty)
    if zoi_dc_opf_lines:
        cs.dPower_Network.loc[zoi_dc_opf_lines, 'pTecRepr'] = 'DC-OPF'
    if buffer_dc_opf_lines:
        cs.dPower_Network.loc[buffer_dc_opf_lines, 'pTecRepr'] = 'DC-OPF'
    if tp_lines:
        cs.dPower_Network.loc[tp_lines, 'pTecRepr'] = 'TP'
    if sn_lines:
        cs.dPower_Network.loc[sn_lines, 'pTecRepr'] = 'SN'

    printer.information(f"Technical representation assignment complete:")
    printer.information(f"  DC-OPF in ZOI: {len(zoi_dc_opf_lines)} lines")
    printer.information(f"  DC-OPF in buffer: {len(buffer_dc_opf_lines)} lines")
    printer.information(f"  TP in buffer: {len(tp_lines)} lines")
    printer.information(f"  SN: {len(sn_lines)} lines")


def prevent_cross_zone_sn(cs):
    """Replace SN connections between buses of different zones with TP."""
    bus_zones = cs.dPower_BusInfo['z']
    net = cs.dPower_Network
    is_sn = net['pTecRepr'] == 'SN'
    i_zones = [bus_zones[i] for i, j, c in net.index]
    j_zones = [bus_zones[j] for i, j, c in net.index]
    is_cross_zone = [iz != jz for iz, jz in zip(i_zones, j_zones)]
    mask = is_sn & pd.Series(is_cross_zone, index=net.index)
    count = mask.sum()
    if count > 0:
        cs.dPower_Network.loc[mask, 'pTecRepr'] = 'TP'
        printer.information(f"  Upgraded {count} cross-zone SN connections to TP")


def load_utilization_from_csv(csv_path):
    """Load line utilization from an EvaluateLines CSV. Returns dict {(i, j, c): mean_util}."""
    df = pd.read_csv(csv_path)
    return {(row['i'], row['j'], row['c']): row['mean_util'] for _, row in df.iterrows()}


def assign_technical_representation_by_utilization(cs: CaseStudy, util_dict: dict, dc_threshold: float, tp_threshold: float) -> None:
    """
    Assigns technical representation based on line utilization thresholds.
    Lines >= dc_threshold -> DC-OPF, >= tp_threshold -> TP, else -> SN.
    Modifies cs.dPower_Network in place.
    """
    dc_lines, tp_lines, sn_lines, missing_lines = [], [], [], []

    for idx in cs.dPower_Network.index:
        i, j, c = idx
        util = util_dict.get((i, j, c))
        if util is None:
            util = util_dict.get((j, i, c))
        if util is None:
            missing_lines.append(idx)
            sn_lines.append(idx)
        elif util >= dc_threshold:
            dc_lines.append(idx)
        elif util >= tp_threshold:
            tp_lines.append(idx)
        else:
            sn_lines.append(idx)

    if missing_lines:
        for idx in missing_lines:
            printer.warning(f"  Line {idx} not found in utilization CSV — defaulting to SN")

    if dc_lines:
        cs.dPower_Network.loc[dc_lines, 'pTecRepr'] = 'DC-OPF'
    if tp_lines:
        cs.dPower_Network.loc[tp_lines, 'pTecRepr'] = 'TP'
    if sn_lines:
        cs.dPower_Network.loc[sn_lines, 'pTecRepr'] = 'SN'

    printer.information(f"Technical representation by utilization (DC>={dc_threshold}, TP>={tp_threshold}):")
    printer.information(f"  DC-OPF: {len(dc_lines)} lines, TP: {len(tp_lines)} lines, SN: {len(sn_lines)} lines ({len(missing_lines)} missing)")


def load_vGenInvest_from_sqlite(sqlite_file):
    """Load vGenInvest values from a solved model's sqlite file. Returns {g: value}."""
    conn = sqlite3.connect(sqlite_file)
    df = pd.read_sql_query('SELECT * FROM vGenInvest', conn)
    conn.close()
    return dict(zip(df['g'], df['values']))


def main(case_study_directory, zoi, limit_k, dc_buffer, tp_buffer, scale_demand, scale_pmax, all_zones, no_overwrite, prevent_cross_zone_merging, utilization=None):
    caseStudyName = case_study_directory.replace("/", "_").replace("\\", "_")

    printer.information(f"Loading case study from '{case_study_directory}'")
    identifier_parts_base = [f"data{caseStudyName}"]  # Build identifier with only non-default parameters
    start_time = time.time()
    cs_base = CaseStudy(case_study_directory)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    printer.information(f"Setting parameters so that it will be solved as rMIP")
    cs_base.dGlobal_Parameters["pEnableRMIP"] = True

    printer.information(f"Removing fixed slack node so that it is calculated based on demand")
    cs_base.dPower_Parameters["is"] = None

    if limit_k is not None:
        printer.information(f"Limiting K values to '{limit_k}'")
        identifier_parts_base.append(f"limitK{limit_k}")
        start, end = limit_k.split("-")
        cs_base.filter_timesteps(start, end, inplace=True)

    if scale_demand != SCALE_DEFAULT:
        printer.information(f"Scaling demand by factor {scale_demand}")
        identifier_parts_base.append(f"demand{scale_demand:g}")
        cs_base.dPower_Demand['value'] *= scale_demand

    if scale_pmax != SCALE_DEFAULT:
        printer.information(f"Scaling pPmax (line capacity) by factor {scale_pmax}")
        identifier_parts_base.append(f"pmax{scale_pmax:g}")
        cs_base.dPower_Network['pPmax'] *= scale_pmax

    # --- Parse utilization args ---
    if utilization is not None:
        util_csv, dc_th, tp_th = utilization
        dc_th, tp_th = float(dc_th), float(tp_th)
        printer.information(f"Loading utilization from '{util_csv}'")
        util_dict = load_utilization_from_csv(util_csv)
    else:
        util_csv = dc_th = tp_th = util_dict = None

    if utilization is None and is_uniform_representation(zoi) and not all_zones:
        if dc_buffer != DC_BUFFER_DEFAULT:
            printer.warning("DC buffer specified but using uniform or unchanged technical representation - DC buffer will be ignored")
        if tp_buffer != TP_BUFFER_DEFAULT:
            printer.warning("TP buffer specified but using uniform or unchanged technical representation - TP buffer will be ignored")

    if utilization is None and dc_buffer == 0 and tp_buffer == 0:
        printer.warning("dcBuffer=0 and tpBuffer=0 would make all cross-boundary lines SN, collapsing ZOI into the merged bus. Setting tpBuffer=1.")
        tp_buffer = 1

    # --- Determine runs ---
    available_zones = sorted(cs_base.dPower_BusInfo['z'].unique().tolist())

    if utilization is not None:
        zones_to_run = [None]
        regret_sources = ['__utilization__']
    elif all_zones:
        zones_to_run = list(UNIFORM_REPR_MAP) + available_zones
        regret_sources = ['TP', 'SN'] + available_zones  # No regret run for DC (would fix DC into DC)
    else:
        zones_to_run = [zoi]
        regret_sources = [zoi] if zoi != 'DC' else []  # No regret run for DC

    # Normal runs first (so their sqlite files exist when regret runs need them), then regret
    runs = [{'type': 'normal', 'zoi': z} for z in zones_to_run]
    runs += [{'type': 'regret', 'zoi': z} for z in regret_sources]

    # Lazy-load caches for regret (populated on demand from sqlite)
    source_vGenInvest = {}  # source name -> {g: value} or None (file not found)

    for run in runs:
        is_regret = run['type'] == 'regret'
        current_zoi = run['zoi']

        # --- Header ---
        if is_regret:
            label = f"utilization (regret)" if current_zoi == '__utilization__' else f"{current_zoi} (regret)"
        else:
            label = str(current_zoi)
        printer.information(f"\n{'=' * 60}")
        printer.information(f"  {label}")
        printer.information(f"{'=' * 60}\n")

        # --- Setup ---
        cs = cs_base.copy()
        identifier_parts = identifier_parts_base.copy()

        if is_regret:
            is_uniform_regret = current_zoi in ('TP', 'SN')
            is_utilization_regret = current_zoi == '__utilization__'

            # Lazy-load the source model's vGenInvest
            if current_zoi not in source_vGenInvest:
                if is_utilization_regret:
                    source_id = "-".join(identifier_parts_base + [f"utilDC{dc_th}-TP{tp_th}"])
                elif is_uniform_regret or current_zoi is None or current_zoi == 'None':
                    source_id = "-".join(identifier_parts_base)  # Don't include buffers for uniform or None
                    if is_uniform_regret:
                        source_id += f"-zoi{current_zoi}"  # But include uniform label (not for None)
                else:
                    source_id = "-".join(identifier_parts_base +
                                         ([f"dcBuffer{dc_buffer}"] if dc_buffer != DC_BUFFER_DEFAULT else []) +
                                         ([f"tpBuffer{tp_buffer}"] if tp_buffer != TP_BUFFER_DEFAULT else []) +
                                         [f"zoi{current_zoi}"])
                source_file = f"TR-{source_id}.sqlite"
                if os.path.exists(source_file):
                    source_vGenInvest[current_zoi] = load_vGenInvest_from_sqlite(source_file)
                else:
                    printer.error(f"  Source model '{source_file}' not found — regret for '{current_zoi}' skipped.")
                    source_vGenInvest[current_zoi] = None
            if source_vGenInvest[current_zoi] is None:
                continue

            # Uniform DC-OPF topology, all buses as ZOI
            cs.dPower_Network['pTecRepr'] = 'DC-OPF'
            cs.dPower_BusInfo['zoi'] = 1
            if is_utilization_regret:
                identifier_parts = [f"{source_id}-regret"]
            else:
                if not is_uniform_regret:
                    if dc_buffer != DC_BUFFER_DEFAULT:
                        identifier_parts.append(f"dcBuffer{dc_buffer}")
                    if tp_buffer != TP_BUFFER_DEFAULT:
                        identifier_parts.append(f"tpBuffer{tp_buffer}")
                identifier_parts.append(f"zoi{current_zoi}")
                identifier_parts.append("regret")

        elif utilization is not None:
            printer.information(f"Assigning technical representations by utilization (DC>={dc_th}, TP>={tp_th})")
            assign_technical_representation_by_utilization(cs, util_dict, dc_th, tp_th)
            printer.information(f"Ensuring no SN connections between different zones (upgrading to TP if needed)")
            prevent_cross_zone_sn(cs)
            identifier_parts = identifier_parts_base.copy() + [f"utilDC{dc_th}-TP{tp_th}"]

        elif current_zoi is None or current_zoi == 'None':
            printer.information("No ZOI adjustment - using original technical representations from data files")

        elif current_zoi in UNIFORM_REPR_MAP:
            printer.information(f"Using uniform technical representation: {current_zoi}")
            identifier_parts.append(f"zoi{current_zoi}")
            cs.dPower_Network['pTecRepr'] = UNIFORM_REPR_MAP[current_zoi]
            cs.dPower_BusInfo['zoi'] = 1
        else:
            printer.information(f"Setting Zone of Interest to '{current_zoi}'")
            if dc_buffer != DC_BUFFER_DEFAULT:
                identifier_parts.append(f"dcBuffer{dc_buffer}")
            if tp_buffer != TP_BUFFER_DEFAULT:
                identifier_parts.append(f"tpBuffer{tp_buffer}")
            identifier_parts.append(f"zoi{current_zoi}")

            cs.dPower_BusInfo['zoi'] = 0
            cs.dPower_BusInfo.loc[cs.dPower_BusInfo['z'] == current_zoi, 'zoi'] = 1
            if (cs.dPower_BusInfo['zoi'] == 1).sum() == 0:
                printer.warning(f"0 buses selected for zone '{current_zoi}'. Available zones: {available_zones}")

            printer.information(f"Assigning technical representations with DC-Buffer={dc_buffer}, TP-Buffer={tp_buffer}")
            assign_technical_representation_by_layers(cs, dc_buffer, tp_buffer)

        # --- Prevent cross-zone SN connections ---
        if prevent_cross_zone_merging:
            printer.information(f"Ensuring no SN connections between different zones (upgrading to TP if needed)")
            identifier_parts.append(f"noCrossZoneSN")
            prevent_cross_zone_sn(cs)

        # --- Filename + noOverwrite check ---
        identifier = "-".join(identifier_parts)
        sqlite_filename = f"TR-{identifier}.sqlite"

        if no_overwrite and os.path.exists(sqlite_filename):
            printer.information(f"  File '{sqlite_filename}' already exists, skipping (--noOverwrite)")
            continue

        # --- Build ---
        cs.merge_single_node_buses()
        lego = LEGO(cs)
        model, timing = lego.build_model()
        printer.information(f"  Build took {timing:.2f} seconds")

        # --- Fix vGenInvest (regret only) ---
        if is_regret:
            for g in model.vGenInvest:
                model.vGenInvest[g].fix(source_vGenInvest[current_zoi].get(g, 0))

        # --- Solve ---
        printer.information(f"  Solving...")
        results, timing, objective_value = lego.solve_model()
        printer.information(f"  Solve took {timing:.2f} seconds")

        match results.solver.termination_condition:
            case pyo.TerminationCondition.optimal:
                printer.success(f"  Optimal solution: {pyo.value(model.objective):.4f}")
            case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                printer.error(f"  Model returned as {results.solver.termination_condition}, logging infeasible constraints:")
                log_infeasible_constraints(model, log_expression=False)
            case _:
                printer.warning(f"  Solver terminated with condition: {results.solver.termination_condition}")

        # --- Export ---
        SQLiteWriter.model_to_sqlite(model, sqlite_filename)
        SQLiteWriter.add_solver_statistics_to_sqlite(sqlite_filename, results, work_units=lego.work_units)
        SQLiteWriter.add_run_parameters_to_sqlite(
            sqlite_filename,
            case_study_directory=case_study_directory,
            limit_k=limit_k,
            scale_demand=scale_demand,
            scale_pmax=scale_pmax,
            zoi=None if utilization is not None else current_zoi,
            dc_buffer=None if utilization is not None else dc_buffer,
            tp_buffer=None if utilization is not None else tp_buffer,
            **(dict(
                util_source_csv=Path(util_csv).name,
                util_dc_threshold=dc_th,
                util_tp_threshold=tp_th,
            ) if utilization is not None else {}),
        )
        printer.information(f"  Saved to '{sqlite_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests different technical representations for network", formatter_class=RichHelpFormatter, fromfile_prefix_chars='@')

    parser.add_argument("caseStudyDirectory", type=str, help="Path to folder containing data for LEGO model")
    parser.add_argument("--zoi", type=str, help="Which Zone (from Power_BusInfo 'z') should be the Zone of Interest ('zoi')? Special values: 'DC' for uniform DC-OPF, 'TP' for uniform Transport Model, 'SN' for uniform Single Node. If not specified (default None), uses original settings from Excel files without adjustment. Ignored if --all is specified.", nargs="?", default=None)
    parser.add_argument("--all", action="store_true", help="Run for all zones: DC, TP, SN, and all zones found in Power_BusInfo. Overrides --zoi.")
    parser.add_argument("--limitK", type=str, help="Limit the ks, format: 'k0025-k0048'", nargs="?", default=None)
    parser.add_argument("--dcBuffer", type=int, help=f"Number of network layers outside ZOI to assign as DC-OPF (default: {DC_BUFFER_DEFAULT})", nargs="?", default=DC_BUFFER_DEFAULT)
    parser.add_argument("--tpBuffer", type=int, help=f"Number of network layers after DC buffer to assign as TP (default: {TP_BUFFER_DEFAULT})", nargs="?", default=TP_BUFFER_DEFAULT)
    parser.add_argument("--scaleDemand", type=float, help=f"Scaling factor for demand (default: {SCALE_DEFAULT} = no scaling)", nargs="?", default=SCALE_DEFAULT)
    parser.add_argument("--scalePMax", type=float, help=f"Scaling factor for pPmax (line capacity) (default: {SCALE_DEFAULT} = no scaling)", nargs="?", default=SCALE_DEFAULT)
    parser.add_argument("--noOverwrite", action="store_true", help="Skip cases where the output .sqlite file already exists")
    parser.add_argument("--preventCrossZoneMerging", action="store_true", help="Prevent SN connections between different zones by upgrading them to TP. This keeps zones electrically separated even in the SN representation, preventing them from being merged into one bus. This is especially relevant when using a ZOI that does not encompass all buses of a zone, or when using multiple zones.")
    parser.add_argument("--utilization", nargs=3, metavar=("CSV_FILE", "DC_THRESHOLD", "TP_THRESHOLD"),
                        help="Assign line representations based on utilization from CSV. Lines >= DC_THRESHOLD get DC-OPF, >= TP_THRESHOLD get TP, else SN. Mutually exclusive with --zoi and --all.")
    args = parser.parse_args()

    if args.utilization:
        conflicts = []
        if args.zoi:
            conflicts.append("--zoi")
        if args.all:
            conflicts.append("--all")
        if args.dcBuffer != DC_BUFFER_DEFAULT:
            conflicts.append("--dcBuffer")
        if args.tpBuffer != TP_BUFFER_DEFAULT:
            conflicts.append("--tpBuffer")
        if conflicts:
            raise ValueError(f"--utilization cannot be combined with: {', '.join(conflicts)}")

    main(args.caseStudyDirectory, args.zoi, args.limitK, args.dcBuffer, args.tpBuffer, args.scaleDemand, args.scalePMax, args.all, args.noOverwrite, args.preventCrossZoneMerging, utilization=args.utilization)
