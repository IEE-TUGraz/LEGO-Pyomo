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


def is_uniform_representation(zone: str | None) -> bool:
    """Check if zone represents a uniform technical representation (DC, TP, SN, or None)."""
    return zone in ('DC', 'TP', 'SN', None) or zone == "None"


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


def main(case_study_directory, zoi, limit_k, dc_buffer, tp_buffer, scale_demand, scale_pmax, all_zones, no_overwrite=False):
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
        identifier_parts_base.append(f"demand{scale_demand:.1f}")
        cs_base.dPower_Demand['value'] *= scale_demand

    if scale_pmax != SCALE_DEFAULT:
        printer.information(f"Scaling pPmax (line capacity) by factor {scale_pmax}")
        identifier_parts_base.append(f"pmax{scale_pmax:.1f}")
        cs_base.dPower_Network['pPmax'] *= scale_pmax

    if is_uniform_representation(zoi) and not all_zones:
        if dc_buffer != DC_BUFFER_DEFAULT:
            printer.warning("DC buffer specified but using uniform or unchanged technical representation - DC buffer will be ignored")
        if tp_buffer != TP_BUFFER_DEFAULT:
            printer.warning("TP buffer specified but using uniform or unchanged technical representation - TP buffer will be ignored")

    # Determine which zones to run
    if all_zones:
        printer.information("Running for all zones (--all specified)")
        # Get all unique zones from the data
        available_zones = sorted(cs_base.dPower_BusInfo['z'].unique().tolist())
        zones_to_run = ['DC', 'TP', 'SN'] + available_zones
        printer.information(f"Will run for zones: {zones_to_run}")
    else:
        zones_to_run = [zoi]

    legos = {}
    for current_zoi in zones_to_run:
        printer.information(f"\n{'=' * 80}")
        printer.information(f"Processing zone: {current_zoi}")
        printer.information(f"{'=' * 80}\n")

        # Create a copy of the base case study for this zone
        cs = cs_base.copy()
        identifier_parts = identifier_parts_base.copy()

        # Apply zone-specific settings
        if current_zoi is None or current_zoi == 'None':
            printer.information(f"No ZOI adjustment - using original technical representations from Power_Network.xlsx")
        elif current_zoi in ['DC', 'TP', 'SN']:
            printer.information(f"Using uniform technical representation: {current_zoi}")
            identifier_parts.append(f"zoi{current_zoi}")
            cs.dPower_Network['pTecRepr'] = 'DC-OPF' if current_zoi == 'DC' else current_zoi
        else:
            printer.information(f"Setting Zone of Interest (zoi) to zone '{current_zoi}'")

            # Add buffer info before zoi for easy grouping
            if dc_buffer != DC_BUFFER_DEFAULT:
                identifier_parts.append(f"dcBuffer{dc_buffer}")
            if tp_buffer != TP_BUFFER_DEFAULT:
                identifier_parts.append(f"tpBuffer{tp_buffer}")

            identifier_parts.append(f"zoi{current_zoi}")
            cs.dPower_BusInfo['zoi'] = 0
            cs.dPower_BusInfo.loc[cs.dPower_BusInfo['z'] == current_zoi, 'zoi'] = 1

            # Check if any buses were assigned to ZOI
            num_zoi_buses = (cs.dPower_BusInfo['zoi'] == 1).sum()
            if num_zoi_buses == 0:
                available_zones = cs.dPower_BusInfo['z'].unique().tolist()
                printer.warning(f"0 buses selected for zone '{current_zoi}'. Available zones: {available_zones}")

            printer.information(f"Assigning technical representations with DC-Buffer={dc_buffer}, TP-Buffer={tp_buffer}")
            assign_technical_representation_by_layers(cs, dc_buffer, tp_buffer)

        identifier = "-".join(identifier_parts)
        sqlite_filename = f"TR-{identifier}.sqlite"

        if no_overwrite and os.path.exists(sqlite_filename):
            printer.information(f"  File '{sqlite_filename}' already exists, skipping (--noOverwrite)")
            continue

        cs.merge_single_node_buses()
        printer.information(f"Building LEGO model for case study with '{current_zoi}' as zoi...")
        lego = LEGO(cs)
        model, timing = lego.build_model()
        printer.information(f"Building LEGO model for case study with '{current_zoi}' as zoi took {timing:.2f} seconds")
        legos[current_zoi] = (lego, model)

        printer.information(f"Solving LEGO model for case study with '{current_zoi}' as zoi...")
        results, timing, objective_value = lego.solve_model()
        printer.information(f"Solving LEGO model for case study with '{current_zoi}' as zoi took {timing:.2f} seconds")

        match results.solver.termination_condition:
            case pyo.TerminationCondition.optimal:
                printer.success(f"Optimal solution: {pyo.value(model.objective):.4f}")
            case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                printer.error(f"Model returned as {results.solver.termination_condition}, logging infeasible constraints:")
                log_infeasible_constraints(model, log_expression=False)
            case _:
                printer.warning(f"Solver terminated with condition: {results.solver.termination_condition}")

        SQLiteWriter.model_to_sqlite(model, sqlite_filename)
        SQLiteWriter.add_solver_statistics_to_sqlite(sqlite_filename, results, work_units=lego.work_units)
        SQLiteWriter.add_run_parameters_to_sqlite(
            sqlite_filename,
            case_study_directory=case_study_directory,
            zoi=current_zoi,
            limit_k=limit_k,
            dc_buffer=dc_buffer,
            tp_buffer=tp_buffer,
            scale_demand=scale_demand,
            scale_pmax=scale_pmax
        )
        printer.information(f"Saved LEGO model to '{sqlite_filename}'")


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
    args = parser.parse_args()

    main(args.caseStudyDirectory, args.zoi, args.limitK, args.dcBuffer, args.tpBuffer, args.scaleDemand, args.scalePMax, args.all, args.noOverwrite)
