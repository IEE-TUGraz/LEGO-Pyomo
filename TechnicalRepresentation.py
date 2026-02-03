import argparse
import logging
import time
from collections import deque

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


def main(case_study_directory, zoi, limit_k, dc_buffer, tp_buffer, scale_demand, scale_pmax):
    caseStudyName = case_study_directory.replace("/", "_").replace("\\", "_")

    # Build identifier with only non-default parameters
    identifier_parts = [f"data{caseStudyName}"]

    printer.information(f"Loading case study from '{case_study_directory}'")
    start_time = time.time()
    cs = CaseStudy(case_study_directory)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    if limit_k is not None:
        printer.information(f"Limiting K values to '{limit_k}'")
        identifier_parts.append(f"limitK{limit_k}")
        start, end = limit_k.split("-")
        cs.filter_timesteps(start, end, inplace=True)

    # Check if zone is uniform representation (where buffers are unused)
    is_uniform_repr = zoi in ['TP', 'SN'] or zoi is None or zoi == 'None'

    if scale_demand != SCALE_DEFAULT:
        printer.information(f"Scaling demand by factor {scale_demand}")
        identifier_parts.append(f"demand{scale_demand:.1f}")
        cs.dPower_Demand['value'] *= scale_demand

    if scale_pmax != SCALE_DEFAULT:
        printer.information(f"Scaling pPmax (line capacity) by factor {scale_pmax}")
        identifier_parts.append(f"pmax{scale_pmax:.1f}")
        cs.dPower_Network['pPmax'] *= scale_pmax

    if dc_buffer != DC_BUFFER_DEFAULT:
        if is_uniform_repr:
            printer.warning("DC buffer specified but using uniform technical representation - DC buffer will be ignored")
        else:
            identifier_parts.append(f"dcBuffer{dc_buffer}")

    if tp_buffer != TP_BUFFER_DEFAULT:
        if is_uniform_repr:
            printer.warning("TP buffer specified but using uniform technical representation - TP buffer will be ignored")
        else:
            identifier_parts.append(f"tpBuffer{tp_buffer}")

    if zoi is not None:
        printer.information(f"Setting Zone of Interest (zoi) to zone '{zoi}'")
        identifier_parts.append(f"zoi{zoi}")
        cs.dPower_BusInfo['zoi'] = 0
        cs.dPower_BusInfo.loc[cs.dPower_BusInfo['z'] == zoi, 'zoi'] = 1

        # Check if any buses were assigned to ZOI
        num_zoi_buses = (cs.dPower_BusInfo['zoi'] == 1).sum()
        if num_zoi_buses == 0:
            available_zones = cs.dPower_BusInfo['z'].unique().tolist()
            printer.warning(f"0 buses selected for zone '{zoi}'. Available zones: {available_zones}")
    else:
        printer.warning("No Zone of Interest (zoi) specified, proceeding with original setting from Power_BusInfo")

    # Check for special zone names that apply uniform technical representation
    if zoi == 'TP':
        printer.information(f"Zone 'TP' specified: Setting all lines to Transport Model (TP)")
        cs.dPower_Network['pTecRepr'] = 'TP'
    elif zoi == 'SN':
        printer.information(f"Zone 'SN' specified: Setting all lines to Single Node (SN)")
        cs.dPower_Network['pTecRepr'] = 'SN'
    else:
        # Normal layer-based algorithm (including zoi='None' which gives all DC-OPF)
        printer.information(f"Assigning technical representations with DC-Buffer={dc_buffer}, TP-Buffer={tp_buffer}")
        assign_technical_representation_by_layers(cs, dc_buffer, tp_buffer)

    printer.information(f"Setting parameters so that it will be solved as rMIP")
    cs.dGlobal_Parameters["pEnableRMIP"] = True

    printer.information(f"Removing fixed slack node so that it is calculated based on demand")
    cs.dPower_Parameters["is"] = None

    cs.merge_single_node_buses()

    printer.information("Building LEGO model")
    legos = {}
    for name, cs in [(zoi, cs)]:
        printer.information(f"Building LEGO model for case study with '{name}' as zoi")
        lego = LEGO(cs)
        model, timing = lego.build_model()
        printer.information(f"Building LEGO model for case study with '{name}' as zoi took {timing:.2f} seconds")
        legos[name] = (lego, model)

    printer.information("Solving LEGO model(s)")
    for name, (lego, model) in legos.items():
        printer.information(f"Solving LEGO model for case study with '{name}' as zoi")
        results, timing, objective_value = lego.solve_model()
        printer.information(f"Solving LEGO model for case study with '{name}' as zoi took {timing:.2f} seconds")

        match results.solver.termination_condition:
            case pyo.TerminationCondition.optimal:
                printer.success(f"Optimal solution: {pyo.value(model.objective):.4f}")
            case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                printer.error(f"Model returned as {results.solver.termination_condition}, logging infeasible constraints:")
                log_infeasible_constraints(model, log_expression=False)
            case _:
                printer.warning(f"Solver terminated with condition: {results.solver.termination_condition}")

        identifier = "-".join(identifier_parts)

        sqlite_filename = f"TR-{identifier}.sqlite"
        SQLiteWriter.model_to_sqlite(model, sqlite_filename)
        SQLiteWriter.add_solver_statistics_to_sqlite(sqlite_filename, results, work_units=lego.work_units)
        SQLiteWriter.add_run_parameters_to_sqlite(
            sqlite_filename,
            case_study_directory=case_study_directory,
            zoi=zoi,
            limit_k=limit_k,
            dc_buffer=dc_buffer,
            tp_buffer=tp_buffer,
            scale_demand=scale_demand,
            scale_pmax=scale_pmax
        )
        printer.information(f"Saved LEGO model to '{sqlite_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests different technical representations for network", formatter_class=RichHelpFormatter)

    parser.add_argument("caseStudyDirectory", type=str, help="Path to folder containing data for LEGO model")
    parser.add_argument("--zoi", type=str, help="Which Zone (from Power_BusInfo 'z') should be the Zone of Interest ('zoi')? Special values: 'TP' or 'SN' to set all lines uniformly to that technical representation. Use 'None' for uniform DC-OPF (baseline for comparisons).", nargs="?", default=None)
    parser.add_argument("--limitK", type=str, help="Limit the ks, format: 'k0025-k0048'", nargs="?", default=None)
    parser.add_argument("--dcBuffer", type=int, help=f"Number of network layers outside ZOI to assign as DC-OPF (default: {DC_BUFFER_DEFAULT})", nargs="?", default=DC_BUFFER_DEFAULT)
    parser.add_argument("--tpBuffer", type=int, help=f"Number of network layers after DC buffer to assign as TP (default: {TP_BUFFER_DEFAULT})", nargs="?", default=TP_BUFFER_DEFAULT)
    parser.add_argument("--scaleDemand", type=float, help=f"Scaling factor for demand (default: {SCALE_DEFAULT} = no scaling)", nargs="?", default=SCALE_DEFAULT)
    parser.add_argument("--scalePMax", type=float, help=f"Scaling factor for pPmax (line capacity) (default: {SCALE_DEFAULT} = no scaling)", nargs="?", default=SCALE_DEFAULT)
    args = parser.parse_args()

    main(args.caseStudyDirectory, args.zoi, args.limitK, args.dcBuffer, args.tpBuffer, args.scaleDemand, args.scalePMax)
