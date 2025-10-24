import argparse
import logging
import os
import time

import pyomo.environ as pyo
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from pyomo.core import NameLabeler
from pyomo.core.base.var import IndexedVar
from rich_argparse import RichHelpFormatter

from InOutModule import SQLiteWriter, ExcelWriter
from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO, ModelType
from LEGO.LEGOUtilities import analyze_infeasible_constraints

printer = Printer.getInstance()

# Set up logging so that infeasible constraints are logged by pyomo
logger = logging.getLogger("pyomo")
logger.setLevel("INFO")

# Parse command line arguments and automatically check for correct usage
parser = argparse.ArgumentParser(description="Starts LEGO for given case study", formatter_class=RichHelpFormatter)


# Check if given string path is a directory
def directory_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise argparse.ArgumentTypeError(f"Directory path not valid: '{string}'")


def process_results(model_results):
    logger = logging.getLogger('pyomo.util.infeasible')
    logger.setLevel(logging.INFO)

    # Ensure there is a handler attached
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)

    match model_results.solver.termination_condition:
        case pyo.TerminationCondition.optimal:
            match args.modelType:
                case ModelType.DETERMINISTIC:
                    printer.success(f"Optimal solution: {pyo.value(model.objective):.4f}\n")
                case ModelType.EXTENSIVE_FORM:
                    printer.success(f"Optimal solution: {lego._extensive_form.get_objective_value():.4f}\n")
                case _:
                    printer.warning(f"Model type {args.modelType} not fully tested yet, no objective value reported.\n")
        case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
            printer.error(f"Model returned as {model_results.solver.termination_condition}")
            analyze_infeasible_constraints(model)
        case _:
            printer.warning(f"Solver terminated with condition: {model_results.solver.termination_condition}")


parser.add_argument("caseStudyDirectory", type=directory_path, help="Path to folder containing data for LEGO model")
parser.add_argument("modelType", default=ModelType.DETERMINISTIC, type=lambda s: ModelType[s], choices=list(ModelType), nargs="?", help="ModelType of first model")
args = parser.parse_args()

# Load case study
printer.information(f"Loading case study from '{args.caseStudyDirectory}'\n")
start_time = time.time()
cs = CaseStudy(args.caseStudyDirectory)

rh_length = cs.dGlobal_Parameters["pMovingWindowLength"]
rh_overlap = cs.dGlobal_Parameters["pMovingWindowOverlap"]

# Check if moving window is disabled (both parameters are 0)
use_moving_window = rh_length > 0 and rh_overlap >= 0

if not use_moving_window:
    printer.information("Moving window disabled - running entire problem at once\n")

    lego = LEGO(cs)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    # Build LEGO model
    printer.information("Building LEGO model")
    model, timing = lego.build_model(model_type=args.modelType)
    printer.information(f"Building LEGO model took {timing:.2f} seconds")

    # Solve LEGO model
    printer.information("Solving LEGO model")
    try:
        results, timing, objective_value = lego.solve_model(model_type=args.modelType)
        printer.information(f"Solving LEGO model took {timing:.2f} seconds\n")
        process_results(results)
    except NoFeasibleSolutionError:
        printer.error("No feasible solution found!")
        analyze_infeasible_constraints(model)
        exit(1)

else:
    printer.information(f"Using moving window: length={rh_length}, overlap={rh_overlap}\n")

    model_old = None
    total_timesteps = len(cs.dPower_WeightsK.index.unique())
    k_padding = len(cs.dPower_WeightsK.index.unique()[0]) - 1

    start_timestep = 1
    while start_timestep <= total_timesteps:
        start_time_iteration = time.time()

        # Calculate the end of the window
        end_timestep = min(start_timestep + rh_length - 1, total_timesteps)

        # Format timestep strings for filtering and printing
        start_k = f"k{start_timestep:0{k_padding}}"
        end_k = f"k{end_timestep:0{k_padding}}"
        print(f"Start k: {start_k}, End k: {end_k}")

        cs.constraints_active_k = [f"k{i:0{k_padding}}" for i in range(start_timestep, end_timestep + 1)]
        printer.information(f"Processing window from {start_k} to {end_k}...")

        cut_cs = cs.filter_timesteps(cs.dPower_WeightsK.index.unique()[0], end_k)

        lego = LEGO(cut_cs)
        printer.information(f"Loading case study took {time.time() - start_time_iteration:.2f} seconds")

        # Build LEGO model
        printer.information("Building LEGO model")
        model, timing = lego.build_model(model_type=args.modelType)

        printer.information(f"Building LEGO model took {timing:.2f} seconds")

        if model_old is not None:
            new_end = f"k{start_timestep-1:05}"
            print(f"New end: {new_end}")
            for component in list(model_old.component_objects()):
                if isinstance(component, IndexedVar):
                    indices = [str(i) for i in component.index_set().subsets()]

                    if "k" in indices:
                        new_component = getattr(model, str(component))
                        for n, v in list(component.items()):
                            if n[(indices.index('k'))] <= new_end:
                                if v.value is not None:
                                    new_component[n].fix(pyo.value(v))  # TODO skip validation

        # Solve LEGO model
        printer.information("Solving LEGO model")
        try:
            results, timing, objective_value = lego.solve_model(model_type=args.modelType)
            printer.information(f"Solving LEGO model took {timing:.2f} seconds")
            process_results(results)
        except NoFeasibleSolutionError:
            printer.error(f"No feasible solution found for window {start_k} to {end_k}!")
            analyze_infeasible_constraints(model)
            exit(1)

        if start_timestep + rh_length >= total_timesteps:
            break
        model_old = model
        start_timestep += rh_length - rh_overlap

printer.information(f"Finished in {time.time() - start_time:.2f} seconds")

SQLiteWriter.model_to_sqlite(model, "model.sqlite")
ExcelWriter.model_to_excel(model, "model.xlsx")
model.write("model.mps", io_options={'labeler': NameLabeler()})
