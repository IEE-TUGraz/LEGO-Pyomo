import argparse
import logging
import os
import time

import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from rich_argparse import RichHelpFormatter

from InOutModule import SQLiteWriter, ExcelWriter
from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO

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


parser.add_argument("caseStudyDirectory", type=directory_path, help="Path to folder containing data for LEGO model")
args = parser.parse_args()

# Load case study
printer.information(f"Loading case study from '{args.caseStudyDirectory}'")
start_time = time.time()
cs = CaseStudy(args.caseStudyDirectory)
lego = LEGO(cs)
printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

# Build LEGO model
printer.information("Building LEGO model")
model, timing = lego.build_model()
printer.information(f"Building LEGO model took {timing:.2f} seconds")

# Solve LEGO model
printer.information("Solving LEGO model")
results, timing, objective_value = lego.solve_model()
printer.information(f"Solving LEGO model took {timing:.2f} seconds")

#model.vLinkExpPower.pprint()

match results.solver.termination_condition:
    case pyo.TerminationCondition.optimal:
        printer.success(f"Optimal solution: {pyo.value(model.objective):.4f}")
    case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
        printer.error(f"Model returned as {results.solver.termination_condition}, logging infeasible constraints:")
        log_infeasible_constraints(model)
    case _:
        printer.warning(f"Solver terminated with condition: {results.solver.termination_condition}")

# SQLiteWriter.model_to_sqlite(model, "model.sqlite")
ExcelWriter.model_to_excel(model, "model.xlsx")
