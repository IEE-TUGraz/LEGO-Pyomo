import argparse
import os
import time
import logging

import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from rich_argparse import RichHelpFormatter

from InOutModule.CaseStudy import CaseStudy
from LEGO.LEGO import LEGO
from tools.printer import Printer

printer = Printer.getInstance()

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
#lego.cs.dPower_Demand = lego.cs.dPower_Demand["k" >= 24]
model, timing = lego.build_model()
# fix solved variables
printer.information(f"Building LEGO model took {timing:.2f} seconds")

# Solve LEGO model
printer.information("Solving LEGO model")
results, timing = lego.solve_model()
printer.information(f"Solving LEGO model took {timing:.2f} seconds")

logger = logging.getLogger('pyomo.util.infeasible')
logger.setLevel(logging.INFO)

# Ensure there is a handler attached
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

match results.solver.termination_condition:
    case pyo.TerminationCondition.optimal:
        printer.success(f"Optimal solution: {pyo.value(model.objective):.4f}")
    case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
        printer.error(f"Model returned as {results.solver.termination_condition}, logging infeasible constraints:")
        log_infeasible_constraints(model, log_expression= False)
    case _:
        printer.warning(f"Solver terminated with condition: {results.solver.termination_condition}")
