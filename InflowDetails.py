import argparse
import logging
import os
import time

import pyomo.environ as pyo
from pyomo.core import NameLabeler
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

# Parse command line arguments and automatically check for correct usage
parser = argparse.ArgumentParser(description="Shows difference of detailed and averaged inflow data for given case study", formatter_class=RichHelpFormatter)


# Check if given string path is a directory
def directory_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise argparse.ArgumentTypeError(f"Directory path not valid: '{string}'")


parser.add_argument("caseStudyDirectory", type=directory_path, help="Path to folder containing data for LEGO model")
parser.add_argument("--numberOfRPs", type=int, help="Number of representative periods to cluster data into", default=1)
parser.add_argument("--lengthOfRPs", type=int, help="Length of representative periods (in number of time steps)", default=24)
parser.add_argument("--scaleDemand", type=float, default=1.0, help="Scaling factor for demand (default: 1.0 = no scaling)")
args = parser.parse_args()

caseStudyName = args.caseStudyDirectory.replace("/", "_").replace("\\", "_")

if args.numberOfRPs < 1:
    printer.error("numberOfRPs must be at least 1")
    exit(1)
if args.lengthOfRPs < 1:
    printer.error("lengthOfRPs must be at least 1")
    exit(1)

printer.information(f"Loading original case study from '{args.caseStudyDirectory}'")
start_time = time.time()
cs_inflow_detailed = CaseStudy(args.caseStudyDirectory)
printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

if args.scaleDemand != 1.0:
    printer.information(f"Scaling demand by factor {args.scaleDemand}")
    cs_inflow_detailed.dPower_Demand['value'] *= args.scaleDemand

printer.information("Creating copy of case study with simplified inflow data")
cs_inflow_simplified = cs_inflow_detailed.copy()
mean_inflows = cs_inflow_simplified.dPower_Inflows["value"].groupby(["rp", "g"]).mean()  # Calculate mean inflow per representative and generator
cs_inflow_simplified.dPower_Inflows.loc[:, "value"] = mean_inflows[cs_inflow_simplified.dPower_Inflows.index.droplevel("k")].values  # Set all inflows to mean inflow of respective representative and generator
printer.information("Copy of case study with simplified inflow data created")

if args.numberOfRPs > 1:
    printer.information(f"Clustering data into {args.numberOfRPs} representative periods of length {args.lengthOfRPs}")
    cs_inflow_detailed.apply_kmedoids_aggregation(args.numberOfRPs, args.lengthOfRPs)
    printer.information("Clustering completed for detailed case study, now clustering simplified case study")
    cs_inflow_simplified.apply_kmedoids_aggregation(args.numberOfRPs, args.lengthOfRPs)
    printer.information("Clustering of simplified case study completed")
else:  # args.numberOfRPs < 1 already handled when parsing arguments
    printer.information("Skipping clustering of case studies since numberOfRPs == 1")

printer.information("Building LEGO models")
lego_detailed = LEGO(cs_inflow_detailed)
model_inflow_detailed, timing = lego_detailed.build_model()
printer.information(f"Building detailed LEGO model took {timing:.2f} seconds")
lego_simplified = LEGO(cs_inflow_simplified)
model_inflow_simplified, timing = lego_simplified.build_model()
printer.information(f"Building simplified LEGO model took {timing:.2f} seconds")

# Solve LEGO model
printer.information("Solving LEGO models")
results_detailed, timing, objective_value_detailed = lego_detailed.solve_model()
printer.information(f"Solving detailed LEGO model took {timing:.2f} seconds")

match results_detailed.solver.termination_condition:
    case pyo.TerminationCondition.optimal:
        printer.success(f"Optimal solution: {pyo.value(model_inflow_detailed.objective):.4f}")
    case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
        printer.error(f"Model returned as {results_detailed.solver.termination_condition}, logging infeasible constraints:")
        log_infeasible_constraints(model_inflow_detailed, log_expression=False)
    case _:
        printer.warning(f"Solver terminated with condition: {results_detailed.solver.termination_condition}")

SQLiteWriter.model_to_sqlite(model_inflow_detailed, f"model_detailed-{caseStudyName}-rps{args.numberOfRPs}-ks{args.lengthOfRPs}-demand{args.scaleDemand}.sqlite")
model_inflow_detailed.write(f"model_detailed-{caseStudyName}-rps{args.numberOfRPs}-ks{args.lengthOfRPs}-demand{args.scaleDemand}.mps", io_options={'labeler': NameLabeler()})

results_simplified, timing, objective_value_simplified = lego_simplified.solve_model()
printer.information(f"Solving simplified LEGO model took {timing:.2f} seconds")

match results_simplified.solver.termination_condition:
    case pyo.TerminationCondition.optimal:
        printer.success(f"Optimal solution: {pyo.value(model_inflow_simplified.objective):.4f}")
    case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
        printer.error(f"Model returned as {results_simplified.solver.termination_condition}, logging infeasible constraints:")
        log_infeasible_constraints(model_inflow_simplified, log_expression=False)
    case _:
        printer.warning(f"Solver terminated with condition: {results_simplified.solver.termination_condition}")

SQLiteWriter.model_to_sqlite(model_inflow_simplified, f"model_simplified-{caseStudyName}-rps{args.numberOfRPs}-ks{args.lengthOfRPs}-demand{args.scaleDemand}.sqlite")
model_inflow_simplified.write(f"model_simplified-{caseStudyName}-rps{args.numberOfRPs}-ks{args.lengthOfRPs}-demand{args.scaleDemand}.mps", io_options={'labeler': NameLabeler()})

logger = logging.getLogger('pyomo.util.infeasible')
logger.setLevel(logging.INFO)
