import argparse
import logging
import time

import pyomo.environ as pyo
from pyomo.core import NameLabeler
from pyomo.util.infeasible import log_infeasible_constraints
from rich_argparse import RichHelpFormatter

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO
from tools.mpsCompare import compare_mps

########################################################################################################################
# Setup
########################################################################################################################

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)
printer = Printer.getInstance()


def modelSelection(string: str) -> str:
    if string.endswith(".mps"):
        return string
    elif string in ["gams", "pyomoSimpleSolve", "pyomoSimpleNoSolve", "pyomoExtensive", "pyomoBenders", "pyomoProgressiveHedging"]:
        return string
    else:
        raise argparse.ArgumentTypeError(f"Model selection not valid: '{string}'. Options are 'gams', 'pyomoSimpleSolve', 'pyomoSimpleNoSolve', 'pyomoExtensive', 'pyomoBenders', 'pyomoProgressiveHedging' or a path to a .mps file")


parser = argparse.ArgumentParser(description="Compares two given models using the MPS format", formatter_class=RichHelpFormatter)
parser.add_argument("scenarioFolder", default="data/exampleStochastic", type=str, nargs="?", help="Path to folder containing data for LEGO model")
parser.add_argument("model1", default="pyomoSimpleSolve", type=modelSelection, nargs="?", help="Path to first model to compare (can be a .mps file or 'gams', 'pyomoSimpleSolve', 'pyomoSimpleNoSolve', 'pyomoExtensive', 'pyomoBenders' or 'pyomoProgressiveHedging')")
parser.add_argument("model2", default="data/mps-archive/model-feac8422246633f43b3c98cf402798ca07a7109b.mps", type=modelSelection, nargs="?", help="Path to second model to compare (can be a .mps file or 'gams', 'pyomoSimpleSolve', 'pyomoSimpleNoSolve', 'pyomoExtensive', 'pyomoBenders' or 'pyomoProgressiveHedging')")

parser.add_argument("--constraintSkipModel1", default=[], nargs='+', help="Constraints to skip from model 1 (either fill this or constraintKeepModel1)")
parser.add_argument("--constraintKeepModel1", default=[], nargs='+', help="Constraints to keep from model 1 (either fill this or constraintSkipModel1)")
parser.add_argument("--coefficientsSkipModel1", default=[], nargs='+', help="Coefficients to skip from model 1")

parser.add_argument("--constraintSkipModel2", default=[], nargs='+', help="Constraints to skip from model 2 (either fill this or constraintKeepModel2)")
parser.add_argument("--constraintKeepModel2", default=[], nargs='+', help="Constraints to keep from model 2 (either fill this or constraintSkipModel2)")
parser.add_argument("--coefficientsSkipModel2", default=[], nargs='+', help="Coefficients to skip from model 2")
parser.add_argument("--constraintEnforceModel2", default=[""], nargs='+', help="Constraints to enforce from model 2 (default is to enforce all)")

parser.add_argument("--skipComparisonOverall", action="store_true", help="Skip comparison of MPS files overall")
parser.add_argument("--skipVariableComparison", action="store_true", help="Skip comparison of variables")
parser.add_argument("--skipConstraintComparison", action="store_true", help="Skip comparison of constraints")
parser.add_argument("--printAdditionalInformation", action="store_true", help="Print additional information")

parser.add_argument("--gamsConsoleLogPath", default="LEGO-GAMS/gams_console.log", type=str, help="Path to GAMS console log file")
parser.add_argument("--gamsPath", default="C:/GAMS/49/gams.exe", type=str, help="Path to GAMS executable")
parser.add_argument("--legoGamsPath", default="LEGO.gms", type=str, help="Path to LEGO GAMS model")
parser.add_argument("--maxGamsRuntimeInSeconds", default=60, type=int, help="Maximum runtime in seconds for GAMS model")

args = parser.parse_args()


def execute_gams(args: argparse.Namespace) -> (str, float):
    import subprocess
    import psutil

    with open(args.gamsConsoleLogPath, "w") as GAMSConsoleLogFile:

        # Create subprocess to execute LEGO model
        # Executing with argument string instead of list since GAMS has problems with double-quotes
        printer.information(f"Starting LEGO-GAMS with scenario folder \"{args.scenarioFolder}\"")
        start_time = time.time()
        lego_process = subprocess.Popen(f"cd LEGO-GAMS && {args.gamsPath} {args.legoGamsPath} --scenarioFolder=\"../{args.scenarioFolder}\"",
                                        stdout=GAMSConsoleLogFile, stderr=subprocess.STDOUT, shell=True)
        try:
            return_value = lego_process.wait(args.maxGamsRuntimeInSeconds)
            stop_time = time.time()

        except subprocess.TimeoutExpired:  # If it exceeds args.maxGamsRuntimeInSeconds, kill it incl. all child processes
            child_processes = psutil.Process(lego_process.pid).children(recursive=True)
            for child in child_processes:
                child.kill()
            printer.error(f"Runtime exceeded {args.maxGamsRuntimeInSeconds} seconds, killing (all) LEGO process(es)")
            gone, still_alive = psutil.wait_procs(child_processes, timeout=5)
            printer.information(f"Status child processes:\n{gone}\n{still_alive}")
            lego_process.kill()

            exit(-1)

    if return_value != 0:
        printer.error(f"Return value of process is {return_value} - please check log files")
        return None, -1
    else:
        timing = stop_time - start_time
        printer.information(f"Executing GAMS took {timing:.2f} seconds")

        with open("LEGO-Gams/gams_console.log", "r") as file:
            for line in file:
                if "Objective:" in line:
                    objective_value_gams = float(line.split()[-1])
                    printer.information(f"Objective value: {objective_value_gams}")
                    return "LEGO-GAMS/model.mps", objective_value_gams
    return "LEGO-GAMS/model.mps", -1


def build_pyomo_simple(args: argparse.Namespace) -> (str, LEGO):
    cs = CaseStudy(args.scenarioFolder, do_not_merge_single_node_buses=True)
    mps_file_path = f"pyomoSimple-{time.strftime('%y%m%d-%H%M%S')}.mps"

    lego = LEGO(cs.filter_scenario("ScenarioA"))
    model, timing = lego.build_model()
    printer.information(f"Building LEGO model took {timing:.2f} seconds")
    model.write(mps_file_path, io_options={'labeler': NameLabeler()})

    return mps_file_path, lego


def solve_pyomo_simple(lego: LEGO, args: argparse.Namespace) -> float:
    results, timing = lego.solve_model()
    match results.solver.termination_condition:
        case pyo.TerminationCondition.optimal:
            printer.information(f"Optimal solution found after {timing:.2f} seconds")
            printer.information(f"Objective value Pyomo: {pyo.value(lego.model.objective):.4f}")
            return pyo.value(lego.model.objective)
        case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
            printer.error(f"ERROR: Model is {results.solver.termination_condition}, logging infeasible constraints:")
            log_infeasible_constraints(lego.model)
            return -1
        case _:
            printer.error("Solver terminated with condition:", results.solver.termination_condition)
            return -1


def execute_extensive_form(args: argparse.Namespace) -> (str, float):
    cs = CaseStudy(args.scenarioFolder, do_not_merge_single_node_buses=True)
    mps_file_path = f"pyomoExtensiveForm-{time.strftime('%y%m%d-%H%M%S')}.mps"
    lego = LEGO(cs)

    model, timing, obj_value = lego.execute_extensive_form()
    printer.information(f"Executing extensive form took {timing:.2f} seconds")
    model.write(mps_file_path, io_options={'labeler': NameLabeler()})

    return mps_file_path, obj_value


def execute_benders(args: argparse.Namespace) -> (str, float):
    cs = CaseStudy(args.scenarioFolder, do_not_merge_single_node_buses=True)
    # mps_file_path = f"pyomoExtensiveForm-{time.strftime('%y%m%d-%H%M%S')}.mps"
    lego = LEGO(cs)

    model, timing, obj_value = lego.execute_benders()
    printer.information(f"Executing Benders decomposition took {timing:.2f} seconds")
    # model.write(mps_file_path, io_options={'labeler': NameLabeler()}) # TODO: Check how to write MPS file for Benders decomposition

    return None, obj_value


def execute_progressive_hedging(args: argparse.Namespace) -> (str, float):
    cs = CaseStudy(args.scenarioFolder, do_not_merge_single_node_buses=True)
    # mps_file_path = f"pyomoExtensiveForm-{time.strftime('%y%m%d-%H%M%S')}.mps"
    lego = LEGO(cs)

    model, timing, obj_val = lego.execute_progressive_hedging()
    printer.information(f"Executing progressive hedging took {timing:.2f} seconds")
    # model.write(mps_file_path, io_options={'labeler': NameLabeler()}) # TODO: Check how to write MPS file for Progressive Hedging

    return None, obj_val


########################################################################################################################
# Re-run with GAMS
########################################################################################################################

match args.model1:
    case "gams":
        model1_path, objective_value_1 = execute_gams(args)
    case "pyomoSimpleSolve" | "pyomoSimpleNoSolve":
        model1_path, lego = build_pyomo_simple(args)
        objective_value_1 = solve_pyomo_simple(lego, args) if args.model1 == "pyomoSimpleSolve" else -1
    case "pyomoExtensive":
        model1_path, objective_value_1 = execute_extensive_form(args)
    case "pyomoBenders":
        model1_path, objective_value_1 = execute_benders(args)
    case "pyomoProgressiveHedging":
        model1_path, objective_value_1 = execute_progressive_hedging(args)
    case s if s.endswith(".mps"):
        model1_path = args.model1
        objective_value_1 = -1
    case _:
        raise ValueError(f"Selection for Model 1 is not valid: '{args.model1}'")

match args.model2:
    case "gams":
        model2_path, objective_value_2 = execute_gams(args)
    case "pyomoSimpleSolve" | "pyomoSimpleNoSolve":
        model2_path, lego = build_pyomo_simple(args)
        objective_value_2 = solve_pyomo_simple(lego, args) if args.model2 == "pyomoSimpleSolve" else -1
    case "pyomoExtensive":
        model2_path, objective_value_2 = execute_extensive_form(args)
    case "pyomoBenders":
        model2_path, objective_value_2 = execute_benders(args)
    case "pyomoProgressiveHedging":
        model2_path, objective_value_2 = execute_progressive_hedging(args)
    case s if s.endswith(".mps"):
        model2_path = args.model2
        objective_value_2 = -1
    case _:
        raise ValueError(f"Selection for Model 2 is not valid: '{args.model2}'")

printer.information(f"Model 1: {args.model1} - {model1_path} - Objective value: {objective_value_1}")
printer.information(f"Model 2: {args.model2} - {model2_path} - Objective value: {objective_value_2}")

if objective_value_1 != -1 and objective_value_2 != -1:
    printer.information(f"Objective difference : {objective_value_1 - objective_value_2:.2f} | {100 * (objective_value_1 - objective_value_2) / objective_value_2:.2f}%")

if not args.skipComparisonOverall:
    if model1_path is not None and model2_path is not None:
        compare_mps(model1_path, args.model1 != "gams", model2_path, args.model2 != "gams", check_vars=not args.skipVariableComparison, check_constraints=not args.skipConstraintComparison, print_additional_information=args.printAdditionalInformation,
                    constraints_to_skip_from1=args.constraintSkipModel1, constraints_to_keep_from1=args.constraintKeepModel1, coefficients_to_skip_from1=args.coefficientsSkipModel1,
                    constraints_to_skip_from2=args.constraintSkipModel2, constraints_to_keep_from2=args.constraintKeepModel2, coefficients_to_skip_from2=args.coefficientsSkipModel2, constraints_to_enforce_from2=args.constraintEnforceModel2)
    else:
        if model1_path is None:
            printer.error(f"Model 1 path is None, cannot compare models")
        if model2_path is None:
            printer.error(f"Model 2 path is None, cannot compare models")
        printer.error("Skipping comparison")

print("Done")
