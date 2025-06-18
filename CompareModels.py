import argparse
import enum
import logging
import pathlib
import time
from typing import Optional

import pyomo.environ as pyo
from pyomo.core import NameLabeler
from pyomo.util.infeasible import log_infeasible_constraints
from rich_argparse import RichHelpFormatter

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO
from LEGO.LEGO import ModelType as LEGOModelType
from tools.mpsCompare import compare_mps

########################################################################################################################
# Setup
########################################################################################################################

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)
printer = Printer.getInstance()


class ModelTypeForComparison(enum.Enum):
    """Enum for model types used in comparison. Similar to LEGO.ModelType, but adds options for GAMS and .mps-files."""
    DETERMINISTIC = "deterministic"
    EXTENSIVE_FORM = "extensive_form"
    BENDERS = "benders"
    PROGRESSIVE_HEDGING = "progressive_hedging"
    GAMS = "gams"
    MPS_FILE = "mps"


def modelSelection(string: str) -> str:
    if string.endswith(".mps"):
        return string
    elif string in ["gams", "pyomoSimpleSolve", "pyomoSimpleNoSolve", "pyomoExtensive", "pyomoBenders", "pyomoProgressiveHedging"]:
        return string
    else:
        raise argparse.ArgumentTypeError(f"Model selection not valid: '{string}'. Options are 'gams', 'pyomoSimpleSolve', 'pyomoSimpleNoSolve', 'pyomoExtensive', 'pyomoBenders', 'pyomoProgressiveHedging' or a path to a .mps file")


def execute_gams(data_folder: str, gams_console_log_path: str, gams_executable_path: str, lego_gams_path: str, max_gams_runtime_in_seconds: int) -> (str, float):
    """
    Executes the GAMS model with the given parameters and returns the path to the MPS file and the objective value.
    :param data_folder: Folder containing the data for the GAMS model.
    :param gams_console_log_path: Path to the GAMS console log file.
    :param gams_executable_path: Path to the GAMS executable.
    :param lego_gams_path: Path to the LEGO GAMS model file.
    :param max_gams_runtime_in_seconds: Maximum runtime in seconds for the GAMS model.
    :return: Tuple containing the path to the MPS file and the objective value.
    """
    import subprocess
    import psutil

    with open(gams_console_log_path, "w") as GAMSConsoleLogFile:

        # Create subprocess to execute LEGO model
        # Executing with argument string instead of list since GAMS has problems with double-quotes
        printer.information(f"Starting LEGO-GAMS with scenario folder \"{data_folder}\"")
        start_time = time.time()
        lego_process = subprocess.Popen(f"cd LEGO-GAMS && {gams_executable_path} {lego_gams_path} --scenarioFolder=\"../{data_folder}\"",
                                        stdout=GAMSConsoleLogFile, stderr=subprocess.STDOUT, shell=True)
        try:
            return_value = lego_process.wait(max_gams_runtime_in_seconds)
            stop_time = time.time()

        except subprocess.TimeoutExpired:  # If it exceeds max_gams_runtime_in_seconds, kill it incl. all child processes
            child_processes = psutil.Process(lego_process.pid).children(recursive=True)
            for child in child_processes:
                child.kill()
            printer.error(f"Runtime exceeded {max_gams_runtime_in_seconds} seconds, killing (all) LEGO process(es)")
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


def build_and_solve_model(model_type: ModelTypeForComparison, data_path: str | pathlib.Path, solve_model: bool,
                          gams_console_log_path: Optional[str] = None, gams_executable_path: Optional[str] = None, lego_gams_path: Optional[str] = None, max_gams_runtime_in_seconds: Optional[int] = None) -> (str, float):
    """
    Build and solve a model based on the given model type and data path. Returns the path to the MPS file and the objective value.
    :param model_type: Model type to be used for building the model. Can be one of ModelTypeForComparison.
    :param data_path: Path to the folder containing data for the model or to a .mps file if model_type is MPS_FILE.
    :param gams_console_log_path: Path to the GAMS console log file. Required if model_type is GAMS.
    :param gams_executable_path: Path to the GAMS executable. Required if model_type is GAMS.
    :param lego_gams_path: Path to the LEGO GAMS model file. Required if model_type is GAMS.
    :param max_gams_runtime_in_seconds: Maximum runtime in seconds for the GAMS model. Required if model_type is GAMS.
    :return: Tuple containing the path to the MPS file and the objective value.
    """
    match model_type:
        case ModelTypeForComparison.GAMS:
            mps_path, objective_value = execute_gams(data_path, gams_console_log_path, gams_executable_path, lego_gams_path, max_gams_runtime_in_seconds)
        case ModelTypeForComparison.DETERMINISTIC | ModelTypeForComparison.EXTENSIVE_FORM | ModelTypeForComparison.BENDERS | ModelTypeForComparison.PROGRESSIVE_HEDGING:
            mps_path = f"pyomo-{model_type}-{time.strftime('%y%m%d-%H%M%S')}.mps" if model_type in [ModelTypeForComparison.DETERMINISTIC, ModelTypeForComparison.EXTENSIVE_FORM] else None
            cs = CaseStudy(data_path, do_not_merge_single_node_buses=True)
            lego = LEGO(cs)
            model, timing = lego.build_model()
            printer.information(f"Building LEGO model took {timing:.2f} seconds")

            # Write MPS file
            if mps_path is not None:
                printer.information(f"Writing MPS file to {mps_path}")
                model.write(mps_path, io_options={'labeler': NameLabeler()})
            else:
                printer.information("No MPS file will be written")  # TODO: Check how to write MPS file for non-deterministic models

            # Solve model if requested
            if solve_model:
                results, timing, objective_value = lego.solve_model(LEGOModelType(model_type.value))
                match results.solver.termination_condition:
                    case pyo.TerminationCondition.optimal:
                        printer.information(f"Optimal solution found after {timing:.2f} seconds")
                        printer.information(f"Objective value Pyomo: {objective_value:.4f}")
                    case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                        printer.error(f"ERROR: Model is {results.solver.termination_condition}, logging infeasible constraints:")
                        log_infeasible_constraints(lego.model)
                    case _:
                        printer.error("Solver terminated with condition:", results.solver.termination_condition)
            else:
                objective_value = -1
                printer.information("As requested, model will not be solved, objective value is set to -1")

        case ModelTypeForComparison.MPS_FILE:
            mps_path = data_path
            objective_value = -1
        case _:
            raise ValueError(f"Selection for model_type is not valid: '{model_type}'")

    return mps_path, objective_value


def compareModels(model_type1: ModelTypeForComparison, folder_path1: str | pathlib.Path, solve_model1: bool,
                  model_type2: ModelTypeForComparison, folder_path2: str | pathlib.Path, solve_model2: bool,
                  skip_comparison_overall: bool = False, skip_variable_comparison: bool = False, skip_constraint_comparison: bool = False,
                  constraint_skip_model1: list[str] = None, constraint_keep_model1: list[str] = None, coefficients_skip_model1: list[str] = None,
                  constraint_skip_model2: list[str] = None, constraint_keep_model2: list[str] = None, coefficients_skip_model2: list[str] = None, constraint_enforce_model2: list[str] = None,
                  print_additional_information: bool = False,
                  gams_console_log_path: Optional[str] = None, gams_executable_path: Optional[str] = None, lego_gams_path: Optional[str] = None, max_gams_runtime_in_seconds: Optional[int] = None) -> bool:
    """
    Compare two models based on their types and folder paths. Returns true if the models are equivalent, false otherwise.
    :param model_type1: Model that should be used for path 1.
    :param folder_path1: Path to folder containing data (or to .mps file) for model1.
    :param solve_model1: Whether to solve model1 or not.
    :param model_type2: Model that should be used for path 2.
    :param folder_path2: Path to folder containing data (or to .mps file) for model2.
    :param solve_model2: Whether to solve model2 or not.
    :param skip_comparison_overall: Whether to skip overall comparison of MPS files.
    :param skip_variable_comparison: Whether to skip comparison of variables.
    :param skip_constraint_comparison: Whether to skip comparison of constraints.
    :param constraint_skip_model1: Constraints to skip from model1 (either fill this or constraint_keep_model1).
    :param constraint_keep_model1: Constraints to keep from model1 (either fill this or constraint_skip_model1).
    :param coefficients_skip_model1: Coefficients to skip from model1.
    :param constraint_skip_model2: Constraints to skip from model2 (either fill this or constraint_keep_model2).
    :param constraint_keep_model2: Constraints to keep from model2 (either fill this or constraint_skip_model2).
    :param coefficients_skip_model2: Coefficients to skip from model2.
    :param constraint_enforce_model2: Constraints to enforce from model2 (default is to enforce all).
    :param print_additional_information: Whether to print additional information during comparison.
    :param gams_console_log_path: Path to GAMS console log file.
    :param gams_executable_path: Path to GAMS executable.
    :param lego_gams_path: Path to LEGO GAMS model.
    :param max_gams_runtime_in_seconds: Maximum runtime in seconds for GAMS model.
    :return: True if models are equivalent, False otherwise.
    """

    if constraint_skip_model1 is None:
        constraint_skip_model1 = []
    if constraint_keep_model1 is None:
        constraint_keep_model1 = []
    if coefficients_skip_model1 is None:
        coefficients_skip_model1 = []
    if constraint_skip_model2 is None:
        constraint_skip_model2 = []
    if constraint_keep_model2 is None:
        constraint_keep_model2 = []
    if coefficients_skip_model2 is None:
        coefficients_skip_model2 = []
    if constraint_enforce_model2 is None:
        constraint_enforce_model2 = [""]

    printer.information(f"--------- Working on model1: '{model_type1}' ---------")
    if model_type1 == ModelTypeForComparison.GAMS and not solve_model1:
        printer.information("GAMS model will always be solved, ignoring 'solve_model1' argument")

    mps_path1, objective_value1 = build_and_solve_model(model_type1, folder_path1, solve_model1,
                                                        gams_console_log_path, gams_executable_path, lego_gams_path, max_gams_runtime_in_seconds)

    printer.information(f"--------- Working on model2: '{model_type2}' ---------")
    if model_type2 == ModelTypeForComparison.GAMS and not solve_model2:
        printer.information("GAMS model will always be solved, ignoring 'solve_model2' argument")

    mps_path2, objective_value2 = build_and_solve_model(model_type2, folder_path2, solve_model2,
                                                        gams_console_log_path, gams_executable_path, lego_gams_path, max_gams_runtime_in_seconds)

    printer.information("--------- Comparing models ---------")
    printer.information(f"Model1: '{model_type1}' - {mps_path1} - Objective value: {objective_value1}")
    printer.information(f"Model2: '{model_type2}' - {mps_path2} - Objective value: {objective_value2}")

    if objective_value1 != -1 and objective_value2 != -1:
        printer.information(f"Objective difference : {objective_value1 - objective_value2:.2f} | {100 * (objective_value1 - objective_value2) / objective_value2:.2f}%")

    if not skip_comparison_overall:
        if mps_path1 is not None and mps_path2 is not None:
            mps_equal = compare_mps(mps_path1, model_type1 != ModelTypeForComparison.GAMS, model_type2 == ModelTypeForComparison.DETERMINISTIC and not (model_type1 == ModelTypeForComparison.DETERMINISTIC or model_type1 == ModelTypeForComparison.GAMS),
                                    mps_path2, model_type2 != ModelTypeForComparison.GAMS, model_type1 == ModelTypeForComparison.DETERMINISTIC and not (model_type2 == ModelTypeForComparison.DETERMINISTIC or model_type2 == ModelTypeForComparison.GAMS),
                                    check_vars=not skip_variable_comparison, check_constraints=not skip_constraint_comparison, print_additional_information=print_additional_information,
                                    constraints_to_skip_from1=constraint_skip_model1, constraints_to_keep_from1=constraint_keep_model1, coefficients_to_skip_from1=coefficients_skip_model1,
                                    constraints_to_skip_from2=constraint_skip_model2, constraints_to_keep_from2=constraint_keep_model2, coefficients_to_skip_from2=coefficients_skip_model2, constraints_to_enforce_from2=constraint_enforce_model2)
        else:
            if mps_path1 is None:
                printer.error(f"Model 1 path is None, will not compare models")
            if mps_path2 is None:
                printer.error(f"Model 2 path is None, will not compare models")
            mps_equal = False
    else:
        printer.information("Skipping overall comparison of MPS files as requested")
        mps_equal = False

    if mps_equal and (objective_value1 - objective_value2) / objective_value2 > 0.01:
        printer.warning(f"Models are equal, but objective values differ by more than 1%: {objective_value1} vs {objective_value2}")

    return mps_equal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compares two given models (using MPS files)", formatter_class=RichHelpFormatter)
    parser.add_argument("modelType1", default=ModelTypeForComparison.DETERMINISTIC, type=ModelTypeForComparison, choices=list(ModelTypeForComparison), nargs="?", help="ModelType of first model")
    parser.add_argument("data1", default="data/example", type=str, nargs="?", help="Path to folder containing data for model1 (or to .mps file if model1 is an MPS file)")
    parser.add_argument("solveModel1", default=True, help="Whether to solve model1 or not (default is True)", type=bool, nargs="?")
    parser.add_argument("modelType2", default=ModelTypeForComparison.EXTENSIVE_FORM, type=ModelTypeForComparison, choices=list(ModelTypeForComparison), nargs="?", help="ModelType of second model")
    parser.add_argument("data2", default=None, type=str, nargs="?", help="Path to folder containing data for model2 (or to .mps file if model2 is an MPS file). If not given, the same folder as for model1 will be used.")
    parser.add_argument("solveModel2", default=True, help="Whether to solve model2 or not (default is True)", type=bool, nargs="?")

    parser.add_argument("--constraintSkipModel1", default=[], nargs='+', help="Constraints to skip from model1 (either fill this or constraintKeepModel1)")
    parser.add_argument("--constraintKeepModel1", default=[], nargs='+', help="Constraints to keep from model1 (either fill this or constraintSkipModel1)")
    parser.add_argument("--coefficientsSkipModel1", default=[], nargs='+', help="Coefficients to skip from model1")

    parser.add_argument("--constraintSkipModel2", default=[], nargs='+', help="Constraints to skip from model2 (either fill this or constraintKeepModel2)")
    parser.add_argument("--constraintKeepModel2", default=[], nargs='+', help="Constraints to keep from model2 (either fill this or constraintSkipModel2)")
    parser.add_argument("--coefficientsSkipModel2", default=[], nargs='+', help="Coefficients to skip from model2")
    parser.add_argument("--constraintEnforceModel2", default=[""], nargs='+', help="Constraints to enforce from model2 (default is to enforce all)")

    parser.add_argument("--skipComparisonOverall", action="store_true", help="Skip comparison of MPS files overall")
    parser.add_argument("--skipVariableComparison", action="store_true", help="Skip comparison of variables")
    parser.add_argument("--skipConstraintComparison", action="store_true", help="Skip comparison of constraints")
    parser.add_argument("--printAdditionalInformation", action="store_true", help="Print additional information")

    parser.add_argument("--gamsConsoleLogPath", default="LEGO-GAMS/gams_console.log", type=str, help="Path to GAMS console log file")
    parser.add_argument("--gamsPath", default="C:/GAMS/49/gams.exe", type=str, help="Path to GAMS executable")
    parser.add_argument("--legoGamsPath", default="LEGO.gms", type=str, help="Path to LEGO GAMS model")
    parser.add_argument("--maxGamsRuntimeInSeconds", default=60, type=int, help="Maximum runtime in seconds for GAMS model")

    args = parser.parse_args()

    result = compareModels(args.modelType1, args.data1, args.solveModel1,
                           args.modelType2, args.data2 if args.data2 is not None else args.data1, args.solveModel2,
                           args.skipComparisonOverall, args.skipVariableComparison, args.skipConstraintComparison,
                           args.constraintSkipModel1, args.constraintKeepModel1, args.coefficientsSkipModel1,
                           args.constraintSkipModel2, args.constraintKeepModel2, args.coefficientsSkipModel2, args.constraintEnforceModel2,
                           args.printAdditionalInformation,
                           args.gamsConsoleLogPath, args.gamsPath, args.legoGamsPath, args.maxGamsRuntimeInSeconds)
