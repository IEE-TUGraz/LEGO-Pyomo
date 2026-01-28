import argparse
import logging
import os
import time

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


def main(case_study_directory, part):
    caseStudyName = case_study_directory.replace("/", "_").replace("\\", "_")

    printer.information(f"Loading original case study from '{case_study_directory}'")
    start_time = time.time()
    cs = CaseStudy(case_study_directory)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    printer.information(f"Setting parameters so that it will be solved as rMIP")
    cs.dGlobal_Parameters["pEnableRMIP"] = True

    printer.information("Creating copies of case study with different formulations for network constraints")
    caseStudy_objects = {}
    if part == 0 or part == 1:
        cs.dPower_Network["pTecRepr"] = "DC-OPF"
        caseStudy_objects["DC-OPF"] = cs
    if part == 0 or part == 2:
        cs_transportProblem = cs if part == 2 else cs.copy()  # Re-use the original case study if possible to save memory
        cs_transportProblem.dPower_Network["pTecRepr"] = "TP"
        caseStudy_objects["Transport Problem"] = cs_transportProblem
    if part == 0 or part == 3:
        cs_singleNode = cs if part == 3 else cs.copy()  # Re-use the original case study if possible to save memory
        cs_singleNode.dPower_Network["pTecRepr"] = "SN"
        cs_singleNode.merge_single_node_buses()
        caseStudy_objects["Single Node"] = cs_singleNode

    printer.information("Creation of case study copies completed")

    printer.information("Building LEGO models")
    legos = {}
    for name, cs in caseStudy_objects.items():
        printer.information(f"Building LEGO model for case study with {name} representation")
        lego = LEGO(cs)
        model, timing = lego.build_model()
        printer.information(f"Building LEGO model for case study with {name} representation took {timing:.2f} seconds")
        legos[name] = (lego, model)

    printer.information("Solving LEGO models")
    for name, (lego, model) in legos.items():
        printer.information(f"Solving LEGO model for case study with {name} representation")
        results, timing, objective_value = lego.solve_model()
        printer.information(f"Solving LEGO model for case study with {name} representation took {timing:.2f} seconds")

        match results.solver.termination_condition:
            case pyo.TerminationCondition.optimal:
                printer.success(f"Optimal solution: {pyo.value(model.objective):.4f}")
            case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                printer.error(f"Model returned as {results.solver.termination_condition}, logging infeasible constraints:")
                log_infeasible_constraints(model, log_expression=False)
            case _:
                printer.warning(f"Solver terminated with condition: {results.solver.termination_condition}")

        SQLiteWriter.model_to_sqlite(model, f"model_{name}-{caseStudyName}.sqlite")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests different ", formatter_class=RichHelpFormatter)


    def directory_path(string) -> str:
        """
        Check if given string path is a directory
        :param string: Path string to be checked
        :return: Validated directory path
        :raises argparse.ArgumentTypeError: If the path is not a valid directory
        """
        if os.path.isdir(string):
            return string
        else:
            raise argparse.ArgumentTypeError(f"Directory path not valid: '{string}'")


    parser.add_argument("caseStudyDirectory", type=directory_path, help="Path to folder containing data for LEGO model")
    parser.add_argument("--part", type=int, help="Part of the case study to be run (if the case study is split into multiple parts)", nargs="?", default=0)
    args = parser.parse_args()

    main(args.caseStudyDirectory, args.part)
