import argparse
import logging
import os
import time

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints
from rich_argparse import RichHelpFormatter

from InOutModule.CaseStudy import CaseStudy
from LEGO.LEGO import LEGO
from LEGO.LEGOUtilities import plot_unit_commitment, add_UnitCommitmentSlack_And_FixVariables, getUnitCommitmentSlackCost
from tools.printer import Printer

########################################################################################################################
# Setup
########################################################################################################################

printer = Printer.getInstance()
printer.console.width = 300

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)


def execute_case_studies(case_study_path: str, unit_commitment_result_file: str = "markov.xlsx"):
    ########################################################################################################################
    # Data input from case study
    ########################################################################################################################

    # Load case study from Excels
    printer.information(f"Loading case study from '{case_study_path}'")
    start_time = time.time()
    cs_notEnforced = CaseStudy(case_study_path)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    # Create varied case studies
    start_time = time.time()
    cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "notEnforced"
    cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "notEnforced"

    cs_cyclic = cs_notEnforced.copy()
    cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "cyclic"
    cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "cyclic"

    cs_markov = cs_notEnforced.copy()
    cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "markov"
    cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "markov"
    printer.information(f"Creating varied case studies took {time.time() - start_time:.2f} seconds")

    # Create "truth" case study for comparison
    start_time = time.time()
    cs_truth = cs_notEnforced.to_full_hourly_model(inplace=False)  # Create a full hourly model (which copies from notEnforced)
    printer.information(f"Creating truth case study (full-hourly) took {time.time() - start_time:.2f} seconds")

    # Build truth model already as it is used later for regret calculations as well
    start_time = time.time()
    truth_lego = LEGO(cs_truth)
    truth_lego_model, truth_timing_building = truth_lego.build_model()  # Build the truth model (for regret calculations later)
    printer.information(f"Building regret model took {time.time() - start_time:.2f} seconds (will be used later for regret calculations)")

    start_time = time.time()
    lego_models = [("NoEnf.", LEGO(cs_notEnforced)), ("Cyclic", LEGO(cs_cyclic)), ("Markov", LEGO(cs_markov)), ("Truth ", truth_lego)]
    printer.information(f"Creating the rest of the LEGO models took {time.time() - start_time:.2f} seconds")

    ########################################################################################################################
    # Evaluation
    ########################################################################################################################

    optimizer = SolverFactory("gurobi")

    results = []

    df = pd.DataFrame()
    for caseName, lego in lego_models:
        printer.information(f"\n\n{'=' * 60}\n{caseName}\n{'=' * 60}")

        if caseName != "Truth ":
            model, timing_building = lego.build_model()
            printer.information(f"Building model took {timing_building:.2f} seconds")
        else:
            model = truth_lego.model
            timing_building = truth_timing_building
            printer.information(f"Using pre-built model for truth case study")

        result, timing_solving = lego.solve_model(optimizer)
        printer.information(f"Solving model took {timing_solving:.2f} seconds")

        match result.solver.termination_condition:
            case pyo.TerminationCondition.optimal:
                printer.success("Optimal solution found")
            case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                printer.error(f"Model is {result.solver.termination_condition}, logging infeasible constraints:")
                log_infeasible_constraints(model)
            case _:
                printer.warning("Solver terminated with condition:", result.solver.termination_condition)

        # Count binary variables within all variables
        variables = list(model.component_objects(pyo.Var))
        counter_binaries = 0
        for v in variables:
            indices = [i for i in v]
            for i in indices:
                if v[i].domain == pyo.Binary:
                    counter_binaries += 1

        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            for x in [pd.Series({"case": caseName, "rp": i[0], "k": i[1], "g": i[2],
                                 "vCommit": pyo.value(model.vCommit[i]),
                                 "vStartup": pyo.value(model.vStartup[i]),
                                 "vShutdown": pyo.value(model.vShutdown[i]) if not model.vShutdown[i].stale else None,
                                 "vGenP": pyo.value(model.vGenP[i]),
                                 "vGenP1": pyo.value(model.vGenP1[i]),
                                 "pMinUpTime": pyo.value(model.pMinUpTime[i[2]]),
                                 "pMinDownTime": pyo.value(model.pMinDownTime[i[2]]),
                                 "pDemandP": sum([pyo.value(model.pDemandP[i[0], i[1], node]) for node in model.i])}) for i in list(model.vCommit)]:
                df = pd.concat([df, x], axis=1)

            if caseName != "Truth ":
                regret_lego = truth_lego.copy()

                slack_cost = 0.1 * cs_notEnforced.dPower_Parameters["pENSCost"]
                add_UnitCommitmentSlack_And_FixVariables(cs_notEnforced, regret_lego, model, slack_cost)

                # Re-solve the model
                printer.information("Re-solving model with fixed variables for regret calculation")
                regret_result, regret_timing_solving = regret_lego.solve_model(optimizer, already_solved_ok=True)
                printer.information(f"Solving regret model took {regret_timing_solving:.2f} seconds")

                match regret_result.solver.termination_condition:
                    case pyo.TerminationCondition.optimal:
                        printer.success("Optimal solution found")
                        for x in [pd.Series({"case": f"{caseName}-regret", "rp": i[0], "k": i[1], "g": i[2],
                                             "vCommit": pyo.value(regret_lego.model.vCommit[i]),
                                             "vStartup": pyo.value(regret_lego.model.vStartup[i]),
                                             "vShutdown": pyo.value(regret_lego.model.vShutdown[i]) if regret_lego.model.vShutdown[i].fixed or not regret_lego.model.vShutdown[i].stale else None,
                                             "vGenP": pyo.value(regret_lego.model.vGenP[i]),
                                             "vGenP1": pyo.value(regret_lego.model.vGenP1[i]),
                                             "pMinUpTime": pyo.value(regret_lego.model.pMinUpTime[i[2]]),
                                             "pMinDownTime": pyo.value(regret_lego.model.pMinDownTime[i[2]]),
                                             "pDemandP": sum([pyo.value(regret_lego.model.pDemandP[i[0], i[1], node]) for node in regret_lego.model.i]),
                                             "vPNS regr.": sum([pyo.value(regret_lego.model.vPNS[i[0], i[1], node]) for node in regret_lego.model.i]),
                                             "vEPS regr.": sum([pyo.value(regret_lego.model.vEPS[i[0], i[1], node]) for node in regret_lego.model.i]),
                                             "vStartupCorrectHigher": pyo.value(regret_lego.model.vStartupCorrectHigher[i]),
                                             "vStartupCorrectLower": pyo.value(regret_lego.model.vStartupCorrectLower[i]),
                                             "vShutdownCorrectHigher": pyo.value(regret_lego.model.vShutdownCorrectHigher[i]),
                                             "vShutdownCorrectLower": pyo.value(regret_lego.model.vShutdownCorrectLower[i])}) for i in list(regret_lego.model.vCommit)]:
                            df = pd.concat([df, x], axis=1)
                    case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                        printer.error(f"Model is {regret_result.solver.termination_condition}, logging infeasible constraints:")
                        log_infeasible_constraints(regret_lego.model)
                    case _:
                        printer.warning("Solver terminated with condition:", regret_result.solver.termination_condition)

        results.append({
            "Case": caseName,
            "Objective": pyo.value(model.objective) if result.solver.termination_condition == pyo.TerminationCondition.optimal else -1,
            "Objective Regret": pyo.value(regret_lego.model.objective) - getUnitCommitmentSlackCost(regret_lego, slack_cost) if regret_result.solver.termination_condition == pyo.TerminationCondition.optimal and caseName != "Truth " else -1,
            "Correction Cost": getUnitCommitmentSlackCost(regret_lego, slack_cost) if caseName != "Truth " else -1,
            "Solution": result.solver.termination_condition,
            "Build Time": timing_building,
            "Solve Time": timing_solving,
            "# Variables Overall": model.nvariables(),
            "# Binary Variables": counter_binaries,
            "# Constraints": model.nconstraints(),
            "PNS": sum(model.vPNS[rp, k, i].value if model.vPNS[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i),
            "EPS": sum(model.vEPS[rp, k, i].value if model.vEPS[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i),
            "PNS regr.": sum(regret_lego.model.vPNS[rp, k, i].value if regret_lego.model.vPNS[rp, k, i].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for i in regret_lego.model.i) if caseName != "Truth " else -1,
            "EPS regr.": sum(regret_lego.model.vEPS[rp, k, i].value if regret_lego.model.vEPS[rp, k, i].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for i in regret_lego.model.i) if caseName != "Truth " else -1,
            "Startup Correction +": sum(regret_lego.model.vStartupCorrectHigher[rp, k, t].value if regret_lego.model.vStartupCorrectHigher[rp, k, t].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for t in regret_lego.model.thermalGenerators) if caseName != "Truth " else -1,
            "Startup Correction -": sum(regret_lego.model.vStartupCorrectLower[rp, k, t].value if regret_lego.model.vStartupCorrectLower[rp, k, t].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for t in regret_lego.model.thermalGenerators) if caseName != "Truth " else -1,
            "Shutdown Correction +": sum(regret_lego.model.vShutdownCorrectHigher[rp, k, t].value if regret_lego.model.vShutdownCorrectHigher[rp, k, t].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for t in regret_lego.model.thermalGenerators) if caseName != "Truth " else -1,
            "Shutdown Correction -": sum(regret_lego.model.vShutdownCorrectLower[rp, k, t].value if regret_lego.model.vShutdownCorrectLower[rp, k, t].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for t in regret_lego.model.thermalGenerators) if caseName != "Truth " else -1,
            "model": model
        })

    printer.information("Case   |  Objective  | Objective Regret | Correction Cost | Solution | Build Time | Solve Time | # Variables Overall | # Binary Variables | # Constraints | PNS     | EPS     | PNS regr. | EPS regr. | Startup Correction + | Startup Correction - | Shutdown Correction + | Shutdown Correction - |")
    for result in results:
        printer.information(
            f"{result['Case']} | {result['Objective']:11.2f} | {result['Objective Regret']:16.2f} | {result['Correction Cost']:15.2f} | {result['Solution']}  | {result['Build Time']:10.2f} | {result['Solve Time']:10.2f} | {result['# Variables Overall']:>19} | {result['# Binary Variables']:>18} | {result['# Constraints']:>13} | {result['PNS']:>7.2f} | {result['EPS']:>7.2f} | {result['PNS regr.']:>9.2f} | {result['EPS regr.']:>9.2f} | {result['Startup Correction +']:>20.2f} | {result['Startup Correction -']:>20.2f} | {result['Shutdown Correction +']:>21.2f} | {result['Shutdown Correction -']:>21.2f}")
    df.T.to_excel(unit_commitment_result_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare edge-handling for given case-study", formatter_class=RichHelpFormatter)
    parser.add_argument("caseStudyFolder", type=str, default=None, help="Path to folder containing data for LEGO model", nargs="?")
    args = parser.parse_args()

    if args.caseStudyFolder is None:
        args.caseStudyFolder = "data/markov/"
    printer.information(f"Loading case study from '{args.caseStudyFolder}'")

    unit_commitment_result_file = f"unitCommitmentResult-{os.path.basename(os.path.normpath(args.caseStudyFolder))}.xlsx"
    printer.information(f"Unit commitment result file: '{unit_commitment_result_file}'")
    execute_case_studies(args.caseStudyFolder, unit_commitment_result_file)

    printer.information("Plotting unit commitment")
    plot_unit_commitment(unit_commitment_result_file, args.caseStudyFolder, 6 * 24, 1)

    printer.success("Done")
