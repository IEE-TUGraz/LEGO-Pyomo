import logging
import time

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints

from InOutModule.CaseStudy import CaseStudy
from LEGO.LEGO import LEGO
from LEGO.LEGOUtilities import plot_unit_commitment, calculate_unit_commitment_regret
from tools.printer import Printer

########################################################################################################################
# Setup
########################################################################################################################

printer = Printer.getInstance()
printer.console.width = 180

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

    # Create "truth" case study for comparison
    cs_truth = cs_notEnforced.to_full_hourly_model(inplace=False)  # Create a full hourly model (which copies from notEnforced)

    # Build truth model already as it is used later for regret calculations as well
    truth_lego = LEGO(cs_truth)
    truth_lego_model, truth_timing_building = truth_lego.build_model()  # Build the truth model (for regret calculations later)
    printer.information(f"Building regret model took {truth_timing_building:.2f} seconds (will be used later for regret calculations)")

    lego_models = [("NoEnf.", LEGO(cs_notEnforced)), ("Cyclic", LEGO(cs_cyclic)), ("Markov", LEGO(cs_markov)), ("Truth ", truth_lego)]

    printer.information(f"Creating varied case studies took {time.time() - start_time:.2f} seconds")

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

                # Iterate over hindex to fix vCommit, vStartup and vShutdown
                for row in cs_notEnforced.dPower_Hindex.iterrows():  # NOTE: Iterating cs_notEnforced, as Hindex is not adjusted there
                    regret_k = row[0][0].replace("h", "k")
                    for g in regret_lego.model.g:

                        regret_lego.model.vStartup["rp01", regret_k, g].fix(model.vStartup[row[0][1], row[0][2], g].value, skip_validation=True)
                        if not model.vShutdown[row[0][1], row[0][2], g].stale:  # and model.vShutdown[row[0][1], row[0][2], g].value > 0:
                            regret_lego.model.vShutdown["rp01", regret_k, g].fix(model.vShutdown[row[0][1], row[0][2], g].value, skip_validation=True)

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
                                             "pDemandP": sum([pyo.value(regret_lego.model.pDemandP[i[0], i[1], node]) for node in regret_lego.model.i])}) for i in list(regret_lego.model.vCommit)]:
                            df = pd.concat([df, x], axis=1)
                    case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                        printer.error(f"Model is {regret_result.solver.termination_condition}, logging infeasible constraints:")
                        log_infeasible_constraints(regret_lego.model)
                    case _:
                        printer.warning("Solver terminated with condition:", regret_result.solver.termination_condition)

        results.append({
            "Case": caseName,
            "Objective": pyo.value(model.objective) if result.solver.termination_condition == pyo.TerminationCondition.optimal else -1,
            "Objective Regret": pyo.value(regret_lego.model.objective) if regret_result.solver.termination_condition == pyo.TerminationCondition.optimal and caseName != "Truth " else -1,
            "Solution": result.solver.termination_condition,
            "Build Time": timing_building,
            "Solve Time": timing_solving,
            "# Variables Overall": model.nvariables(),
            "# Binary Variables": counter_binaries,
            "# Constraints": model.nconstraints(),
            "PNS": sum(model.vPNS[rp, k, i].value if model.vPNS[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i),
            "EPS": sum(model.vEPS[rp, k, i].value if model.vEPS[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i),
            "model": model
        })

    printer.information("Case   |  Objective  | Objective Regret | Solution | Build Time | Solve Time | # Variables Overall | # Binary Variables | # Constraints | PNS     | EPS    |")
    for result in results:
        printer.information(f"{result['Case']} | {result['Objective']:11.2f} | {result['Objective Regret']:16.2f} | {result['Solution']}  | {result['Build Time']:10.2f} | {result['Solve Time']:10.2f} | {result['# Variables Overall']:>19} | {result['# Binary Variables']:>18} | {result['# Constraints']:>13} | {result['PNS']:>7.2f} | {result['EPS']:>7.2f}")
    df.T.to_excel(unit_commitment_result_file)


if __name__ == "__main__":
    case_study_folder = "data/markov/"
    unit_commitment_result_file = "markov_quick.xlsx"
    execute_case_studies(case_study_folder, unit_commitment_result_file)

    calculate_unit_commitment_regret(unit_commitment_result_file, case_study_folder)

    printer.information("Plotting unit commitment")
    plot_unit_commitment(unit_commitment_result_file, case_study_folder, 6 * 24, 1)

    printer.success("Done")
