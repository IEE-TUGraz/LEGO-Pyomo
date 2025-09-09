import argparse
import logging
import os
import time
import typing

import pandas as pd
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from rich_argparse import RichHelpFormatter

from InOutModule import SQLiteWriter, ExcelWriter
from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO
from LEGO.LEGOUtilities import plot_unit_commitment, add_UnitCommitmentSlack_And_FixVariables, getUnitCommitmentSlackCost

########################################################################################################################
# Setup
########################################################################################################################

printer = Printer.getInstance()
printer.set_width(300)

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)


def execute_case_studies(case_study_path: str, unit_commitment_result_file: str = "markov.xlsx", no_sqlite: bool = False, no_excel: bool = False, calculate_regret: bool = False, write_unit_commitment_result_file: bool = True, step_size: int = 0):
    ########################################################################################################################
    # Data input from case study
    ########################################################################################################################

    # Load case study from Excels
    printer.information(f"Loading case study from '{case_study_path}'")
    start_time = time.time()
    cs = CaseStudy(case_study_path, clip_method="none", clip_value=0)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    # Create varied case studies
    start_time = time.time()
    printer.information(f"Creating varied case studies")
    cs_notEnforced = cs.copy()
    cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "notEnforced"
    cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "notEnforced"
    cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingIntraDayStorage"] = "notEnforced"

    cs_cyclic = cs_notEnforced.copy()
    cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "cyclic"
    cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "cyclic"
    cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingIntraDayStorage"] = "cyclic"

    cs_markov = cs_notEnforced.copy()
    cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "markov"
    cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "markov"
    cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingIntraDayStorage"] = "markov"
    printer.information(f"Creating varied case studies took {time.time() - start_time:.2f} seconds")

    # Create "truth" case study for comparison
    start_time = time.time()
    printer.information(f"Creating truth case study (full-hourly)")
    cs_truth = cs.to_full_hourly_model(inplace=False)  # Create a full hourly model (which copies from notEnforced)
    printer.information(f"Creating truth case study (full-hourly) took {time.time() - start_time:.2f} seconds")

    thermalGenerators = cs.dPower_ThermalGen.index.tolist()

    if step_size == 0:
        printer.information(f"Not relaxing any unit commitment variables, all {len(thermalGenerators)} thermal generators stay binary")
    else:
        printer.information(f"Relaxing with step size: {step_size}")

    for count_relaxed in range(len(thermalGenerators), -1, -step_size if step_size > 0 else -1):
        if step_size == 0:
            count_relaxed = 0
        start_time = time.time()
        printer.information(f"Building the LEGO models for adjustments")  # Note this is actually faster (1.5-2x) than copying already built models to re-use them
        lego_models = {"NoEnf.": LEGO(cs_notEnforced), "Cyclic": LEGO(cs_cyclic), "Markov": LEGO(cs_markov), "Truth ": LEGO(cs_truth)}
        for name, lego in lego_models.items():
            _, build_time = lego.build_model()
            printer.information(f"Building model for case study '{name}' took {build_time:.2f} seconds")
        printer.information(f"Building the LEGO models took {time.time() - start_time:.2f} seconds overall")

        start_time = time.time()
        printer.information(f"Relaxing {count_relaxed} thermal generators, keeping {len(thermalGenerators) - count_relaxed} binary")
        thermalGeneratorRelaxed = {g: False for g in thermalGenerators}
        offset = 0
        for i in range(count_relaxed):
            if all(thermalGeneratorRelaxed[g] == True for g in thermalGenerators):
                break  # Safety check if all generators are already relaxed
            index = (i * (len(thermalGenerators) // step_size) + offset) % len(thermalGenerators)
            while thermalGeneratorRelaxed[thermalGenerators[index]]:  # Skip already relaxed generators
                index = (index + 1) % len(thermalGenerators)
                offset += 1
            thermalGeneratorRelaxed[thermalGenerators[index]] = True

        printer.information(f"Relaxing {count_relaxed} thermal generators: {[g for g in thermalGenerators if thermalGeneratorRelaxed[g]]}")
        for case_name, lego in lego_models.items():
            for rp in lego.model.rp:
                for k in lego.model.k:
                    for g in lego.model.thermalGenerators:
                        lego.model.vCommit[rp, k, g].domain = pyo.PercentFraction if thermalGeneratorRelaxed[g] else pyo.Binary
                        lego.model.vStartup[rp, k, g].domain = pyo.PercentFraction if thermalGeneratorRelaxed[g] else pyo.Binary
                        lego.model.vShutdown[rp, k, g].domain = pyo.PercentFraction if thermalGeneratorRelaxed[g] else pyo.Binary
        printer.information(f"Relaxing {count_relaxed} thermal generators took {time.time() - start_time:.2f} seconds")

        execute_case_study(lego_models, f"{os.path.basename(os.path.normpath(case_study_path))}-relaxed{count_relaxed}", unit_commitment_result_file.replace(".xlsx", f"-relaxed{count_relaxed}.xlsx"), no_sqlite, no_excel, calculate_regret, write_unit_commitment_result_file)

        if step_size == 0:
            break


def execute_case_study(lego_models: typing.Dict[str, LEGO], case_name: str, unit_commitment_result_file: str, no_sqlite: bool, no_excel: bool, calculate_regret: bool, write_unit_commitment_result_file: bool):
    ########################################################################################################################
    # Evaluation
    ########################################################################################################################
    results = []

    truth_lego = lego_models["Truth "]

    df = pd.DataFrame()
    for edgeHandlingType, lego in lego_models.items():
        printer.information(f"\n\n{'=' * 60}\n{edgeHandlingType}\n{'=' * 60}")
        model = lego.model

        # Solve model
        optimizer = pyo.SolverFactory('gurobi_persistent')
        optimizer.set_instance(model)
        start_time = time.time()
        result = optimizer.solve(tee=True)
        objective_value = pyo.value(model.objective) if result.solver.termination_condition == pyo.TerminationCondition.optimal else -1
        timing_solving = time.time() - start_time
        work_time = optimizer._solver_model.Work
        printer.information(f"Solving model took {timing_solving:.2f} seconds ({work_time:.2f} work units)")

        if not no_sqlite:
            sqlite_timer = time.time()
            sqlite_file = f"{case_name}-{edgeHandlingType.replace(".", "")}.sqlite"
            printer.information(f"Writing model to SQLite database: {sqlite_file}")
            SQLiteWriter.model_to_sqlite(lego.model, sqlite_file)
            printer.information(f"Writing model to SQLite database took {time.time() - sqlite_timer:.2f} seconds")

        if not no_excel:
            excel_timer = time.time()
            excel_file = f"{case_name}-{edgeHandlingType.replace('.', '')}.xlsx"
            printer.information(f"Writing model to Excel file: {excel_file}")
            ExcelWriter.model_to_excel(lego.model, excel_file)
            printer.information(f"Writing model to Excel file took {time.time() - excel_timer:.2f} seconds")

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

        if result.solver.termination_condition == pyo.TerminationCondition.optimal and write_unit_commitment_result_file:
            for x in [pd.Series({"case": edgeHandlingType, "rp": i[0], "k": i[1], "g": i[2],
                                 "vCommit": pyo.value(model.vCommit[i]),
                                 "vStartup": pyo.value(model.vStartup[i]),
                                 "vShutdown": pyo.value(model.vShutdown[i]) if not model.vShutdown[i].stale else None,
                                 "vGenP": pyo.value(model.vGenP[i]),
                                 "vGenP1": pyo.value(model.vGenP1[i]),
                                 "pMinUpTime": pyo.value(model.pMinUpTime[i[2]]),
                                 "pMinDownTime": pyo.value(model.pMinDownTime[i[2]]),
                                 "pDemandP": sum([pyo.value(model.pDemandP[i[0], i[1], node]) for node in model.i]),
                                 "vPNS": sum([pyo.value(model.vPNS[i[0], i[1], node]) for node in model.i]),
                                 "vEPS": sum([pyo.value(model.vEPS[i[0], i[1], node]) for node in model.i])}) for i in list(model.vCommit)]:
                df = pd.concat([df, x], axis=1)

            if calculate_regret and edgeHandlingType != "Truth ":
                regret_lego = truth_lego.copy()

                add_UnitCommitmentSlack_And_FixVariables(regret_lego, model, lego.cs.dPower_Hindex, lego.cs.dPower_ThermalGen, lego.cs.dPower_Parameters["pENSCost"])

                # Re-solve the model
                printer.information("Re-solving model with fixed variables for regret calculation")
                regret_result, regret_timing_solving, regret_objective_value = regret_lego.solve_model(already_solved_ok=True)
                printer.information(f"Solving regret model took {regret_timing_solving:.2f} seconds")

                if not no_sqlite:
                    sqlite_timer = time.time()
                    sqlite_file = f"{case_name}-{edgeHandlingType.replace(".", "")}-regret.sqlite"
                    printer.information(f"Writing model to SQLite database: {sqlite_file}")
                    SQLiteWriter.model_to_sqlite(regret_lego.model, sqlite_file)
                    printer.information(f"Writing model to SQLite database took {time.time() - sqlite_timer:.2f} seconds")

                if not no_excel:
                    excel_timer = time.time()
                    excel_file = f"{case_name}-{edgeHandlingType.replace('.', '')}-regret.xlsx"
                    printer.information(f"Writing model to Excel file: {excel_file}")
                    ExcelWriter.model_to_excel(regret_lego.model, excel_file)
                    printer.information(f"Writing model to Excel file took {time.time() - excel_timer:.2f} seconds")

                match regret_result.solver.termination_condition:
                    case pyo.TerminationCondition.optimal:
                        printer.success("Optimal solution found")
                        for x in [pd.Series({"case": f"{edgeHandlingType}-regret", "rp": i[0], "k": i[1], "g": i[2],
                                             "vCommit": pyo.value(regret_lego.model.vCommit[i]),
                                             "vStartup": pyo.value(regret_lego.model.vStartup[i]),
                                             "vShutdown": pyo.value(regret_lego.model.vShutdown[i]) if regret_lego.model.vShutdown[i].fixed or not regret_lego.model.vShutdown[i].stale else None,
                                             "vGenP": pyo.value(regret_lego.model.vGenP[i]),
                                             "vGenP1": pyo.value(regret_lego.model.vGenP1[i]),
                                             "pMinUpTime": pyo.value(regret_lego.model.pMinUpTime[i[2]]),
                                             "pMinDownTime": pyo.value(regret_lego.model.pMinDownTime[i[2]]),
                                             "pDemandP": sum([pyo.value(regret_lego.model.pDemandP[i[0], i[1], node]) for node in regret_lego.model.i]),
                                             "vPNS": sum([pyo.value(regret_lego.model.vPNS[i[0], i[1], node]) for node in regret_lego.model.i]),
                                             "vEPS": sum([pyo.value(regret_lego.model.vEPS[i[0], i[1], node]) for node in regret_lego.model.i]),
                                             "vCommitCorrectHigher": pyo.value(regret_lego.model.vCommitCorrectHigher[i]) if not regret_lego.model.vCommitCorrectHigher[i].stale else None,
                                             "vCommitCorrectLower": pyo.value(regret_lego.model.vCommitCorrectLower[i]) if not regret_lego.model.vCommitCorrectLower[i].stale else None, }) for i in list(regret_lego.model.vCommit)]:
                            df = pd.concat([df, x], axis=1)
                    case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                        printer.error(f"Model is {regret_result.solver.termination_condition}, logging infeasible constraints:")
                        log_infeasible_constraints(regret_lego.model)
                    case _:
                        printer.warning("Solver terminated with condition:", regret_result.solver.termination_condition)

        entry = {
            "Case": f"{case_name}-{edgeHandlingType}",
            "Objective": objective_value if result.solver.termination_condition == pyo.TerminationCondition.optimal else -1,
            "Objective Regret": -1 if not calculate_regret else (regret_objective_value - getUnitCommitmentSlackCost(regret_lego, lego.cs.dPower_ThermalGen, lego.cs.dPower_Parameters["pENSCost"]) if regret_result.solver.termination_condition == pyo.TerminationCondition.optimal and edgeHandlingType != "Truth " else -1),
            "Correction Cost": -1 if not calculate_regret else (getUnitCommitmentSlackCost(regret_lego, lego.cs.dPower_ThermalGen, lego.cs.dPower_Parameters["pENSCost"]) if edgeHandlingType != "Truth " else -1),
            "Solution": result.solver.termination_condition,
            # "Build Time": timing_building,
            "Solve Time": timing_solving,
            "Work Time": work_time,
            "# Variables Overall": model.nvariables(),
            "# Binary Variables": counter_binaries,
            "# Constraints": model.nconstraints(),
            "PNS regr.": -1 if not calculate_regret else (sum(regret_lego.model.vPNS[rp, k, i].value if regret_lego.model.vPNS[rp, k, i].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for i in regret_lego.model.i) if edgeHandlingType != "Truth " else -1),
            "EPS regr.": -1 if not calculate_regret else (sum(regret_lego.model.vEPS[rp, k, i].value if regret_lego.model.vEPS[rp, k, i].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for i in regret_lego.model.i) if edgeHandlingType != "Truth " else -1),
            "Commit Correction +": -1 if not calculate_regret else (sum(regret_lego.model.vCommitCorrectHigher[rp, k, t].value if regret_lego.model.vCommitCorrectHigher[rp, k, t].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for t in regret_lego.model.thermalGenerators) if edgeHandlingType != "Truth " else -1),
            "Commit Correction -": -1 if not calculate_regret else (sum(regret_lego.model.vCommitCorrectLower[rp, k, t].value if regret_lego.model.vCommitCorrectLower[rp, k, t].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for t in regret_lego.model.thermalGenerators) if edgeHandlingType != "Truth " else -1),
            "vGenP": sum(model.vGenP[rp, k, g].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vGenP[rp, k, g].value is not None else 0 for rp in model.rp for k in model.k for g in model.g),
            "vCommit": sum(model.vCommit[rp, k, g].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vCommit[rp, k, g].value is not None else 0 for rp in model.rp for k in model.k for g in model.thermalGenerators),
            "vStartup": sum(model.vStartup[rp, k, g].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vStartup[rp, k, g].value is not None else 0 for rp in model.rp for k in model.k for g in model.thermalGenerators),
            "vShutdown": sum(model.vShutdown[rp, k, g].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vShutdown[rp, k, g].value is not None else 0 for rp in model.rp for k in model.k for g in model.thermalGenerators),
            "vPNS": sum(model.vPNS[rp, k, i].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vPNS[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i),
            "vEPS": sum(model.vEPS[rp, k, i].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vEPS[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i),
            "model": model
        }
        results.append(entry)

        # Write entry to solutions logfile
        log_file = printer.get_logfile().replace(".log", "-solutions.csv")
        if os.path.exists(log_file):
            with open(log_file, "a") as f:
                f.write(",".join([f"{v}" for v in entry.values()]) + "\n")
        else:
            with open(log_file, "w") as f:
                f.write(",".join(entry.keys()) + "\n")
                f.write(",".join([f"{v}" for v in entry.values()]) + "\n")

    values = ["Case", "Objective", "Solve Time", "vGenP", "vCommit", "vStartup", "vShutdown", "vPNS", "vEPS", "Objective Regret"]
    table = []
    for v in values:
        column = [v]
        for result in results:
            value = result[v]
            if isinstance(value, float):
                value = f"{value:.2f}"
            elif isinstance(value, int):
                value = f"{value:d}"
            else:
                value = f"{value}"
            column.append(value)
        table.append(column)

    for i in range(len(table[0])):
        printer.information(" | ".join(f"{table[j][i]:{">" if i != 0 else ""}{max(len(table[j][i2]) for i2 in range(len(table[j])))}}" for j in range(len(table))))

    if write_unit_commitment_result_file:
        df.T.to_excel(unit_commitment_result_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare edge-handling for given case-study", formatter_class=RichHelpFormatter)
    parser.add_argument("caseStudyFolder", type=str, help="Path to folder containing data for LEGO model. Can be a comma-separated list of multiple folders (executed after each other)")
    parser.add_argument("--plot", action="store_true", help="Plot unit commitment results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode where exceptions are passed on")
    parser.add_argument("--no-sqlite", action="store_true", help="Do not save results to SQLite database")
    parser.add_argument("--no-excel", action="store_true", help="Do not save results to Excel file")
    parser.add_argument("--no-regret-plot", action="store_true", help="Do not plot regret results")
    parser.add_argument("--calculate-regret", action="store_true", help="Calculate regret by re-solving the truth model with fixed unit commitment from the other models (can take a while)")
    parser.add_argument("--dont-write-unit-commitment-result-file", action="store_true", help="Write unit commitment result file (can take a while)")
    parser.add_argument("--relax-step-size", type=int, default=0, help="Step size for relaxing unit commitment variables (default: 0 = no relaxation, all binary)")
    args = parser.parse_args()

    for folder in args.caseStudyFolder.split(","):
        try:
            if not folder.endswith("/"):
                folder += "/"
            folder_name = os.path.basename(os.path.normpath(folder))
            printer.set_logfile(f"markov-{folder_name}.log")
            printer.information(f"Loading case study from '{folder}'")
            printer.information(f"Logfile: '{printer.get_logfile()}'")

            unit_commitment_result_file = f"unitCommitmentResult-{folder_name}.xlsx"
            printer.information(f"Unit commitment result file: '{unit_commitment_result_file}'")
            execute_case_studies(folder, unit_commitment_result_file, args.no_sqlite, args.no_excel, args.calculate_regret, not args.dont_write_unit_commitment_result_file, args.relax_step_size)

            if args.plot:
                printer.information("Plotting unit commitment")
                plot_unit_commitment(unit_commitment_result_file, folder, 6 * 24, 1, not args.no_regret_plot)
        except Exception as e:
            printer.error(f"Exception while executing case study '{folder}': {e}")
            if args.debug:
                raise e
            else:
                printer.error(f"Continuing with next case study")

    printer.success("Done")
