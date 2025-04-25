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
    cs_truth = cs_notEnforced.copy()
    cs_truth.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "notEnforced"
    cs_truth.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "notEnforced"

    # cs_truth.dPower_Hindex = cs_truth.dPower_Hindex.iloc[:-8000]  # TODO: Remove this line when using real data
    # Adjust Demand
    adjusted_demand = []
    for i, _ in cs_truth.dPower_BusInfo.iterrows():
        for h, row in cs_truth.dPower_Hindex.iterrows():
            adjusted_demand.append(["rp01", h[0].replace("h", "k"), i, cs_truth.dPower_Demand.loc[(h[1], h[2], i), "Demand"]])

    cs_truth.dPower_Demand = pd.DataFrame(adjusted_demand, columns=["rp", "k", "i", "Demand"])
    cs_truth.dPower_Demand = cs_truth.dPower_Demand.set_index(["rp", "k", "i"])

    # Adjust VRESProfiles
    if hasattr(cs_truth, "dPower_VRESProfiles"):
        adjusted_vresprofiles = []
        cs_truth.dPower_VRESProfiles.sort_index(inplace=True)
        for g in cs_truth.dPower_VRESProfiles.index.get_level_values('g').unique().tolist():
            if len(cs_truth.dPower_VRESProfiles.loc[:, :, g]) > 0:  # Check if VRESProfiles has entries for g
                for h, row in cs_truth.dPower_Hindex.iterrows():
                    adjusted_vresprofiles.append(["rp01", h[0].replace("h", "k"), g, cs_truth.dPower_VRESProfiles.loc[(h[1], h[2], g), "Capacity"]])

        cs_truth.dPower_VRESProfiles = pd.DataFrame(adjusted_vresprofiles, columns=["rp", "k", "g", "Capacity"])
        cs_truth.dPower_VRESProfiles = cs_truth.dPower_VRESProfiles.set_index(["rp", "k", "g"])

    # Adjust Hindex
    cs_truth.dPower_Hindex = cs_truth.dPower_Hindex.reset_index()
    for i, row in cs_truth.dPower_Hindex.iterrows():
        cs_truth.dPower_Hindex.loc[i] = f"h{i + 1:0>4}", f"rp01", f"k{i + 1:0>4}", None, None, None
    cs_truth.dPower_Hindex = cs_truth.dPower_Hindex.set_index(["p", "rp", "k"])

    # Adjust WeightsK
    cs_truth.dPower_WeightsK = cs_truth.dPower_WeightsK.reset_index()
    cs_truth.dPower_WeightsK = cs_truth.dPower_WeightsK.drop(cs_truth.dPower_WeightsK.index)
    for i in range(len(cs_truth.dPower_Hindex)):
        cs_truth.dPower_WeightsK.loc[i] = f"k{i + 1:0>4}", None, 1, None, None
    cs_truth.dPower_WeightsK = cs_truth.dPower_WeightsK.set_index("k")

    # Adjust WeightsRP
    cs_truth.dPower_WeightsRP = cs_truth.dPower_WeightsRP.drop(cs_truth.dPower_WeightsRP.index)
    cs_truth.dPower_WeightsRP.loc["rp01"] = 1

    lego_models = [("NoEnf.", LEGO(cs_notEnforced)), ("Cyclic", LEGO(cs_cyclic)), ("Markov", LEGO(cs_markov)), ("Truth ", LEGO(cs_truth))]
    printer.information(f"Creating varied case studies took {time.time() - start_time:.2f} seconds")

    ########################################################################################################################
    # Evaluation
    ########################################################################################################################

    optimizer = SolverFactory("gurobi")

    results = []

    df = pd.DataFrame()
    for caseName, lego in lego_models:
        printer.information(f"\n\n{'=' * 60}\n{caseName}\n{'=' * 60}")

        model, timing_building = lego.build_model()
        printer.information(f"Building model took {timing_building:.2f} seconds")

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

        results.append({
            "Case": caseName,
            "Objective": pyo.value(model.objective) if result.solver.termination_condition == pyo.TerminationCondition.optimal else -1,
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

        print(df.head())

    printer.information("Case   |  Objective  | Solution | Build Time | Solve Time | # Variables Overall | # Binary Variables | # Constraints | PNS     | EPS    |")
    for result in results:
        printer.information(f"{result['Case']} | {result['Objective']:11.2f} | {result['Solution']}  | {result['Build Time']:10.2f} | {result['Solve Time']:10.2f} | {result['# Variables Overall']:>19} | {result['# Binary Variables']:>18} | {result['# Constraints']:>13} | {result['PNS']:>7.2f} | {result['EPS']:>7.2f}")

    df.T.to_excel(unit_commitment_result_file)


if __name__ == "__main__":
    case_study_folder = "data/markov/"
    unit_commitment_result_file = "markov_quick.xlsx"
    execute_case_studies(case_study_folder, unit_commitment_result_file)

    calculate_unit_commitment_regret(unit_commitment_result_file, case_study_folder)

    printer.information("Plotting unit commitment")
    plot_unit_commitment(unit_commitment_result_file, case_study_folder, 6 * 24, 1)

    printer.success("Done")
