import logging
import time

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints
from tabulate import tabulate

import LEGOUtilities
from CaseStudy import CaseStudy
from LEGO import LEGO, build_from_clone_with_fixed_results
from PyomoResult import model_to_sqlite
from tools.printer import pprint_var

########################################################################################################################
# Setup
########################################################################################################################

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)

########################################################################################################################
# Data input from case study
########################################################################################################################

cs = CaseStudy("data/example/", do_not_merge_single_node_buses=True)

########################################################################################################################
# Case study preparation
########################################################################################################################

caseStudies = []

### All DC-OPF
csAllDCOPF = cs.copy()
csAllDCOPF.dPower_Network['Technical Representation'] = 'DC-OPF'

### Mixed DC-OPF & TP
csMixed = cs.copy()
technicalRepresentations = [(("Node_1", "Node_6"), "DC-OPF"),
                            (("Node_2", "Node_3"), "DC-OPF"),
                            (("Node_2", "Node_6"), "DC-OPF"),
                            (("Node_3", "Node_4"), "DC-OPF"),
                            (("Node_3", "Node_6"), "TP"),
                            (("Node_4", "Node_5"), "DC-OPF"),
                            (("Node_4", "Node_6"), "TP"),
                            (("Node_4", "Node_9"), "TP"),
                            (("Node_6", "Node_7"), "DC-OPF"),
                            (("Node_6", "Node_8"), "DC-OPF"),
                            (("Node_7", "Node_8"), "DC-OPF"),
                            (("Node_8", "Node_9"), "DC-OPF"),
                            (("Node_1", "Node_4"), "TP")]
for (i, j), tr in technicalRepresentations:
    csMixed.dPower_Network.loc[(i, j), 'Technical Representation'] = tr
csMixed.merge_single_node_buses()

### All TP
csAllTP = cs.copy()
csAllTP.dPower_Network['Technical Representation'] = 'TP'

### Mixed TP & SN
csMixedTPSN = cs.copy()
technicalRepresentations = [(("Node_1", "Node_6"), "TP"),
                            (("Node_2", "Node_3"), "TP"),
                            (("Node_2", "Node_6"), "TP"),
                            (("Node_3", "Node_4"), "TP"),
                            (("Node_3", "Node_6"), "SN"),
                            (("Node_4", "Node_5"), "TP"),
                            (("Node_4", "Node_6"), "TP"),
                            (("Node_4", "Node_9"), "TP"),
                            (("Node_6", "Node_7"), "SN"),
                            (("Node_6", "Node_8"), "SN"),
                            (("Node_7", "Node_8"), "SN"),
                            (("Node_8", "Node_9"), "TP"),
                            (("Node_1", "Node_4"), "TP")]
for (i, j), tr in technicalRepresentations:
    csMixedTPSN.dPower_Network.loc[(i, j), 'Technical Representation'] = tr
csMixedTPSN.merge_single_node_buses()

### All SN
csAllSN = cs.copy()
csAllSN.dPower_Network['Technical Representation'] = 'SN'
csAllSN.merge_single_node_buses()

caseStudies.append(("All DC-OPF", csAllDCOPF))
caseStudies.append(("Mixed DC-OPF & TP", csMixed))
caseStudies.append(("All TP", csAllTP))
caseStudies.append(("Mixed TP & SN", csMixedTPSN))
caseStudies.append(("All SN", csAllSN))

########################################################################################################################
# Evaluation
########################################################################################################################

resultslist = pd.DataFrame(columns=["Objective Value ZOI", "Model building time [s]", "Solving time [s]", "Regret ZOI", "Regret ZOI [%]", "Objective Value Overall", "Regret Overall [%]"])
modelList = {}

for caseName, cs in caseStudies:
    print(f"\n\n{'=' * 60}\n{caseName}\n{'=' * 60}")
    startModelBuilding = time.time()
    model = LEGO(cs).build_model()
    endModelBuilding = time.time()
    print(f"Building model for {caseName} took {startModelBuilding - endModelBuilding:.2f} seconds")

    opt = SolverFactory("gurobi")
    startSolve = time.time()
    results = opt.solve(model)
    endSolve = time.time()
    print(f"Solving model for {caseName} took {startSolve - endSolve:.2f} seconds")
    results.write()

    match results.solver.termination_condition:
        case pyo.TerminationCondition.optimal:
            print("Optimal solution found")
        case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
            print(f"ERROR: Model is {results.solver.termination_condition}, logging infeasible constraints:")
            log_infeasible_constraints(model)
            exit(-1)
        case _:
            print("Solver terminated with condition:", results.solver.termination_condition)

    print("\nDisplaying Solution\n" + '-' * 60)
    pprint_var(model.p, model.zoi_g)
    pprint_var(model.t, model.zoi_i, index_positions=[0, 1])
    pprint_var(model.delta, model.zoi_i)

    # Print sum of slack variables
    print("\nSlack Variables\n" + '-' * 60)
    print("Slack variable sum of demand not served:", sum(model.vSlack_DemandNotServed[rp, k, i].value if model.vSlack_DemandNotServed[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i))
    print("Slack variable sum of overproduction:", sum(model.vSlack_OverProduction[rp, k, i].value if model.vSlack_OverProduction[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i))

    print("\nObjective Function Value\n" + '-' * 60)
    print("Objective value:", f"{pyo.value(model.objective):>8.2f}")

    # Compare with DC-OPF
    if caseName != "All DC-OPF":
        comparisonModelDC = build_from_clone_with_fixed_results(modelList["All DC-OPF"], model, ["bUC"])
        comparisonResults = opt.solve(comparisonModelDC)
        comparison_value_zoi = LEGOUtilities.get_objective_zoi(comparisonModelDC)

    objective_value_zoi = LEGOUtilities.get_objective_zoi(model)

    resultslist.loc[caseName] = {"Objective Value ZOI": objective_value_zoi,
                                 "Model building time [s]": endModelBuilding - startModelBuilding,
                                 "Solving time [s]": endSolve - startSolve,
                                 "Regret ZOI": comparison_value_zoi - objective_value_zoi if caseName != "All DC-OPF" else None,
                                 "Regret ZOI [%]": objective_value_zoi / comparison_value_zoi * 100 - 100 if caseName != "All DC-OPF" else None,
                                 "Objective Value Overall": pyo.value(model.objective),
                                 "Regret Overall [%]": (pyo.value(comparisonModelDC.objective) - pyo.value(model.objective)) / pyo.value(comparisonModelDC.objective) * 100 - 100 if caseName != "All DC-OPF" else None}

    modelList.update({caseName: model})
    model_to_sqlite(model, f"results/{caseName}.sqlite")

# Print results in pretty table
print("\n\nResults")
print(tabulate(resultslist, headers='keys', floatfmt=(".2f", ".2f", ".2f", ".2f", ".2f", ".0f", ".2f", ".0f"), colalign=("left", "right", "right", "right", "right", "right")))

print("Done")
