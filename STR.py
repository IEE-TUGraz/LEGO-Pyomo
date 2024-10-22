import logging

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints
from tabulate import tabulate

from CaseStudy import CaseStudy
from LEGO import LEGO, build_from_clone_with_fixed_results
from PyomoResult import model_to_sqlite
from tools.printer import pprint_var, Printer

########################################################################################################################
# Setup
########################################################################################################################

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)
printer = Printer.getInstance()
print_variable_results = False

########################################################################################################################
# Data input from case study
########################################################################################################################

cs = CaseStudy("data/example/", do_not_merge_single_node_buses=True)

########################################################################################################################
# Case study preparation
########################################################################################################################

legoModels = []

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

legoModels.append(("All DC-OPF", LEGO(csAllDCOPF)))
legoModels.append(("Mixed DC-OPF & TP", LEGO(csMixed)))
legoModels.append(("All TP", LEGO(csAllTP)))
legoModels.append(("Mixed TP & SN", LEGO(csMixedTPSN)))
legoModels.append(("All SN", LEGO(csAllSN)))

########################################################################################################################
# Evaluation
########################################################################################################################

resultslist = pd.DataFrame(columns=["Obj. value in original (ZOI)", "Model building time [s]", "Solving time [s]", "Regret to original (ZOI)", "Regret to original (ZOI) [%]", "Obj. value in original (Overall)", "Regret to original (Overall) [%]", "# Variables", "# Constraints"])
modelList = {}
optimizer = SolverFactory("gurobi")

for caseName, lego in legoModels:
    print(f"\n\n{'=' * 60}\n{caseName}\n{'=' * 60}")

    model, timing = lego.build_model()
    printer.information(f"Building model took {timing:.2f} seconds")

    results, timing = lego.solve_model(optimizer)
    print(f"Solving model took {timing:.2f} seconds")
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

    if print_variable_results:
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
    if caseName == "All DC-OPF":
        resultslist.loc[caseName] = {"Obj. value in original (ZOI)": lego.get_objective_value(zoi=True),
                                     "Model building time [s]": lego.timings["model_building"],
                                     "Solving time [s]": lego.timings["model_solving"],
                                     "Regret to original (ZOI)": None,
                                     "Regret to original (ZOI) [%]": None,
                                     "Obj. value in original (Overall)": lego.get_objective_value(zoi=False),
                                     "Regret to original (Overall) [%]": None,
                                     "# Variables": lego.get_number_of_variables(),
                                     "# Constraints": lego.get_number_of_constraints()
                                     }
    else:
        # Re-calculate DC-OPF model with fixed Unit Commitment
        comparisonModelDC = build_from_clone_with_fixed_results(model_to_be_cloned=modelList["All DC-OPF"], model_with_fixed_results=lego.model, variables_to_fix=["bUC"])
        comparisonResults, _ = comparisonModelDC.solve_model(optimizer)
        comparison_objective_value_overall = comparisonModelDC.get_objective_value(zoi=False)
        comparison_objective_value_zoi = comparisonModelDC.get_objective_value(zoi=True)

        objective_value_overall = lego.get_objective_value(False)
        objective_value_zoi = lego.get_objective_value(True)

        original_objective_value_overall = LEGO.get_objective_value(LEGO(model=modelList["All DC-OPF"]), zoi=False)
        original_objective_value_zoi = LEGO.get_objective_value(LEGO(model=modelList["All DC-OPF"]), zoi=True)

        resultslist.loc[caseName] = {"Obj. value in original (ZOI)": comparison_objective_value_zoi,
                                     "Model building time [s]": lego.timings["model_building"],
                                     "Solving time [s]": lego.timings["model_solving"],
                                     "Regret to original (ZOI)": comparison_objective_value_zoi - original_objective_value_zoi,
                                     "Regret to original (ZOI) [%]": (comparison_objective_value_zoi - original_objective_value_zoi) / original_objective_value_zoi * 100,
                                     "Obj. value in original (Overall)": comparison_objective_value_overall,
                                     "Regret to original (Overall) [%]": (comparison_objective_value_overall - original_objective_value_overall) / comparison_objective_value_overall * 100,
                                     "# Variables": lego.get_number_of_variables(),
                                     "# Constraints": lego.get_number_of_constraints()
                                     }

    modelList.update({caseName: lego.model})
    model_to_sqlite(lego.model, f"results/{caseName}.sqlite")

# Print results in pretty table
print("\n\nResults")
print(tabulate(resultslist, headers='keys', floatfmt=(".2f", ".2f", ".2f", ".2f", ".2f", ".0f", ".2f", ".0f", ".0f", ".0f"), colalign=("left", "right", "right", "right", "right", "right", "right", "right")))

print("Done")
