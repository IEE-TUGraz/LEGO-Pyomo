import logging
import time

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints

from CaseStudy import CaseStudy
from LEGO import LEGO
from Pretty import pprint_var

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
caseStudies.append(("All DC-OPF", csAllDCOPF))

### Mixed DC-OPF, TP & SN
csMixed = cs.copy()
technicalRepresentations = [(("Node_1", "Node_6"), "DC-OPF"),
                            (("Node_2", "Node_3"), "DC-OPF"),
                            (("Node_2", "Node_6"), "TP"),
                            (("Node_3", "Node_4"), "TP"),
                            (("Node_3", "Node_6"), "SN"),
                            (("Node_4", "Node_5"), "DC-OPF"),
                            (("Node_4", "Node_6"), "TP"),
                            (("Node_4", "Node_9"), "TP"),
                            (("Node_6", "Node_7"), "DC-OPF"),
                            (("Node_6", "Node_8"), "SN"),
                            (("Node_7", "Node_8"), "SN"),
                            (("Node_8", "Node_9"), "DC-OPF"),
                            (("Node_1", "Node_4"), "TP")]
for (i, j), tr in technicalRepresentations:
    csMixed.dPower_Network.loc[(i, j), 'Technical Representation'] = tr
csMixed.merge_single_node_buses()
caseStudies.append(("Mixed DC-OPF, TP & SN", csMixed))

### All TP
csAllTP = cs.copy()
csAllTP.dPower_Network['Technical Representation'] = 'TP'
caseStudies.append(("All TP", csAllTP))

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
caseStudies.append(("Mixed TP & SN", csMixedTPSN))

### All SN
csAllSN = cs.copy()
csAllSN.dPower_Network['Technical Representation'] = 'SN'
csAllSN.merge_single_node_buses()
caseStudies.append(("All SN", csAllSN))

########################################################################################################################
# Evaluation
########################################################################################################################

resultslist = pd.DataFrame(columns=["Objective Value", "Model building time", "Solving time", "Slack variable sum of demand not served", "Slack variable sum of overproduction"])

for caseName, cs in caseStudies:
    print(f"\n\n{'=' * 60}\n{caseName}\n{'=' * 60}")
    startModelBUilding = time.time()
    model = LEGO(cs).build_model()
    endModelBuilding = time.time()
    print(f"Building model for {caseName} took {time.time() - endModelBuilding:.2f} seconds")

    opt = SolverFactory("gurobi")
    startSolve = time.time()
    results = opt.solve(model)
    endSolve = time.time()
    print(f"Solving model for {caseName} took {time.time() - endSolve:.2f} seconds")
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

    resultslist.loc[caseName] = {"Objective Value": pyo.value(model.objective),
                                 "Model building time": endModelBuilding - startModelBUilding,
                                 "Solving time": endSolve - startSolve,
                                 "Slack variable sum of demand not served": sum(model.vSlack_DemandNotServed[rp, k, i].value if model.vSlack_DemandNotServed[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i),
                                 "Slack variable sum of overproduction": sum(model.vSlack_OverProduction[rp, k, i].value if model.vSlack_OverProduction[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i)}

# Print results in pretty table
print("\n\nResults\n" + '-' * 60)
print(resultslist)

print("Done")
