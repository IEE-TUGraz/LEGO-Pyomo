import logging

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints

from CaseStudy import CaseStudy

########################################################################################################################
# Setup
########################################################################################################################

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)

########################################################################################################################
# Data input from case study
########################################################################################################################

cs = CaseStudy("data/example/", merge_single_node_buses=True)

pMaxAngleDCOPF = cs.dPower_Parameters.loc["pMaxAngleDCOPF"].iloc[0] * np.pi / 180  # Read and convert to radians
pSBase = cs.dPower_Parameters.loc["pSBase"].iloc[0]


########################################################################################################################
# Model creation
########################################################################################################################

model = pyo.ConcreteModel()

# Sets
model.i = pyo.Set(doc='Buses', initialize=cs.dPower_BusInfo.index.tolist())
model.e = pyo.Set(doc='Lines', initialize=cs.dPower_Network.index.tolist())
model.thermalGenerators = pyo.Set(doc='Thermal Generators', initialize=cs.dPower_ThermalGen.index.tolist())
model.rorGenerators = pyo.Set(doc='Run-of-river generators', initialize=cs.dPower_RoR.index.tolist())
model.vresGenerators = pyo.Set(doc='Variable renewable energy sources', initialize=cs.dPower_VRES.index.tolist())
model.g = pyo.Set(doc='Generators', initialize=model.thermalGenerators | model.rorGenerators | model.vresGenerators)
model.rp = pyo.Set(doc='Representative periods', initialize=cs.dPower_Demand.index.get_level_values('rp').unique().tolist())
model.k = pyo.Set(doc='Timestep within representative period', initialize=cs.dPower_Demand.index.get_level_values('k').unique().tolist())

# Helper Sets for zone of interest
model.zoi_i = pyo.Set(doc="Buses in zone of interest", initialize=cs.dPower_BusInfo.loc[cs.dPower_BusInfo["ZoneOfInterest"] == "yes"].index.tolist(), within=model.i)
model.zoi_g = pyo.Set(doc="Generators in zone of interest", initialize=cs.hGenerators_to_Buses.loc[cs.hGenerators_to_Buses["i"].isin(model.zoi_i)].index.tolist(), within=model.g)

# Variables
model.delta = pyo.Var(model.i, model.rp, model.k, doc='Angle of bus i', bounds=(-pMaxAngleDCOPF, pMaxAngleDCOPF))  # TODO: Discuss impact on runtime etc.(based on discussion with Prof. Renner)
model.vSlack_DemandNotServed = pyo.Var(model.rp, model.k, model.i, doc='Slack variable demand not served', bounds=(0, None))
model.vSlack_OverProduction = pyo.Var(model.rp, model.k, model.i, doc='Slack variable overproduction', bounds=(0, None))

model.p = pyo.Var(model.g, model.rp, model.k, doc='Power output of generator g', bounds=(0, None))
for g in model.thermalGenerators:
    model.p[g, :, :].setub(cs.dPower_ThermalGen.loc[g, 'MaxProd'] * cs.dPower_ThermalGen.loc[g, 'ExisUnits'])
    # TODO: Add min production (needs unit commitment)

for g in model.rorGenerators:
    for rp in model.rp:
        for k in model.k:
            model.p[g, rp, k].setub(min(cs.dPower_RoR.loc[g, 'MaxProd'], cs.dPower_Inflows.loc[rp, g, k]['Inflow']))  # TODO: Check and adapt for storage

for g in model.vresGenerators:
    for rp in model.rp:
        for k in model.k:
            maxProd = cs.dPower_VRES.loc[g, 'MaxProd']
            capacity = cs.dPower_VRESProfiles.loc[rp, cs.dPower_VRES.loc[g, 'i'], k, cs.dPower_VRES.loc[g, 'tec']]['Capacity']
            capacity = capacity.values[0] if isinstance(capacity, pd.Series) else capacity
            exisUnits = cs.dPower_VRES.loc[g, 'ExisUnits']
            model.p[g, rp, k].setub(maxProd * capacity * exisUnits)
            if maxProd * capacity * exisUnits == 0:
                model.p[g, rp, k].fix(0)

model.t = pyo.Var(model.e, model.rp, model.k, doc='Power flow from bus i to j', bounds=(None, None))
for (i, j) in model.e:
    match cs.dPower_Network.loc[i, j]["Technical Representation"]:
        case "DC-OPF" | "TP":
            model.t[(i, j), :, :].setlb(-cs.dPower_Network.loc[i, j]['Pmax'])
            model.t[(i, j), :, :].setub(cs.dPower_Network.loc[i, j]['Pmax'])
        case "SN":
            assert False  # Should not happen, as we merged all "Single Node" representations
        case _:
            raise ValueError(f"Technical representation '{cs.dPower_Network.loc[i, j]["Technical Representation"]}' "
                             f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

# Parameters
model.pDemand = pyo.Param(model.rp, model.i, model.k, initialize=cs.dPower_Demand['Demand'], doc='Demand at bus i in representative period rp and timestep k')

# Helper for FuelCost that has dPower_ThermalGen['FuelCost'] for ThermalGen, and 0 for all gs in ror and vres
hFuelCost = pd.concat([cs.dPower_ThermalGen['FuelCost'].copy(), pd.Series(0, index=model.rorGenerators), pd.Series(0, index=model.vresGenerators)])
model.pProductionCost = pyo.Param(model.g, initialize=hFuelCost, doc='Production cost of generator g')

model.pReactance = pyo.Param(model.e, initialize=cs.dPower_Network['X'], doc='Reactance of line e')
model.pSlackPrice = pyo.Param(initialize=max(model.pProductionCost.values()) * 100, doc='Price of slack variable')

# For each DC-OPF "island", set node with highest demand as slack node
dDCOPFIslands = pd.DataFrame(index=cs.dPower_BusInfo.index, columns=[cs.dPower_BusInfo.index], data=False)

for index, entry in cs.dPower_Network.iterrows():
    if cs.dPower_Network.loc[(index[0], index[1])]["Technical Representation"] == "DC-OPF":
        dDCOPFIslands.loc[index[0], index[1]] = True
        dDCOPFIslands.loc[index[1], index[0]] = True

completed_buses = set()  # Set of buses that have been looked at already
i = 0
for index, entry in dDCOPFIslands.iterrows():
    if index in completed_buses or entry[entry == True].empty:  # Skip if bus has already been looked at or has no connections
        continue

    connected_buses = cs.get_connected_buses(dDCOPFIslands, str(index))

    for bus in connected_buses:
        completed_buses.add(bus)

    # Set slack node
    slack_node = cs.dPower_Demand.loc[:, connected_buses, :].groupby('i').sum().idxmax().values[0]
    if i == 0: print("Setting slack nodes for DC-OPF zones:")
    print(f"DC-OPF Zone {i:>2} - Slack node: {slack_node}")
    i += 1
    model.delta[slack_node, :, :].fix(0)

# Constraint(s)
model.cPower_Balance = pyo.ConstraintList(doc='Power balance constraint for each bus')
for i in model.i:
    for rp in model.rp:
        for k in model.k:
            model.cPower_Balance.add(
                sum(model.p[g, rp, k] for g in model.g if cs.hGenerators_to_Buses.loc[g]['i'] == i) -  # Production of generators at bus i
                sum(model.t[e, rp, k] for e in model.e if (e[0] == i)) +  # Power flow from bus i to bus j
                sum(model.t[e, rp, k] for e in model.e if (e[1] == i)) ==  # Power flow from bus j to bus i
                model.pDemand[rp, i, k] -  # Demand at bus i
                model.vSlack_DemandNotServed[rp, k, i] +  # Slack variable for demand not served
                model.vSlack_OverProduction[rp, k, i])  # Slack variable for overproduction

model.cReactance = pyo.ConstraintList(doc='Reactance constraint for each line (for DC-OPF)')
for (i, j) in model.e:
    match cs.dPower_Network.loc[i, j]["Technical Representation"]:
        case "DC-OPF":
            for rp in model.rp:
                for k in model.k:
                    model.cReactance.add(model.t[(i, j), rp, k] == (model.delta[i, rp, k] - model.delta[j, rp, k]) * pSBase / model.pReactance[(i, j)])
        case "TP" | "SN":
            continue
        case _:
            raise ValueError(f"Technical representation '{cs.dPower_Network.loc[i, j]["Technical Representation"]}' "
                             f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

# Objective function
model.objective = pyo.Objective(doc='Total production cost (Objective Function)', sense=pyo.minimize, expr=sum(model.pProductionCost[g] * sum(model.p[g, :, :]) for g in model.g) +
                                                                                                           (sum(model.vSlack_DemandNotServed[:, :, :]) + sum(model.vSlack_OverProduction[:, :, :])) * model.pSlackPrice)


# Helper function to pretty-print the values of a Pyomo indexed variable within zone of interest
def pprint_var(var, zoi, index_positions: list = None, decimals: int = 2):
    if index_positions is None:
        index_positions = [0]

    key_list = ["Key"]
    lower_list = ["Lower"]
    value_list = ["Value"]
    upper_list = ["Upper"]
    fixed_list = ["Fixed"]
    stale_list = ["Stale"]
    domain_list = ["Domain"]

    for index in var:
        # check if at least one index is in zone of interest
        if not any(i in zoi for i in index):
            continue
        key_list.append(str(index))
        lower_list.append(f"{var[index].lb:.2f}" if var[index].has_lb() else str(var[index].lb))
        value_list.append(f"{pyo.value(var[index]):.2f}" if not var[index].value is None else str(var[index].value))
        upper_list.append(f"{var[index].ub:.2f}" if var[index].has_ub() else str(var[index].ub))
        fixed_list.append(str(var[index].fixed))
        stale_list.append(str(var[index].stale))
        domain_list.append(str(var[index].domain.name))

    key_spacer = len(max(key_list, key=len))
    lower_spacer = len(max(lower_list, key=len))
    value_spacer = len(max(value_list, key=len))
    upper_spacer = len(max(upper_list, key=len))
    fixed_spacer = len(max(fixed_list, key=len))
    stale_spacer = len(max(stale_list, key=len))
    domain_spacer = len(max(domain_list, key=len))

    print(f"{var.name} : {var.doc}")
    print(f"    Size={len(var)}, In Zone of Interest={len(key_list) - 1}, Index={var.index_set()}")

    # Iterate over all lists and print the values
    for i in range(len(value_list)):
        print(f"    {key_list[i]:>{key_spacer}} : {lower_list[i]:>{lower_spacer}} : {value_list[i]:>{value_spacer}} : {upper_list[i]:>{upper_spacer}} : {fixed_list[i]:>{fixed_spacer}} : {stale_list[i]:>{stale_spacer}} : {domain_list[i]:>{domain_spacer}}")


if __name__ == '__main__':
    from pyomo.opt import SolverFactory

    opt = SolverFactory("gurobi")
    results = opt.solve(model)
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
    print("Objective value:", pyo.value(model.objective))

    print("Done")
