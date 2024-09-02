import logging

import pandas as pd
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints

# Settings
example_folder = "data/example/"

# Setup
pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)

# Data
dPower_BusInfo = pd.read_excel(example_folder + "Power_BusInfo.xlsx", skiprows=[0, 1, 3, 4, 5])
dPower_BusInfo = dPower_BusInfo.drop(dPower_BusInfo.columns[0], axis=1)
dPower_BusInfo = dPower_BusInfo.rename(columns={dPower_BusInfo.columns[0]: "i", dPower_BusInfo.columns[1]: "System"})
dPower_BusInfo = dPower_BusInfo.set_index('i')

dPower_Network = pd.read_excel(example_folder + "Power_Network.xlsx", skiprows=[0, 1, 3, 4, 5])
dPower_Network = dPower_Network.drop(dPower_Network.columns[0], axis=1)
dPower_Network = dPower_Network.rename(columns={dPower_Network.columns[0]: "i", dPower_Network.columns[1]: "j", dPower_Network.columns[2]: "Circuit ID"})
dPower_Network = dPower_Network.set_index(['i', 'j'])


# Function to read generator data
def read_generator_data(file_path):
    d_generator = pd.read_excel(file_path, skiprows=[0, 1, 3, 4, 5])
    d_generator = d_generator.drop(d_generator.columns[0], axis=1)
    d_generator = d_generator.rename(columns={d_generator.columns[0]: "g", d_generator.columns[1]: "tec", d_generator.columns[2]: "i"})
    d_generator = d_generator.set_index('g')
    return d_generator


dPower_ThermalGen = read_generator_data(example_folder + "Power_ThermalGen.xlsx")
dPower_RoR = read_generator_data(example_folder + "Power_RoR.xlsx")
dPower_VRES = read_generator_data(example_folder + "Power_VRES.xlsx")

dPower_Demand = pd.read_excel(example_folder + "Power_Demand.xlsx", skiprows=[0, 1, 3, 4, 5])
dPower_Demand = dPower_Demand.drop(dPower_Demand.columns[0], axis=1)
dPower_Demand = dPower_Demand.rename(columns={dPower_Demand.columns[0]: "rp", dPower_Demand.columns[1]: "i"})
dPower_Demand = dPower_Demand.melt(id_vars=['rp', 'i'], var_name='k', value_name='Demand')
dPower_Demand = dPower_Demand.set_index(['rp', 'i', 'k'])

dPower_Inflows = pd.read_excel(example_folder + "Power_Inflows.xlsx", skiprows=[0, 1, 3, 4, 5])
dPower_Inflows = dPower_Inflows.drop(dPower_Inflows.columns[0], axis=1)
dPower_Inflows = dPower_Inflows.rename(columns={dPower_Inflows.columns[0]: "rp", dPower_Inflows.columns[1]: "g"})
dPower_Inflows = dPower_Inflows.melt(id_vars=['rp', 'g'], var_name='k', value_name='Inflow')
dPower_Inflows = dPower_Inflows.set_index(['rp', 'g', 'k'])

dPower_VRESProfiles = pd.read_excel(example_folder + "Power_VRESProfiles.xlsx", skiprows=[0, 1, 3, 4, 5])
dPower_VRESProfiles = dPower_VRESProfiles.drop(dPower_VRESProfiles.columns[0], axis=1)
dPower_VRESProfiles = dPower_VRESProfiles.rename(columns={dPower_VRESProfiles.columns[0]: "rp", dPower_VRESProfiles.columns[1]: "i", dPower_VRESProfiles.columns[2]: "tec"})
dPower_VRESProfiles = dPower_VRESProfiles.melt(id_vars=['rp', 'i', 'tec'], var_name='k', value_name='Capacity')
dPower_VRESProfiles = dPower_VRESProfiles.set_index(['rp', 'i', 'k', 'tec'])

# dataframe that shows connections between g and i, only concatenating g and i from the dataframes
hGenerators_to_Buses = pd.concat([dPower_ThermalGen[['i']], dPower_RoR[['i']], dPower_VRES[['i']]])

########################################################################################################################
# Model creation
########################################################################################################################

model = pyo.ConcreteModel()

# Sets
model.i = pyo.Set(doc='Buses', initialize=dPower_BusInfo.index.tolist())
model.e = pyo.Set(doc='Lines', initialize=dPower_Network.index.tolist())
model.thermalGenerators = pyo.Set(doc='Thermal Generators', initialize=dPower_ThermalGen.index.tolist())
model.rorGenerators = pyo.Set(doc='Run-of-river generators', initialize=dPower_RoR.index.tolist())
model.vresGenerators = pyo.Set(doc='Variable renewable energy sources', initialize=dPower_VRES.index.tolist())
model.g = pyo.Set(doc='Generators', initialize=model.thermalGenerators | model.rorGenerators | model.vresGenerators)
model.rp = pyo.Set(doc='Representative periods', initialize=dPower_Demand.index.get_level_values('rp').unique().tolist())
model.k = pyo.Set(doc='Timestep within representative period', initialize=dPower_Demand.index.get_level_values('k').unique().tolist())

# Helper Sets for zone of interest
model.zoi_i = pyo.Set(doc="Buses in zone of interest", initialize=dPower_BusInfo.loc[dPower_BusInfo["ZoneOfInterest"] == "yes"].index.tolist(), within=model.i)
model.zoi_g = pyo.Set(doc="Generators in zone of interest", initialize=hGenerators_to_Buses.loc[hGenerators_to_Buses["i"].isin(model.zoi_i)].index.tolist(), within=model.g)

# Variables
model.delta = pyo.Var(model.i, model.rp, model.k, doc='Angle of bus i', bounds=(-60, 60))

model.p = pyo.Var(model.g, model.rp, model.k, doc='Power output of generator g', bounds=(0, None))
for g in model.thermalGenerators:
    model.p[g, :, :].setub(dPower_ThermalGen.loc[g, 'MaxProd'])
    # TODO: Add min production (needs unit commitment)

for g in model.rorGenerators:
    for rp in model.rp:
        for k in model.k:
            model.p[g, rp, k].setub(min(dPower_RoR.loc[g, 'MaxProd'], dPower_Inflows.loc[rp, g, k]['Inflow']))  # TODO: Check if this is correct

for g in model.vresGenerators:
    for rp in model.rp:
        for k in model.k:
            model.p[g, rp, k].setub(dPower_VRES.loc[g, 'MaxProd'] * dPower_VRESProfiles.loc[rp, dPower_VRES.loc[g, 'i'], k, dPower_VRES.loc[g, 'tec']]['Capacity'])

model.vSlack_DemandNotServed = pyo.Var(model.rp, model.k, doc='Slack variable demand not served', bounds=(0, None))
model.vSlack_OverProduction = pyo.Var(model.rp, model.k, doc='Slack variable overproduction', bounds=(0, None))
model.t = pyo.Var(model.e, model.rp, model.k, doc='Power flow from bus i to j', bounds=(None, None))
for (i, j) in model.e:
    match dPower_Network.loc[i, j]["Technical Representation"]:
        case "DC-OPF" | "TP":
            model.t[(i, j), :, :].setlb(-dPower_Network.loc[i, j]['Pmax'])
            model.t[(i, j), :, :].setub(dPower_Network.loc[i, j]['Pmax'])
        case "SN":
            continue  # No bounds for power flow in "Single Node" representation
        case _:
            raise ValueError(f"Technical representation '{dPower_Network.loc[i, j]["Technical Representation"]}' "
                             f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

# Parameters
model.pDemand = pyo.Param(model.rp, model.i, model.k, initialize=dPower_Demand['Demand'], doc='Demand at bus i in representative period rp and timestep k')

# Helper for FuelCost that has dPower_ThermalGen['FuelCost'] for ThermalGen, and 0 for all gs in ror and vres
hFuelCost = pd.concat([dPower_ThermalGen['FuelCost'].copy(), pd.Series(0, index=model.rorGenerators), pd.Series(0, index=model.vresGenerators)])
model.pProductionCost = pyo.Param(model.g, initialize=hFuelCost, doc='Production cost of generator g')

model.pReactance = pyo.Param(model.e, initialize=dPower_Network['R'], doc='Reactance of line e')
model.pSlackPrice = pyo.Param(initialize=max(model.pProductionCost.values()) * 100, doc='Price of slack variable')

# Select slack node with highest demand (TODO: Check if this is the best way to select slack node)
slack_node = dPower_Demand.groupby('i').sum().idxmax().values[0]
print("Slack node:", slack_node)
model.delta[slack_node, :, :].fix(0)

# Constraint(s)
model.cPower_Balance = pyo.ConstraintList(doc='Power balance constraint for each bus')
for i in model.i:
    for rp in model.rp:
        for k in model.k:
            model.cPower_Balance.add(
                sum(model.p[g, rp, k] for g in model.g if hGenerators_to_Buses.loc[g]['i'] == i) -  # Production of generators at bus i
                sum(model.t[e, rp, k] for e in model.e if (e[0] == i)) +  # Power flow from bus i to bus j
                sum(model.t[e, rp, k] for e in model.e if (e[1] == i)) ==  # Power flow from bus j to bus i
                model.pDemand[rp, i, k] -  # Demand at bus i
                model.vSlack_DemandNotServed[rp, k] +  # Slack variable for demand not served
                model.vSlack_OverProduction[rp, k])  # Slack variable for overproduction

model.cReactance = pyo.ConstraintList(doc='Reactance constraint for each line (for DC-OPF)')
for (i, j) in model.e:
    match dPower_Network.loc[i, j]["Technical Representation"]:
        case "DC-OPF":
            for rp in model.rp:
                for k in model.k:
                    model.cReactance.add(model.t[(i, j), rp, k] == model.pReactance[(i, j)] * (model.delta[i, rp, k] - model.delta[j, rp, k]))
        case "TP" | "SN":
            continue
        case _:
            raise ValueError(f"Technical representation '{dPower_Network.loc[i, j]["Technical Representation"]}' "
                             f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

# Objective function
model.objective = pyo.Objective(doc='Total production cost (Objective Function)', sense=pyo.minimize, expr=sum(sum(model.pProductionCost[g] * model.p[g, rp, k] for g in model.g) + (model.vSlack_DemandNotServed[rp, k] + model.vSlack_OverProduction[rp, k]) * model.pSlackPrice for rp in model.rp for k in model.k))


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
        value_list.append(f"{pyo.value(var[index]):.2f}")
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
    print("Slack variable sum of demand not served:", sum(pyo.value(model.vSlack_DemandNotServed[rp, k]) for rp in model.rp for k in model.k))
    print("Slack variable sum of overproduction:", sum(pyo.value(model.vSlack_OverProduction[rp, k]) for rp in model.rp for k in model.k))

    print("\nObjective Function Value\n" + '-' * 60)
    print("Objective value:", pyo.value(model.objective))

    print("Done")
