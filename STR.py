import pandas as pd
import pyomo.environ as pyo

example_folder = "data/example/"

# Data
dPower_BusInfo = pd.read_excel(example_folder + "Power_BusInfo.xlsx", skiprows=[0, 1, 3, 4, 5])
dPower_BusInfo = dPower_BusInfo.drop(dPower_BusInfo.columns[0], axis=1)
dPower_BusInfo = dPower_BusInfo.rename(columns={dPower_BusInfo.columns[0]: "i", dPower_BusInfo.columns[1]: "System"})
dPower_BusInfo = dPower_BusInfo.set_index('i')

dPower_Network = pd.read_excel(example_folder + "Power_Network.xlsx", skiprows=[0, 1, 3, 4, 5])
dPower_Network = dPower_Network.drop(dPower_Network.columns[0], axis=1)
dPower_Network = dPower_Network.rename(columns={dPower_Network.columns[0]: "i", dPower_Network.columns[1]: "j", dPower_Network.columns[2]: "Circuit ID"})
dPower_Network = dPower_Network.set_index(['i', 'j'])

# TODO: Also include other generators (as subsets)
dPower_ThermalGen = pd.read_excel(example_folder + "Power_ThermalGen.xlsx", skiprows=[0, 1, 3, 4, 5])
dPower_ThermalGen = dPower_ThermalGen.drop(dPower_ThermalGen.columns[0], axis=1)
dPower_ThermalGen = dPower_ThermalGen.rename(columns={dPower_ThermalGen.columns[0]: "g", dPower_ThermalGen.columns[1]: "tec", dPower_ThermalGen.columns[2]: "i"})
dPower_ThermalGen = dPower_ThermalGen.set_index('g')

dPower_Demand = pd.read_excel(example_folder + "Power_Demand.xlsx", skiprows=[0, 1, 3, 4, 5])
dPower_Demand = dPower_Demand.drop(dPower_Demand.columns[0], axis=1)
dPower_Demand = dPower_Demand.rename(columns={dPower_Demand.columns[0]: "rp", dPower_Demand.columns[1]: "i"})
dPower_Demand = dPower_Demand.melt(id_vars=['rp', 'i'], var_name='k', value_name='Demand')
dPower_Demand = dPower_Demand.set_index(['rp', 'i', 'k'])

########################################################################################################################
# Model creation
########################################################################################################################

model = pyo.ConcreteModel()

# Sets
model.i = pyo.Set(doc='Buses', initialize=dPower_BusInfo.index.tolist())
model.e = pyo.Set(doc='Lines', initialize=dPower_Network.index.tolist())
model.g = pyo.Set(doc='Generators', initialize=dPower_ThermalGen.index.tolist())
model.rp = pyo.Set(doc='Representative periods', initialize=dPower_Demand.index.get_level_values('rp').unique().tolist())
model.k = pyo.Set(doc='Timestep within representative period', initialize=dPower_Demand.index.get_level_values('k').unique().tolist())

# Variables
model.delta = pyo.Var(model.i, model.rp, model.k, doc='Angle of bus i', bounds=(None, None))
model.p = pyo.Var(model.g, model.rp, model.k, doc='Power output of generator g', bounds=(0, None))
for g in model.g:
    model.p[g, :, :].setub(dPower_ThermalGen.loc[g, 'MaxProd'])

model.vSlack_DemandNotServed = pyo.Var(model.rp, model.k, doc='Slack variable demand not served', bounds=(0, None))
model.vSlack_OverProduction = pyo.Var(model.rp, model.k, doc='Slack variable overproduction', bounds=(0, None))
model.t = pyo.Var(model.e, model.rp, model.k, doc='Power flow from bus i to j', bounds=(None, None))
for (i, j) in model.e:
    for rp in model.rp:
        for k in model.k:
            model.t[(i, j), rp, k].setlb(-dPower_Network.loc[i, j]['Pmax'])
            model.t[(i, j), rp, k].setub(dPower_Network.loc[i, j]['Pmax'])

# Parameters
model.pDemand = pyo.Param(model.rp, model.i, model.k, initialize=dPower_Demand['Demand'], doc='Demand at bus i in representative period rp and timestep k')
model.pProductionCost = pyo.Param(model.g, initialize=dPower_ThermalGen['FuelCost'], doc='Production cost of generator g')
model.pReactance = pyo.Param(model.e, initialize=dPower_Network['R'], doc='Reactance of line e')
model.pSlackPrice = pyo.Param(initialize=max(model.pProductionCost.values()) * 100, doc='Price of slack variable')

# Slack node (TODO: Make selection of slack node dynamic)
for rp in model.rp:
    for k in model.k:
        model.delta[model.i[1], rp, k].fix(0)

# Constraint(s)
model.cPower_Balance = pyo.ConstraintList(doc='Power balance constraint for each bus')
for i in model.i:
    for rp in model.rp:
        for k in model.k:
            model.cPower_Balance.add(
                sum(model.p[g, rp, k] for g in model.g if dPower_ThermalGen.loc[g, 'i'] == i) -  # Production of generators at bus i
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
        case "TP":
            continue
        case _:
            raise ValueError(f"Technical representation '{dPower_Network.loc[i, j]["Technical Representation"]}' "
                             f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

# Objective function
model.objective = pyo.Objective(doc='Total production cost (Objective Function)', sense=pyo.minimize, expr=sum(sum(model.pProductionCost[g] * model.p[g, rp, k] for g in model.g) + (model.vSlack_DemandNotServed[rp, k] + model.vSlack_OverProduction[rp, k]) * model.pSlackPrice for rp in model.rp for k in model.k))

if __name__ == '__main__':
    from pyomo.opt import SolverFactory

    opt = SolverFactory("gurobi")
    results = opt.solve(model)
    results.write()

    print("\nDisplaying Solution\n" + '-' * 60)
    model.p.pprint()
    model.t.pprint()
    model.delta.pprint()

    # Print sum of slack variables
    print("\nSlack Variables\n" + '-' * 60)
    print("Slack variable sum of demand not served:", sum(pyo.value(model.vSlack_DemandNotServed[rp, k]) for rp in model.rp for k in model.k))
    print("Slack variable sum of overproduction:", sum(pyo.value(model.vSlack_OverProduction[rp, k]) for rp in model.rp for k in model.k))

    print("\nObjective Function Value\n" + '-' * 60)
    print("Objective value:", pyo.value(model.objective))

    print("Done")
