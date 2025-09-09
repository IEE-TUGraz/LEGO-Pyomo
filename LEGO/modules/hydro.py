import pyomo.environ as pyo
from pyomo.core import ConcreteModel

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGO, LEGOUtilities

@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Matthias Jannach Adaptations
    model = ConcreteModel()

    # Sets
    model.Hydroplants = pyo.Set(initialize=['P1', 'P2', 'P3', 'P4'], doc='Hydro plants')  # Example hydro plants, replace with actual data
    model.T = pyo.Set(initialize=[1, 2, 3])
    model.PumpPairs = pyo.Set(dimen=2, initialize=[('P2', 'P1'), ('P4', 'P1')], doc='Allowed pump connections between hydro plants')# Example time steps, replace with actual data
    model.CascadeNodes = pyo.Set(dimen=2, initialize=[('P1', 'P2'), ('P1', 'P3'), ('P2', 'P4'), ('P3', 'P4')], doc='Cascade nodes for hydro plants')

    # Parameters
    model.pMaxProdWater = pyo.Param(model.Hydroplants, initialize={'P1': 100, 'P2': 150, 'P3': 200, 'P4': 250}, doc='Maximum production rate for hydro plants [water amount]')  # Example max inflow rates
    model.pInflowRiver = pyo.Param(model.Hydroplants, model.T, initialize={
        ('P1', 1): 50, ('P1', 2): 60, ('P1', 3): 70,
        ('P2', 1): 80, ('P2', 2): 90, ('P2', 3): 100,
        ('P3', 1): 110, ('P3', 2): 120, ('P3', 3): 130,
        ('P4', 1): 140, ('P4', 2): 150, ('P4', 3): 160
    }, doc='Flow of river for hydro plants at certain time steps [water amount]')  # Example flow rates, replace with actual data
    model.pCapacityReservoir = pyo.Param(model.Hydroplants, initialize={
        'P1': 5000, 'P2': 6000, 'P3': 7000, 'P4': 8000}, doc='Capacity of reservoirs for hydro plants [water amount]')  # not yet used, but can be added if needed
    model.pLowerLimitReservoir = pyo.Param(model.Hydroplants, initialize={
        'P1': 50, 'P2': 60, 'P3': 70, 'P4': 80}, doc='Lower limit of reservoir levels for hydro plants [water amount]')  # not yet used, but can be added if needed
    model.pInitialStorage = pyo.Param(model.Hydroplants, initialize={
        'P1': 50, 'P2': 100, 'P3': 140, 'P4': 280}, doc='Initial storage levels for hydro plants [water amount]')  # Initial storage levels, replace with
    model.pPowerFactor = pyo.Param(model.Hydroplants, initialize={ 'P1': 1.5, 'P2': 1.4, 'P3': 1.5, 'P4': 1.6
    }, doc='Power factor for hydro plants (water amount to energy output)')  # Example power factors, replace with actual data
    model.pDemand = pyo.Param(model.T, initialize={1: 100, 2: 1000, 3: 900}, doc='Demand for each time step')  # Example demand, replace with actual data
    model.pCost = pyo.Param(model.Hydroplants, initialize={'P1': 20, 'P2': 25, 'P3': 30, 'P4': 35}, doc='Cost of production for hydro plants')  # Example costs, replace with actual data
    model.pCostPumps = pyo.Param(model.PumpPairs, initialize={
        ('P2', 'P1'): 1,
        ('P4', 'P1'): 1
    }, doc='Pump costs between hydro plants')
    model.pDistributionFactor = pyo.Param(model.CascadeNodes, initialize={
        ('P1', 'P2'): 0.5,  # 50% of the inflow from P1 goes to P2
        ('P1', 'P3'): 0.5,  # 50% of the inflow from P1 goes to P3
        ('P2', 'P4'): 1.0,  # 100% of the inflow from P2 goes to P4
        ('P3', 'P4'): 1.0  # 100% of the inflow from P3 goes to P4
    }, doc='Distribution factors for cascade nodes')
    model.pPowerFactorPumps = pyo.Param(model.PumpPairs, initialize={
        ('P2', 'P1'): 1.2,  # Example power factor for pumping from P2 to P1
        ('P4', 'P1'): 1.3   # Example power factor for pumping from P4 to P1
    }, doc='Power factors for pumps between hydro plants')

    # Variables
    model.vGenP = pyo.Var(model.Hydroplants, model.T, domain=pyo.NonNegativeReals, doc='Production of hydro plants [energy]')  # Production of hydro plants; Domain cannot be changed?
    second_stage_variables += [model.vGenP]
    model.vTotalIntake = pyo.Var(model.Hydroplants, model.T, bounds=lambda m, i, t: (0, m.pMaxProdWater[i]), doc='Inflow rate into the hydro plants / actual intake of the hydro plant [water amount]')  # Inflow rate for hydro plants
    second_stage_variables += [model.vTotalIntake]
    model.vStorage = pyo.Var(model.Hydroplants, model.T, bounds=lambda m, i, t: (m.pLowerLimitReservoir[i], m.pCapacityReservoir[i]), doc='Storage level of the reservoir at the hydro plants [water amount]')  # Storage level for hydro plants at certain time steps
    second_stage_variables += [model.vStorage]
    model.vPumpedWater = pyo.Var(model.PumpPairs, model.T, domain=pyo.NonNegativeReals, doc='Pumped water between hydro plants [water amount]')  # Pumped water between hydro plants
    second_stage_variables += [model.vPumpedWater]
    model.vConsumptionPumps = pyo.Var(model.PumpPairs, model.T, domain=pyo.NonNegativeReals, doc='Energy used for pumping between hydro plants [energy]')  # Energy used for pumping between hydro plants
    second_stage_variables += [model.vConsumptionPumps]
    model.vSlackPNS = pyo.Var(model.T, domain=pyo.NonNegativeReals, doc='Slack variable for unmet demand')
    second_stage_variables += [model.vSlackPNS]
    model.vSlackEPS = pyo.Var(model.T, domain=pyo.NonNegativeReals, doc='Slack variable for excess power served')
    second_stage_variables += [model.vSlackEPS]

    # Constraints
    def prod_def_rule(model, i, t):
        return model.vGenP[i, t] == model.vTotalIntake[i, t] * model.pPowerFactor[i]
    model.eProdDef = pyo.Constraint(model.Hydroplants, model.T, rule=prod_def_rule, doc='Production definition for hydro plants')  # Production definition constraint

    def use_pump(model, i, j, t):
        return model.vConsumptionPumps[i, j, t] == model.vPumpedWater[i, j, t] * model.pPowerFactorPumps[i, j]
    model.eUsePump = pyo.Constraint(model.PumpPairs, model.T, rule=use_pump, doc='Energy use definition for pumps between hydro plants')

    def demand_rule(model, t):
        return sum(model.vGenP[i, t] for i in model.Hydroplants) + model.vSlackPNS[t] - model.vSlackEPS[t] == model.pDemand[t] + sum(model.vConsumptionPumps[i, j, t] for (i, j) in model.PumpPairs)
    model.eDemand_con = pyo.Constraint(model.T, rule=demand_rule)

    # Cascade for Network of HPP (P1 -> P2, P1 -> P3, P2 -> P4, P3 -> P4)
    # Find upstream plants for each hydro plant based on the cascade network structure
    def upstream_rule(model, i):
        return [u for (u, d) in model.CascadeNodes if d == i]
    model.UpstreamPlants = pyo.Set(model.Hydroplants, initialize=upstream_rule)

    def cascade_rule_graph(model, i, t):
        pumped_out = sum(model.vPumpedWater[j, k, t] for (j, k) in model.PumpPairs if j == i)
        if t == 1:
            return model.vStorage[i, t] == model.pInitialStorage[i] + model.pInflowRiver[i, t] - model.vTotalIntake[i, t] - pumped_out
        else:
            # Sum of all inflows from upstream plants
            inflow_from_upstream = sum(model.vTotalIntake[u, model.T.prev(t)] * model.pDistributionFactor[u, i] for u in model.UpstreamPlants[i])
            # Pumps water
            pumped_in = sum(model.vPumpedWater[j, k, model.T.prev(t)] for (j, k) in model.PumpPairs if k == i)
            return model.vStorage[i, t] == model.vStorage[i, t - 1] + inflow_from_upstream + model.pInflowRiver[i, t] - model.vTotalIntake[i, t] + pumped_in - pumped_out
    model.eCascadeGraph = pyo.Constraint(model.Hydroplants, model.T, rule=cascade_rule_graph, doc='Cascade constraints for hydro plants based on graph structure')

    # Objectives
    def objective_rule(model):
        prod_cost = sum(model.vGenP[i, t] * model.pCost[i] for i in model.Hydroplants for t in model.T)
        pump_cost = sum(model.vPumpedWater[i, j, t] * model.pCostPumps[i, j] for (i, j) in model.PumpPairs for t in model.T)
        slack_cost = sum(model.vSlackPNS[t] * 10000 for t in model.T)
        excess_cost = sum(model.vSlackEPS[t] * 1 for t in model.T)
        return prod_cost + pump_cost + slack_cost + excess_cost
    model.obj_cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize, doc='Objective function for hydro plants including pump costs')

    print("Model created, trying to solve it...")
    optimizer = pyo.SolverFactory("highs")  # Use HiGHS solver for optimization
    results = optimizer.solve(model)
    model.pprint()
    exit(0)

@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):  # TODO: Adapt to Hydro

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = 0.0

    # No additional objective expressions for storage units (already handled in "power" module)

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
