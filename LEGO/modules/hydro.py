import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGO, LEGOUtilities

DEFHydroplants = ['P1', 'P2', 'P3', 'P4']
DEFHydroplantsToNodes = [('P1', 'Node_5'), ('P2', 'Node_6'), ('P3', 'Node_7'), ('P4', 'Node_8')]
DEFPumpNetwork = [('P2', 'P1'), ('P4', 'P1')]
DEFHydroDownstreamNetwork = [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P4'), ('P3', 'P4')]


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.Hydroplants = pyo.Set(initialize=DEFHydroplants, doc='Hydro plants')
    LEGO.addToSet(model, "g", DEFHydroplants)
    LEGO.addToSet(model, "gi", DEFHydroplantsToNodes)  # Note: Add gi after g since it depends on g

    model.PumpNetwork = pyo.Set(dimen=2, initialize=DEFPumpNetwork, doc='Allowed pump connections between hydro plants')
    model.HydroDownstreamNetwork = pyo.Set(dimen=2, initialize=DEFHydroDownstreamNetwork, doc='Downstream network for hydro plants')

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
    model.pPowerFactor = pyo.Param(model.Hydroplants, initialize={'P1': 1.5, 'P2': 1.4, 'P3': 1.5, 'P4': 1.6
                                                                  }, doc='Power factor for hydro plants (water amount to energy output)')  # Example power factors, replace with actual data
    model.pDemand = pyo.Param(model.T, initialize={1: 100, 2: 1000, 3: 900}, doc='Demand for each time step')  # Example demand, replace with actual data
    model.pCost = pyo.Param(model.Hydroplants, initialize={'P1': 20, 'P2': 25, 'P3': 30, 'P4': 35}, doc='Cost of production for hydro plants')  # Example costs, replace with actual data
    model.pCostPumps = pyo.Param(model.PumpNetwork, initialize={
        ('P2', 'P1'): 1,
        ('P4', 'P1'): 1
    }, doc='Pump costs between hydro plants')
    model.pDistributionFactor = pyo.Param(model.HydroDownstreamNetwork, initialize={
        ('P1', 'P2'): 0.5,  # 50% of the inflow from P1 goes to P2
        ('P1', 'P3'): 0.5,  # 50% of the inflow from P1 goes to P3
        ('P2', 'P4'): 1.0,  # 100% of the inflow from P2 goes to P4
        ('P3', 'P4'): 1.0  # 100% of the inflow from P3 goes to P4
    }, doc='Distribution factors for cascade nodes')
    model.pPowerFactorPumps = pyo.Param(model.PumpNetwork, initialize={
        ('P2', 'P1'): 1.2,  # Example power factor for pumping from P2 to P1
        ('P4', 'P1'): 1.3  # Example power factor for pumping from P4 to P1
    }, doc='Power factors for pumps between hydro plants')

    # Variables
    model.vTotalIntake = pyo.Var(model.rp, model.k, model.Hydroplants, bounds=lambda m, rp, k, i: (0, m.pMaxProdWater[i]), doc='Inflow rate into the hydro plants / actual intake of the hydro plant [water amount]')
    second_stage_variables += [model.vTotalIntake]
    model.vStorage = pyo.Var(model.rp, model.k, model.Hydroplants, bounds=lambda m, rp, k, i: (m.pLowerLimitReservoir[i], m.pCapacityReservoir[i]), doc='Storage level of the reservoir at the hydro plants [water amount]')
    second_stage_variables += [model.vStorage]
    model.vPumpedWater = pyo.Var(model.rp, model.k, model.PumpNetwork, domain=lambda m, rp, k, i, j: (0, m.pPumplimit[i, j]), doc='Pumped water between hydro plants [water amount]')
    second_stage_variables += [model.vPumpedWater]
    model.vConsumptionPumps = pyo.Var(model.rp, model.k, model.PumpNetwork, domain=pyo.NonNegativeReals, doc='Energy used for pumping between hydro plants [energy]')
    second_stage_variables += [model.vConsumptionPumps]
    model.vSlackWNS = pyo.Var(model.rp, model.k, model.Hydroplants, domain=pyo.NonNegativeReals, doc='Slack variable for water not served (unmet water inflow)')
    second_stage_variables += [model.vSlackWNS]
    model.vSlackWES = pyo.Var(model.rp, model.k, model.Hydroplants, domain=pyo.NonNegativeReals, doc='Slack variable for excess water served (overflowing reservoir)')
    second_stage_variables += [model.vSlackWES]

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    model.eHydroConversion = pyo.Constraint(model.rp, model.k, model.Hydroplants, rule=lambda m, rp, k, g: m.vGenP[g, rp, k] == m.vTotalIntake[g, rp, k] * m.pPowerFactor[g], doc='Water to energy conversion for Hydroplants')

    model.ePumpConversion = pyo.Constraint(model.rp, model.k, model.PumpNetwork, rule=lambda m, rp, k, i, j: m.vConsumptionPumps[rp, k, i, j] == m.vPumpedWater[rp, k, i, j] * m.pPowerFactorPumps[i, j], doc='Energy use definition for pumps between hydro plants')

    # Add consumption from pumping to power balance
    for rp in model.rp:
        for k in model.k:
            for (i, j) in model.PumpNetwork:
                model.eDC_BalanceP_expr[rp, k, i] -= model.vConsumptionPumps[rp, k, i, j]

    # Upstream plants for each hydro plant based on the cascade network structure
    pyo.Set(model.Hydroplants, initialize={i: [u for (u, d) in model.HydroDownstreamNetwork if d == i] for i in model.Hydroplants})

    def eHydroBalance_rule(model, rp, k, g):
        pumped_out = sum(model.vPumpedWater[rp, k, i, j] for (i, j) in model.PumpNetwork if g == i)
        if model.rp.card() == 1 and model.k.first() == k:
            return model.vStorage[rp, k, g] == model.pInitialStorage[g] + model.pInflowRiver[rp, k, g] - model.vTotalIntake[rp, k, g] - pumped_out + model.vSlackWNS[rp, k, g] - model.vSlackWES[rp, k, g]
        elif model.rp.card() > 1:

            inflow_from_upstream = sum(model.vTotalIntake[rp, model.k.prevw(k), u] * model.pDistributionFactor[u, g] for u in model.UpstreamPlants[g])

            pumped_in = sum(model.vPumpedWater[rp, model.k.prevw(k), i, j] for (i, j) in model.PumpNetwork if g == j)
            return model.vStorage[rp, k, g] == model.vStorage[rp, model.k.prevw(k), g] + inflow_from_upstream + model.pInflowRiver[rp, k, g] - model.vTotalIntake[rp, k, g] + pumped_in - pumped_out + model.vSlackWNS[rp, k, g] - model.vSlackWES[rp, k, g]

    model.eHydroBalance = pyo.Constraint(model.rp, model.k, model.Hydroplants, rule=eHydroBalance_rule, doc='Hydro balance for hydro plants based on graph structure')

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = (sum(model.vPumpedWater[rp, k, i, j] * model.pCostPumps[i, j] for (i, j) in model.PumpNetwork for rp in model.rp for k in model.k)  # Cost for pumping
                              + sum(model.vSlackWES[rp, k, g] * 10000 for g in model.Hydroplants for rp in model.rp for k in model.k)  # Cost for excess water served
                              + sum(model.vSlackWNS[rp, k, g] * 10000 for g in model.Hydroplants for rp in model.rp for k in model.k)  # Cost for unmet water inflow
                              )

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
