import numpy as np
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGO, LEGOUtilities

DEFHydroplants = ['P1', 'P2', 'P3', 'P4']
DEFHydroplantsToNodes = {'P1': 'Node_5', 'P2': 'Node_6', 'P3': 'Node_7', 'P4': 'Node_8'}
DEFPumpNetwork = [('P2', 'P1'), ('P4', 'P1')]
DEFHydroDownstreamNetwork = [('P1', 'P2'), ('P1', 'P3'), ('P2', 'P4'), ('P3', 'P4')]
DEFMaxProdWater = {'P1': 100, 'P2': 150, 'P3': 200, 'P4': 250}  # Example max inflow rates
DEFCapacityReservoir = {'P1': 5000, 'P2': 6000, 'P3': 7000, 'P4': 8000}  # Example reservoir capacities
DEFLowerLimitReservoir = {'P1': 50, 'P2': 60, 'P3': 70, 'P4': 80}  # Example lower limits for reservoirs
DEFInitialStorage = {'P1': 50, 'P2': 100, 'P3': 140, 'P4': 280}  # Initial storage levels
DEFPowerFactor = {'P1': 1.5, 'P2': 1.4, 'P3': 1.5, 'P4': 1.6}  # Example power factors
DEFHydroOMVarCost = {'P1': 5, 'P2': 6, 'P3': 7, 'P4': 8}  # Example O&M variable costs
DEFHydroPumpVarCost = {('P2', 'P1'): 1, ('P4', 'P1'): 1}  # Example pump variable costs
DEFHydroDistributionFactor = {('P1', 'P2'): 0.5,  # 50% of the inflow from P1 goes to P2
                              ('P1', 'P3'): 0.5,  # 50% of the inflow from P1 goes to P3
                              ('P2', 'P4'): 1.0,  # 100% of the inflow from P2 goes to P4
                              ('P3', 'P4'): 1.0}  # 100% of the inflow from P3 goes to P4
DEFPowerFactorPumps = {('P2', 'P1'): 1.2,  # Example power factor for pumping from P2 to P1
                       ('P4', 'P1'): 1.3}  # Example power factor for pumping from P4 to P1
DEFHydroPumplimit = {('P2', 'P1'): 80,  # Example maximum pumping capacity from P2 to P1
                     ('P4', 'P1'): 90}  # Example maximum pumping capacity from P4 to P1

printer = Printer.getInstance()


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # From Frauental Tool
    printer.error("USING RANDOM INFLOWS FOR HYDRO - REPLACE WITH REAL DATA")
    DEFInflows = {(rp, k, g): np.random.randint(50, 160) for rp in model.rp for k in model.k for g in DEFHydroplants}

    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.Hydroplants = pyo.Set(initialize=DEFHydroplants, doc='Hydro plants')
    LEGO.addToSet(model, "g", DEFHydroplants)
    LEGO.addToSet(model, "gi", [(k, v) for k, v in DEFHydroplantsToNodes.items()])  # Note: Add gi after g since it depends on g

    model.PumpNetwork = pyo.Set(dimen=2, initialize=DEFPumpNetwork, doc='Allowed pump connections between hydro plants')
    model.HydroDownstreamNetwork = pyo.Set(dimen=2, initialize=DEFHydroDownstreamNetwork, doc='Downstream network for hydro plants')

    # Parameters
    model.pMaxProdWater = pyo.Param(model.Hydroplants, initialize=DEFMaxProdWater, doc='Maximum production rate for hydro plants [water amount]')
    LEGO.addToParameter(model, "pMaxProd", {g: maxWater * DEFPowerFactor[g] for g, maxWater in DEFMaxProdWater.items()}, 'Maximum production rate for hydro plants [energy amount]')
    LEGO.addToParameter(model, "pExisUnits", {g: 1 for g in DEFHydroplants}, 'Existing units for hydro plants')
    LEGO.addToParameter(model, "pMaxInvest", {g: 0 for g in DEFHydroplants}, 'Maximum investable units for hydro plants')
    LEGO.addToParameter(model, "pEnabInv", {g: 0 for g in DEFHydroplants}, 'Enable investment for hydro plants')
    LEGO.addToParameter(model, "pInvestCost", {g: 0 for g in DEFHydroplants}, 'Investment cost for hydro plants [€/energy amount]')
    model.pInflowRiver = pyo.Param(model.rp, model.k, model.Hydroplants, initialize=DEFInflows, doc='Flow of river for hydro plants at certain time steps [water amount]')
    model.pCapacityReservoir = pyo.Param(model.Hydroplants, initialize=DEFCapacityReservoir, doc='Capacity of reservoirs for hydro plants [water amount]')  # TODO: not yet used, but can be added if needed
    model.pLowerLimitReservoir = pyo.Param(model.Hydroplants, initialize=DEFLowerLimitReservoir, doc='Lower limit of reservoir levels for hydro plants [water amount]')  # TODO: not yet used, but can be added if needed
    model.pInitialStorage = pyo.Param(model.Hydroplants, initialize=DEFInitialStorage, doc='Initial storage levels for hydro plants [water amount]')
    model.pPowerFactor = pyo.Param(model.Hydroplants, initialize=DEFPowerFactor, doc='Power factor for hydro plants (water amount to energy output)')
    model.pCostPumps = pyo.Param(model.PumpNetwork, initialize=DEFHydroPumpVarCost, doc='Pump costs between hydro plants')
    LEGO.addToParameter(model, "pOMVarCost", DEFHydroOMVarCost, 'Variable O&M cost for hydro plants')
    model.pDistributionFactor = pyo.Param(model.HydroDownstreamNetwork, initialize=DEFHydroDistributionFactor, doc='Distribution factors for cascade nodes')
    model.pPowerFactorPumps = pyo.Param(model.PumpNetwork, initialize=DEFPowerFactorPumps, doc='Power factors for pumps between hydro plants')
    model.pPumplimit = pyo.Param(model.PumpNetwork, initialize=DEFHydroPumplimit, doc='Maximum pumping capacity between hydro plants [water amount]')

    # Variables
    model.vTotalIntake = pyo.Var(model.rp, model.k, model.Hydroplants, bounds=lambda m, rp, k, i: (0, m.pMaxProdWater[i]), doc='Inflow rate into the hydro plants / actual intake of the hydro plant [water amount]')
    second_stage_variables += [model.vTotalIntake]
    model.vStorage = pyo.Var(model.rp, model.k, model.Hydroplants, bounds=lambda m, rp, k, i: (m.pLowerLimitReservoir[i], m.pCapacityReservoir[i]), doc='Storage level of the reservoir at the hydro plants [water amount]')
    second_stage_variables += [model.vStorage]
    model.vPumpedWater = pyo.Var(model.rp, model.k, model.PumpNetwork, bounds=lambda m, rp, k, i, j: (0, m.pPumplimit[i, j]), doc='Pumped water between hydro plants [water amount]')
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
    model.eHydroConversion = pyo.Constraint(model.rp, model.k, model.Hydroplants, rule=lambda m, rp, k, g: m.vGenP[rp, k, g] == m.vTotalIntake[rp, k, g] * m.pPowerFactor[g], doc='Water to energy conversion for Hydroplants')

    model.ePumpConversion = pyo.Constraint(model.rp, model.k, model.PumpNetwork, rule=lambda m, rp, k, i, j: m.vConsumptionPumps[rp, k, i, j] == m.vPumpedWater[rp, k, i, j] * m.pPowerFactorPumps[i, j], doc='Energy use definition for pumps between hydro plants')

    # Add consumption from pumping to power balance
    for rp in model.rp:
        for k in model.k:
            for (i, j) in model.PumpNetwork:
                model.eDC_BalanceP_expr[rp, k, DEFHydroplantsToNodes[i]] -= model.vConsumptionPumps[rp, k, i, j]

    # Upstream plants for each hydro plant based on the cascade network structure
    model.UpstreamPlants = pyo.Set(model.Hydroplants, initialize={i: [u for (u, d) in model.HydroDownstreamNetwork if d == i] for i in model.Hydroplants})

    def eHydroBalance_rule(model, rp, k, g):
        pumped_out = sum(model.vPumpedWater[rp, k, i, j] for (i, j) in model.PumpNetwork if g == i)
        if len(model.rp) == 1 and model.k.first() == k:
            return model.vStorage[rp, k, g] == model.pInitialStorage[g] + model.pInflowRiver[rp, k, g] - model.vTotalIntake[rp, k, g] - pumped_out + model.vSlackWNS[rp, k, g] - model.vSlackWES[rp, k, g]
        elif len(model.rp) > 1:

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
