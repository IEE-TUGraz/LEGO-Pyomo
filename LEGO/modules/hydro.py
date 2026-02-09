import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGO, LEGOUtilities

printer = Printer.getInstance()


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.Hydroplants = pyo.Set(initialize=cs.dPower_HydroAssets.index, doc='Hydro plants')
    LEGO.addToSet(model, "g", model.Hydroplants)
    LEGO.addToSet(model, "gi", cs.dPower_HydroAssets.reset_index().set_index(['g', 'i']).index)

    model.PumpNetwork = pyo.Set(dimen=2, initialize=cs.dPower_HydroNetwork[cs.dPower_HydroNetwork["TurbineOrPump"] == 1].index, doc='Allowed pump connections between hydro plants', domain=model.Hydroplants * model.Hydroplants)
    model.HydroDownstreamNetwork = pyo.Set(dimen=2, initialize=cs.dPower_HydroNetwork[cs.dPower_HydroNetwork["TurbineOrPump"] == 0].index, doc='Downstream network for hydro plants', domain=model.Hydroplants * model.Hydroplants)

    # Parameters
    model.pMaxProdWater = pyo.Param(model.Hydroplants, initialize=cs.dPower_HydroAssets["MaxProdWater"], doc='Maximum production rate for hydro plants [water amount]')
    LEGO.addToParameter(model, "pMaxProd", cs.dPower_HydroAssets["MaxProdWater"] * cs.dPower_HydroAssets["PowerFactorTurbine"], 'Maximum production rate for hydro plants [energy amount]')
    LEGO.addToParameter(model, "pExisUnits", cs.dPower_HydroAssets["ExisUnits"], 'Existing units for hydro plants')
    LEGO.addToParameter(model, "pMaxInvest", cs.dPower_HydroAssets["MaxInvest"], 'Maximum investable units for hydro plants')
    LEGO.addToParameter(model, "pEnabInv", cs.dPower_HydroAssets["EnableInvest"], 'Enable investment for hydro plants')
    LEGO.addToParameter(model, "pInvestCost", cs.dPower_HydroAssets["InvestCostPerMW"], 'Investment cost for hydro plants [€/energy amount]')
    model.pEnergy2PowerRatio = pyo.Param(model.Hydroplants, initialize=cs.dPower_HydroAssets["Ene2PowRatio"], doc='Energy to Power Ratio (how many time-steps can the unit discharge from a full reservoir)')
    model.pLowerLimitReservoir = pyo.Param(model.Hydroplants, initialize=cs.dPower_HydroAssets["MinReserve"] * cs.dPower_HydroAssets["StorageCapacity"], doc='Lower limit of reservoir levels for hydro plants [water amount]')  # TODO: not yet used, but can be added if needed
    model.pInitialStorage = pyo.Param(model.Hydroplants, initialize=cs.dPower_HydroAssets["IniReserve"] * cs.dPower_HydroAssets["StorageCapacity"], doc='Initial storage levels for hydro plants [water amount]')
    model.pPowerFactorTurbine = pyo.Param(model.Hydroplants, initialize=cs.dPower_HydroAssets["PowerFactorTurbine"], doc='Power factor for hydro plants (water amount to energy output)')
    model.pPowerFactorPumps = pyo.Param(model.Hydroplants, initialize=cs.dPower_HydroAssets["PowerFactorPump"], doc='Power factors for pumps between hydro plants')
    LEGO.addToParameter(model, "pOMVarCost", cs.dPower_HydroAssets["OMVarCostTurbine"], 'Variable O&M cost for hydro plants')
    model.pCostPumps = pyo.Param(model.Hydroplants, initialize=cs.dPower_HydroAssets["OMVarCostPump"], doc='Pump costs between hydro plants')
    model.pPumplimit = pyo.Param(model.Hydroplants, initialize=cs.dPower_HydroAssets["MaxPumpWater"], doc='Maximum pumping capacity between hydro plants [water amount]')

    model.pDistributionFactor = pyo.Param(model.HydroDownstreamNetwork, initialize=cs.dPower_HydroNetwork[cs.dPower_HydroNetwork["TurbineOrPump"] == 0]["DistributionFactor"], doc='Distribution factors for cascade nodes')

    model.pInflowRiver = pyo.Param(model.rp, model.k, model.Hydroplants, initialize=cs.dPower_Inflows_WaterAmount["value"], default=0, doc='Flow of river for hydro plants at certain time steps [water amount]')

    # Variables
    model.vTotalIntake = pyo.Var(model.rp, model.k, model.Hydroplants, bounds=lambda m, rp, k, i: (0, m.pMaxProdWater[i]), doc='Inflow rate into the hydro plants / actual intake of the hydro plant [water amount]')
    second_stage_variables += [model.vTotalIntake]
    model.vStorage = pyo.Var(model.rp, model.k, model.Hydroplants, bounds=lambda m, rp, k, i: (m.pLowerLimitReservoir[i], m.pEnergy2PowerRatio[i] * model.pMaxProdWater[i]), doc='Storage level of the reservoir at the hydro plants [water amount]')
    second_stage_variables += [model.vStorage]
    model.vPumpedWater = pyo.Var(model.rp, model.k, model.PumpNetwork, bounds=lambda m, rp, k, i, j: (0, m.pPumplimit[i]), doc='Pumped water between hydro plants [water amount]')
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
    model.eHydroConversion = pyo.Constraint(model.rp, model.k, model.Hydroplants, rule=lambda m, rp, k, g: m.vGenP[rp, k, g] == m.vTotalIntake[rp, k, g] * m.pPowerFactorTurbine[g], doc='Water to energy conversion for Hydroplants')

    model.ePumpConversion = pyo.Constraint(model.rp, model.k, model.PumpNetwork, rule=lambda m, rp, k, i, j: m.vConsumptionPumps[rp, k, i, j] == m.vPumpedWater[rp, k, i, j] * m.pPowerFactorPumps[i], doc='Energy use definition for pumps between hydro plants')

    HydroPlantsToNodes = {g: i for g, i in cs.dPower_HydroAssets.reset_index().set_index(['g', 'i']).index}
    # Add consumption from pumping to power balance
    for rp in model.rp:
        for k in model.k:
            for (i, j) in model.PumpNetwork:
                model.eDC_BalanceP_expr[rp, k, HydroPlantsToNodes[i]] -= model.vConsumptionPumps[rp, k, i, j]

    # Upstream plants for each hydro plant based on the cascade network structure
    model.UpstreamPlants = pyo.Set(model.Hydroplants, initialize={i: [u for (u, d) in model.HydroDownstreamNetwork if d == i] for i in model.Hydroplants})

    def eHydroBalance_rule(model, rp, k, g):
        pumped_out = sum(model.vPumpedWater[rp, k, i, j] for (i, j) in model.PumpNetwork if g == i)
        if len(model.rp) == 1 and model.k.first() == k:
            return model.vStorage[rp, k, g] == model.pInitialStorage[g] + model.pInflowRiver[rp, k, g] - model.vTotalIntake[rp, k, g] - pumped_out + model.vSlackWNS[rp, k, g] - model.vSlackWES[rp, k, g]
        elif len(model.rp) > 1:
            raise NotImplementedError("Multiple RPs not yet implemented for hydro balance.")
        else:
            inflow_from_upstream = sum(model.vTotalIntake[rp, model.k.prevw(k), u] * model.pDistributionFactor[u, g] for u in model.UpstreamPlants[g])

            pumped_in = sum(model.vPumpedWater[rp, model.k.prevw(k), i, j] for (i, j) in model.PumpNetwork if g == j)
            return model.vStorage[rp, k, g] == model.vStorage[rp, model.k.prevw(k), g] + inflow_from_upstream + model.pInflowRiver[rp, k, g] - model.vTotalIntake[rp, k, g] + pumped_in - pumped_out + model.vSlackWNS[rp, k, g] - model.vSlackWES[rp, k, g]

    model.eHydroBalance = pyo.Constraint(model.rp, model.k, model.Hydroplants, rule=eHydroBalance_rule, doc='Hydro balance for hydro plants based on graph structure')

    def eStorageLeveling_rule(model, rp, k, g):
        # enforce that storage levels at the end of the time horizon are the same as at the beginning for each hydro plant and each scenario, to avoid end-of-horizon effects and ensure cyclic operation
        if k == model.k.last():
            return model.vStorage[rp, k, g] == model.pInitialStorage[g]
        else:
            return pyo.Constraint.Skip

    model.eStorageLeveling = pyo.Constraint(model.rp, model.k, model.Hydroplants,
                                        rule=eStorageLeveling_rule,
                                        doc='Cyclic storage constraint to ensure storage levels at the end of the time horizon are the same as at the beginning')

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    # TODO: Add weight of RP and K
    second_stage_objective = (sum(model.vPumpedWater[rp, k, i, j] * model.pCostPumps[i] for (i, j) in model.PumpNetwork for rp in model.rp for k in model.k)  # Cost for pumping
                              + sum(model.vSlackWES[rp, k, g] * 0 for g in model.Hydroplants for rp in model.rp for k in model.k)  # Cost for excess water served  # TODO: Scale slack cost with cost factor
                              + sum(model.vSlackWNS[rp, k, g] * 10000 for g in model.Hydroplants for rp in model.rp for k in model.k)  # Cost for unmet water inflow  # TODO: Scale slack cost with cost factor
                              )

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
