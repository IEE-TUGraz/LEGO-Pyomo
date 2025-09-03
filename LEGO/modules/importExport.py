import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGOUtilities


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.hubs = pyo.Set(doc='Import/Export hubs', initialize=cs.dPower_ImpExpHubs.index.unique(level=0))
    model.hubConnections = pyo.Set(doc='Nodes connected to hub', initialize=cs.dPower_ImpExpHubs.index, within=model.hubs * model.i)

    # Parameters
    model.pHubConnectionPmaxImport = pyo.Param(model.hubConnections, doc='Maximum import power at hub connection', initialize=cs.dPower_ImpExpHubs['Pmax Import'])
    model.pHubConnectionPmaxExport = pyo.Param(model.hubConnections, doc='Maximum export power at hub connection', initialize=cs.dPower_ImpExpHubs['Pmax Export'])
    model.pImpExpPrice = pyo.Param(model.rp, model.k, model.hubs, doc='Imp-/Export price at hub', initialize=cs.dPower_ImpExpProfiles['Price'].reorder_levels(["rp", "k", "hub"]))
    model.pImpExpValue = pyo.Param(model.rp, model.k, model.hubs, doc='Imp-/Export value at hub', initialize=cs.dPower_ImpExpProfiles['ImpExp'].reorder_levels(["rp", "k", "hub"]))
    model.pImpType = pyo.Param(model.hubs, doc='Import type at hub', initialize=cs.dPower_ImpExpHubs['Import Type'].groupby('hub').first(), within=pyo.Any)
    model.pExpType = pyo.Param(model.hubs, doc='Export type at hub', initialize=cs.dPower_ImpExpHubs['Export Type'].groupby('hub').first(), within=pyo.Any)

    # Variables
    def vImpExp_bounds(model, rp, k, hub, i):
        if model.pImpExpValue[rp, k, hub] >= 0:  # Only importing allowed
            return 0, min(model.pImpExpValue[rp, k, hub], model.pHubConnectionPmaxImport[hub, i])
        else:  # Only exporting allowed
            return max(model.pImpExpValue[rp, k, hub], -model.pHubConnectionPmaxExport[hub, i]), 0

    model.vImpExp = pyo.Var(model.rp, model.k, model.hubConnections, doc='Import/Export at hub connection', bounds=vImpExp_bounds)
    second_stage_variables += [model.vImpExp]

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    # Enforce ImpFix/ImpMax and ExpFix/ExpMax
    def eImpExp_rule(model, rp, k, hub, hub_i):
        if model.pImpExpValue[rp, k, hub] >= 0:  # Only importing allowed
            if model.pImpType[hub] == 'ImpFix':
                return sum(model.vImpExp[rp, k, hub, i] for i in hub_i[hub]) == model.pImpExpValue[rp, k, hub]
            elif model.pImpType[hub] == 'ImpMax':
                return sum(model.vImpExp[rp, k, hub, i] for i in hub_i[hub]) <= model.pImpExpValue[rp, k, hub]
            else:
                raise ValueError(f"Unknown import type for hub '{hub}': '{model.pImpType[hub]}'")
        else:  # Only exporting allowed
            if model.pExpType[hub] == 'ExpFix':
                return sum(model.vImpExp[rp, k, hub, i] for i in hub_i[hub]) == model.pImpExpValue[rp, k, hub]
            elif model.pExpType[hub] == 'ExpMax':
                return sum(model.vImpExp[rp, k, hub, i] for i in hub_i[hub]) >= model.pImpExpValue[rp, k, hub]
            else:
                raise ValueError(f"Unknown export type for hub '{hub}': '{model.pExpType[hub]}'")

    hub_i = {}  # Precompute list of nodes connected to each hub
    for hub, i in model.hubConnections:
        if hub not in hub_i:
            hub_i[hub] = []
        hub_i[hub].append(i)
    model.eImpExp = pyo.Constraint(model.rp, model.k, model.hubs, doc='Imp-/Export sum at each hub', rule=lambda m, rp, k, hub: eImpExp_rule(m, rp, k, hub, hub_i))

    # Add import/export to power-balance of each node in each hour
    for rp in model.rp:
        for k in model.k:
            for hub, i in model.hubConnections:
                model.eDC_BalanceP_expr[rp, k, i] += model.vImpExp[rp, k, hub, i]

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0

    # Add import/export cost/revenues to total cost
    second_stage_objective = sum(model.vImpExp[rp, k, hub, i] * model.pImpExpPrice[rp, k, hub] for rp in model.rp for k in model.k for hub, i in model.hubConnections)

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
