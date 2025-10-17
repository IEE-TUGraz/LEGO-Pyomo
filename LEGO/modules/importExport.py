import typing

import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGOUtilities


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> typing.Tuple[typing.List[pyo.Var], typing.List[pyo.Var]]:
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.hubs = pyo.Set(doc='Import/Export hubs', initialize=cs.dPower_ImportExport.index.get_level_values("hub").unique())
    model.hubConnections = pyo.Set(doc='Nodes connected to hub', initialize=cs.dPower_ImportExport.reset_index().set_index(["hub", "i"]).index.unique(), within=model.hubs * model.i)

    # Parameters
    model.pImpExpMinimum = pyo.Param(model.rp, model.k, model.hubConnections, doc='Minimum Imp-/Export value at hub', initialize=cs.dPower_ImportExport['ImpExpMinimum'].reorder_levels(["rp", "k", "hub", "i"]))
    model.pImpExpMaximum = pyo.Param(model.rp, model.k, model.hubConnections, doc='Maximum Imp-/Export value at hub', initialize=cs.dPower_ImportExport['ImpExpMaximum'].reorder_levels(["rp", "k", "hub", "i"]))
    model.pImpExpPrice = pyo.Param(model.rp, model.k, model.hubConnections, doc='Imp-/Export price at hub', initialize=cs.dPower_ImportExport['ImpExpPrice'].reorder_levels(["rp", "k", "hub", "i"]))
    if not cs.dPower_Parameters['pEnableSOCP']:
        model.vImpExp = pyo.Var(model.rp, model.k, model.hubConnections, doc='Import/Export at hub connection', bounds=lambda m, rp, k, hub, i: (model.pImpExpMinimum[rp, k, hub, i], model.pImpExpMaximum[rp, k, hub, i]))
        second_stage_variables += [model.vImpExp]
    else:
        model.vImpExp = pyo.Var(model.rp, model.k, model.hubConnections, doc='Import/Export at hub connection', bounds=lambda m, rp, k, hub, i: (model.pImpExpMinimum[rp, k, hub, i], model.pImpExpMaximum[rp, k, hub, i]))
        model.vImpExpQ = pyo.Var(model.rp, model.k, model.hubConnections, doc='Reactive power associated with Import/Export at hub connection', bounds=lambda m, rp, k, hub, i: (model.pImpExpMinimum[rp, k, hub, i], model.pImpExpMaximum[rp, k, hub, i]))
        second_stage_variables += [model.vImpExp, model.vImpExpQ]

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    # Add import/export to power-balance of each node in each hour
    for rp in model.rp:
        for k in model.constraintsActiveK:
            for hub, i in model.hubConnections:
                model.eDC_BalanceP_expr[rp, k, i] += model.vImpExp[rp, k, hub, i]
                if cs.dPower_Parameters['pEnableSOCP']:
                    model.eSOCP_BalanceQ_expr[rp, k, i] += model.vImpExpQ[rp, k, hub, i]
    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0

    # Add import/export cost/revenues to total cost
    if not cs.dPower_Parameters['pEnableSOCP']:
        second_stage_objective = sum(model.vImpExp[rp, k, hub, i] * model.pImpExpPrice[rp, k, hub, i] for rp in model.rp for k in model.constraintsActiveK for hub, i in model.hubConnections)
    else:
        second_stage_objective = sum((model.vImpExp[rp, k, hub, i] + model.vImpExpQ[rp, k, hub, i]) * model.pImpExpPrice[rp, k, hub, i] for rp in model.rp for k in model.constraintsActiveK for hub, i in model.hubConnections)


    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
