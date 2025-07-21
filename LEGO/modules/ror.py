import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGOUtilities, LEGO


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.rorGenerators = pyo.Set(doc='Run-of-river generators', initialize=cs.dPower_RoR.index.tolist())
    LEGO.addToSet(model, "g", model.rorGenerators)
    LEGO.addToSet(model, "gi", cs.dPower_RoR.reset_index().set_index(['g', 'i']).index)

    # Parameters
    LEGO.addToParameter(model, "pOMVarCost", cs.dPower_RoR['OMVarCost'])
    LEGO.addToParameter(model, "pEnabInv", cs.dPower_RoR['EnableInvest'])
    LEGO.addToParameter(model, "pMaxInvest", cs.dPower_RoR['MaxInvest'])
    LEGO.addToParameter(model, "pInvestCost", cs.dPower_RoR['InvestCostEUR'])
    LEGO.addToParameter(model, "pMaxProd", cs.dPower_RoR['MaxProd'])
    LEGO.addToParameter(model, "pMinProd", cs.dPower_RoR['MinProd'])
    LEGO.addToParameter(model, "pExisUnits", cs.dPower_RoR['ExisUnits'])

    LEGO.addToParameter(model, 'pMaxGenQ', cs.dPower_RoR['Qmax'])
    LEGO.addToParameter(model, 'pMinGenQ', cs.dPower_RoR['Qmin'])

    # Variables
    for g in model.rorGenerators:
        for rp in model.rp:
            for k in model.k:
                model.vGenP[rp, k, g].setub(min(model.pMaxProd[g], cs.dPower_Inflows.loc[rp, g, k]['Inflow']))  # TODO: Check and adapt for storage

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = 0.0

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
