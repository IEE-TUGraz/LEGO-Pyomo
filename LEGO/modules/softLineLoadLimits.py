import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGOUtilities


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Parameters
    model.pMaxLineLoad = pyo.Param(doc='Scaling factor for pMax (rest to 100% is soft limit)', initialize=cs.dPower_Parameters["pMaxLineLoad"])
    model.pLOLCost = pyo.Param(doc='Cost of exceeding line load limit [€/MWh]', initialize=cs.dPower_Parameters["pLOLCost"])

    # Variables
    model.vLineOverload = pyo.Var(model.rp, model.k, model.la, doc='Line overload [% above Pmax]', bounds=lambda model, rp, k, i, j, c: (0, 1 - model.pMaxLineLoad))
    second_stage_variables += [model.vLineOverload]

    # Checks
    if not 0 <= pyo.value(model.pMaxLineLoad) <= 1:
        raise ValueError(f"pMaxLineLoad must be between 0 and 1 (got {pyo.value(model.pMaxLineLoad)})")

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    model.eSoftLineLoadLimitPos = pyo.Constraint(model.rp, model.constraintsActiveK, model.le, doc='Positive soft limit for line load of existing lines',
                                                 rule=lambda model, rp, k, i, j, c: model.vLineP[rp, k, i, j, c] <= model.pPmax[i, j, c] * model.pMaxLineLoad + model.vLineOverload[rp, k, i, j, c] * model.pPmax[i, j, c])

    model.eSoftLineLoadLimitNeg = pyo.Constraint(model.rp, model.constraintsActiveK, model.le, doc='Negative soft limit for line load of existing lines',
                                                 rule=lambda model, rp, k, i, j, c: model.vLineP[rp, k, i, j, c] >= -model.pPmax[i, j, c] * model.pMaxLineLoad - model.vLineOverload[rp, k, i, j, c] * model.pPmax[i, j, c])

    model.eSoftLineLoadLimitCanPos = pyo.Constraint(model.rp, model.constraintsActiveK, model.lc, doc='Positive soft limit for line load of candidate lines',
                                                    rule=lambda model, rp, k, i, j, c: model.vLineP[rp, k, i, j, c] <= model.pPmax[i, j, c] * model.pMaxLineLoad * model.vLineInvest[i, j, c] + model.vLineOverload[rp, k, i, j, c] * model.pPmax[i, j, c])

    model.eSoftLineLoadLimitCanNeg = pyo.Constraint(model.rp, model.constraintsActiveK, model.lc, doc='Negative soft limit for line load of candidate lines',
                                                    rule=lambda model, rp, k, i, j, c: model.vLineP[rp, k, i, j, c] >= -model.pPmax[i, j, c] * model.pMaxLineLoad * model.vLineInvest[i, j, c] - model.vLineOverload[rp, k, i, j, c] * model.pPmax[i, j, c])

    # Note: The term (1 - pMaxLineLoad) ensures that this also makes sense in an rMIP context (e.g., 50% vLineInvest with 70% pMaxLineLoad (=> 30% vLineOverload) limits vLineOverload to 15% (= 30% * 50%))
    model.eSoftLineLoadLimitCanInv = pyo.Constraint(model.rp, model.constraintsActiveK, model.lc, doc='Limit for vLineOverload if candidate line is not invested in',
                                                    rule=lambda model, rp, k, i, j, c: model.vLineInvest[i, j, c] >= model.vLineOverload[rp, k, i, j, c] / (1 - model.pMaxLineLoad))

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0

    # Add cost when exceeding soft line load limit
    second_stage_objective = sum(model.pWeight_rp[rp] *
                                 sum(model.pWeight_k[k] *
                                     sum(model.vLineOverload[rp, k, i, j, c]
                                         for i, j, c in model.la)
                                     for k in model.constraintsActiveK)
                                 for rp in model.rp) * model.pLOLCost

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
