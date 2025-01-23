import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(lego: LEGO):
    # Parameters
    lego.model.pMaxLineLoad = pyo.Param(doc='Scaling factor for pMax (rest to 100% is soft limit)', initialize=lego.cs.dPower_Parameters["pMaxLineLoad"])
    lego.model.pLOLCost = pyo.Param(doc='Cost of exceeding line load limit [€/MWh]', initialize=lego.cs.dPower_Parameters["pLOLCost"])

    # Variables
    lego.model.vLineOverload = pyo.Var(lego.model.rp, lego.model.k, lego.model.la, doc='Line overload [% above Pmax]', bounds=lambda model, rp, k, la: (0, 1 - model.pMaxLineLoad))

    # Checks
    if not 0 <= pyo.value(lego.model.pMaxLineLoad) <= 1:
        raise ValueError(f"pMaxLineLoad must be between 0 and 1 (got {pyo.value(lego.model.pMaxLineLoad)})")


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):
    lego.model.eSoftLineLoadLimitPos = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le, doc='Positive soft limit for line load of existing lines',
                                                      rule=lambda model, rp, k, i, j, c: model.vLineP[rp, k, i, j, c] <= model.pPmax[i, j, c] * model.pMaxLineLoad + model.vLineOverload[rp, k, i, j, c] * model.pPmax[i, j, c])

    lego.model.eSoftLineLoadLimitNeg = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le, doc='Negative soft limit for line load of existing lines',
                                                      rule=lambda model, rp, k, i, j, c: model.vLineP[rp, k, i, j, c] >= -model.pPmax[i, j, c] * model.pMaxLineLoad - model.vLineOverload[rp, k, i, j, c] * model.pPmax[i, j, c])

    lego.model.eSoftLineLoadLimitCanPos = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc='Positive soft limit for line load of candidate lines',
                                                         rule=lambda model, rp, k, i, j, c: model.vLineP[rp, k, i, j, c] <= model.pPmax[i, j, c] * model.pMaxLineLoad * model.vLineInvest[i, j, c] + model.vLineOverload[rp, k, i, j, c] * model.pPmax[i, j, c])

    lego.model.eSoftLineLoadLimitCanNeg = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc='Negative soft limit for line load of candidate lines',
                                                         rule=lambda model, rp, k, i, j, c: model.vLineP[rp, k, i, j, c] >= -model.pPmax[i, j, c] * model.pMaxLineLoad * model.vLineInvest[i, j, c] - model.vLineOverload[rp, k, i, j, c] * model.pPmax[i, j, c])

    # Note: The term (1 - pMaxLineLoad) ensures that this also makes sense in an rMIP context (e.g., 50% vLineInvest with 70% pMaxLineLoad (=> 30% vLineOverload) limits vLineOverload to 15% (= 30% * 50%))
    lego.model.eSoftLineLoadLimitCanInv = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc='Limit for vLineOverload if candidate line is not invested in',
                                                         rule=lambda model, rp, k, i, j, c: model.vLineInvest[i, j, c] >= model.vLineOverload[rp, k, i, j, c] / (1 - model.pMaxLineLoad))

    # Add cost when exceeding soft line load limit
    lego.model.objective.expr += sum(lego.model.pWeight_rp(rp) * lego.model.pWeight_k(k) * lego.model.vLineOverload(rp, k, i, j, c) * lego.model.pLOLCost for rp in lego.model.rp for k in lego.model.k for i, j, c in lego.model.ca)
