import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.secondReserveGenerators = pyo.Set(doc="Second reserve providing generators", within=model.g)
    if hasattr(model, "thermalGenerators"):
        LEGO.addToSet(model, "secondReserveGenerators", model.thermalGenerators)
    if hasattr(model, "storageUnits"):
        LEGO.addToSet(model, "secondReserveGenerators", model.storageUnits)

    # Variables
    max2ndResUP = {}
    max2ndResDW = {}
    if hasattr(model, "thermalGenerators"):
        for t in model.thermalGenerators:
            max2ndResUP[t] = (model.pMaxProd[t] - model.pMinProd[t]) * (model.pExisUnits[t] + (model.pMaxInvest[t] * model.pEnabInv[t]))
            max2ndResDW[t] = (model.pMaxProd[t] - model.pMinProd[t]) * (model.pExisUnits[t] + (model.pMaxInvest[t] * model.pEnabInv[t]))

    if hasattr(model, "storageUnits"):
        for s in model.storageUnits:
            max2ndResUP[s] = model.pMaxProd[s] * (model.pExisUnits[s] + (model.pMaxInvest[s] * model.pEnabInv[s]))
            max2ndResDW[s] = max(model.pMaxCons[s], model.pMaxProd[s]) * (model.pExisUnits[s] + (model.pMaxInvest[s] * model.pEnabInv[s]))

    model.v2ndResUP = pyo.Var(model.rp, model.k, model.secondReserveGenerators, doc="2nd reserve up allocation [GW]", bounds=lambda m, rp, k, g: (0, max2ndResUP[g]))
    second_stage_variables += [model.v2ndResUP]
    model.v2ndResDW = pyo.Var(model.rp, model.k, model.secondReserveGenerators, doc="2nd reserve down allocation [GW]", bounds=lambda m, rp, k, g: (0, max2ndResDW[g]))
    second_stage_variables += [model.v2ndResDW]

    # Parameters
    model.p2ndResUp = pyo.Param(initialize=cs.dPower_Parameters["p2ndResUp"], doc="2nd reserve factor up")
    model.p2ndResDW = pyo.Param(initialize=cs.dPower_Parameters["p2ndResDW"], doc="2nd reserve factor down")

    model.p2ndResUpCost = pyo.Param(initialize=cs.dPower_Parameters["p2ndResUpCost"], doc="2nd reserve up cost [$/GW]")
    model.p2ndResDWCost = pyo.Param(initialize=cs.dPower_Parameters["p2ndResDWCost"], doc="2nd reserve down cost [$/GW]")

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    def e2ReserveUp_rule(model, rp, k):  # TODO: Check if we need to multiply with ExisUnite or InvestedUnits here
        return ((sum(model.v2ndResUP[rp, k, t] for t in model.thermalGenerators) if hasattr(model, "thermalGenerators") else 0) +
                (sum(model.v2ndResUP[rp, k, s] for s in model.storageUnits) if hasattr(model, "storageUnits") else 0) >= model.p2ndResUp * sum(model.pDemandP[rp, k, i] for i in model.i))

    model.e2ReserveUp = pyo.Constraint(model.rp, model.k, doc="2nd reserve up", rule=e2ReserveUp_rule)

    def e2ReserveDw_rule(model, rp, k):
        return ((sum(model.v2ndResDW[rp, k, t] for t in model.thermalGenerators) if hasattr(model, "thermalGenerators") else 0) +
                (sum(model.v2ndResDW[rp, k, s] for s in model.storageUnits) if hasattr(model, "storageUnits") else 0) >= model.p2ndResDW * sum(model.pDemandP[rp, k, i] for i in model.i))

    model.e2ReserveDw = pyo.Constraint(model.rp, model.k, doc="2nd reserve down", rule=e2ReserveDw_rule)

    if hasattr(model, "thermalGenerators"):
        model.eUCMinOut = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc="Output limit of a committed unit", rule=lambda model, rp, k, t: model.vGenP1[rp, k, t] - model.v2ndResDW[rp, k, t] >= 0)

    # Add 2nd reserve to power balance and unit commitment constraints
    if hasattr(model, "thermalGenerators"):
        for rp in model.rp:
            for k in model.k:
                for g in model.thermalGenerators:
                    model.eUCMaxOut1_expr[rp, k, g] += model.v2ndResUP[rp, k, g]

                    match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
                        case "notEnforced":
                            if k != model.k.first():  # Skip first timestep if constraint is not enforced
                                model.eThRampDw_expr[rp, k, g] -= model.v2ndResDW[rp, k, g]
                                model.eThRampUp_expr[rp, k, g] += model.v2ndResUP[rp, k, g]

                            if k != model.k.last():  # Skip last timestep if constraint is not enforced
                                model.eUCMaxOut2_expr[rp, k, g] += model.v2ndResUP[rp, k, g]

                        case "cyclic" | "markov":
                            model.eThRampDw_expr[rp, k, g] -= model.v2ndResDW[rp, k, g]
                            model.eThRampUp_expr[rp, k, g] += model.v2ndResUP[rp, k, g]
                            model.eUCMaxOut2_expr[rp, k, g] += model.v2ndResUP[rp, k, g]

    # Add 2nd reserve to storage constraints
    if hasattr(model, "storageUnits"):
        for rp in model.rp:
            for k in model.k:
                for s in model.storageUnits:
                    model.eStMaxProd_expr[rp, k, s] += model.v2ndResUP[rp, k, s]
                    model.eStMaxCons_expr[rp, k, s] -= model.v2ndResDW[rp, k, s]
                for s in model.intraStorageUnits:
                    model.eStMaxIntraRes_expr[rp, k, s] += model.v2ndResDW[rp, k, s] + model.v2ndResDW[rp, model.k.prevw(k), s] * model.pWeight_k[k]
                    model.eStMinIntraRes_expr[rp, k, s] -= model.v2ndResUP[rp, k, s] + model.v2ndResUP[rp, model.k.prevw(k), s] * model.pWeight_k[k]

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = 0.0

    # Add 2nd reserve cost to objective
    second_stage_objective += sum(model.pWeight_rp[rp] *  # Weight for representative period
                                  sum(model.pWeight_k[k] *  # Weight for time step
                                      sum(model.pOMVarCost[g] *  # Variable O&M cost of generator
                                          (+ model.v2ndResUP[rp, k, g] * model.p2ndResUpCost  # Cost for 2nd reserve up
                                           + model.v2ndResDW[rp, k, g] * model.p2ndResDWCost)  # Cost for 2nd reserve down
                                          for g in model.secondReserveGenerators)
                                      for k in model.k)
                                  for rp in model.rp)

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
