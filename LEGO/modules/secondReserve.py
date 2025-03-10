import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(lego: LEGO):
    # Sets
    lego.model.secondReserveGenerators = pyo.Set(doc="Second reserve providing generators", within=lego.model.g)
    if hasattr(lego.model, "thermalGenerators"):
        lego.addToSet("secondReserveGenerators", lego.model.thermalGenerators)
    if hasattr(lego.model, "storageUnits"):
        lego.addToSet("secondReserveGenerators", lego.model.storageUnits)

    # Variables
    lego.model.v2ndResUP = pyo.Var(lego.model.rp, lego.model.k, lego.model.secondReserveGenerators, doc="2nd reserve up allocation [GW]", bounds=(0, None))
    lego.model.v2ndResDW = pyo.Var(lego.model.rp, lego.model.k, lego.model.secondReserveGenerators, doc="2nd reserve down allocation [GW]", bounds=(0, None))

    if hasattr(lego.model, "thermalGenerators"):
        lego.model.eUCMinOut = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc="Output limit of a committed unit", rule=lambda model, rp, k, t: model.vGenP1[rp, k, t] - model.v2ndResDW[rp, k, t] >= 0)
        for t in lego.model.thermalGenerators:
            lego.model.v2ndResUP[:, :, t].setub((lego.model.pMaxProd[t] - lego.model.pMinProd[t]) * (lego.model.pExisUnits[t] + (lego.model.pMaxInvest[t] * lego.model.pEnabInv[t])))
            lego.model.v2ndResDW[:, :, t].setub((lego.model.pMaxProd[t] - lego.model.pMinProd[t]) * (lego.model.pExisUnits[t] + (lego.model.pMaxInvest[t] * lego.model.pEnabInv[t])))

    if hasattr(lego.model, "storageUnits"):
        for s in lego.model.storageUnits:
            lego.model.v2ndResUP[:, :, s].setub(lego.model.pMaxProd[s] * (lego.model.pExisUnits[s] + (lego.model.pMaxInvest[s] * lego.model.pEnabInv[s])))
            lego.model.v2ndResDW[:, :, s].setub(max(lego.model.pMaxCons[s], lego.model.pMaxProd[s]) * (lego.model.pExisUnits[s] + (lego.model.pMaxInvest[s] * lego.model.pEnabInv[s])))

    # Parameters
    lego.model.p2ndResUp = pyo.Param(initialize=lego.cs.dPower_Parameters["p2ndResUp"], doc="2nd reserve factor up")
    lego.model.p2ndResDW = pyo.Param(initialize=lego.cs.dPower_Parameters["p2ndResDW"], doc="2nd reserve factor down")

    lego.model.p2ndResUpCost = pyo.Param(initialize=lego.cs.dPower_Parameters["p2ndResUpCost"], doc="2nd reserve up cost [$/GW]")
    lego.model.p2ndResDWCost = pyo.Param(initialize=lego.cs.dPower_Parameters["p2ndResDWCost"], doc="2nd reserve down cost [$/GW]")


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):
    def e2ReserveUp_rule(model, rp, k):  # TODO: Check if we need to multiply with ExisUnite or InvestedUnits here
        return ((sum(model.v2ndResUP[rp, k, t] for t in model.thermalGenerators) if hasattr(model, "thermalGenerators") else 0) +
                (sum(model.v2ndResUP[rp, k, s] for s in model.storageUnits) if hasattr(model, "storageUnits") else 0) >= model.p2ndResUp * sum(model.pDemandP[rp, k, i] for i in model.i))

    lego.model.e2ReserveUp = pyo.Constraint(lego.model.rp, lego.model.k, doc="2nd reserve up", rule=e2ReserveUp_rule)

    def e2ReserveDw_rule(model, rp, k):
        return ((sum(model.v2ndResDW[rp, k, t] for t in model.thermalGenerators) if hasattr(model, "thermalGenerators") else 0) +
                (sum(model.v2ndResDW[rp, k, s] for s in model.storageUnits) if hasattr(model, "storageUnits") else 0) >= model.p2ndResDW * sum(model.pDemandP[rp, k, i] for i in model.i))

    lego.model.e2ReserveDw = pyo.Constraint(lego.model.rp, lego.model.k, doc="2nd reserve down", rule=e2ReserveDw_rule)

    # Add 2nd reserve to power balance and unit commitment constraints
    if hasattr(lego.model, "thermalGenerators"):
        for rp in lego.model.rp:
            for k in lego.model.k:
                for g in lego.model.thermalGenerators:
                    lego.model.eUCMaxOut1_expr[rp, k, g] += lego.model.v2ndResUP[rp, k, g]

                    match lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
                        case "notEnforced":
                            if k != lego.model.k.first():  # Skip first timestep if constraint is not enforced
                                lego.model.eThRampDw_expr[rp, k, g] -= lego.model.v2ndResDW[rp, k, g]
                                lego.model.eThRampUp_expr[rp, k, g] += lego.model.v2ndResUP[rp, k, g]

                            if k != lego.model.k.last():  # Skip last timestep if constraint is not enforced
                                lego.model.eUCMaxOut2_expr[rp, k, g] += lego.model.v2ndResUP[rp, k, g]

                        case "cyclic" | "markov":
                            lego.model.eThRampDw_expr[rp, k, g] -= lego.model.v2ndResDW[rp, k, g]
                            lego.model.eThRampUp_expr[rp, k, g] += lego.model.v2ndResUP[rp, k, g]
                            lego.model.eUCMaxOut2_expr[rp, k, g] += lego.model.v2ndResUP[rp, k, g]

    # Add 2nd reserve to storage constraints
    if hasattr(lego.model, "storageUnits"):
        for rp in lego.model.rp:
            for k in lego.model.k:
                for s in lego.model.storageUnits:
                    lego.model.eStMaxProd_expr[rp, k, s] += lego.model.v2ndResUP[rp, k, s]
                    lego.model.eStMaxCons_expr[rp, k, s] -= lego.model.v2ndResDW[rp, k, s]
                    lego.model.eStMaxIntraRes_expr[rp, k, s] += lego.model.v2ndResDW[rp, k, s] + lego.model.v2ndResDW[rp, lego.model.k.prevw(k), s] * lego.model.pWeight_k[k]
                    lego.model.eStMinIntraRes_expr[rp, k, s] -= lego.model.v2ndResUP[rp, k, s] + lego.model.v2ndResUP[rp, lego.model.k.prevw(k), s] * lego.model.pWeight_k[k]

    # Add 2nd reserve cost to objective
    if hasattr(lego.model, "thermalGenerators"):
        lego.model.objective.expr += sum(lego.model.v2ndResUP[rp, k, t] * lego.model.pOMVarCost[t] * lego.model.p2ndResUpCost * lego.model.pWeight_rp[rp] * lego.model.pWeight_k[k] for rp in lego.model.rp for k in lego.model.k for t in lego.model.thermalGenerators)
        lego.model.objective.expr += sum(lego.model.v2ndResDW[rp, k, t] * lego.model.pOMVarCost[t] * lego.model.p2ndResDWCost * lego.model.pWeight_rp[rp] * lego.model.pWeight_k[k] for rp in lego.model.rp for k in lego.model.k for t in lego.model.thermalGenerators)

    if hasattr(lego.model, "storageUnits"):
        lego.model.objective.expr += sum(lego.model.v2ndResUP[rp, k, s] * lego.model.pOMVarCost[s] * lego.model.p2ndResUpCost * lego.model.pWeight_rp[rp] * lego.model.pWeight_k[k] for rp in lego.model.rp for k in lego.model.k for s in lego.model.storageUnits)
        lego.model.objective.expr += sum(lego.model.v2ndResDW[rp, k, s] * lego.model.pOMVarCost[s] * lego.model.p2ndResDWCost * lego.model.pWeight_rp[rp] * lego.model.pWeight_k[k] for rp in lego.model.rp for k in lego.model.k for s in lego.model.storageUnits)
