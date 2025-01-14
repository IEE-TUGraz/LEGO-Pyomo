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
        for t in lego.model.thermalGenerators:
            lego.model.v2ndResUP[:, :, t].setub((lego.model.pMaxProd[t] - lego.model.pMinProd[t]) * (lego.model.pExisUnits[t] + (lego.model.pMaxInvest[t] * lego.model.pEnabInv[t])))
            lego.model.v2ndResDW[:, :, t].setub((lego.model.pMaxProd[t] - lego.model.pMinProd[t]) * (lego.model.pExisUnits[t] + (lego.model.pMaxInvest[t] * lego.model.pEnabInv[t])))

    if hasattr(lego.model, "storageUnits"):
        for s in lego.model.storageUnits:
            lego.model.v2ndResUP[:, :, s].setub(lego.model.pMaxProd[s] * (lego.model.pExisUnits[s] + (lego.model.pMaxInvest[s] * lego.model.pEnabInv[s])))
            lego.model.v2ndResDW[:, :, s].setub(max(lego.model.pMaxCons[s], lego.model.pMaxProd[s]) * (lego.model.pExisUnits[s] + (lego.model.pMaxInvest[s] * lego.model.pEnabInv[s])))

    # Parameters
    lego.model.p2ndResUp = pyo.Param(initialize=lego.cs.dPower_Parameters["p2ndResUp"], doc="2nd reserve factor up")
    lego.model.p2ndResDw = pyo.Param(initialize=lego.cs.dPower_Parameters["p2ndResDw"], doc="2nd reserve factor down")


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):
    def e2ReserveUp_rule(model, rp, k):  # TODO: Check if we need to multiply with ExisUnite or InvestedUnits here
        return sum(model.v2ndResUP[rp, k, t] for t in model.thermalGenerators) + sum(model.v2ndResUP[rp, k, s] for s in model.storageUnits) >= model.p2ndResUp * sum(model.pDemandP[rp, k, i] for i in model.i)

    lego.model.e2ReserveUp = pyo.Constraint(lego.model.rp, lego.model.k, doc="2nd reserve up", rule=e2ReserveUp_rule)

    def e2ReserveDw_rule(model, rp, k):
        return sum(model.v2ndResDW[rp, k, t] for t in model.thermalGenerators) + sum(model.v2ndResDW[rp, k, s] for s in model.storageUnits) >= model.p2ndResDw * sum(model.pDemandP[rp, k, i] for i in model.i)

    lego.model.e2ReserveDw = pyo.Constraint(lego.model.rp, lego.model.k, doc="2nd reserve down", rule=e2ReserveDw_rule)

    # Add 2nd reserve to power balance
    if hasattr(lego.model, "thermalGenerators"):
        for rp in lego.model.rp:
            for k in lego.model.k:
                for g in lego.model.thermalGenerators:
                    lego.model.eThRampDw_expr[rp, k, g] -= lego.model.v2ndResDW[rp, k, g]
                    lego.model.eThRampUp_expr[rp, k, g] += lego.model.v2ndResUP[rp, k, g]

    # Add 2nd reserve to storage constraints
    if hasattr(lego.model, "storageUnits"):
        for rp in lego.model.rp:
            for k in lego.model.k:
                for s in lego.model.storageUnits:
                    lego.model.eStMaxProd_expr[rp, k, s] += lego.model.v2ndResUP[rp, k, s]
                    lego.model.eStMaxCons_expr[rp, k, s] -= lego.model.v2ndResDW[rp, k, s]
                    lego.model.eStMaxIntraRes_expr[rp, k, s] += lego.model.v2ndResDW[rp, k, s] + lego.model.v2ndResDW[rp, lego.model.k.prevw(k), s] * lego.model.pWeight_k[k]
                    lego.model.eStMinIntraRes_expr[rp, k, s] -= lego.model.v2ndResUP[rp, k, s] + lego.model.v2ndResUP[rp, lego.model.k.prevw(k), s] * lego.model.pWeight_k[k]
