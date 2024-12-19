import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(lego: LEGO):
    # Sets
    secondReserveGenerators = set()
    if hasattr(lego.model, "thermalGenerators"):
        secondReserveGenerators.update(lego.model.thermalGenerators)
    if hasattr(lego.model, "storageUnits"):
        secondReserveGenerators.update(lego.model.storageUnits)
    lego.model.secondReserveGenerators = pyo.Set(initialize=secondReserveGenerators, doc="Second reserve providing generators")

    # Variables
    lego.model.v2ndResUP = pyo.Var(lego.model.rp, lego.model.k, lego.model.secondReserveGenerators, doc="2nd reserve up allocation [GW]", bounds=(0, None))
    lego.model.v2ndResDW = pyo.Var(lego.model.rp, lego.model.k, lego.model.secondReserveGenerators, doc="2nd reserve down allocation [GW]", bounds=(0, None))

    if hasattr(lego.model, "thermalGenerators"):
        for t in lego.model.thermalGenerators:
            lego.model.v2ndResUP[:, :, t].setub((lego.model.pMaxProd[t] - lego.model.pMinProd[t]) * lego.model.pExisUnits[t])  # TODO: Add MaxInvestUnits to ExisUnits for bound
            lego.model.v2ndResDW[:, :, t].setub((lego.model.pMaxProd[t] - lego.model.pMinProd[t]) * lego.model.pExisUnits[t])  # TODO: Add MaxInvestUnits to ExisUnits for bound

    if hasattr(lego.model, "storageUnits"):
        for s in lego.model.storageUnits:
            lego.model.v2ndResUP[:, :, s].setub(lego.model.pMaxProd[s] * lego.model.pExisUnits[s])  # TODO: Add MaxInvestUnits to ExisUnits for bound
            lego.model.v2ndResDW[:, :, s].setub(max(lego.model.pMaxCons[s], lego.model.pMaxProd[s]) * lego.model.pExisUnits[s])  # TODO: Add MaxInvestUnits to ExisUnits for bound

    # Parameters
    lego.model.p2ndResUp = pyo.Param(initialize=lego.cs.dPower_Parameters["p2ndResUp"], doc="2nd reserve factor up")
    lego.model.p2ndResDw = pyo.Param(initialize=lego.cs.dPower_Parameters["p2ndResDw"], doc="2nd reserve factor down")


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):
    def e2ReserveUp_rule(model, rp, k):  # TODO: Check if we need to multiply with ExisUnite or InvestedUnits here
        return sum(model.v2ndResUP[rp, k, t] for t in model.thermalGenerators) + sum(model.v2ndResUP[rp, k, s] for s in model.storageUnits) >= model.p2ndResUp * sum(model.pDemandP[rp, i, k] for i in model.i)

    lego.model.e2ReserveUp = pyo.Constraint(lego.model.rp, lego.model.k, doc="2nd reserve up", rule=e2ReserveUp_rule)

    def e2ReserveDw_rule(model, rp, k):
        return sum(model.v2ndResDW[rp, k, t] for t in model.thermalGenerators) + sum(model.v2ndResDW[rp, k, s] for s in model.storageUnits) >= model.p2ndResDw * sum(model.pDemandP[rp, i, k] for i in model.i)

    lego.model.e2ReserveDw = pyo.Constraint(lego.model.rp, lego.model.k, doc="2nd reserve down", rule=e2ReserveDw_rule)
