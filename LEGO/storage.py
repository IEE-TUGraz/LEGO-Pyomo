import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_variable_definitions(lego: LEGO):
    # Sets
    storageUnits = lego.cs.dPower_Storage.index.tolist()
    lego.model.storageUnits = pyo.Set(doc='Storage units', initialize=storageUnits)
    lego.update_generators(storageUnits)

    # Variables
    lego.model.vCharge = pyo.Var(lego.model.storageUnits, lego.model.rp, lego.model.k, doc='Charging of storage unit g', bounds=(0, None))
    lego.model.vStIntraRes = pyo.Var(lego.model.storageUnits, lego.model.rp, lego.model.k, doc='Intra-reserve of storage unit g', bounds=(0, None))
    lego.model.bCharge = pyo.Var(lego.model.storageUnits, lego.model.rp, lego.model.k, doc='Binary variable for charging of storage unit g', domain=pyo.Binary)

    # Parameters
    lego.model.pOMVarCost = pyo.Param(lego.model.storageUnits, initialize=lego.cs.dPower_Storage['pOMVarCostEUR'], doc='Variable O&M cost of storage unit g')
    lego.model.pWeight_k = pyo.Param(lego.model.k, initialize=lego.cs.dPower_WeightsK, doc='Weight of time step k')


@LEGOUtilities.checkExecutionLog([add_variable_definitions])
def add_variable_bounds(lego: LEGO):
    for g in lego.model.storageUnits:
        for rp in lego.model.rp:
            for k in lego.model.k:
                lego.model.p[g, rp, k].setub(lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])
                lego.model.vCharge[g, rp, k].setub(lego.cs.dPower_Storage.loc[g, 'MaxCons'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])
                lego.model.vStIntraRes[g, rp, k].setub(lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'] * lego.cs.dPower_Storage.loc[g, 'Ene2PowRatio'])


@LEGOUtilities.checkExecutionLog([add_variable_definitions, add_variable_bounds])
def add_constraints(lego: LEGO):
    # TODO: Check if we should add Hydro here as well

    # Storage unit charging and discharging
    lego.model.eStIntraRes = pyo.ConstraintList(doc='Intra-reserve constraint for storage units')
    lego.model.eExclusiveChargeDischarge = pyo.ConstraintList(doc='Enforce exclusive charge or discharge for storage units')
    for g in lego.model.storageUnits:
        for rp in lego.model.rp:
            for k in lego.model.k:
                if len(lego.model.rp) == 1:  # Only cyclic if it has multiple representative periods
                    if LEGOUtilities.k_to_int(k) == 1:  # Adding IniReserve if it is the first time step (instead of 'prev' value)
                        lego.model.eStIntraRes.add(0 == lego.cs.dPower_Storage.loc[g, 'IniReserve'] - lego.model.vStIntraRes[g, rp, k] - lego.model.p[g, rp, k] * lego.model.pWeight_k[k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] + lego.model.vCharge[g, rp, k] * lego.model.pWeight_k[k] * lego.cs.dPower_Storage.loc[g, 'ChEffic'])
                    else:
                        lego.model.eStIntraRes.add(0 == lego.model.vStIntraRes[g, rp, lego.model.k.prev(k)] - lego.model.vStIntraRes[g, rp, k] - lego.model.p[g, rp, k] * lego.model.pWeight_k[k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] + lego.model.vCharge[g, rp, k] * lego.model.pWeight_k[k] * lego.cs.dPower_Storage.loc[g, 'ChEffic'])
                elif len(lego.model.rp) > 1:
                    lego.model.eStIntraRes.add(0 == lego.model.vStIntraRes[g, rp, lego.model.k.prevw(k)] - lego.model.vStIntraRes[g, rp, k] - lego.model.p[g, rp, k] * lego.model.pWeight_k[k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] + lego.model.vCharge[g, rp, k] * lego.model.pWeight_k[k] * lego.cs.dPower_Storage.loc[g, 'ChEffic'])

                # TODO: Check if we should rather do a +/- value and calculate charge/discharge ex-post
                lego.model.eExclusiveChargeDischarge.add(lego.model.vCharge[g, rp, k] <= lego.model.bCharge[g, rp, k] * lego.cs.dPower_Storage.loc[g, 'MaxCons'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])
                lego.model.eExclusiveChargeDischarge.add(lego.model.p[g, rp, k] <= (1 - lego.model.bCharge[g, rp, k]) * lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])
