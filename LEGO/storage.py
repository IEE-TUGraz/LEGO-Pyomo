import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addExecutionLog
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


@LEGOUtilities.addExecutionLog
def add_variable_bounds(lego: LEGO):
    for g in lego.model.storageUnits:
        for rp in lego.model.rp:
            for k in lego.model.k:
                lego.model.p[g, rp, k].setub(lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])
                lego.model.vCharge[g, rp, k].setub(lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])
                lego.model.vStIntraRes[g, rp, k].setub(lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'] * lego.cs.dPower_Storage.loc[g, 'Ene2PowRatio'])


@LEGOUtilities.checkExecutionLog([add_variable_definitions, add_variable_bounds])
def add_constraints(lego: LEGO):
    # Storage unit charging and discharging
    lego.model.cStIntraRes = pyo.ConstraintList(doc='Intra-reserve constraint for storage units')
    lego.model.cExclusiveChargeDischarge = pyo.ConstraintList(doc='Enforce exclusive charge or discharge for storage units')
    for g in lego.model.storageUnits:
        for rp in lego.model.rp:
            for k in lego.model.k:
                if LEGOUtilities.rp_to_int(rp) == 1 and LEGOUtilities.k_to_int(k) != 1:  # Only cyclic if it has multiple representative periods (and skipping first timestep)
                    lego.model.cStIntraRes.add(lego.model.vStIntraRes[g, rp, k] == lego.model.vStIntraRes[g, rp, lego.model.k.prev(k)] - lego.model.p[g, rp, k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] + lego.model.vCharge[g, rp, k] * lego.cs.dPower_Storage.loc[g, 'ChEffic'])
                elif LEGOUtilities.rp_to_int(rp) > 1:
                    lego.model.cStIntraRes.add(lego.model.vStIntraRes[g, rp, k] == lego.model.vStIntraRes[g, rp, lego.model.k.prevw(k)] - lego.model.p[g, rp, k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] + lego.model.vCharge[g, rp, k] * lego.cs.dPower_Storage.loc[g, 'ChEffic'])

                # TODO: Check if we should rather do a +/- value and calculate charge/discharge ex-post
                lego.model.cExclusiveChargeDischarge.add(lego.model.vCharge[g, rp, k] <= lego.model.bCharge[g, rp, k] * lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])
                lego.model.cExclusiveChargeDischarge.add(lego.model.p[g, rp, k] <= (1 - lego.model.bCharge[g, rp, k]) * lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])
