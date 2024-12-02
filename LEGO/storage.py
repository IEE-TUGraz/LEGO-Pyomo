import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_variable_definitions(lego: LEGO):
    # Sets
    storageUnits = lego.cs.dPower_Storage.index.tolist()
    lego.model.storageUnits = pyo.Set(doc='Storage units', initialize=storageUnits)
    lego.update_generators(storageUnits)

    # Variables
    lego.model.vConsump = pyo.Var(lego.model.storageUnits, lego.model.rp, lego.model.k, doc='Charging of storage unit g', bounds=(0, None))
    lego.model.vStIntraRes = pyo.Var(lego.model.storageUnits, lego.model.rp, lego.model.k, doc='Intra-reserve of storage unit g', bounds=(None, None))
    lego.model.vStInterRes = pyo.Var(lego.model.p, lego.model.storageUnits, doc='Inter-reserve of storage unit g', bounds=(None, None))
    lego.model.bChargeDisCharge = pyo.Var(lego.model.storageUnits, lego.model.rp, lego.model.k, doc='Binary variable for charging of storage unit g', domain=pyo.Binary)

    # Parameters
    lego.model.pOMVarCost = pyo.Param(lego.model.storageUnits, initialize=lego.cs.dPower_Storage['pOMVarCostEUR'], doc='Variable O&M cost of storage unit g')
    lego.model.pWeight_k = pyo.Param(lego.model.k, initialize=lego.cs.dPower_WeightsK, doc='Weight of time step k')
    lego.model.pEnableChDisPower = lego.cs.dPower_Parameters.loc['pEnableChDisPower', 'Value']  # Avoid simultaneous charging and discharging


@LEGOUtilities.checkExecutionLog([add_variable_definitions])
def add_variable_bounds(lego: LEGO):
    for g in lego.model.storageUnits:
        for rp in lego.model.rp:
            for k in lego.model.k:
                lego.model.vGenP[g, rp, k].setub(lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])
                lego.model.vConsump[g, rp, k].setub(lego.cs.dPower_Storage.loc[g, 'MaxCons'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])

                lego.model.vStIntraRes[g, rp, k].setub(lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'] * lego.cs.dPower_Storage.loc[g, 'Ene2PowRatio'])
                lego.model.vStIntraRes[g, rp, k].setlb(lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'] * lego.cs.dPower_Storage.loc[g, 'Ene2PowRatio'] * lego.cs.dPower_Storage.loc[g, 'MinReserve'])

    for p in lego.model.p:
        for g in lego.model.storageUnits:
            if LEGOUtilities.p_to_int(p) % lego.model.pMovWindow == 0:
                lego.model.vStInterRes[p, g].setub(lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'] * lego.cs.dPower_Storage.loc[g, 'Ene2PowRatio'])
                lego.model.vStInterRes[p, g].setlb(lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'] * lego.cs.dPower_Storage.loc[g, 'Ene2PowRatio'] * lego.cs.dPower_Storage.loc[g, 'MinReserve'])


@LEGOUtilities.checkExecutionLog([add_variable_definitions, add_variable_bounds])
def add_constraints(lego: LEGO):
    # TODO: Check if we should add Hydro here as well

    # Constraint definitions
    lego.model.eStIntraRes = pyo.ConstraintList(doc='Intra-day reserve constraint for storage units')
    lego.model.eStInterRes = pyo.ConstraintList(doc='Inter-day reserve constraint for storage units')
    lego.model.eExclusiveChargeDischarge = pyo.ConstraintList(doc='Enforce exclusive charge or discharge for storage units')

    # Constraint implementations
    for g in lego.model.storageUnits:
        for rp in lego.model.rp:
            for k in lego.model.k:
                if len(lego.model.rp) == 1:  # Only cyclic if it has multiple representative periods
                    if LEGOUtilities.k_to_int(k) == 1:  # Adding IniReserve if it is the first time step (instead of 'prev' value)
                        lego.model.eStIntraRes.add(0 == lego.cs.dPower_Storage.loc[g, 'IniReserve'] - lego.model.vStIntraRes[g, rp, k] - lego.model.vGenP[g, rp, k] * lego.model.pWeight_k[k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] + lego.model.vConsump[g, rp, k] * lego.model.pWeight_k[k] * lego.cs.dPower_Storage.loc[g, 'ChEffic'])
                    else:
                        lego.model.eStIntraRes.add(0 == lego.model.vStIntraRes[g, rp, lego.model.k.prev(k)] - lego.model.vStIntraRes[g, rp, k] - lego.model.vGenP[g, rp, k] * lego.model.pWeight_k[k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] + lego.model.vConsump[g, rp, k] * lego.model.pWeight_k[k] * lego.cs.dPower_Storage.loc[g, 'ChEffic'])
                elif len(lego.model.rp) > 1:
                    lego.model.eStIntraRes.add(0 == lego.model.vStIntraRes[g, rp, lego.model.k.prevw(k)] - lego.model.vStIntraRes[g, rp, k] - lego.model.vGenP[g, rp, k] * lego.model.pWeight_k[k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] + lego.model.vConsump[g, rp, k] * lego.model.pWeight_k[k] * lego.cs.dPower_Storage.loc[g, 'ChEffic'])

                # TODO: Check if we should rather do a +/- value and calculate charge/discharge ex-post
                if lego.model.pEnableChDisPower:
                    lego.model.eExclusiveChargeDischarge.add(lego.model.vConsump[g, rp, k] <= lego.model.bChargeDisCharge[g, rp, k] * lego.cs.dPower_Storage.loc[g, 'MaxCons'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])
                    lego.model.eExclusiveChargeDischarge.add(lego.model.vGenP[g, rp, k] <= (1 - lego.model.bChargeDisCharge[g, rp, k]) * lego.cs.dPower_Storage.loc[g, 'MaxProd'] * lego.cs.dPower_Storage.loc[g, 'ExisUnits'])

    if len(lego.model.rp) > 1:
        for g in lego.model.storageUnits:
            for p, rp, k in lego.model.hindex:
                # If current p is a multiple of moving window, add constraint
                if LEGOUtilities.p_to_int(p) % lego.model.pMovWindow == 0 and LEGOUtilities.p_to_int(p) >= lego.model.pMovWindow:
                    relevant_hindeces = lego.model.hindex[LEGOUtilities.p_to_int(p) - lego.model.pMovWindow:LEGOUtilities.p_to_int(p)]
                    hindex_count = relevant_hindeces.to_frame(index=False).groupby(['rp', 'k']).size()

                    lego.model.eStInterRes.add(0 == lego.model.vStInterRes[LEGOUtilities.int_to_p(LEGOUtilities.p_to_int(p) - lego.model.pMovWindow + 1), g] +
                                               (lego.cs.dPower_Storage.loc[g, 'IniReserve'] if LEGOUtilities.p_to_int(p) == lego.model.pMovWindow + 1 else 0) -
                                               lego.model.vStInterRes[p, g] +
                                               sum(-lego.model.vGenP[g, rp, k] * lego.model.pWeight_k[k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] * hindex_count.loc[rp, k] +
                                                   lego.model.vConsump[g, rp2, k2] * lego.model.pWeight_k[k2] * lego.cs.dPower_Storage.loc[g, 'ChEffic'] * hindex_count.loc[rp2, k2] for rp2, k2 in hindex_count.index if LEGOUtilities.p_to_int(p) - lego.model.pMovWindow < LEGOUtilities.p_to_int(p) <= LEGOUtilities.p_to_int(p)))
