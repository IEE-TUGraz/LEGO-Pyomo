import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(lego: LEGO):
    # Sets
    storageUnits = lego.cs.dPower_Storage.index.tolist()
    lego.model.storageUnits = pyo.Set(doc='Storage units', initialize=storageUnits)
    lego.addToSet("g", storageUnits)

    # Parameters
    lego.model.pOMVarCost = pyo.Param(lego.model.storageUnits, initialize=lego.cs.dPower_Storage['pOMVarCostEUR'], doc='Variable O&M cost of storage unit g')
    lego.model.pWeight_k = pyo.Param(lego.model.k, initialize=lego.cs.dPower_WeightsK, doc='Weight of time step k')
    lego.model.pEnableChDisPower = lego.cs.dPower_Parameters['pEnableChDisPower']  # Avoid simultaneous charging and discharging
    lego.model.pE2PRatio = pyo.Param(lego.model.storageUnits, initialize=lego.cs.dPower_Storage['Ene2PowRatio'], doc='Energy to power ratio of storage unit g')
    lego.model.pMinReserve = pyo.Param(lego.model.storageUnits, initialize=lego.cs.dPower_Storage['MinReserve'], doc='Minimum reserve of storage unit g')

    lego.addToParameter("pMaxCons", lego.cs.dPower_Storage['MaxCons'], indices=[lego.model.storageUnits], doc='Maximum consumption of storage unit')

    lego.addToParameter("pMaxProd", lego.cs.dPower_Storage['MaxProd'])
    lego.addToParameter("pMinProd", lego.cs.dPower_Storage['MinProd'])
    lego.addToParameter("pExisUnits", lego.cs.dPower_Storage['ExisUnits'])

    lego.addToParameter("pMaxInvest", lego.cs.dPower_Storage['MaxInvest'])
    lego.addToParameter("pEnabInv", lego.cs.dPower_Storage['EnableInvest'])
    lego.addToParameter("pInvestCost", lego.cs.dPower_Storage['InvestCostEUR'])

    # Variables
    lego.model.bChargeDisCharge = pyo.Var(lego.model.storageUnits, lego.model.rp, lego.model.k, doc='Binary variable for charging of storage unit g', domain=pyo.Binary)

    lego.model.vConsump = pyo.Var(lego.model.rp, lego.model.k, lego.model.storageUnits, doc='Charging of storage unit g', bounds=(0, None))
    lego.model.vStIntraRes = pyo.Var(lego.model.rp, lego.model.k, lego.model.storageUnits, doc='Intra-reserve of storage unit g', bounds=(None, None))
    lego.model.vStInterRes = pyo.Var(lego.model.p, lego.model.storageUnits, doc='Inter-reserve of storage unit g', bounds=(None, None))

    for p in lego.model.p:
        for g in lego.model.storageUnits:
            if LEGOUtilities.p_to_int(p) == len(lego.model.p):
                if lego.cs.dPower_Parameters['pFixStInterResToIniReserve']:
                    lego.model.vStInterRes[p, g].fix(lego.cs.dPower_Storage.loc[g, 'IniReserve'])
                else:
                    lego.model.vStInterRes[p, g].setlb(lego.cs.dPower_Storage.loc[g, 'IniReserve'])

    for g in lego.model.storageUnits:
        lego.model.vGenInvest[g].setub(lego.model.pMaxProd[g] * lego.model.pMaxInvest[g])
        if not lego.model.pEnabInv[g]:
            lego.model.vGenInvest[g].fix(0)


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):
    # TODO: Check if we should add Hydro here as well

    # Constraint definitions
    lego.model.eStIntraRes = pyo.ConstraintList(doc='Intra-day reserve constraint for storage units')
    lego.model.eExclusiveChargeDischarge = pyo.ConstraintList(doc='Enforce exclusive charge or discharge for storage units')

    # Constraint implementations
    for g in lego.model.storageUnits:
        for rp in lego.model.rp:
            for k in lego.model.k:
                if len(lego.model.rp) == 1:  # Only cyclic if it has multiple representative periods
                    if LEGOUtilities.k_to_int(k) == 1:  # Adding IniReserve if it is the first time step (instead of 'prev' value)
                        lego.model.eStIntraRes.add(0 == lego.cs.dPower_Storage.loc[g, 'IniReserve'] - lego.model.vStIntraRes[rp, k, g] - lego.model.vGenP[rp, k, g] * lego.model.pWeight_k[k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] + lego.model.vConsump[rp, k, g] * lego.model.pWeight_k[k] * lego.cs.dPower_Storage.loc[g, 'ChEffic'])
                    else:
                        lego.model.eStIntraRes.add(0 == lego.model.vStIntraRes[rp, lego.model.k.prev(k), g] - lego.model.vStIntraRes[rp, k, g] - lego.model.vGenP[rp, k, g] * lego.model.pWeight_k[k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] + lego.model.vConsump[rp, k, g] * lego.model.pWeight_k[k] * lego.cs.dPower_Storage.loc[g, 'ChEffic'])
                elif len(lego.model.rp) > 1:
                    lego.model.eStIntraRes.add(0 == lego.model.vStIntraRes[rp, lego.model.k.prevw(k), g] - lego.model.vStIntraRes[rp, k, g] - lego.model.vGenP[rp, k, g] * lego.model.pWeight_k[k] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] + lego.model.vConsump[rp, k, g] * lego.model.pWeight_k[k] * lego.cs.dPower_Storage.loc[g, 'ChEffic'])

                # TODO: Check if we should rather do a +/- value and calculate charge/discharge ex-post
                if lego.model.pEnableChDisPower:
                    lego.model.eExclusiveChargeDischarge.add(lego.model.vConsump[rp, k, g] <= lego.model.bChargeDisCharge[rp, k, g] * lego.model.pMaxCons[g] * lego.model.pExisUnits[g])
                    lego.model.eExclusiveChargeDischarge.add(lego.model.vGenP[rp, k, g] <= (1 - lego.model.bChargeDisCharge[rp, k, g]) * lego.model.pMaxProd[g] * lego.model.pExisUnits[g])

    def eStMaxProd_rule(model, rp, k, s):
        return model.vGenP[rp, k, s] - model.vConsump[rp, k, s] <= model.pMaxProd[s] * (model.vGenInvest[s] + model.pExisUnits[s])

    lego.model.eStMaxProd = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.storageUnits, doc='Max production constraint for storage units', rule=eStMaxProd_rule)

    def eStMaxCons_rule(model, rp, k, s):
        return model.vGenP[rp, k, s] - model.vConsump[rp, k, s] >= -model.pMaxCons[s] * (model.vGenInvest[s] + model.pExisUnits[s])

    lego.model.eStMaxCons = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.storageUnits, doc='Max consumption constraint for storage units', rule=eStMaxCons_rule)

    def eStMaxIntraRes_rule(model, rp, k, s):
        return model.vStIntraRes[rp, k, s] <= model.pMaxProd[s] * (model.vGenInvest[s] + model.pExisUnits[s]) * model.pE2PRatio[s]

    lego.model.eStMaxIntraRes = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.storageUnits, doc='Max intra-reserve constraint for storage units', rule=eStMaxIntraRes_rule)

    def eStMinIntraRes(model, rp, k, s):
        return model.vStIntraRes[rp, k, s] >= model.pMaxProd[s] * (model.vGenInvest[s] + model.pExisUnits[s]) * model.pE2PRatio[s] * model.pMinReserve[s]

    lego.model.eStMinIntraRes = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.storageUnits, doc='Min intra-reserve constraint for storage units', rule=eStMinIntraRes)

    def eStMaxInterRes_rule(model, p, s):
        # If current p is a multiple of moving window, add constraint
        if LEGOUtilities.p_to_int(p) % model.pMovWindow == 0:
            return model.vStInterRes[p, s] <= model.pMaxProd[s] * (model.vGenInvest[s] + model.pExisUnits[s]) * model.pE2PRatio[s]
        else:
            return pyo.Constraint.Skip

    lego.model.eStMaxInterRes = pyo.Constraint(lego.model.p, lego.model.storageUnits, doc='Max inter-reserve constraint for storage units', rule=eStMaxInterRes_rule)

    def eStMinInterRes_rule(model, p, s):
        # If current p is a multiple of moving window, add constraint
        if LEGOUtilities.p_to_int(p) % model.pMovWindow == 0:
            return model.vStInterRes[p, s] >= model.pMaxProd[s] * (model.vGenInvest[s] + model.pExisUnits[s]) * model.pE2PRatio[s] * model.pMinReserve[s]
        else:
            return pyo.Constraint.Skip

    lego.model.eStMinInterRes = pyo.Constraint(lego.model.p, lego.model.storageUnits, doc='Min inter-reserve constraint for storage units', rule=eStMinInterRes_rule)

    def eStInterRes_rule(model, p, storage_unit):
        # If current p is a multiple of moving window, add constraint
        if LEGOUtilities.p_to_int(p) % model.pMovWindow == 0:
            relevant_hindeces = model.hindex[LEGOUtilities.p_to_int(p) - model.pMovWindow:LEGOUtilities.p_to_int(p)]
            hindex_count = relevant_hindeces.to_frame(index=False).groupby(['rp', 'k']).size()

            return (0 ==
                    (model.vStInterRes[LEGOUtilities.int_to_p(LEGOUtilities.p_to_int(p) - model.pMovWindow), storage_unit] if LEGOUtilities.p_to_int(p) - model.pMovWindow > 0 else 0)
                    + (lego.cs.dPower_Storage.loc[storage_unit, 'IniReserve'] if LEGOUtilities.p_to_int(p) == model.pMovWindow else 0)
                    - model.vStInterRes[p, storage_unit]
                    + sum(- model.vGenP[rp2, k2, storage_unit] * model.pWeight_k[k2] / lego.cs.dPower_Storage.loc[storage_unit, 'DisEffic'] * hindex_count.loc[rp2, k2]
                          + model.vConsump[rp2, k2, storage_unit] * model.pWeight_k[k2] * lego.cs.dPower_Storage.loc[storage_unit, 'ChEffic'] * hindex_count.loc[rp2, k2] for rp2, k2 in hindex_count.index))

        else:  # Skip otherwise
            return pyo.Constraint.Skip

    if len(lego.model.rp) > 1:
        lego.model.eStInterRes = pyo.Constraint(lego.model.p, lego.model.storageUnits, doc='Inter-day reserve constraint for storage units', rule=eStInterRes_rule)

    # Add vConsump to eDC_BalanceP (vGenP should already be there, since it gets added for all generators)
    for rp in lego.model.rp:
        for k in lego.model.k:
            for i in lego.model.i:
                for g in lego.model.storageUnits:
                    if lego.cs.hGenerators_to_Buses.loc[g]['i'] == i:
                        lego.model.eDC_BalanceP_expr[rp, k, i] -= lego.model.vConsump[rp, k, g]
