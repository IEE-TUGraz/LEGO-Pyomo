import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(lego: LEGO):
    # Sets
    lego.model.importNodes = pyo.Set(doc='Import nodes', initialize=lego.cs.dPower_ImpExp[lego.cs.dPower_ImpExp.index.isin(['ImportFix', 'ImportMax'], level=2)].index.unique(level="i").tolist(), within=lego.model.i)
    lego.model.exportNodes = pyo.Set(doc='Export nodes', initialize=lego.cs.dPower_ImpExp[lego.cs.dPower_ImpExp.index.isin(['ExportFix', 'ExportMax'], level=2)].index.unique(level="i").tolist(), within=lego.model.i)

    # Variables
    lego.model.vImport = pyo.Var(lego.model.rp, lego.model.k, lego.model.importNodes, doc='Import at node i', bounds=(0, None))
    lego.model.vExport = pyo.Var(lego.model.rp, lego.model.k, lego.model.exportNodes, doc='Export at node i', bounds=(0, None))
    for rp in lego.model.rp:
        for k in lego.model.k:
            for i in lego.model.importNodes:
                if any(lego.cs.dPower_ImpExp.index.isin([(i, rp, 'ImportFix', k)])):
                    lego.model.vImport[rp, k, i].fix(lego.cs.dPower_ImpExp.loc[(i, rp, 'ImportFix', k)].iloc[0])
                elif any(lego.cs.dPower_ImpExp.index.isin([(i, rp, 'ImportMax', k)])):
                    lego.model.vImport[rp, k, i].setub(lego.cs.dPower_ImpExp.loc[(i, rp, 'ImportMax', k)].iloc[0])
                else:
                    lego.model.vImport[rp, k, i].fix(0)

                if any(lego.cs.dPower_ImpExp.index.isin([(i, rp, 'ExportFix', k)])):
                    lego.model.vExport[rp, k, i].fix(lego.cs.dPower_ImpExp.loc[(i, rp, 'ExportFix', k)].iloc[0])
                elif any(lego.cs.dPower_ImpExp.index.isin([(i, rp, 'ExportMax', k)])):
                    lego.model.vExport[rp, k, i].setub(lego.cs.dPower_ImpExp.loc[(i, rp, 'ExportMax', k)].iloc[0])
                else:
                    lego.model.vExport[rp, k, i].fix(0)

    # Parameters
    lego.model.pImpExpPrice = pyo.Var(lego.model.rp, lego.model.k, lego.model.importNodes | lego.model.exportNodes, doc='Imp-/Export price at node i',
                                      initialize=lego.cs.dPower_ImpExp[lego.cs.dPower_ImpExp.index.isin(['Price'], level=2)].droplevel("Type").reorder_levels(["rp", "k", "i"]))


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):
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
                if LEGOUtilities.p_to_int(p) % lego.model.pMovWindow == 0 and LEGOUtilities.p_to_int(p) > lego.model.pMovWindow:
                    relevant_hindeces = lego.model.hindex[LEGOUtilities.p_to_int(p) - lego.model.pMovWindow:LEGOUtilities.p_to_int(p)]
                    hindex_count = relevant_hindeces.to_frame(index=False).groupby(['rp', 'k']).size()

                    lego.model.eStInterRes.add(0 == lego.model.vStInterRes[LEGOUtilities.int_to_p(LEGOUtilities.p_to_int(p) - lego.model.pMovWindow), g] +
                                               (lego.cs.dPower_Storage.loc[g, 'IniReserve'] if LEGOUtilities.p_to_int(p) == lego.model.pMovWindow else 0) -
                                               lego.model.vStInterRes[p, g] +
                                               sum(-lego.model.vGenP[g, rp2, k2] * lego.model.pWeight_k[k2] / lego.cs.dPower_Storage.loc[g, 'DisEffic'] * hindex_count.loc[rp2, k2] +
                                                   lego.model.vConsump[g, rp2, k2] * lego.model.pWeight_k[k2] * lego.cs.dPower_Storage.loc[g, 'ChEffic'] * hindex_count.loc[rp2, k2] for rp2, k2 in hindex_count.index if LEGOUtilities.p_to_int(p) - lego.model.pMovWindow < LEGOUtilities.p_to_int(p) <= LEGOUtilities.p_to_int(p)))
