import pyomo.environ as pyo

import LEGOUtilities
from LEGO import LEGO


def add_module(lego: LEGO):
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
    pass
