import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    storageUnits = cs.dPower_Storage.index.tolist()
    model.storageUnits = pyo.Set(doc='Storage units', initialize=storageUnits)
    LEGO.addToSet(model, "g", storageUnits)
    LEGO.addToSet(model, "gi", cs.dPower_Storage.reset_index().set_index(['g', 'i']).index)  # Note: Add gi after g since it depends on g
    model.longDurationStorageUnits = pyo.Set(doc='Long-duration storage units (subset of storage units)', initialize=cs.dPower_Storage[cs.dPower_Storage['IsLDS'] == 1].index.tolist(), within=model.storageUnits)

    # Parameters
    model.pEnableChDisPower = cs.dPower_Parameters['pEnableChDisPower']  # Avoid simultaneous charging and discharging
    model.pE2PRatio = pyo.Param(model.storageUnits, initialize=cs.dPower_Storage['Ene2PowRatio'], doc='Energy to power ratio of storage unit g')
    model.pMinReserve = pyo.Param(model.storageUnits, initialize=cs.dPower_Storage['MinReserve'] * cs.dPower_Storage['MaxProd'] * cs.dPower_Storage['Ene2PowRatio'], doc='Minimum reserve of storage unit g [power]')
    model.pIniReserve = pyo.Param(model.storageUnits, initialize=cs.dPower_Storage['IniReserve'] * cs.dPower_Storage['MaxProd'] * cs.dPower_Storage['Ene2PowRatio'], doc='Initial reserve of storage unit g [power]')
    model.pMaxReserve = pyo.Param(model.storageUnits, initialize=cs.dPower_Storage['MaxProd'] * cs.dPower_Storage['Ene2PowRatio'], doc='Maximum reserve of storage unit g [power]')

    dInflows = []
    for g in model.longDurationStorageUnits:
        if g in cs.dPower_Inflows.index.get_level_values("g"):
            dInflows.append(cs.dPower_Inflows.loc[(slice(None), slice(None), g), 'value'])
    #dInflows = pd.concat(dInflows, axis=0)

    model.pLDSInflows = pyo.Param(model.rp, model.k, model.longDurationStorageUnits, initialize=dInflows, doc="Inflows of long-duration storage units [power/timestep]", default=0)

    LEGO.addToParameter(model, "pMaxCons", cs.dPower_Storage['MaxCons'], indices=[model.storageUnits], doc='Maximum consumption of storage unit')

    LEGO.addToParameter(model, "pMaxProd", cs.dPower_Storage['MaxProd'])
    LEGO.addToParameter(model, "pMinProd", cs.dPower_Storage['MinProd'])
    LEGO.addToParameter(model, "pExisUnits", cs.dPower_Storage['ExisUnits'])

    LEGO.addToParameter(model, "pOMVarCost", cs.dPower_Storage['pOMVarCostEUR'])
    LEGO.addToParameter(model, "pMaxInvest", cs.dPower_Storage['MaxInvest'])
    LEGO.addToParameter(model, "pEnabInv", cs.dPower_Storage['EnableInvest'])
    LEGO.addToParameter(model, "pInvestCost", cs.dPower_Storage['InvestCostEUR'])

    LEGO.addToParameter(model, 'pMaxGenQ', cs.dPower_Storage['Qmax'])
    LEGO.addToParameter(model, 'pMinGenQ', cs.dPower_Storage['Qmin'])

    # Variables
    if model.pEnableChDisPower:
        model.bChargeDisCharge = pyo.Var(model.storageUnits, model.rp, model.k, doc='Binary variable for charging of storage unit g', domain=pyo.Binary)
        second_stage_variables += [model.bChargeDisCharge]

    model.vConsump = pyo.Var(model.rp, model.k, model.storageUnits, doc='Charging of storage unit g', bounds=lambda model, rp, k, g: (0, model.pMaxCons[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g]))))
    second_stage_variables += [model.vConsump]

    model.vStIntraRes = pyo.Var(model.rp, model.k, model.storageUnits, doc='Intra-reserve of storage unit g', bounds=(None, None))
    second_stage_variables += [model.vStIntraRes]
    for rp in model.rp:
        for k in model.k:
            for g in model.storageUnits:
                model.vStIntraRes[rp, k, g].setub(model.pMaxReserve[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g])))
                model.vStIntraRes[rp, k, g].setlb(model.pMinReserve[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g])))

    model.vStInterRes = pyo.Var(model.p, model.longDurationStorageUnits, doc='Inter-reserve of storage unit g', bounds=(None, None))
    second_stage_variables += [model.vStInterRes]
    for p in model.p:
        for g in model.longDurationStorageUnits:
            if model.p.ord(p) % model.pMovWindow == 0:
                model.vStInterRes[p, g].setub(model.pMaxReserve[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g])))
                model.vStInterRes[p, g].setlb(model.pMinReserve[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g])))

    model.vLDSSpillage = pyo.Var(model.rp, model.k, model.longDurationStorageUnits, doc='Spillage of long-duration storage units', bounds=(0, None))
    second_stage_variables += [model.vLDSSpillage]

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    # Constraint definitions
    model.eStIntraRes = pyo.ConstraintList(doc='Intra-day reserve constraint for storage units')
    model.eExclusiveChargeDischarge = pyo.ConstraintList(doc='Enforce exclusive charge or discharge for storage units')

    # Constraint implementations
    for g in model.storageUnits:
        for rp in model.rp:
            for k in model.k:
                if len(model.rp) == 1:
                    if model.k.ord(k) == 1:  # Adding IniReserve if it is the first time step (instead of 'prev' value)
                        model.eStIntraRes.add(0 == model.pIniReserve[g] * (model.pExisUnits[g] + model.vGenInvest[g])
                                              - model.vStIntraRes[rp, k, g]
                                              - model.vGenP[rp, k, g] * model.pWeight_k[k] / cs.dPower_Storage.loc[g, 'DisEffic'] + model.vConsump[rp, k, g] * model.pWeight_k[k] * cs.dPower_Storage.loc[g, 'ChEffic']
                                              + ((model.pLDSInflows[rp, k, g] * model.pWeight_k[k] - model.vLDSSpillage[rp, k, g] * model.pWeight_k[k]) if g in model.longDurationStorageUnits else 0))
                    else:
                        model.eStIntraRes.add(0 == model.vStIntraRes[rp, model.k.prev(k), g]
                                              - model.vStIntraRes[rp, k, g]
                                              - model.vGenP[rp, k, g] * model.pWeight_k[k] / cs.dPower_Storage.loc[g, 'DisEffic'] + model.vConsump[rp, k, g] * model.pWeight_k[k] * cs.dPower_Storage.loc[g, 'ChEffic']
                                              + ((model.pLDSInflows[rp, k, g] * model.pWeight_k[k] - model.vLDSSpillage[rp, k, g] * model.pWeight_k[k]) if g in model.longDurationStorageUnits else 0))
                elif len(model.rp) > 1:  # Only cyclic if it has multiple representative periods
                    model.eStIntraRes.add(0 == model.vStIntraRes[rp, model.k.prevw(k), g]
                                          - model.vStIntraRes[rp, k, g]
                                          - model.vGenP[rp, k, g] * model.pWeight_k[k] / cs.dPower_Storage.loc[g, 'DisEffic'] + model.vConsump[rp, k, g] * model.pWeight_k[k] * cs.dPower_Storage.loc[g, 'ChEffic']
                                          + ((model.pLDSInflows[rp, k, g] * model.pWeight_k[k] - model.vLDSSpillage[rp, k, g] * model.pWeight_k[k]) if g in model.longDurationStorageUnits else 0))

                # TODO: Check if we should rather do a +/- value and calculate charge/discharge ex-post
                if model.pEnableChDisPower:
                    model.eExclusiveChargeDischarge.add(model.vConsump[rp, k, g] <= model.bChargeDisCharge[rp, k, g] * model.pMaxCons[g] * model.pExisUnits[g])
                    model.eExclusiveChargeDischarge.add(model.vGenP[rp, k, g] <= (1 - model.bChargeDisCharge[rp, k, g]) * model.pMaxProd[g] * model.pExisUnits[g])

    model.eStMaxProd_expr = pyo.Expression(model.rp, model.k, model.storageUnits, doc='Max production expression for storage units', rule=lambda model, rp, k, s: model.vGenP[rp, k, s] - model.vConsump[rp, k, s] - model.pMaxProd[s] * (model.vGenInvest[s] + model.pExisUnits[s]))
    model.eStMaxProd = pyo.Constraint(model.rp, model.k, model.storageUnits, doc='Max production constraint for storage units', rule=lambda model, rp, k, s: model.eStMaxProd_expr[rp, k, s] <= 0)

    model.eStMaxCons_expr = pyo.Expression(model.rp, model.k, model.storageUnits, doc='Max consumption expression for storage units', rule=lambda model, rp, k, s: model.vGenP[rp, k, s] - model.vConsump[rp, k, s] + model.pMaxCons[s] * (model.vGenInvest[s] + model.pExisUnits[s]))
    model.eStMaxCons = pyo.Constraint(model.rp, model.k, model.storageUnits, doc='Max consumption constraint for storage units', rule=lambda model, rp, k, s: model.eStMaxCons_expr[rp, k, s] >= 0)

    model.eStMaxIntraRes_expr = pyo.Expression(model.rp, model.k, model.storageUnits, doc='Max intra-reserve expression for storage units', rule=lambda model, rp, k, s: model.vStIntraRes[rp, k, s] - model.pMaxReserve[s] * (model.vGenInvest[s] + model.pExisUnits[s]))
    model.eStMaxIntraRes = pyo.Constraint(model.rp, model.k, model.storageUnits, doc='Max intra-reserve constraint for storage units', rule=lambda model, rp, k, s: model.eStMaxIntraRes_expr[rp, k, s] <= 0)

    model.eStMinIntraRes_expr = pyo.Expression(model.rp, model.k, model.storageUnits, doc='Min intra-reserve expression for storage units', rule=lambda model, rp, k, s: model.vStIntraRes[rp, k, s] - model.pMinReserve[s] * (model.vGenInvest[s] + model.pExisUnits[s]))
    model.eStMinIntraRes = pyo.Constraint(model.rp, model.k, model.storageUnits, doc='Min intra-reserve constraint for storage units', rule=lambda model, rp, k, s: model.eStMinIntraRes_expr[rp, k, s] >= 0)

    def eStFinIntraRes_rule(model, rp, k, s):
        # If there is only one rp and k is the last period of the representative period, add constraint
        if len(model.rp) == 1 and model.k.ord(k) == len(model.k):
            return model.vStIntraRes[rp, k, s] >= model.pIniReserve[s] * (model.pExisUnits[s] + model.vGenInvest[s])
        else:
            return pyo.Constraint.Skip

    model.eStFinIntraRes = pyo.Constraint(model.rp, model.k, model.storageUnits, doc='Final intra-reserve storage level constraint', rule=eStFinIntraRes_rule)

    def eStMaxInterRes_rule(model, p, s):
        # If current p is a multiple of moving window, add constraint
        if model.p.ord(p) % model.pMovWindow == 0:
            return 0 >= model.vStInterRes[p, s] - model.pMaxReserve[s] * (model.vGenInvest[s] + model.pExisUnits[s])
        else:
            return pyo.Constraint.Skip

    model.eStMaxInterRes = pyo.Constraint(model.p, model.longDurationStorageUnits, doc='Max inter-reserve constraint for storage units', rule=eStMaxInterRes_rule)

    def eStMinInterRes_rule(model, p, s):
        # If current p is a multiple of moving window, add constraint
        if model.p.ord(p) % model.pMovWindow == 0:
            return 0 <= model.vStInterRes[p, s] - model.pMinReserve[s] * (model.vGenInvest[s] + model.pExisUnits[s])
        else:
            return pyo.Constraint.Skip

    model.eStMinInterRes = pyo.Constraint(model.p, model.longDurationStorageUnits, doc='Min inter-reserve constraint for storage units', rule=eStMinInterRes_rule)

    def eStFinInterRes_rule(model, p, s):
        # If current p is the last period of the representative period, add constraint
        if len(model.rp) > 1 and model.p.ord(p) == len(model.p):
            if cs.dPower_Parameters['pFixStInterResToIniReserve']:
                return model.vStInterRes[p, s] == model.pIniReserve[s] * (model.pExisUnits[s] + model.vGenInvest[s])
            else:
                return model.vStInterRes[p, s] >= model.pIniReserve[s] * (model.pExisUnits[s] + model.vGenInvest[s])
        else:
            return pyo.Constraint.Skip

    model.eStFinInterRes = pyo.Constraint(model.p, model.longDurationStorageUnits, doc='Final inter-reserve storage level constraint', rule=eStFinInterRes_rule)

    def eStInterRes_rule(model, p, storage_unit):
        # If current p is a multiple of moving window, add constraint
        if model.p.ord(p) % model.pMovWindow == 0 and len(model.rp) > 1:
            relevant_hindeces = model.hindex[model.p.ord(p) - model.pMovWindow:model.p.ord(p)]
            hindex_count = relevant_hindeces.to_frame(index=False).groupby(['rp', 'k']).size()

            return (0 ==
                    (model.vStInterRes[model.p.at(model.p.ord(p) - model.pMovWindow), storage_unit] if model.p.ord(p) - model.pMovWindow > 0 else 0)
                    + (model.pIniReserve[storage_unit] * (model.pExisUnits[storage_unit] + model.vGenInvest[storage_unit]) if model.p.ord(p) == model.pMovWindow else 0)
                    - model.vStInterRes[p, storage_unit]
                    + sum(- model.vGenP[rp2, k2, storage_unit] * model.pWeight_k[k2] / cs.dPower_Storage.loc[storage_unit, 'DisEffic'] * hindex_count.loc[rp2, k2]
                          + model.vConsump[rp2, k2, storage_unit] * model.pWeight_k[k2] * cs.dPower_Storage.loc[storage_unit, 'ChEffic'] * hindex_count.loc[rp2, k2]
                          + model.pLDSInflows[rp2, k2, storage_unit] * model.pWeight_k[k2] * hindex_count.loc[rp2, k2]
                          - model.vLDSSpillage[rp2, k2, storage_unit] * model.pWeight_k[k] * hindex_count.loc[rp2, k2] for rp2, k2 in hindex_count.index))

        else:  # Skip otherwise
            return pyo.Constraint.Skip

    if len(model.rp) > 1:
        model.eStInterRes = pyo.Constraint(model.p, model.longDurationStorageUnits, doc='Inter-day reserve constraint for storage units', rule=eStInterRes_rule)

    # Add vConsump to eDC_BalanceP (vGenP should already be there, since it gets added for all generators)
    for rp in model.rp:
        for k in model.k:
            for i in model.i:
                for g in model.storageUnits:
                    if (g, i) in model.gi:
                        model.eDC_BalanceP_expr[rp, k, i] -= model.vConsump[rp, k, g]

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = 0.0

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
