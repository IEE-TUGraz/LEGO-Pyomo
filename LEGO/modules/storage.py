import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGO, LEGOUtilities

printer = Printer.getInstance()


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    storageUnits = cs.dPower_Storage.index.tolist()
    model.storageUnits = pyo.Set(doc='Storage units', initialize=storageUnits)
    LEGO.addToSet(model, "g", storageUnits)
    model.gi_storage = pyo.Set(doc="Storage unit g connected to node i", initialize=cs.dPower_Storage.reset_index().set_index(['g', 'i']).index, within=model.storageUnits * model.i)
    LEGO.addToSet(model, "gi", model.gi_storage)  # Note: Add gi after g since it depends on g
    model.longDurationEnergyStorageUnits = pyo.Set(doc='Long-duration energy storage units (subset of storage units)', initialize=cs.dPower_Storage[cs.dPower_Storage['IsLDES'] == 1].index.tolist(), within=model.storageUnits)

    model.intraStorageUnits = pyo.Set(doc='Intra-day storage units (subset of storage units)', initialize=model.storageUnits if len(model.rp) == 1 else (model.storageUnits - model.longDurationEnergyStorageUnits), within=model.storageUnits)
    model.interStorageUnits = pyo.Set(doc='Inter-day storage units (subset of storage units)', initialize=model.longDurationEnergyStorageUnits if len(model.rp) > 1 else [], within=model.storageUnits)
    model.hydroStorageUnits = pyo.Set(doc='Hydro storage units (subset of storage units)', initialize=cs.dPower_Storage[cs.dPower_Storage['tec'].str.lower() == "hydro"].index.tolist(), within=model.storageUnits)
    if len(model.hydroStorageUnits) > 0:
        printer.information(f"The following storage units are hydro storage units: {list(model.hydroStorageUnits)}")
    else:
        printer.information("No hydro storage units defined.")

    # Subset of p with only the elements at 'movingWindow' intervals
    model.movingWindowP = pyo.Set(doc='Set of periods at moving window intervals', initialize=[p for p in model.p if model.p.ord(p) % model.pMovWindow == 0])

    # Parameters
    model.pEnableChDisPower = cs.dPower_Parameters['pEnableChDisPower']  # Avoid simultaneous charging and discharging
    model.pE2PRatio = pyo.Param(model.storageUnits, initialize=cs.dPower_Storage['Ene2PowRatio'], doc='Energy to power ratio of storage unit g')
    model.pMinReserve = pyo.Param(model.storageUnits, initialize=cs.dPower_Storage['MinReserve'] * cs.dPower_Storage['MaxProd'] * cs.dPower_Storage['Ene2PowRatio'], doc='Minimum reserve of storage unit g [energy]')
    model.pIniReserve = pyo.Param(model.storageUnits, initialize=cs.dPower_Storage['IniReserve'] * cs.dPower_Storage['MaxProd'] * cs.dPower_Storage['Ene2PowRatio'], doc='Initial reserve of storage unit g [energy]')
    model.pMaxReserve = pyo.Param(model.storageUnits, initialize=cs.dPower_Storage['MaxProd'] * cs.dPower_Storage['Ene2PowRatio'], doc='Maximum reserve of storage unit g [energy]')

    dInflows = []
    for g in model.hydroStorageUnits:
        if g in cs.dPower_Inflows.index.get_level_values("g"):
            dInflows.append(cs.dPower_Inflows.loc[(slice(None), slice(None), g), 'value'])
    if len(dInflows) != 0:
        dInflows = pd.concat(dInflows, axis=0)
    model.pStorageInflows = pyo.Param(model.rp, model.k, model.storageUnits, initialize=dInflows, doc="Inflows of long-duration energy storage units [energy/timestep]", default=0)

    model.pMaxCons = pyo.Param(model.storageUnits, initialize=cs.dPower_Storage['MaxCons'], doc='Maximum consumption of storage unit [power]')

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

    model.vStIntraRes = pyo.Var(model.rp, model.k, model.intraStorageUnits, doc='Intra-reserve of storage unit g', bounds=lambda m, rp, k, g: (m.pMinReserve[g] * (m.pExisUnits[g] + (m.pMaxInvest[g] * m.pEnabInv[g])), m.pMaxReserve[g] * (m.pExisUnits[g] + (m.pMaxInvest[g] * m.pEnabInv[g]))))
    second_stage_variables += [model.vStIntraRes]

    model.vStInterRes = pyo.Var(model.movingWindowP, model.interStorageUnits, doc='Inter-reserve of storage unit g', bounds=lambda m, p, g: (m.pMinReserve[g] * (m.pExisUnits[g] + (m.pMaxInvest[g] * m.pEnabInv[g])), m.pMaxReserve[g] * (m.pExisUnits[g] + (m.pMaxInvest[g] * m.pEnabInv[g]))))
    second_stage_variables += [model.vStInterRes]

    model.vStorageSpillage = pyo.Var(model.rp, model.k, model.hydroStorageUnits, doc='Spillage of hydro storage unit [power]', bounds=lambda m, rp, k, s: (0, (m.pMaxReserve[s] * (m.pExisUnits[s] + (m.pMaxInvest[s] * m.pEnabInv[s]))) + m.pStorageInflows[rp, k, s]))
    second_stage_variables += [model.vStorageSpillage]

    if cs.dPower_Parameters["pForcePrimitiveStorageUsage"]:
        model.vBin_DyLeR_StBe_NoChGrid = pyo.Var(model.rp, model.k, model.i, doc="TODO Alex", domain=pyo.Binary)
        second_stage_variables.append(model.vBin_DyLeR_StBe_NoChGrid)

        model.vBin_DyLeR_StBe_BeFull = pyo.Var(model.rp, model.k, model.i, doc="TODO Alex", domain=pyo.Binary)
        second_stage_variables.append(model.vBin_DyLeR_StBe_BeFull)

        model.vBin_DyLeR_StBe_NoDischGrid = pyo.Var(model.rp, model.k, model.i, doc="TODO Alex", domain=pyo.Binary)
        second_stage_variables.append(model.vBin_DyLeR_StBe_NoDischGrid)

        model.vBin_DyLeR_StBe_BeEmpty = pyo.Var(model.rp, model.k, model.i, doc="TODO Alex", domain=pyo.Binary)
        second_stage_variables.append(model.vBin_DyLeR_StBe_BeEmpty)

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    def eStIntraRes_rule(m, rp, k, g):
        return (((m.pIniReserve[g] * (m.pExisUnits[g] + m.vGenInvest[g])) if (len(m.rp) == 1 and m.k.ord(k) == 1) else m.vStIntraRes[rp, m.k.prevw(k), g])  # If single representative period and first time step, use initial reserve, otherwise use previous time step
                ==
                + m.vStIntraRes[rp, k, g]
                + m.vGenP[rp, k, g] * m.pWeight_k[k] / cs.dPower_Storage.loc[g, 'DisEffic']
                - m.vConsump[rp, k, g] * m.pWeight_k[k] * cs.dPower_Storage.loc[g, 'ChEffic']
                - ((m.pStorageInflows[rp, k, g] - m.vStorageSpillage[rp, k, g] * m.pWeight_k[k]) if g in m.hydroStorageUnits else 0))

    model.eStIntraRes = pyo.Constraint(model.rp, model.constraintsActiveK, model.intraStorageUnits, doc='Intra-day reserve constraint for storage units', rule=eStIntraRes_rule)

    if model.pEnableChDisPower:
        # TODO: Check if we should rather do a +/- value and calculate charge/discharge ex-post
        model.eExclusiveChargeDischarge = pyo.ConstraintList(doc='Enforce exclusive charge or discharge for storage units')
        for rp in model.rp:
            for k in model.constraintsActiveK:
                for g in model.storageUnits:
                    model.eExclusiveChargeDischarge.add(model.vConsump[rp, k, g] <= model.bChargeDisCharge[rp, k, g] * model.pMaxCons[g] * (model.pExisUnits[g] + model.vGenInvest[g]))
                    model.eExclusiveChargeDischarge.add(model.vGenP[rp, k, g] <= (1 - model.bChargeDisCharge[rp, k, g]) * model.pMaxProd[g] * (model.pExisUnits[g] + model.vGenInvest[g]))

    model.eStMaxProd_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.storageUnits, doc='Max production expression for storage units', rule=lambda model, rp, k, s: model.vGenP[rp, k, s] - model.vConsump[rp, k, s] - model.pMaxProd[s] * (model.pExisUnits[s] + model.vGenInvest[s]))
    model.eStMaxProd = pyo.Constraint(model.rp, model.constraintsActiveK, model.storageUnits, doc='Max production constraint for storage units', rule=lambda model, rp, k, s: model.eStMaxProd_expr[rp, k, s] <= 0)

    model.eStMaxCons_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.storageUnits, doc='Max consumption expression for storage units', rule=lambda model, rp, k, s: model.vGenP[rp, k, s] - model.vConsump[rp, k, s] + model.pMaxCons[s] * (model.pExisUnits[s] + model.vGenInvest[s]))
    model.eStMaxCons = pyo.Constraint(model.rp, model.constraintsActiveK, model.storageUnits, doc='Max consumption constraint for storage units', rule=lambda model, rp, k, s: model.eStMaxCons_expr[rp, k, s] >= 0)

    model.eStMaxIntraRes_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.intraStorageUnits, doc='Max intra-reserve expression for storage units', rule=lambda model, rp, k, s: model.vStIntraRes[rp, k, s] - model.pMaxReserve[s] * (model.pExisUnits[s] + model.vGenInvest[s]))
    model.eStMaxIntraRes = pyo.Constraint(model.rp, model.constraintsActiveK, model.intraStorageUnits, doc='Max intra-reserve constraint for storage units', rule=lambda model, rp, k, s: model.eStMaxIntraRes_expr[rp, k, s] <= 0)

    model.eStMinIntraRes_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.intraStorageUnits, doc='Min intra-reserve expression for storage units', rule=lambda model, rp, k, s: model.vStIntraRes[rp, k, s] - model.pMinReserve[s] * (model.pExisUnits[s] + model.vGenInvest[s]))
    model.eStMinIntraRes = pyo.Constraint(model.rp, model.constraintsActiveK, model.intraStorageUnits, doc='Min intra-reserve constraint for storage units', rule=lambda model, rp, k, s: model.eStMinIntraRes_expr[rp, k, s] >= 0)

    if len(model.rp) == 1:
        # If there is only one rp and k is the last period of the representative period, limit the final storage level to initial storage level
        model.eStFinIntraRes = pyo.Constraint(model.rp, model.constraintsActiveK.at(-1), model.intraStorageUnits, doc='Final intra-reserve storage level constraint', rule=lambda m, rp, k, g: (m.vStIntraRes[rp, k, g] >= m.pIniReserve[g] * (m.pExisUnits[g] + m.vGenInvest[g])))

    if len(model.rp) > 1:  # Only add inter-day constraints if there are multiple representative periods
        model.eStMaxInterRes = pyo.Constraint(model.movingWindowP, model.interStorageUnits, doc='Max inter-reserve constraint for storage units', rule=lambda m, p, s: m.vStInterRes[p, s] <= m.pMaxReserve[s] * (m.pExisUnits[s] + m.vGenInvest[s]))
        model.eStMinInterRes = pyo.Constraint(model.movingWindowP, model.interStorageUnits, doc='Min inter-reserve constraint for storage units', rule=lambda m, p, s: m.vStInterRes[p, s] >= m.pMinReserve[s] * (m.pExisUnits[s] + m.vGenInvest[s]))

        model.eStFinInterRes = pyo.Constraint([model.movingWindowP.at(-1)], model.interStorageUnits, doc='Final inter-reserve storage level constraint', rule=lambda m, p, s: (m.vStInterRes[p, s] == m.pIniReserve[s] * (m.pExisUnits[s] + m.vGenInvest[s])) if cs.dPower_Parameters['pFixStInterResToIniReserve'] else (m.vStInterRes[p, s] >= m.pIniReserve[s] * (m.pExisUnits[s] + m.vGenInvest[s])))

        def eStInterRes_rule(model, p, storage_unit):
            if model.movingWindowP.ord(p) == 1:
                return model.vStInterRes[p, storage_unit] == model.pIniReserve[storage_unit] * (model.pExisUnits[storage_unit] + model.vGenInvest[storage_unit])
            else:
                relevant_hindeces = model.hindex[model.p.ord(p) - model.pMovWindow:model.p.ord(p)]
                hindex_count = relevant_hindeces.to_frame(index=False).groupby(['rp', 'k']).size()

                return (model.vStInterRes[model.movingWindowP.prev(p), storage_unit]
                        ==
                        model.vStInterRes[p, storage_unit]
                        + sum((+ model.vGenP[rp, k, storage_unit] * model.pWeight_k[k] / cs.dPower_Storage.loc[storage_unit, 'DisEffic']
                               - model.vConsump[rp, k, storage_unit] * model.pWeight_k[k] * cs.dPower_Storage.loc[storage_unit, 'ChEffic']
                               - model.pStorageInflows[rp, k, storage_unit])
                              * hindex_count.loc[rp, k] for rp, k in hindex_count.index)
                        + (0 if storage_unit not in model.hydroStorageUnits else sum(model.vStorageSpillage[rp, k, storage_unit] * model.pWeight_k[k] * hindex_count.loc[rp, k] for rp, k in hindex_count.index)))

        model.eStInterRes = pyo.Constraint(model.movingWindowP, model.longDurationEnergyStorageUnits, doc='Inter-day reserve constraint for storage units', rule=eStInterRes_rule)

    # Add vConsump to eDC_BalanceP (vGenP is already there, since it gets added for all generators)
    for rp in model.rp:
        for k in model.constraintsActiveK:
            for g, i in model.gi_storage:
                model.eDC_BalanceP_expr[rp, k, i] -= model.vConsump[rp, k, g]

    if cs.dPower_Parameters["pForcePrimitiveStorageUsage"]:
        def eStDyLeR_StBe1_rule(m, rp, k, i, cs):
            return sum(m.vConsump[rp, k, s] for s in m.storageUnits if (s, i) in model.gi) <= sum(m.vGenP[rp, k, solar] for solar in m.vresGenerators if (solar, i) in model.gi and cs.dPower_VRES.loc[solar]["tec"] == "Solar") - m.pDemandP[rp, k, i] + m.pBigM * m.vBin_DyLeR_StBe_NoChGrid[rp, k, i] - m.eps

        model.eStDyLeR_StBe1 = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Don't charge out of the grid", rule=lambda m, rp, k, i: eStDyLeR_StBe1_rule(m, rp, k, i, cs))

        def eStDyLeR_StBe2_rule(m, rp, k, i):
            return sum(m.vConsump[rp, k, s] for s in m.storageUnits if (s, i) in model.gi) <= m.pBigM * (1 - m.vBin_DyLeR_StBe_NoChGrid[rp, k, i])

        model.eStDyLeR_StBe2 = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Charge only overproduction", rule=lambda m, rp, k, i: eStDyLeR_StBe2_rule(m, rp, k, i))

        def eStDyLeR_StBe3_rule(m, rp, k, i):
            return sum(m.vStIntraRes[rp, k, s] for s in m.storageUnits if (s, i) in model.gi) >= sum(m.pMaxCons[s] * m.pE2PRatio[s] for s in m.storageUnits if (s, i) in model.gi) * m.vBin_DyLeR_StBe_BeFull[rp, k, i] - m.eps

        model.eStDyLeR_StBe3 = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Checks if storage is full", rule=lambda m, rp, k, i: eStDyLeR_StBe3_rule(m, rp, k, i))

        def eStDyLeR_StBe4_rule(m, rp, k, i):
            return sum(m.vConsump[rp, k, s] for s in m.storageUnits if (s, i) in model.gi) >= sum(m.vGenP[rp, k, solar] for solar in m.vresGenerators if (solar, i) in model.gi and cs.dPower_VRES.loc[solar]["tec"] == "Solar") - m.pDemandP[rp, k, i] - m.pBigM * m.vBin_DyLeR_StBe_BeFull[rp, k, i] + m.eps

        model.eStDyLeR_StBe4 = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Charge Storage until it is full", rule=lambda m, rp, k, i: eStDyLeR_StBe4_rule(m, rp, k, i))

        def eStDyLeR_StBe5_rule(m, rp, k, i):
            return sum(m.vGenP[rp, k, s] for s in m.storageUnits if (s, i) in model.gi) <= m.pDemandP[rp, k, i] - sum(m.vGenP[rp, k, solar] for solar in m.vresGenerators if (solar, i) in model.gi and cs.dPower_VRES.loc[solar]["tec"] == "Solar") + m.pBigM * m.vBin_DyLeR_StBe_NoDischGrid[rp, k, i] + m.eps

        model.eStDyLeR_StBe5 = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Don't discharge in the grid", rule=lambda m, rp, k, i: eStDyLeR_StBe5_rule(m, rp, k, i))

        def eStDyLeR_StBe6_rule(m, rp, k, i):
            return sum(m.vGenP[rp, k, s] for s in m.storageUnits if (s, i) in model.gi) <= m.pBigM * (1 - m.vBin_DyLeR_StBe_NoDischGrid[rp, k, i])

        model.eStDyLeR_StBe6 = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Discharge only underproduction", rule=lambda m, rp, k, i: eStDyLeR_StBe6_rule(m, rp, k, i))

        def eStDyLeR_StBe7_rule(m, rp, k, i):
            return sum(m.vStIntraRes[rp, k, s] for s in m.storageUnits if (s, i) in model.gi) <= sum(m.pMaxProd[s] * m.pE2PRatio[s] for s in m.storageUnits if (s, i) in model.gi) * (1 - m.vBin_DyLeR_StBe_BeEmpty[rp, k, i]) + m.eps

        model.eStDyLeR_StBe7 = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Checks if storage is empty", rule=lambda m, rp, k, i: eStDyLeR_StBe7_rule(m, rp, k, i))

        def eStDyLeR_StBe8_rule(m, rp, k, i):
            return sum(m.vGenP[rp, k, s] for s in m.storageUnits if (s, i) in model.gi) >= m.pDemandP[rp, k, i] - sum(m.vGenP[rp, k, solar] for solar in m.vresGenerators if (solar, i) in model.gi and cs.dPower_VRES.loc[solar]["tec"] == "Solar") - m.pBigM * m.vBin_DyLeR_StBe_BeEmpty[rp, k, i] + m.eps

        model.eStDyLeR_StBe8 = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Disharge Storage until it is empty", rule=lambda m, rp, k, i: eStDyLeR_StBe8_rule(m, rp, k, i))

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = 0.0

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
