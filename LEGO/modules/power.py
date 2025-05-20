import numpy as np
import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGOUtilities, LEGO


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy):
    # Sets
    model.i = pyo.Set(doc='Buses', initialize=cs.dPower_BusInfo.index.tolist())

    model.c = pyo.Set(doc='Circuits', initialize=cs.dPower_Network.index.get_level_values('c').unique().tolist())
    model.la = pyo.Set(doc='All lines', initialize=cs.dPower_Network.index.tolist(), within=model.i * model.i * model.c)
    model.le = pyo.Set(doc='Existing lines', initialize=cs.dPower_Network[(cs.dPower_Network["pEnableInvest"] == 0)].index.tolist(), within=model.la)
    model.lc = pyo.Set(doc='Candidate lines', initialize=cs.dPower_Network[(cs.dPower_Network["pEnableInvest"] == 1)].index.tolist(), within=model.la)

    model.g = pyo.Set(doc='Generators')
    model.gi = pyo.Set(doc='Generator g connected to bus i', within=model.g * model.i)

    if cs.dPower_Parameters["pEnableThermalGen"]:
        model.thermalGenerators = pyo.Set(doc='Thermal Generators', initialize=cs.dPower_ThermalGen.index.tolist())
        LEGO.addToSet(model, "g", model.thermalGenerators)
        LEGO.addToSet(model, "gi", cs.dPower_ThermalGen.reset_index().set_index(['g', 'i']).index)

    if cs.dPower_Parameters["pEnableRoR"]:
        model.rorGenerators = pyo.Set(doc='Run-of-river generators', initialize=cs.dPower_RoR.index.tolist())
        LEGO.addToSet(model, "g", model.rorGenerators)
        LEGO.addToSet(model, "gi", cs.dPower_RoR.reset_index().set_index(['g', 'i']).index)

    if cs.dPower_Parameters["pEnableVRES"]:
        model.vresGenerators = pyo.Set(doc='Variable renewable energy sources', initialize=cs.dPower_VRES.index.tolist())
        LEGO.addToSet(model, "g", model.vresGenerators)
        LEGO.addToSet(model, "gi", cs.dPower_VRES.reset_index().set_index(['g', 'i']).index)

    model.p = pyo.Set(doc='Periods', initialize=cs.dPower_Hindex.index.get_level_values('p').unique().tolist())
    model.rp = pyo.Set(doc='Representative periods', initialize=cs.dPower_Demand.index.get_level_values('rp').unique().tolist())
    model.k = pyo.Set(doc='Timestep within representative period', initialize=cs.dPower_Demand.index.get_level_values('k').unique().tolist())
    model.hindex = cs.dPower_Hindex.index

    # Parameters
    model.pDemandP = pyo.Param(model.rp, model.k, model.i, initialize=cs.dPower_Demand['value'], doc='Demand at bus i in representative period rp and timestep k')
    model.pMovWindow = cs.dGlobal_Parameters['pMovWindow']

    model.pOMVarCost = pyo.Param(model.g, doc='Production cost of generator g')
    model.pEnabInv = pyo.Param(model.g, doc='Enable investment in thermal generator g')
    model.pMaxInvest = pyo.Param(model.g, doc='Maximum investment in thermal generator g')
    model.pInvestCost = pyo.Param(model.g, doc='Investment cost for thermal generator g')
    model.pMaxProd = pyo.Param(model.g, doc='Maximum production of generator g')
    model.pMinProd = pyo.Param(model.g, doc='Minimum production of generator g')
    model.pExisUnits = pyo.Param(model.g, doc='Existing units of generator g')

    if cs.dPower_Parameters["pEnableThermalGen"]:
        LEGO.addToParameter(model, "pOMVarCost", cs.dPower_ThermalGen['pSlopeVarCostEUR'])
        LEGO.addToParameter(model, "pEnabInv", cs.dPower_ThermalGen['EnableInvest'])
        LEGO.addToParameter(model, "pMaxInvest", cs.dPower_ThermalGen['MaxInvest'])
        LEGO.addToParameter(model, "pInvestCost", cs.dPower_ThermalGen['InvestCostEUR'])
        LEGO.addToParameter(model, "pMaxProd", cs.dPower_ThermalGen['MaxProd'])
        LEGO.addToParameter(model, "pMinProd", cs.dPower_ThermalGen['MinProd'])
        LEGO.addToParameter(model, "pExisUnits", cs.dPower_ThermalGen['ExisUnits'])

        model.pInterVarCost = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['pInterVarCostEUR'], doc='Inter-variable cost of thermal generator g')
        model.pStartupCost = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['pStartupCostEUR'], doc='Startup cost of thermal generator g')
        model.pMinUpTime = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['MinUpTime'], doc='Minimum up time of thermal generator g')
        model.pMinDownTime = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['MinDownTime'], doc='Minimum down time of thermal generator g')
        model.pRampUp = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['RampUp'], doc='Ramp up of thermal generator g')
        model.pRampDw = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['RampDw'], doc='Ramp down of thermal generator g')

    if cs.dPower_Parameters["pEnableRoR"]:
        LEGO.addToParameter(model, "pOMVarCost", cs.dPower_RoR['OMVarCost'])
        LEGO.addToParameter(model, "pEnabInv", cs.dPower_RoR['EnableInvest'])
        LEGO.addToParameter(model, "pMaxInvest", cs.dPower_RoR['MaxInvest'])
        LEGO.addToParameter(model, "pInvestCost", cs.dPower_RoR['InvestCostEUR'])
        LEGO.addToParameter(model, "pMaxProd", cs.dPower_RoR['MaxProd'])
        LEGO.addToParameter(model, "pMinProd", cs.dPower_RoR['MinProd'])
        LEGO.addToParameter(model, "pExisUnits", cs.dPower_RoR['ExisUnits'])

    if cs.dPower_Parameters["pEnableVRES"]:
        LEGO.addToParameter(model, "pOMVarCost", cs.dPower_VRES['OMVarCost'])
        LEGO.addToParameter(model, "pEnabInv", cs.dPower_VRES['EnableInvest'])
        LEGO.addToParameter(model, "pMaxInvest", cs.dPower_VRES['MaxInvest'])
        LEGO.addToParameter(model, "pInvestCost", cs.dPower_VRES['InvestCostEUR'])
        LEGO.addToParameter(model, "pMaxProd", cs.dPower_VRES['MaxProd'])
        LEGO.addToParameter(model, "pMinProd", cs.dPower_VRES['MinProd'])
        LEGO.addToParameter(model, "pExisUnits", cs.dPower_VRES['ExisUnits'])

    model.pXline = pyo.Param(model.la, initialize=cs.dPower_Network['pXline'], doc='Reactance of line la')
    model.pAngle = pyo.Param(model.la, initialize=cs.dPower_Network['pAngle'] * np.pi / 180, doc='Transformer angle shift')
    model.pRatio = pyo.Param(model.la, initialize=cs.dPower_Network['pRatio'], doc='Transformer ratio')
    model.pPmax = pyo.Param(model.la, initialize=cs.dPower_Network['pPmax'], doc='Maximum power flow on line la')
    model.pFixedCost = pyo.Param(model.la, initialize=cs.dPower_Network['pInvestCost'], doc='Fixed cost when investing in line la')  # TODO: Think about renaming this parameter (something related to 'investment cost')
    model.pSBase = pyo.Param(initialize=cs.dPower_Parameters['pSBase'], doc='Base power')
    model.pBigM_Flow = pyo.Param(initialize=1e3, doc="Big M for power flow")
    model.pENSCost = pyo.Param(initialize=cs.dPower_Parameters['pENSCost'], doc='Cost used for Power Not Served (PNS) and Excess Power Served (EPS)')

    model.pWeight_rp = pyo.Param(model.rp, initialize=cs.dPower_WeightsRP["pWeight_rp"], doc='Weight of representative period rp')
    model.pWeight_k = pyo.Param(model.k, initialize=cs.dPower_WeightsK["pWeight_k"], doc='Weight of time step k')

    # Variables
    model.vTheta = pyo.Var(model.rp, model.k, model.i, doc='Angle of bus i', bounds=(-cs.dPower_Parameters["pMaxAngleDCOPF"], cs.dPower_Parameters["pMaxAngleDCOPF"]))  # TODO: Discuss impact on runtime etc.(based on discussion with Prof. Renner)
    model.vAngle = pyo.Var(model.rp, model.k, model.la, doc='Angle phase shifting transformer')
    for i, j, c in model.la:
        if model.pAngle[i, j, c] == 0:
            model.vAngle[:, :, i, j, c].fix(0)
        else:
            model.vAngle[:, :, i, j, c].setub(model.pAngle[i, j, c])
            model.vAngle[:, :, i, j, c].setlb(-model.pAngle[i, j, c])

    model.vLineInvest = pyo.Var(model.la, doc='Transmission line investment', domain=pyo.Binary)
    for i, j, c in model.le:
        model.vLineInvest[i, j, c].fix(0)  # Set existing lines to not investable

    model.vGenInvest = pyo.Var(model.g, doc="Integer generation investment", bounds=lambda model, g: (0, model.pMaxInvest[g] * model.pEnabInv[g]))

    # For each DC-OPF "island", set node with highest demand as slack node
    dDCOPFIslands = pd.DataFrame(index=cs.dPower_BusInfo.index, columns=[cs.dPower_BusInfo.index], data=False)

    for index, entry in cs.dPower_Network.iterrows():
        if cs.dPower_Network.loc[(index[0], index[1], index[2])]["pTecRepr"] == "DC-OPF":
            dDCOPFIslands.loc[index[0], index[1]] = True
            dDCOPFIslands.loc[index[1], index[0]] = True

    completed_buses = set()  # Set of buses that have been looked at already
    i = 0
    for index, entry in dDCOPFIslands.iterrows():
        if index in completed_buses or entry[entry == True].empty:  # Skip if bus has already been looked at or has no connections
            continue

        connected_buses = cs.get_connected_buses(dDCOPFIslands, str(index))

        for bus in connected_buses:
            completed_buses.add(bus)

        # Set slack node
        slack_node = cs.dPower_Demand.loc[:, :, connected_buses].groupby('i').sum().idxmax().values[0]
        slack_node = cs.dPower_Parameters["is"]  # TODO: Switch this again to be calculated (fixed to 'is' for compatibility)
        if i == 0: print("Setting slack nodes for DC-OPF zones:")
        print(f"DC-OPF Zone {i:>2} - Slack node: {slack_node}")
        i += 1
        model.vTheta[:, :, slack_node].fix(0)

    model.vPNS = pyo.Var(model.rp, model.k, model.i, doc='Slack variable power not served', bounds=lambda model, rp, k, i: (0, model.pDemandP[rp, k, i]))
    model.vEPS = pyo.Var(model.rp, model.k, model.i, doc='Slack variable excess power served', bounds=(0, None))

    # Used to relax vCommit, vStartup and vShutdown in the first timesteps of each representative period
    # Required when using Markov-Chains to connect the timesteps of the representative periods - since fractions of the binary variables (which are present due to the transition-probabilities) are otherwise not possible
    def vUC_domain(model, k, relax_duration_from_beginning):
        if model.k.ord(k) <= relax_duration_from_beginning:
            return pyo.PercentFraction  # PercentFraction = Floating point values in the interval [0,1]
        else:
            return pyo.Binary

    model.vGenP = pyo.Var(model.rp, model.k, model.g, doc='Power output of generator g', bounds=lambda model, rp, k, g: (0, model.pMaxProd[g] * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g])))

    if cs.dPower_Parameters["pEnableThermalGen"]:
        model.vCommit = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Unit commitment of generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, max(model.pMinUpTime[t], model.pMinDownTime[t])) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
        model.vStartup = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Start-up of thermal generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, model.pMinDownTime[t]) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
        model.vShutdown = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Shut-down of thermal generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, model.pMinUpTime[t]) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
        model.vGenP1 = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Power output of generator g above minimum production', bounds=lambda model, rp, k, g: (0, (model.pMaxProd[g] - model.pMinProd[g]) * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g])))

    if cs.dPower_Parameters["pEnableRoR"]:
        for g in model.rorGenerators:
            for rp in model.rp:
                for k in model.k:
                    model.vGenP[rp, k, g].setub(min(model.pMaxProd[g], cs.dPower_Inflows.loc[rp, g, k]['Inflow']))  # TODO: Check and adapt for storage

    if cs.dPower_Parameters["pEnableVRES"]:
        for g in model.vresGenerators:
            for rp in model.rp:
                for k in model.k:
                    model.vGenP[rp, k, g].setub((model.pMaxProd[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g])) * cs.dPower_VRESProfiles.loc[rp, k, g]['value']))

    model.vLineP = pyo.Var(model.rp, model.k, model.la, doc='Power flow from bus i to j', bounds=(None, None))
    for (i, j, c) in model.la:
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF" | "TP":
                model.vLineP[:, :, (i, j), c].setlb(-model.pPmax[i, j, c])
                model.vLineP[:, :, (i, j), c].setub(model.pPmax[i, j, c])
            case "SN":
                assert False  # "SN" line found, although all "Single Node" buses should be merged
            case _:
                raise ValueError(f"Technical representation '{cs.dPower_Network.loc[i, j]["pTecRepr"]}' "
                                 f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    # Power balance for nodes
    def eDC_BalanceP_rule(model, rp, k, i):
        return (sum(model.vGenP[rp, k, g] for g in model.g if (g, i) in model.gi) -  # Production of generators at bus i
                sum(model.vLineP[rp, k, e] for e in model.la if (e[0] == i)) +  # Power flow from bus i to bus j
                sum(model.vLineP[rp, k, e] for e in model.la if (e[1] == i)) -  # Power flow from bus j to bus i
                model.pDemandP[rp, k, i] +  # Demand at bus i
                model.vPNS[rp, k, i] -  # Slack variable for demand not served
                model.vEPS[rp, k, i])  # Slack variable for overproduction

    # Note: eDC_BalanceP_expr is defined as expression to enable later adding coefficients to the constraint (e.g., for import/export)
    model.eDC_BalanceP_expr = pyo.Expression(model.rp, model.k, model.i, rule=eDC_BalanceP_rule)
    model.eDC_BalanceP = pyo.Constraint(model.rp, model.k, model.i, doc='Power balance constraint for each bus', rule=lambda model, rp, k, i: model.eDC_BalanceP_expr[rp, k, i] == 0)

    def eDC_ExiLinePij_rule(model, rp, k, i, j, c):
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF":
                return model.vLineP[rp, k, i, j, c] == (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) * model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c])
            case "TP" | "SN":
                return pyo.Constraint.Skip
            case _:
                raise ValueError(f"Technical representation '{cs.dPower_Network.loc[i, j]["pTecRepr"]}' "
                                 f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

    model.eDC_ExiLinePij = pyo.Constraint(model.rp, model.k, model.le, doc="Power flow existing lines (for DC-OPF)", rule=eDC_ExiLinePij_rule)

    def eDC_CanLinePij1_rule(model, rp, k, i, j, c):
        return model.vLineP[rp, k, i, j, c] / (model.pBigM_Flow * model.pPmax[i, j, c]) >= (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) * model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c]) / (model.pBigM_Flow * model.pPmax[i, j, c]) - 1 + model.vLineInvest[i, j, c]

    model.eDC_CanLinePij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij1_rule)

    def eDC_CanLinePij2_rule(model, rp, k, i, j, c):
        return model.vLineP[rp, k, i, j, c] / (model.pBigM_Flow * model.pPmax[i, j, c]) <= (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) * model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c]) / (model.pBigM_Flow * model.pPmax[i, j, c]) + 1 - model.vLineInvest[i, j, c]

    model.eDC_CanLinePij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij2_rule)

    def eDC_LimCanLine1_rule(model, rp, k, i, j, c):
        return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] + model.vLineInvest[i, j, c] >= 0

    model.eDC_LimCanLine1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow limit reverse direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine1_rule)

    def eDC_LimCanLine2_rule(model, rp, k, i, j, c):
        return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] - model.vLineInvest[i, j, c] <= 0

    model.eDC_LimCanLine2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow limit standard direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine2_rule)

    def eReMaxProd_rule(model, rp, k, r):
        capacity = cs.dPower_VRESProfiles.loc[rp, k, r]['value']
        capacity = capacity.values[0] if isinstance(capacity, pd.Series) else capacity
        return model.vGenP[rp, k, r] <= model.pMaxProd[r] * (model.vGenInvest[r] + model.pExisUnits[r]) * capacity

    if cs.dPower_Parameters["pEnableVRES"]:
        model.eReMaxProd = pyo.Constraint(model.rp, model.k, model.vresGenerators, rule=eReMaxProd_rule)

    def eThRampUp_rule(model, rp, k, g, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"]:
            case "notEnforced":
                if model.k.first() == k:
                    return None  # Is not enforced and should therefore be turned into pyo.Constraint.Skip in constraint construction
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prev(k), g] - model.vCommit[rp, k, g] * model.pRampUp[g]
            case "cyclic":
                return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prevw(k), g] - model.vCommit[rp, k, g] * model.pRampUp[g]
            case "markov":
                if model.k.first() == k:
                    return model.vGenP1[rp, k, g] - LEGOUtilities.markov_summand(model.rp, rp, False, model.k.prevw(k), model.vGenP1, transition_matrix, g) - model.vCommit[rp, k, g] * model.pRampUp[g]
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prev(k), g] - model.vCommit[rp, k, g] * model.pRampUp[g]
            case _:
                raise ValueError(f"Period edge handling ramping '{cs.dPower_Parameters['pReprPeriodEdgeHandlingRamping']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eThRampUp_expr = pyo.Expression(model.rp, model.k, model.thermalGenerators, rule=lambda m, rp, k, t: eThRampUp_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))
    model.eThRampUp = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Ramp up for thermal generators (based on doi:10.1007/s10107-015-0919-9)', rule=lambda model, rp, k, t: model.eThRampUp_expr[rp, k, t] <= 0 if not ((cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] == "notEnforced") and (model.k.first() == k)) else pyo.Constraint.Skip)

    def eThRampDw_rule(model, rp, k, g, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"]:
            case "notEnforced":
                if model.k.first() == k:
                    return None  # Is not enforced and should therefore be turned into pyo.Constraint.Skip in constraint construction
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prev(k), g] + model.vCommit[rp, model.k.prevw(k), g] * model.pRampDw[g]
            case "cyclic":
                return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prevw(k), g] + model.vCommit[rp, model.k.prevw(k), g] * model.pRampDw[g]
            case "markov":
                if model.k.first() == k:
                    return model.vGenP1[rp, k, g] - LEGOUtilities.markov_summand(model.rp, rp, False, model.k.prevw(k), model.vGenP1, transition_matrix, g) + LEGOUtilities.markov_summand(model.rp, rp, False, model.k.prevw(k), model.vCommit, transition_matrix, g) * model.pRampDw[g]
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prev(k), g] + model.vCommit[rp, model.k.prev(k), g] * model.pRampDw[g]
            case _:
                raise ValueError(f"Period edge handling ramping '{cs.dPower_Parameters['pReprPeriodEdgeHandlingRamping']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eThRampDw_expr = pyo.Expression(model.rp, model.k, model.thermalGenerators, rule=lambda m, rp, k, t: eThRampDw_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))
    model.eThRampDw = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Ramp down for thermal generators (based on doi:10.1007/s10107-015-0919-9)', rule=lambda model, rp, k, t: model.eThRampDw_expr[rp, k, t] >= 0 if not ((cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] == "notEnforced") and (model.k.first() == k)) else pyo.Constraint.Skip)

    # Thermal Generator production with unit commitment & ramping constraints
    model.eUCTotOut = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Total production of thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, g: model.vGenP[rp, k, g] == model.pMinProd[g] * model.vCommit[rp, k, g] + model.vGenP1[rp, k, g])

    def eThMaxUC_rule(model, rp, k, t):
        return model.vCommit[rp, k, t] <= model.vGenInvest[t] + model.pExisUnits[t]

    model.eThMaxUC = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Maximum number of active units for thermal generators', rule=eThMaxUC_rule)

    def eUCMaxOut1_rule(model, rp, k, t):
        return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vStartup[rp, k, t])

    model.eUCMaxOut1_expr = pyo.Expression(model.rp, model.k, model.thermalGenerators, rule=eUCMaxOut1_rule)
    model.eUCMaxOut1 = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Maximum production for startup of thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, t: model.eUCMaxOut1_expr[rp, k, t] <= 0)

    def eUCMaxOut2_rule(model, rp, k, t, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
            case "notEnforced":
                if model.k.last() == k:
                    return None  # Is not enforced and should therefore be turned into pyo.Constraint.Skip in constraint construction
                else:
                    return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vShutdown[rp, model.k.nextw(k), t])
            case "cyclic":
                return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vShutdown[rp, model.k.nextw(k), t])
            case "markov":
                if model.k.last() == k:  # Markov summand only required for very last timestep
                    return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - LEGOUtilities.markov_summand(model.rp, rp, True, model.k.nextw(k), model.vShutdown, transition_matrix, t))
                else:
                    return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vShutdown[rp, model.k.nextw(k), t])
            case _:
                raise ValueError(f"Period edge handling unit commitment '{cs.dPower_Parameters['pReprPeriodEdgeHandlingUnitCommitment']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eUCMaxOut2_expr = pyo.Expression(model.rp, model.k, model.thermalGenerators, rule=lambda m, rp, k, t: eUCMaxOut2_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeTo))
    model.eUCMaxOut2 = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Maximum production for shutdown of thermal generators (from doi:10.1109/TPWRS.2013.2251373)',
                                      rule=lambda model, rp, k, t: model.eUCMaxOut2_expr[rp, k, t] <= 0 if not ((cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "notEnforced") and (model.k.last() == k)) else pyo.Constraint.Skip)

    def eUCStrShut_rule(model, rp, k, t, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
            case "notEnforced":
                if model.k.ord(k) == 1:
                    return pyo.Constraint.Skip
                else:
                    return model.vCommit[rp, k, t] - model.vCommit[rp, model.k.prevw(k), t] == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
            case "cyclic":
                return model.vCommit[rp, k, t] - model.vCommit[rp, model.k.prevw(k), t] == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
            case "markov":
                if model.k.ord(k) == 1:  # Markov summand only required for very first timestep
                    return model.vCommit[rp, k, t] - LEGOUtilities.markov_summand(model.rp, rp, False, model.k.prevw(k), model.vCommit, transition_matrix, t) == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
                else:
                    return model.vCommit[rp, k, t] - model.vCommit[rp, model.k.prevw(k), t] == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
            case _:
                raise ValueError(f"Period edge handling unit commitment '{cs.dPower_Parameters['pReprPeriodEdgeHandlingUnitCommitment']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eUCStrShut = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Start-up and shut-down logic for thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, t: eUCStrShut_rule(model, rp, k, t, cs.rpTransitionMatrixRelativeFrom))

    def eMinUpTime_rule(model, rp, k, t, transition_matrix):
        if model.pMinUpTime[t] == 0:
            raise ValueError("Minimum up time must be at least 1, got 0 instead")
        elif model.pMinUpTime[t] == 1:
            return pyo.Constraint.Skip
        else:
            match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
                case "notEnforced":
                    if model.k.ord(k) < model.pMinUpTime[t]:
                        return pyo.Constraint.Skip  # Constraint is not active until the minimum up-time is reached
                    else:
                        return sum(model.vStartup[rp, k2, t] for k2 in LEGOUtilities.set_range_non_cyclic(model.k, model.k.ord(k) - model.pMinUpTime[t] + 1, model.k.ord(k))) <= model.vCommit[rp, k, t]
                case "cyclic":
                    return sum(model.vStartup[rp, k2, t] for k2 in LEGOUtilities.set_range_cyclic(model.k, model.k.ord(k) - model.pMinUpTime[t] + 1, model.k.ord(k))) <= model.vCommit[rp, k, t]
                case "markov":
                    return LEGOUtilities.markov_sum(model.rp, rp, model.k, model.k.ord(k) - model.pMinUpTime[t] + 1, model.k.ord(k), model.vStartup, transition_matrix, t) <= model.vCommit[rp, k, t]
                case _:
                    raise ValueError(f"Invalid value for 'pReprPeriodEdgeHandlingUnitCommitment' in 'Global_Parameters.xlsx': {cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]} - please choose from 'notEnforced', 'cyclic' or 'markov'!")

    model.eMinUpTime = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Minimum up time for thermal generators (from doi:10.1109/TPWRS.2013.2251373, adjusted to be cyclic)', rule=lambda m, rp, k, t: eMinUpTime_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))

    def eMinDownTime_rule(model, rp, k, t, transition_matrix):
        if model.pMinDownTime[t] == 0:
            raise ValueError("Minimum down time must be at least 1, got 0 instead")
        elif model.pMinDownTime[t] == 1:
            return pyo.Constraint.Skip
        else:
            match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
                case "notEnforced":
                    if model.k.ord(k) < model.pMinDownTime[t]:
                        return pyo.Constraint.Skip  # Constraint is not active until the minimum down-time is reached
                    else:
                        return sum(model.vShutdown[rp, k2, t] for k2 in LEGOUtilities.set_range_non_cyclic(model.k, model.k.ord(k) - model.pMinDownTime[t] + 1, model.k.ord(k))) <= 1 - model.vCommit[rp, k, t]
                case "cyclic":
                    return sum(model.vShutdown[rp, k2, t] for k2 in LEGOUtilities.set_range_cyclic(model.k, model.k.ord(k) - model.pMinDownTime[t] + 1, model.k.ord(k))) <= 1 - model.vCommit[rp, k, t]
                case "markov":
                    return LEGOUtilities.markov_sum(model.rp, rp, model.k, model.k.ord(k) - model.pMinDownTime[t] + 1, model.k.ord(k), model.vShutdown, transition_matrix, t) <= 1 - model.vCommit[rp, k, t]
                case _:
                    raise ValueError(f"Invalid value for 'pReprPeriodEdgeHandlingUnitCommitment' in 'Global_Parameters.xlsx': {cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]} - please choose from 'notEnforced', 'cyclic' or 'markov'!")

    model.eMinDownTime = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Minimum down time for thermal generators (from doi:10.1109/TPWRS.2013.2251373, adjusted to be cyclic)', rule=lambda m, rp, k, t: eMinDownTime_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))

    # Objective function
    model.objective = pyo.Objective(doc='Total production cost (Objective Function)', sense=pyo.minimize, expr=sum(sum(model.vPNS[rp, k, :]) * model.pWeight_rp[rp] * model.pWeight_k[k] * model.pENSCost for rp in model.rp for k in model.k) +  # Power not served
                                                                                                               sum(sum(model.vEPS[rp, k, :]) * model.pWeight_rp[rp] * model.pWeight_k[k] * model.pENSCost * 2 for rp in model.rp for k in model.k) +  # Excess power served
                                                                                                               sum(model.vStartup[rp, k, t] * model.pStartupCost[t] * model.pWeight_rp[rp] * model.pWeight_k[k] for rp in model.rp for k in model.k for t in model.thermalGenerators) +  # Startup cost of thermal generators
                                                                                                               sum(model.vCommit[rp, k, t] * model.pInterVarCost[t] * model.pWeight_rp[rp] * model.pWeight_k[k] for rp in model.rp for k in model.k for t in model.thermalGenerators) +  # Commit cost of thermal generators
                                                                                                               sum(model.vGenP[rp, k, g] * model.pOMVarCost[g] * model.pWeight_rp[rp] * model.pWeight_k[k] for rp in model.rp for k in model.k for g in model.g) +  # Production cost of generators
                                                                                                               sum(model.pFixedCost[i, j, c] * model.vLineInvest[i, j, c] for i, j, c in model.lc) +  # Investment cost of transmission lines
                                                                                                               sum(model.pInvestCost[g] * model.vGenInvest[g] for g in model.g))  # Investment cost of generators
