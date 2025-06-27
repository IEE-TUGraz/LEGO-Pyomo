import numpy as np
import pandas as pd
import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(lego: LEGO):
    # Sets
    lego.model.i = pyo.Set(doc='Buses', initialize=lego.cs.dPower_BusInfo.index.tolist())

    lego.model.c = pyo.Set(doc='Circuits', initialize=lego.cs.dPower_Network.index.get_level_values('c').unique().tolist())
    lego.model.la = pyo.Set(doc='All lines', initialize=lego.cs.dPower_Network.index.tolist(), within=lego.model.i * lego.model.i * lego.model.c)
    lego.model.le = pyo.Set(doc='Existing lines', initialize=lego.cs.dPower_Network[(lego.cs.dPower_Network["pEnableInvest"] == 0)].index.tolist(), within=lego.model.la)
    lego.model.lc = pyo.Set(doc='Candidate lines', initialize=lego.cs.dPower_Network[(lego.cs.dPower_Network["pEnableInvest"] == 1)].index.tolist(), within=lego.model.la)



    lego.model.g = pyo.Set(doc='Generators')
    lego.model.gi = pyo.Set(doc='Generator g connected to bus i', within=lego.model.g * lego.model.i)

    if lego.cs.dPower_Parameters["pEnableThermalGen"]:
        lego.model.thermalGenerators = pyo.Set(doc='Thermal Generators', initialize=lego.cs.dPower_ThermalGen.index.tolist())
        lego.addToSet("g", lego.model.thermalGenerators)
        lego.addToSet("gi", lego.cs.dPower_ThermalGen.reset_index().set_index(['g', 'i']).index)

    if lego.cs.dPower_Parameters["pEnableRoR"]:
        lego.model.rorGenerators = pyo.Set(doc='Run-of-river generators', initialize=lego.cs.dPower_RoR.index.tolist())
        lego.addToSet("g", lego.model.rorGenerators)
        lego.addToSet("gi", lego.cs.dPower_RoR.reset_index().set_index(['g', 'i']).index)

    if lego.cs.dPower_Parameters["pEnableVRES"]:
        lego.model.vresGenerators = pyo.Set(doc='Variable renewable energy sources', initialize=lego.cs.dPower_VRES.index.tolist())
        lego.addToSet("g", lego.model.vresGenerators)
        lego.addToSet("gi", lego.cs.dPower_VRES.reset_index().set_index(['g', 'i']).index)

    lego.model.p = pyo.Set(doc='Periods', initialize=lego.cs.dPower_Hindex.index.get_level_values('p').unique().tolist())
    lego.model.rp = pyo.Set(doc='Representative periods', initialize=lego.cs.dPower_Demand.index.get_level_values('rp').unique().tolist())
    lego.model.k = pyo.Set(doc='Timestep within representative period', initialize=lego.cs.dPower_Demand.index.get_level_values('k').unique().tolist())
    lego.model.hindex = lego.cs.dPower_Hindex.index

    # Parameters
    lego.model.pDemandP = pyo.Param(lego.model.rp, lego.model.k, lego.model.i, initialize=lego.cs.dPower_Demand['Demand'], doc='Demand at bus i in representative period rp and timestep k')
    lego.model.pMovWindow = lego.cs.dGlobal_Parameters['pMovWindow']

    lego.model.pOMVarCost = pyo.Param(lego.model.g, doc='Production cost of generator g')
    lego.model.pEnabInv = pyo.Param(lego.model.g, doc='Enable investment in thermal generator g')
    lego.model.pMaxInvest = pyo.Param(lego.model.g, doc='Maximum investment in thermal generator g')
    lego.model.pInvestCost = pyo.Param(lego.model.g, doc='Investment cost for thermal generator g')
    lego.model.pMaxProd = pyo.Param(lego.model.g, doc='Maximum production of generator g')
    lego.model.pMinProd = pyo.Param(lego.model.g, doc='Minimum production of generator g')
    lego.model.pExisUnits = pyo.Param(lego.model.g, doc='Existing units of generator g')
    lego.model.pMaxGenQ = pyo.Param(lego.model.g, doc='Maximum reactive production of generator g')
    lego.model.pMinGenQ = pyo.Param(lego.model.g, doc='Minimum reactive production of generator g')

    if lego.cs.dPower_Parameters["pEnableThermalGen"]:
        lego.addToParameter("pOMVarCost", lego.cs.dPower_ThermalGen['pSlopeVarCostEUR'])
        lego.addToParameter("pEnabInv", lego.cs.dPower_ThermalGen['EnableInvest'])
        lego.addToParameter("pMaxInvest", lego.cs.dPower_ThermalGen['MaxInvest'])
        lego.addToParameter("pInvestCost", lego.cs.dPower_ThermalGen['InvestCostEUR'])
        lego.addToParameter("pMaxProd", lego.cs.dPower_ThermalGen['MaxProd'])
        lego.addToParameter("pMinProd", lego.cs.dPower_ThermalGen['MinProd'])
        lego.addToParameter("pExisUnits", lego.cs.dPower_ThermalGen['ExisUnits'])

        # Reactive generator power limits for SOCP formulation
        lego.addToParameter('pMaxGenQ',1e-3 * lego.cs.dPower_ThermalGen['Qmax'].fillna(0))  # Convert from MVar to kVar
        lego.addToParameter('pMinGenQ', 1e-3 * lego.cs.dPower_ThermalGen['Qmin'].fillna(0)) # Convert from MVar to kVar

        lego.model.pInterVarCost = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['pInterVarCostEUR'], doc='Inter-variable cost of thermal generator g')
        lego.model.pStartupCost = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['pStartupCostEUR'], doc='Startup cost of thermal generator g')
        lego.model.pMinUpTime = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['MinUpTime'], doc='Minimum up time of thermal generator g')
        lego.model.pMinDownTime = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['MinDownTime'], doc='Minimum down time of thermal generator g')
        lego.model.pRampUp = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['RampUp'], doc='Ramp up of thermal generator g')
        lego.model.pRampDw = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['RampDw'], doc='Ramp down of thermal generator g')

    if lego.cs.dPower_Parameters["pEnableRoR"]:
        lego.addToParameter("pOMVarCost", lego.cs.dPower_RoR['OMVarCost'])
        lego.addToParameter("pEnabInv", lego.cs.dPower_RoR['EnableInvest'])
        lego.addToParameter("pMaxInvest", lego.cs.dPower_RoR['MaxInvest'])
        lego.addToParameter("pInvestCost", lego.cs.dPower_RoR['InvestCostEUR'])
        lego.addToParameter("pMaxProd", lego.cs.dPower_RoR['MaxProd'])
        lego.addToParameter("pMinProd", lego.cs.dPower_RoR['MinProd'])
        lego.addToParameter("pExisUnits", lego.cs.dPower_RoR['ExisUnits'])

        # Reactive ROR power limits for SOCP formulation
        lego.addToParameter('pMaxGenQ',1e-3 * lego.cs.dPower_RoR['Qmax'].fillna(0))  # Convert from MVar to kVar
        lego.addToParameter('pMinGenQ', 1e-3 * lego.cs.dPower_RoR['Qmin'].fillna(0)) # Convert from MVar to kVar

    if lego.cs.dPower_Parameters["pEnableVRES"]:
        lego.addToParameter("pOMVarCost", lego.cs.dPower_VRES['OMVarCost'])
        lego.addToParameter("pEnabInv", lego.cs.dPower_VRES['EnableInvest'])
        lego.addToParameter("pMaxInvest", lego.cs.dPower_VRES['MaxInvest'])
        lego.addToParameter("pInvestCost", lego.cs.dPower_VRES['InvestCostEUR'])
        lego.addToParameter("pMaxProd", lego.cs.dPower_VRES['MaxProd'])
        lego.addToParameter("pMinProd", lego.cs.dPower_VRES['MinProd'])
        lego.addToParameter("pExisUnits", lego.cs.dPower_VRES['ExisUnits'])

        # Reactive VRES power limits for SOCP formulation
        lego.addToParameter('pMaxGenQ', 1e-3 * lego.cs.dPower_VRES['Qmax'].fillna(0))  # Convert from MVar to kVar
        lego.addToParameter('pMinGenQ', 1e-3 * lego.cs.dPower_VRES['Qmin'].fillna(0))  # Convert from MVar to kVar

    lego.model.pXline = pyo.Param(lego.model.la, initialize=lego.cs.dPower_Network['pXline'], doc='Reactance of line la')
    lego.model.pAngle = pyo.Param(lego.model.la, initialize=lego.cs.dPower_Network['pAngle'] * np.pi / 180, doc='Transformer angle shift')
    lego.model.pRatio = pyo.Param(lego.model.la, initialize=lego.cs.dPower_Network['pRatio'], doc='Transformer ratio')
    lego.model.pPmax = pyo.Param(lego.model.la, initialize=lego.cs.dPower_Network['pPmax'], doc='Maximum power flow on line la')
    lego.model.pFixedCost = pyo.Param(lego.model.la, initialize=lego.cs.dPower_Network['pInvestCost'], doc='Fixed cost when investing in line la')  # TODO: Think about renaming this parameter (something related to 'investment cost')
    lego.model.pSBase = pyo.Param(initialize=lego.cs.dPower_Parameters['pSBase'], doc='Base power')
    lego.model.pBigM_Flow = pyo.Param(initialize=1e3, doc="Big M for power flow")
    lego.model.pENSCost = pyo.Param(initialize=lego.cs.dPower_Parameters['pENSCost'], doc='Cost used for Power Not Served (PNS) and Excess Power Served (EPS)')
    lego.model.pWeight_rp = pyo.Param(lego.model.rp, initialize=lego.cs.dPower_WeightsRP["pWeight_rp"], doc='Weight of representative period rp')
    lego.model.pWeight_k = pyo.Param(lego.model.k, initialize=lego.cs.dPower_WeightsK["pWeight_k"], doc='Weight of time step k')


    # SOCP Parameters
    lego.model.pBusG = pyo.Param(lego.model.i, initialize=lego.cs.dPower_BusInfo['pBusG'], doc='Conductance of bus i')
    lego.model.pBusB = pyo.Param(lego.model.i, initialize=lego.cs.dPower_BusInfo['pBusG'], doc='Susceptance of bus i')
    lego.model.pBus_pf = pyo.Param(lego.model.i, initialize=lego.cs.dPower_BusInfo['pBus_pf'], doc='PowerFactor of bus i')
    lego.model.pRline = pyo.Param(lego.model.la, initialize=lego.cs.dPower_Network['pRline'], doc='Resistance of line la')
    lego.model.pBcline = pyo.Param(lego.model.la, initialize=lego.cs.dPower_Network['pBcline'], doc='Susceptance of line la')
    lego.model.pQmax = pyo.Param(lego.model.la, initialize=lambda model, i, j, c: model.pPmax[i, j, c], doc='Maximum reactive power flow on line la') # It is asumed that Qmax is ident to Pmax
    lego.model.pBigM_SOCP = pyo.Param(initialize=1e3, doc="Big M for SOCP")
    lego.model.pMaxAngleDiff = pyo.Param(initialize=lego.cs.dPower_Parameters["pMaxAngleDiff"] * np.pi / 180, doc='Maximum angle difference between two buses for the SOCP formulation')
    lego.model.pBusMaxV = pyo.Param(lego.model.i, initialize=lego.cs.dPower_BusInfo['pBusMaxV'], doc='Maximum voltage at bus i')
    lego.model.pBusMinV = pyo.Param(lego.model.i, initialize=lambda model, i: max(lego.cs.dPower_BusInfo['pBusMinV'][i], 0.1), doc='Minimum voltage at bus i (with a lower bound of 0.1)')
    lego.model.pGline = pyo.Param(lego.model.la,initialize=lambda model, i, j, c: model.pRline[i, j, c] / ((model.pRline[i, j, c]**2 + model.pXline[i, j, c]**2) if model.pRline[i, j, c] > 1e-6 else 1e-6),doc='Conductance of line la with lower bound')
    lego.model.pBline = pyo.Param(lego.model.la,initialize=lambda model, i, j, c: - model.pXline[i, j, c] / ((model.pRline[i, j, c]**2 + model.pXline[i, j, c]**2) if model.pRline[i, j, c] > 1e-6 else 1e-6),doc='Susceptance of line la with lower bound')
    lego.model.pRatioDemQP = pyo.Param(lego.model.i, initialize=lambda model,i: pyo.tan(pyo.acos(model.pBus_pf[i])))
    lego.model.pDemandQ = pyo.Param(lego.model.rp, lego.model.k, lego.model.i, initialize=lambda model, rp, k, i: model.pDemandP[rp, k, i] * model.pRatioDemQP[i], doc='Reactive demand at bus i in representative period rp and timestep k')


    # Variables
    lego.model.vTheta = pyo.Var(lego.model.rp, lego.model.k, lego.model.i, doc='Angle of bus i', bounds=(-lego.cs.dPower_Parameters["pMaxAngleDCOPF"], lego.cs.dPower_Parameters["pMaxAngleDCOPF"]))  # TODO: Discuss impact on runtime etc.(based on discussion with Prof. Renner)
    lego.model.vAngle = pyo.Var(lego.model.rp, lego.model.k, lego.model.la, doc='Angle phase shifting transformer')
    for i, j, c in lego.model.la:
        if lego.model.pAngle[i, j, c] == 0:
            lego.model.vAngle[:, :, i, j, c].fix(0)
        else:
            lego.model.vAngle[:, :, i, j, c].setub(lego.model.pAngle[i, j, c])
            lego.model.vAngle[:, :, i, j, c].setlb(-lego.model.pAngle[i, j, c])


    lego.model.vLineInvest = pyo.Var(lego.model.la, doc='Transmission line investment', domain=pyo.Binary)
    for i, j, c in lego.model.le:
        lego.model.vLineInvest[i, j, c].fix(0)  # Set existing lines to not investable

    lego.model.vGenInvest = pyo.Var(lego.model.g, doc="Integer generation investment", bounds=lambda model, g: (0, model.pMaxInvest[g] * model.pEnabInv[g]))

    # For each DC-OPF "island", set node with highest demand as slack node
    dDCOPFIslands = pd.DataFrame(index=lego.cs.dPower_BusInfo.index, columns=[lego.cs.dPower_BusInfo.index], data=False)

    for index, entry in lego.cs.dPower_Network.iterrows():
        if lego.cs.dPower_Network.loc[(index[0], index[1], index[2])]["pTecRepr"] == "DC-OPF" or "SOCP":
            dDCOPFIslands.loc[index[0], index[1]] = True
            dDCOPFIslands.loc[index[1], index[0]] = True
    completed_buses = set()  # Set of buses that have been looked at already
    i = 0

    for index, entry in dDCOPFIslands.iterrows():
        if index in completed_buses or entry[entry == True].empty:
            continue
        connected_buses = lego.cs.get_connected_buses(dDCOPFIslands, str(index))
        for bus in connected_buses:
            completed_buses.add(bus)
        completed_buses.add(index)

        # Set slack node
        slack_node = lego.cs.dPower_Demand.loc[:, :, connected_buses].groupby('i').sum().idxmax().values[0]
        slack_node = lego.cs.dPower_Parameters["is"]  # TODO: Switch this again to be calculated (fixed to 'is' for compatibility)
        if not lego.cs.dPower_Parameters["pEnableSOCP"]:

            if i == 0: print("Setting slack nodes for DC-OPF zones:")
            print(f"DC-OPF Zone {i:>2} - Slack node: {slack_node}")
            i += 1
            lego.model.vTheta[:, :, slack_node].fix(0)

    lego.model.vPNS = pyo.Var(lego.model.rp, lego.model.k, lego.model.i, doc='Slack variable power not served', bounds=lambda model, rp, k, i: (0, model.pDemandP[rp, k, i]))
    lego.model.vEPS = pyo.Var(lego.model.rp, lego.model.k, lego.model.i, doc='Slack variable excess power served', bounds=(0, None))

    # Used to relax vCommit, vStartup and vShutdown in the first timesteps of each representative period
    # Required when using Markov-Chains to connect the timesteps of the representative periods - since fractions of the binary variables (which are present due to the transition-probabilities) are otherwise not possible
    def vUC_domain(model, k, relax_duration_from_beginning):
        if model.k.ord(k) <= relax_duration_from_beginning:
            return pyo.PercentFraction  # PercentFraction = Floating point values in the interval [0,1]
        else:
            return pyo.Binary

    lego.model.vGenP = pyo.Var(lego.model.rp, lego.model.k, lego.model.g, doc='Power output of generator g', bounds=lambda model, rp, k, g: (0, lego.model.pMaxProd[g] * (lego.model.pExisUnits[g] + lego.model.pMaxInvest[g] * lego.model.pEnabInv[g])))

    if lego.cs.dPower_Parameters["pEnableThermalGen"]:
        lego.model.vCommit = pyo.Var(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Unit commitment of generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, max(model.pMinUpTime[t], model.pMinDownTime[t])) if lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
        lego.model.vStartup = pyo.Var(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Start-up of thermal generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, model.pMinDownTime[t]) if lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
        lego.model.vShutdown = pyo.Var(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Shut-down of thermal generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, model.pMinUpTime[t]) if lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
        lego.model.vGenP1 = pyo.Var(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Power output of generator g above minimum production', bounds=lambda model, rp, k, g: (0, (lego.model.pMaxProd[g] - lego.model.pMinProd[g]) * (lego.model.pExisUnits[g] + lego.model.pMaxInvest[g] * lego.model.pEnabInv[g])))

    if lego.cs.dPower_Parameters["pEnableRoR"]:
        for g in lego.model.rorGenerators:
            for rp in lego.model.rp:
                for k in lego.model.k:
                    lego.model.vGenP[rp, k, g].setub(min(lego.model.pMaxProd[g], lego.cs.dPower_Inflows.loc[rp, g, k]['Inflow']))  # TODO: Check and adapt for storage

    if lego.cs.dPower_Parameters["pEnableVRES"]:
        for g in lego.model.vresGenerators:
            for rp in lego.model.rp:
                for k in lego.model.k:
                    lego.model.vGenP[rp, k, g].setub((lego.model.pMaxProd[g] * (lego.model.pExisUnits[g] + (lego.model.pMaxInvest[g] * lego.model.pEnabInv[g])) * lego.cs.dPower_VRESProfiles.loc[rp, k, g]['Capacity']))



    if not lego.cs.dPower_Parameters["pEnableSOCP"]: #Bound sonly apply in forward direction for existing and candidate lines

        lego.model.vLineP = pyo.Var(lego.model.rp, lego.model.k, lego.model.la, doc='Power flow from bus i to j', bounds=(None, None))
        for (i, j, c) in lego.model.la:
            match lego.cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
                case "DC-OPF" | "TP":
                    lego.model.vLineP[:, :, (i, j), c].setlb(-lego.model.pPmax[i, j, c])
                    lego.model.vLineP[:, :, (i, j), c].setub(lego.model.pPmax[i, j, c])
                case "SN":
                    assert False  # "SN" line found, although all "Single Node" buses should be merged
                case _:
                    raise ValueError(f"Technical representation '{lego.cs.dPower_Network.loc[i, j]["pTecRepr"]}' "
                                        f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):
    # Power balance for nodes DC ann SOCP
    def eDC_BalanceP_rule(model, rp, k, i):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return pyo.Constraint.Skip  # skip DC constraint if SOCP is enabled

        return (
                sum(model.vGenP[rp, k, g] for g in model.g if (g, i) in model.gi) -
                sum(model.vLineP[rp, k, e] for e in model.la if e[0] == i) +
                sum(model.vLineP[rp, k, e] for e in model.la if e[1] == i) -
                model.pDemandP[rp, k, i] +
                model.vPNS[rp, k, i] -
                model.vEPS[rp, k, i]
        )

    lego.model.eDC_BalanceP = pyo.Constraint(
        lego.model.rp, lego.model.k, lego.model.i, rule=eDC_BalanceP_rule, doc="DC power balance equation"
    )

    lego.model.eDC_BalanceP_expr = pyo.Expression(lego.model.rp, lego.model.k, lego.model.i, rule=eDC_BalanceP_rule)
    lego.model.eDC_BalanceP = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.i, doc='Power balance constraint for each bus', rule=lambda model, rp, k, i: lego.model.eDC_BalanceP_expr[rp, k, i] == 0)

    # Note: eDC_BalanceP_expr is defined as expression to enable later adding coefficients to the constraint (e.g., for import/export)

    def eSOCP_BalanceQ_rule(model, rp, k, i):
        return (
                sum(model.vGenQ[rp, k, g] for g in model.g if (g, i) in model.gi)
                # Only vLineQ where i is the sending end (i → j)
                - sum(model.vLineQ[rp, k, i, j, c] for (i2, j, c) in model.la_full if i2 == i)
                + model.vSOCP_cii[rp, k, i] * model.pBusB[i] * model.pSBase
                - model.pDemandQ[rp, k, i]
                + model.vPNS[rp, k, i] * model.pRatioDemQP[i]
                - model.vEPS[rp, k, i] * model.pRatioDemQP[i]
        )

    def eDC_ExiLinePij_rule(model, rp, k, i, j, c):
        match lego.cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF":
                return model.vLineP[rp, k, i, j, c] == (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) * model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c])
            case "TP" | "SN" | "SOCP":
                return pyo.Constraint.Skip
            case _:
                raise ValueError(f"Unsupported pTecRepr: {lego.cs.dPower_Network.loc[i, j, c]['pTecRepr']}")
    if not lego.cs.dPower_Parameters["pEnableSOCP"]:
        lego.model.eDC_ExiLinePij = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le, doc="Power flow existing lines (for DC-OPF)", rule=eDC_ExiLinePij_rule)

    def eDC_CanLinePij1_rule(model, rp, k, i, j, c):
        match lego.cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF":
                return (
                    model.vLineP[rp, k, i, j, c] / (model.pBigM_Flow * model.pPmax[i, j, c]) >= 
                    (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) * 
                    model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c]) / 
                    (model.pBigM_Flow * model.pPmax[i, j, c]) - 1 + model.vLineInvest[i, j, c]
                )
            case "TP" | "SN" | "SOCP":
                return pyo.Constraint.Skip
            case _:
                raise ValueError(f"Unsupported pTecRepr: {lego.cs.dPower_Network.loc[i, j, c]['pTecRepr']}")
    if not lego.cs.dPower_Parameters["pEnableSOCP"]:
        lego.model.eDC_CanLinePij1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij1_rule)

    def eDC_CanLinePij2_rule(model, rp, k, i, j, c):
        match lego.cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF":
                return (
                    model.vLineP[rp, k, i, j, c] / (model.pBigM_Flow * model.pPmax[i, j, c]) <= 
                    (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) * 
                    model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c]) / 
                    (model.pBigM_Flow * model.pPmax[i, j, c]) + 1 - model.vLineInvest[i, j, c]
                )
            case "TP" | "SN" | "SOCP":
                return pyo.Constraint.Skip
            case _:
                raise ValueError(f"Unsupported pTecRepr: {lego.cs.dPower_Network.loc[i, j, c]['pTecRepr']}")
    if not lego.cs.dPower_Parameters["pEnableSOCP"]:
        lego.model.eDC_CanLinePij2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij2_rule)

    def eDC_LimCanLine1_rule(model, rp, k, i, j, c):
        match lego.cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF" | "TP" | "SN":
                return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] + model.vLineInvest[i, j, c] >= 0
            case 'SOCP':
                return pyo.Constraint.Skip
        return pyo.Constraint.Skip
    if not lego.cs.dPower_Parameters["pEnableSOCP"]:
        lego.model.eDC_LimCanLine1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Power flow limit standart direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine1_rule)

    def eDC_LimCanLine2_rule(model, rp, k, i, j, c):
        match lego.cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF" | "TP" | "SN":
                return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] - model.vLineInvest[i, j, c] <= 0
            case 'SOCP':
                return pyo.Constraint.Skip
        return pyo.Constraint.Skip
    if not lego.cs.dPower_Parameters["pEnableSOCP"]:
        lego.model.eDC_LimCanLine2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Power flow limit reverse direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine2_rule)
    # Reactive power limits

    def eSOCP_QMaxOut_rule(model, rp, k, g): 
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            if model.pMaxGenQ[g] != 0:
                return model.vGenQ[rp, k, g] / model.pMaxGenQ[g] <= model.vCommit[rp, k, g]
            else:
                return pyo.Constraint.Skip
        else: 
            return pyo.Constraint.Skip

    def eSOCP_QMinOut1_rule(model, rp, k, g): 
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            if model.pMinGenQ[g] >= 0:
                return model.vGenQ[rp, k, g] / model.pMinGenQ[g] >= model.vCommit[rp, k, g]
            else:
                return pyo.Constraint.Skip
        else: 
            return pyo.Constraint.Skip

    def eSOCP_QMinOut2_rule(model, rp, k, g): 
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            if model.pMinGenQ[g] <= 0:
                return  model.vGenQ[rp, k, g] / model.pMinGenQ[g] <= model.vCommit[rp, k, g]
            else:
                return pyo.Constraint.Skip
        else: 
            return pyo.Constraint.Skip

    # FACTS (not yet Implemented)
    # TODO: Add FACTS as a set, add FACTS parameters to nodes i
    def eSOCP_QMaxFACTS_rule(model, rp, k, i):
        if lego.cs.dPower_Parameters["pEnableSOCP"] == 99999:
            return (model.vGenQ[rp, k, i]) <= model.pMaxGenQ[i] * (model.pExisUnits[i] + model.vGenInvest[i])
        else:
            return pyo.Constraint.Skip

    def eSOCP_QMinFACTS_rule(model, rp, k, i):
        if lego.cs.dPower_Parameters["pEnableSOCP"] == 99999:
            return (model.vGenQ[rp, k, i]) >= model.pMaxGenQ[i] * (model.pExisUnits[i] + model.vGenInvest[i])
        else:
            return pyo.Constraint.Skip


    # Active and reactive power flow on existing lines SOCP
    # Active power flow on existing lines
    def eSOCP_ExiLinePij_rule(model, rp, k, i, j, c): #Fertig
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return ( model.vLineP[rp, k, i, j, c] == model.pSBase * (
                + model.pGline[i, j, c] * model.vSOCP_cii[rp, k, i] /  (model.pRatio[i, j, c]**2)
                - (1/model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                - (1/model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * - model.vSOCP_sij[rp, k, i, j])
            )
        else:
            return pyo.Constraint.Skip


    def eSOCP_ExiLinePji_rule(model, rp, k, i, j, c): #Fertig
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return( model.vLineP[rp, k, j, i, c] == model.pSBase * (
                + (model.pGline[i, j, c] * model.vSOCP_cii[rp, k, j])
                - (1/model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                - (1/model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j])
            )
        else:
            return pyo.Constraint.Skip


    # Reactive power flow on existing lines
    def eSOCP_ExiLineQij_rule(model, rp, k, i, j, c): 
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return ( model.vLineQ[rp, k, i, j, c] == model.pSBase * (
                - model.vSOCP_cii[rp, k, i] * (model.pBline[i, j, c] + model.pBcline[i, j, c]/2) / (model.pRatio[i, j, c]**2)
                - (1/model.pRatio[i, j, c]) * (model.pGline[i ,j ,c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * -model.vSOCP_sij[rp, k, i, j]
                + (1/model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
            )
        else:
            return pyo.Constraint.Skip

    def eSOCP_ExiLineQji_rule(model, rp, k, i, j, c): # Fertig
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return ( model.vLineQ[rp, k, j, i, c] == model.pSBase * (
                - model.vSOCP_cii[rp, k, j] * (model.pBline[i, j, c] + model.pBcline[i, j, c]/2)
                - (1/model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j]
                + (1/model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
            )
        else:
            return pyo.Constraint.Skip

    # Active Power flow limits for candidate lines c


    
    # Active and reactive power flow on candidte lines SOCP
    # Active power flow on candidate lines

    def eSOCP_CanLinePij1_rule(model, rp, k, i, j, c): #Fertig
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return( model.vLineP[rp, k, i, j, c] >= model.pSBase * (
                + model.pGline[i, j, c] * model.vSOCP_cii[rp, k, i] / (model.pRatio[i, j, c]**2)
                - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                - (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * -model.vSOCP_sij[rp, k, i, j])
                - model.pBigM_Flow* (1-model.vLineInvest[i, j, c])
                )
        else:
            return pyo.Constraint.Skip


    def eSOCP_CanLinePij2_rule(model, rp, k, i, j, c): # Fertig...
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return( model.vLineP[rp, k, i, j, c] <= model.pSBase * (
                + model.pGline[i, j, c] * model.vSOCP_cii[rp, k, i] / (model.pRatio[i, j, c]**2)
                - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                - (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * -model.vSOCP_sij[rp, k, i, j])
                + model.pBigM_Flow * (1-model.vLineInvest[i, j, c])
                )
        else:
            return pyo.Constraint.Skip


    def eSOCP_CanLinePji1_rule(model, rp, k, i, j, c): # Fertig
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return( model.vLineP[rp, k, j, i, c] >= model.pSBase * (
                + model.pGline[i, j, c] * model.vSOCP_cii[rp, k, j]
                - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                - (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j])
                - model.pBigM_Flow * (1-model.vLineInvest[i, j, c])
                )
        else:
            return pyo.Constraint.Skip

    def eSOCP_CanLinePji2_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineP[rp, k, j, i, c] <= model.pSBase * (
                    + model.pGline[i, j, c] * model.vSOCP_cii[rp, k, j]
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                    - (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j])
                    + model.pBigM_Flow * (1 - model.vLineInvest[i, j, c])
                    )
        else:
            return pyo.Constraint.Skip


    # Reactive power flow on candidate lines
    def eSOCP_CanLineQij1_rule(model, rp, k, i, j, c): # Fertig
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return ( model.vLineQ[rp, k, i, j, c] >= model.pSBase * (
                - model.vSOCP_cii[rp, k, i] * (model.pBline[i, j, c] + model.pBcline[i, j, c]/2) / (model.pRatio[i, j, c]**2)
                - (1/model.pRatio[i, j, c]) * (model.pGline[i ,j ,c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * -model.vSOCP_sij[rp, k, i, j]
                + (1/model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
                - model.pBigM_Flow * (1-model.vLineInvest[i, j, c])
            )
        else:
            return pyo.Constraint.Skip

    def eSOCP_CanLineQij2_rule(model, rp, k, i, j, c): # Fertig
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return ( model.vLineQ[rp, k, i, j, c] <= model.pSBase * (
                - model.vSOCP_cii[rp, k, i] * (model.pBline[i, j, c] + model.pBcline[i, j, c]/2) / (model.pRatio[i, j, c]**2)
                - (1/model.pRatio[i, j, c]) * (model.pGline[i ,j ,c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * -model.vSOCP_sij[rp, k, i, j]
                + (1/model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
                + model.pBigM_Flow * (1-model.vLineInvest[i, j, c])
            )
        else:
            return pyo.Constraint.Skip


    def eSOCP_CanLineQji1_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return ( model.vLineQ[rp, k, j, i, c] >= model.pSBase * (
                - model.vSOCP_cii[rp, k, j] * (model.pBline[i, j, c] + model.pBcline[i, j, c]/2)
                - (1/model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j]
                + (1/model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
                - model.pBigM_Flow * (1-model.vLineInvest[i, j, c])
            )
        else:
            return pyo.Constraint.Skip

    def eSOCP_CanLineQji2_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
                return ( model.vLineQ[rp, k, j, i, c] <= model.pSBase * (
                    - model.vSOCP_cii[rp, k, j] * (model.pBline[i, j, c] + model.pBcline[i, j, c]/2)
                    - (1/model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j]
                    + (1/model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
                    + model.pBigM_Flow * (1-model.vLineInvest[i, j, c])
                )
        else:
            return pyo.Constraint.Skip


    # Active and reactive power flow limits for candidates lines
    # Active power flow limits for candidate lines

    def eSOCP_LimCanLinePij1_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] >= - model.vLineInvest[i, j, c]
        else:
            return pyo.Constraint.Skip



    def eSOCP_LimCanLinePij2_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
                return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] <=  model.vLineInvest[i, j, c]
        else:
            return pyo.Constraint.Skip


    def eSOCP_LimCanLinePji1_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineP[rp, k, j, i, c] / model.pPmax[i, j, c] >= - model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip


    def eSOCP_LimCanLinePji2_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineP[rp, k, j, i, c] / model.pPmax[i, j, c] <=  model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip

    #Reactive power flow limits for candidate lines

    def eSOCP_LimCanLineQij1_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineQ[rp, k, i, j, c] / model.pQmax[i, j, c] >=  - model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip

    def eSOCP_LimCanLineQij2_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
                return model.vLineQ[rp, k, i, j, c] / model.pQmax[i, j, c] <=  model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip

    def eSOCP_LimCanLineQji1_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
                return model.vLineQ[rp, k, j, i, c] / model.pQmax[i, j, c] >= - model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip

    def eSOCP_LimCanLineQji2_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineQ[rp, k, j, i, c] / model.pQmax[i, j, c] <=  model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip


    # SCOP constraints for existing and candidate lines
    # SOCP constraints for existing lines

    def eSOCP_ExiLine_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            if any((i, j, c) in lego.model.le for c in lego.model.c):
                return (model.vSOCP_cij[rp, k, i, j]**2 + model.vSOCP_sij[rp, k, i, j] **2 <= model.vSOCP_cii[rp, k, i]* model.vSOCP_cii[rp, k, j])
        return pyo.Constraint.Skip


    # SOCP constraints for candidate lines
    # Does only apply if the line is not in le (existing lines set) for the first circuit and is a candidate line (lc), therefore is not a candidate line in a different circuit while one already exists(parallel lines)   
 
    def eSOCP_CanLine_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            c = lego.first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLine '.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in lego.model.lc:
                if (i, j, c) in lego.model.le:
                    return pyo.Constraint.Skip
                return (
                    model.vSOCP_cij[rp, k, i, j] ** 2 + model.vSOCP_sij[rp, k, i, j] ** 2 <= model.vSOCP_cii[rp, k, i] * model.vSOCP_cii[rp, k, j])
        return pyo.Constraint.Skip

    # Does only apply if the line is in le (existing lines set) for the first circuit and is a candidate line (lc), therefore is a candidate line in a different circuit while one already exists (parallel lines)   
 
    def eSOCP_CanLine_cij_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            c = lego.first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLine_cij'.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in lego.model.lc:
                if (i, j, c) not in lego.model.le:
                    return pyo.Constraint.Skip
                return(
                    model.vSOCP_cij[rp, k, i, j] <= model.pBigM_SOCP * model.vSOCP_IndicConnecNodes[i, j])
        return pyo.Constraint.Skip


    def eSOCP_CanLine_sij1_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            c = lego.first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line for eSOCP_CanLine_sij1 ({}, {})'.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in lego.model.lc:
                if (i, j, c) not in lego.model.le:
                    return pyo.Constraint.Skip
                return (
                    model.vSOCP_sij[rp, k, i, j] <= model.pBigM_SOCP * model.vSOCP_IndicConnecNodes[i, j])
        return pyo.Constraint.Skip

    
    def eSOCP_CanLine_sij2_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            c = lego.first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLine_sij2'.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in lego.model.lc:
                if (i, j, c) not in lego.model.le:
                    return pyo.Constraint.Skip
                return (
                    model.vSOCP_sij[rp, k, i, j] >= -model.pBigM_SOCP * model.vSOCP_IndicConnecNodes[i, j])

        return pyo.Constraint.Skip

    def eSOCP_IndicConnecNodes1_rule(model, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            c = lego.first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_IndicConnecNodes1'.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in lego.model.lc:
                if (i, j, c) not in lego.model.le:
                    return pyo.Constraint.Skip
                return ( sum(
                    model.vSOCP_IndicConnecNodes[i, j, c_]
                    for (i_, j_, c_) in model.lc
                    if i_ == i and j_ == j
                    ) == 1)
        return pyo.Constraint.Skip

    def eSOCP_IndicConnecNodes2_rule(model, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            c = lego.first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_IndicConnecNodes2'.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in lego.model.lc:
                if (i, j, c) in lego.model.le:
                    return pyo.Constraint.Skip
                return (
                    model.vSOCP_IndicConnecNodes[i, j] == model.vLineInvest[i, j, c])
        return pyo.Constraint.Skip



    # Limits for SOCP variables of candidates lines

    def eSOCP_CanLineCijUpLim_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            # Look up the first circuit 'c' for this line (i, j)
            c = lego.first_circuit_map_bidir.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLineCijUpLim'.format(i, j))
                return pyo.Constraint.Skip

            # Check forward direction
            if (i, j, c) in lego.model.lc:
                if (i, j, c) in lego.model.le:
                    return pyo.Constraint.Skip
                return (
                    model.vSOCP_cij[rp, k, i, j]
                        <= ((model.pBusMaxV[i]**2) + model.pBigM_SOCP * (1 - model.vSOCP_IndicConnecNodes[i, j]))
                )


    def eSOCP_CanLineCijLoLim_rule(model, rp, k, i, j):
            if lego.cs.dPower_Parameters["pEnableSOCP"]:
                if lego.cs.dPower_Parameters["pEnableSOCP"]:
                    # Look up the first circuit 'c' for this line (i, j)
                    c = lego.first_circuit_map_bidir.get((i, j))
                    if c is None:
                        # No first circuit info -> skip
                        print('No first circuit info for line ({}, {}) for eSOCP_CanLineCijLoLim'.format(i, j))
                        return pyo.Constraint.Skip
                    if (i, j, c) in lego.model.lc:
                        if (i, j, c) in lego.model.le:
                            return pyo.Constraint.Skip
                        expr = max(0.1, model.pBusMinV[i] ** 2)
                        return model.vSOCP_cij[rp, k, i, j] >= expr - model.pBigM_SOCP * (1 - model.vSOCP_IndicConnecNodes[i, j])

            return pyo.Constraint.Skip

    def eSOCP_CanLineSijUpLim_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            c = lego.first_circuit_map_bidir.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {} for eSOCP_CanLineSijUpLim)'.format(i, j))
                return pyo.Constraint.Skip
            if (i, j, c) in lego.model.lc:
                if (i, j, c) in lego.model.le:
                    return pyo.Constraint.Skip
                return (
                model.vSOCP_sij[rp, k, i, j] <= (model.pBusMaxV[i]**2) + model.pBigM_SOCP * (1 - model.vSOCP_IndicConnecNodes[i, j]))

        return pyo.Constraint.Skip

    def eSOCP_CanLineSijLoLim_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            c = lego.first_circuit_map_bidir.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLineSijLoLim'.format(i, j))
                return pyo.Constraint.Skip
            if (i, j, c) in lego.model.lc:
                if (i, j, c) in lego.model.le:
                    return pyo.Constraint.Skip
                return (
                    model.vSOCP_sij[rp, k, i, j] >= -(model.pBusMaxV[i]**2) - model.pBigM_SOCP * (1 - model.vSOCP_IndicConnecNodes[i, j]))
        return pyo.Constraint.Skip

    # Angle difference constraints for lines
    # Angle difference constraints for existing lines

    def eSOCP_ExiLineAngDif1_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            if any((i, j, c) in lego.model.le for c in lego.model.c):
                return model.vSOCP_sij[rp, k, i, j] <= model.vSOCP_cij[rp, k, i, j] * pyo.tan(model.pMaxAngleDiff)
        return pyo.Constraint.Skip

    def eSOCP_ExiLineAngDif2_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            if any((i, j, c) in lego.model.le for c in lego.model.c):
                return model.vSOCP_sij[rp, k, i, j] >= -model.vSOCP_cij[rp, k, i, j] * pyo.tan(model.pMaxAngleDiff)
        return pyo.Constraint.Skip

    # Angle difference constraints for candidate lines

    def eSOCP_CanLineAngDif1_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            c = lego.first_circuit_map_bidir.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLineAngDif1'.format(i, j))
                return pyo.Constraint.Skip
            if (i, j, c) in lego.model.lc:
                if(i, j, c) in lego.model.le:
                    return pyo.Constraint.Skip
                return model.vSOCP_sij[rp, k, i, j] <= model.vSOCP_cij[rp, k, i, j] * pyo.tan(model.pMaxAngleDiff) + model.pBigM_Flow * (1-model.vSOCP_IndicConnecNodes[i, j])
        return pyo.Constraint.Skip

    def eSOCP_CanLineAngDif2_rule(model, rp, k, i, j):
        if lego.cs.dPower_Parameters["pEnableSOCP"]:
            c = lego.first_circuit_map_bidir.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLineAngDif2'.format(i, j))
                return pyo.Constraint.Skip
            if (i, j, c) in lego.model.lc:
                if(i, j, c) in lego.model.le:
                    return pyo.Constraint.Skip
            return model.vSOCP_sij[rp, k, i, j] >= - model.vSOCP_cij[rp, k, i, j] * pyo.tan(model.pMaxAngleDiff) - model.pBigM_Flow * (1-model.vSOCP_IndicConnecNodes[i, j])
        return pyo.Constraint.Skip


    # Apparent power constraints for existing and candidate lines (Disabled in the LEGO model due to increased solving time) T
    # Constraints might need to be redefined for only the le set

    def eSOCP_ExiLineSLimit_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"] == 9999:
            if (i, j, c) in lego.model.le:
                return (model.vLineP[rp, k, i, j, c] ** 2
                        + model.vLineQ[rp, k, i, j, c] ** 2
                        <= pyo.sqrt(model.pPmax[i, j, c] ** 2
                                    + model.pQmax[i, j, c] ** 2))
            elif (j, i, c) in lego.model.le:
                # For the Reverse direction, the rhs should be 0 but this does not work with the solver. If this constraint is to be used, skip the reverse direction (will lead to error in the compare tool)
                rhs = 1
            else:
                return pyo.Constraint.Skip  # Not in le or le_reverse

            return (
                    model.vLineP[rp, k, i, j, c] ** 2
                    + model.vLineQ[rp, k, i, j, c] ** 2
                    <= rhs
            )
        return pyo.Constraint.Skip

    def eSOCP_CanLineSLimit_rule(model, rp, k, i, j, c):
        if lego.cs.dPower_Parameters["pEnableSOCP"] == 9999:
            if (i, j, c) in lego.model.lc:
                return (
                        model.vLineP[rp, k, i, j, c] ** 2
                        + model.vLineQ[rp, k, i, j, c] ** 2
                        <= pyo.sqrt(model.pPmax[i, j, c] ** 2
                                    + model.pQmax[i, j, c] ** 2) * model.vLineInvest[i, j, c])
            elif (j, i, c) in lego.model.lc:
                # For the Reverse direction, the rhs should be 0 but this does not work with the solver. If this constraint is to be used skip the reverse direction (will lead to error in the compare tool)
                rhs = 1
            else:
                return pyo.Constraint.Skip  # Not in le or le_reverse

            return (model.vLineP[rp, k, i, j, c] ** 2
                    + model.vLineQ[rp, k, i, j, c] ** 2
                    <= rhs)
        return pyo.Constraint.Skip

    if lego.cs.dPower_Parameters["pEnableSOCP"]:
        lego.model.eSOCP_QMaxOut = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc=" max reactive power output of generator unit", rule=eSOCP_QMaxOut_rule)
        lego.model.eSOCP_QMinOut1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc=" min postive reactive power output of generator unit", rule=eSOCP_QMinOut1_rule)
        lego.model.eSOCP_QMinOut2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc=" min negative reactive power output of generator unit ", rule=eSOCP_QMinOut2_rule)
        lego.model.eSOCP_BalanceQ_expr = pyo.Expression(lego.model.rp, lego.model.k, lego.model.i, rule=eSOCP_BalanceQ_rule)
        lego.model.eSOCP_BalanceQ = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.i, doc='Reactive power balance for each bus (SOCP)', rule=lambda model, rp, k, i: lego.model.eSOCP_BalanceQ_expr[rp, k, i] == 0)
        lego.model.eSOCP_ExiLinePij = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le, doc=" Active power flow existing lines from i to j (for SOCP)", rule=eSOCP_ExiLinePij_rule)
        lego.model.eSOCP_ExiLinePji = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le, doc="Active power flow existing lines from j to i (for SOCP)", rule=eSOCP_ExiLinePji_rule)
        lego.model.eSOCP_ExiLineQij = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le, rule=eSOCP_ExiLineQij_rule, doc="Reactive power flow existing lines from i to j (for SOCP)")
        lego.model.eSOCP_ExiLineQji = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le, rule=eSOCP_ExiLineQji_rule, doc="Reactive power flow existing lines from j to i (for SOCP)")
        lego.model.eSOCP_CanLinePij1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Active power flow candidate lines from i to j (for SOCP) Big-M 1", rule=eSOCP_CanLinePij1_rule)
        lego.model.eSOCP_CanLinePij2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Active power flow candidate lines from i to j (for SOCP) Big-M 2", rule=eSOCP_CanLinePij2_rule)
        lego.model.eSOCP_CanLinePji1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Active power flow candidate lines from j to i (for SOCP) Big-M 1", rule=eSOCP_CanLinePji1_rule)
        lego.model.eSOCP_CanLinePji2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Active power flow candidate lines from j to i (for SOCP) Big-M 1", rule=eSOCP_CanLinePji2_rule)
        lego.model.eSOCP_CanLineQij1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Reactive power flow candidate lines from i to j (for SOCP) Big-M 1", rule=eSOCP_CanLineQij1_rule)
        lego.model.eSOCP_CanLineQij2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Reactive power flow candidate lines from i to j (for SOCP) Big-M 2", rule=eSOCP_CanLineQij2_rule)
        lego.model.eSOCP_CanLineQji1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Reactive power flow candidate lines from j to i (for SOCP) Big-M 1", rule=eSOCP_CanLineQji1_rule)
        lego.model.eSOCP_CanLineQji2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Reactive power flow candidate lines from j to i (for SOCP) Big-M 2", rule=eSOCP_CanLineQji2_rule)
        lego.model.eSOCP_LimCanLinePij1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Active power candidate lower limit i to j candidate lines (for SOCP)", rule=eSOCP_LimCanLinePij1_rule)
        lego.model.eSOCP_LimCanLinePij2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Active power candidate upper limit i to j candidate lines (for SOCP)", rule=eSOCP_LimCanLinePij2_rule)
        lego.model.eSOCP_LimCanLinePji1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Active power candidate lower limit j to i candidate lines (for SOCP)", rule=eSOCP_LimCanLinePji1_rule)
        lego.model.eSOCP_LimCanLinePji2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Active power candidate upper limit j to i candidate lines (for SOCP)", rule=eSOCP_LimCanLinePji2_rule)
        lego.model.eSOCP_LimCanLineQij1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Reactive power candidate lower limit i to j candidate lines (for SOCP)", rule=eSOCP_LimCanLineQij1_rule)
        lego.model.eSOCP_LimCanLineQij2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Reactive power candidate upper limit i to j candidate lines (for SOCP)", rule=eSOCP_LimCanLineQij2_rule)
        lego.model.eSOCP_LimCanLineQji1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Reactive power candidate lower limit j to i candidate lines (for SOCP)", rule=eSOCP_LimCanLineQji1_rule)
        lego.model.eSOCP_LimCanLineQji2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, doc="Reactive power candidate upper limit j to i candidate lines (for SOCP)", rule=eSOCP_LimCanLineQji2_rule)
        lego.model.eSOCP_ExiLine = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le_no_c, doc="SCOP constraints for existing lines (for AC-OPF) original set", rule=eSOCP_ExiLine_rule)
        lego.model.eSOCP_CanLine = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc_no_c, doc="SOCP constraint for candidate lines (only if not in le)", rule=eSOCP_CanLine_rule)
        lego.model.eSOCP_CanLine_cij = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=eSOCP_CanLine_cij_rule)
        lego.model.eSOCP_CanLine_sij1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=eSOCP_CanLine_sij1_rule)
        lego.model.eSOCP_CanLine_sij2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=eSOCP_CanLine_sij2_rule)
        lego.model.eSOCP_IndicConnecNodes1 = pyo.Constraint(lego.model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=eSOCP_IndicConnecNodes1_rule)
        lego.model.eSOCP_IndicConnecNodes2 = pyo.Constraint(lego.model.lc_no_c, doc="SOCP constraint for candidate lines (only if not in le)", rule=eSOCP_IndicConnecNodes2_rule)
        lego.model.eSOCP_CanLineCijUpLim = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc_no_c, doc="Limits for SOCP variables lines (only if not in le)", rule=eSOCP_CanLineCijUpLim_rule)
        lego.model.eSOCP_CanLineCijLoLim = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc_no_c, doc="Limits for SOCP variables (only if not in le)", rule=eSOCP_CanLineCijLoLim_rule)
        lego.model.eSOCP_CanLineSijUpLim = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc_no_c, doc="Limits for SOCP variables (only if not in le)", rule=eSOCP_CanLineSijUpLim_rule)
        lego.model.eSOCP_CanLineSijLoLim = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc_no_c, doc="Limits for SOCP variables (only if not in le)", rule=eSOCP_CanLineSijLoLim_rule)
        lego.model.eSOCP_ExiLineAngDif1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le_no_c, doc="Angle difference upper bounds existing lines", rule=eSOCP_ExiLineAngDif1_rule)
        lego.model.eSOCP_ExiLineAngDif2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le_no_c, doc="Angle difference lower bounds existing lines", rule=eSOCP_ExiLineAngDif2_rule)
        lego.model.eSOCP_CanLineAngDif1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc_no_c, doc="Angle difference upper bounds candidate lines", rule=eSOCP_CanLineAngDif1_rule)
        lego.model.eSOCP_CanLineAngDif2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc_no_c, doc="Angle difference lowerf bounds candidate lines ", rule=eSOCP_CanLineAngDif2_rule)

        if lego.cs.dPower_Parameters["pEnableSOCP"] == 99999: # Not used in the GAMS model as well
            lego.model.eSOCP_ExiLineSLimit = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le_full, doc="Apparent power constraints for existing lines ", rule=eSOCP_ExiLineSLimit_rule)
            lego.model.eSOCP_CanLineSLimit = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc_full, doc="Apparent power constraints for existing lines ", rule=eSOCP_CanLineSLimit_rule)

        if lego.cs.dPower_Parameters["pEnableSOCP"] == 99999:  # FACTS are not implemented yet
            lego.model.eSOCP_QMinFACTS = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.facts, doc='min reactive power output of FACTS unit', rule=eSOCP_QMinFACTS_rule)
            lego.model.eSOCP_QMaxFACTS = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.facts, doc='max reactive power output of FACTS unit', rule=eSOCP_QMaxFACTS_rule)



    # lego.model.vSOCP_sij['rp01', 'k0001', 'Node_1', 'Node_4'].fix(0)
    # lego.model.vSOCP_sij['rp01', 'k0001', 'Node_1', 'Node_6'].fix(0)
    # lego.model.vSOCP_sij['rp01','k0001','Node_2','Node_3'].fix(-0.0502779)
    # lego.model.vSOCP_sij['rp01','k0001','Node_2','Node_6'].fix(-0.0645962)
    # lego.model.vSOCP_sij['rp01','k0001','Node_3','Node_4'].fix(0.00593706)
    # lego.model.vSOCP_sij['rp01', 'k0001', 'Node_3', 'Node_6'].fix(0.00768445)
    # lego.model.vSOCP_sij['rp01', 'k0001', 'Node_4', 'Node_5'].fix(0)
    # lego.model.vSOCP_sij['rp01','k0001','Node_4','Node_6'].fix(-0.0108786)
    # lego.model.vSOCP_sij['rp01','k0001','Node_6','Node_7'].fix(-0.00179561)
    #
    # lego.model.vSOCP_cii['rp01', 'k0001', 'Node_1'].fix(1)
    # lego.model.vSOCP_cii['rp01','k0001','Node_2'].fix(1.21)
    # lego.model.vSOCP_cii['rp01','k0001','Node_3'].fix(1.16522)
    # lego.model.vSOCP_cii['rp01', 'k0001', 'Node_4'].fix(1.21)
    # lego.model.vSOCP_cii['rp01','k0001','Node_5'].fix(1.04313)
    # lego.model.vSOCP_cii['rp01','k0001','Node_6'].fix(1.12218)
    # lego.model.vSOCP_cii['rp01', 'k0001', 'Node_7'].fix(1.12824)
    #
    # lego.model.vSOCP_cij['rp01', 'k0001', 'Node_1', 'Node_4'].fix(0.670921)
    # lego.model.vSOCP_cij['rp01', 'k0001', 'Node_1', 'Node_6'].fix(0.657579)
    # lego.model.vSOCP_cij['rp01','k0001','Node_2','Node_3'].fix(1.18633)
    # lego.model.vSOCP_cij['rp01','k0001','Node_2','Node_6'].fix(1.16347)
    # lego.model.vSOCP_cij['rp01','k0001','Node_3','Node_4'].fix(1.18738)
    # lego.model.vSOCP_cij['rp01', 'k0001', 'Node_3', 'Node_6'].fix(1.14347)
    # lego.model.vSOCP_cij['rp01', 'k0001', 'Node_4', 'Node_5'].fix(0.680759)
    # lego.model.vSOCP_cij['rp01','k0001','Node_4','Node_6'].fix(1.16521)
    # lego.model.vSOCP_cij['rp01','k0001','Node_6','Node_7'].fix(1.12521)

    # lego.model.vLineQ['rp01', 'k0001', 'Node_1', 'Node_4', 'c1'].fix(0)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_1', 'Node_6', 'c1'].fix(0)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_2', 'Node_3', 'c1'].fix(0.0522752)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_2', 'Node_6', 'c1'].fix(0.0857078)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_3', 'Node_2', 'c1'].fix(-0.0503305)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_3', 'Node_4', 'c1'].fix(-0.0466141)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_4', 'Node_1', 'c1'].fix(0)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_4', 'Node_3', 'c1'].fix(0.036806)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_4', 'Node_5', 'c1'].fix(0)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_4', 'Node_6', 'c1'].fix(0.0466796)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_5', 'Node_4', 'c1'].fix(0)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_6', 'Node_1', 'c1'].fix(0)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_6', 'Node_2', 'c1'].fix(-0.0812982)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_6', 'Node_3', 'c1'].fix(-0.0425447)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_6', 'Node_4', 'c1'].fix(-0.0612795)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_6', 'Node_7', 'c1'].fix(-0.0116414)
    # lego.model.vLineQ['rp01', 'k0001', 'Node_7', 'Node_6', 'c1'].fix(-0.000496099)


    # lego.model.eSOCP_QMaxOut.deactivate()
    #lego.model.eSOCP_QMinOut1.deactivate()
    # lego.model.eSOCP_QMinOut2.deactivate()
    # lego.model.eSOCP_BalanceQ.deactivate()
    # lego.model.eSOCP_ExiLinePij.deactivate()
    # lego.model.eSOCP_ExiLinePji.deactivate()
    # lego.model.eSOCP_ExiLineQij.deactivate()
    # lego.model.eSOCP_ExiLineQji.deactivate()
    # lego.model.eSOCP_CanLinePij1.deactivate()
    # lego.model.eSOCP_CanLinePij2.deactivate()
    # lego.model.eSOCP_CanLinePji1.deactivate()
    # lego.model.eSOCP_CanLinePji2.deactivate()
    # lego.model.eSOCP_CanLineQij1.deactivate()
    # lego.model.eSOCP_CanLineQij2.deactivate()
    # lego.model.eSOCP_CanLineQji1.deactivate()
    # lego.model.eSOCP_CanLineQji2.deactivate()
    # lego.model.eSOCP_LimCanLinePij1.deactivate()
    # lego.model.eSOCP_LimCanLinePij2.deactivate()
    # lego.model.eSOCP_LimCanLinePji1.deactivate()
    # lego.model.eSOCP_LimCanLinePji2.deactivate()
    # lego.model.eSOCP_LimCanLineQij1.deactivate()
    # lego.model.eSOCP_LimCanLineQij2.deactivate()
    # lego.model.eSOCP_LimCanLineQji1.deactivate()
    # lego.model.eSOCP_LimCanLineQji2.deactivate()
    # lego.model.eSOCP_ExiLine.deactivate()
    # lego.model.eSOCP_CanLine.deactivate()
    # lego.model.eSOCP_CanLine_cij.deactivate()
    # lego.model.eSOCP_CanLine_sij1.deactivate()
    # lego.model.eSOCP_CanLine_sij2.deactivate()
    # lego.model.eSOCP_IndicConnecNodes1.deactivate()
    # lego.model.eSOCP_IndicConnecNodes2.deactivate()
    # lego.model.eSOCP_CanLineCijUpLim.deactivate()
    # lego.model.eSOCP_CanLineCijLoLim.deactivate()
    # lego.model.eSOCP_CanLineSijUpLim.deactivate()
    # lego.model.eSOCP_CanLineSijLoLim.deactivate()
    # lego.model.eSOCP_ExiLineAngDif1.deactivate()
    # lego.model.eSOCP_ExiLineAngDif2.deactivate()
    # lego.model.eSOCP_CanLineAngDif1.deactivate()
    # lego.model.eSOCP_CanLineAngDif2.deactivate()
    # lego.model.eSOCP_ExiLineSLimit.deactivate()
    # lego.model.eSOCP_CanLineSLimit.deactivate()

    # Production constraints

    def eReMaxProd_rule(model, rp, k, r):
        capacity = lego.cs.dPower_VRESProfiles.loc[rp, k, r]['Capacity']
        capacity = capacity.values[0] if isinstance(capacity, pd.Series) else capacity
        return model.vGenP[rp, k, r] <= model.pMaxProd[r] * (model.vGenInvest[r] + model.pExisUnits[r]) * capacity

    if lego.cs.dPower_Parameters["pEnableVRES"]:
        lego.model.eReMaxProd = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.vresGenerators, rule=eReMaxProd_rule)

    def eThRampUp_rule(model, rp, k, g, transition_matrix):
        match lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"]:
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
                raise ValueError(f"Period edge handling ramping '{lego.cs.dPower_Parameters['pReprPeriodEdgeHandlingRamping']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    lego.model.eThRampUp_expr = pyo.Expression(lego.model.rp, lego.model.k, lego.model.thermalGenerators, rule=lambda m, rp, k, t: eThRampUp_rule(m, rp, k, t, lego.cs.rpTransitionMatrixRelativeFrom))
    lego.model.eThRampUp = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Ramp up for thermal generators (based on doi:10.1007/s10107-015-0919-9)', rule=lambda model, rp, k, t: lego.model.eThRampUp_expr[rp, k, t] <= 0 if not ((lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] == "notEnforced") and (model.k.first() == k)) else pyo.Constraint.Skip)

    def eThRampDw_rule(model, rp, k, g, transition_matrix):
        match lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"]:
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
                raise ValueError(f"Period edge handling ramping '{lego.cs.dPower_Parameters['pReprPeriodEdgeHandlingRamping']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    lego.model.eThRampDw_expr = pyo.Expression(lego.model.rp, lego.model.k, lego.model.thermalGenerators, rule=lambda m, rp, k, t: eThRampDw_rule(m, rp, k, t, lego.cs.rpTransitionMatrixRelativeFrom))
    lego.model.eThRampDw = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Ramp down for thermal generators (based on doi:10.1007/s10107-015-0919-9)', rule=lambda model, rp, k, t: lego.model.eThRampDw_expr[rp, k, t] >= 0 if not ((lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] == "notEnforced") and (model.k.first() == k)) else pyo.Constraint.Skip)

    # Thermal Generator production with unit commitment & ramping constraints
    lego.model.eUCTotOut = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Total production of thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, g: model.vGenP[rp, k, g] == model.pMinProd[g] * model.vCommit[rp, k, g] + model.vGenP1[rp, k, g])

    def eThMaxUC_rule(model, rp, k, t):
        return model.vCommit[rp, k, t] <= model.vGenInvest[t] + model.pExisUnits[t]

    lego.model.eThMaxUC = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Maximum number of active units for thermal generators', rule=eThMaxUC_rule)

    def eUCMaxOut1_rule(model, rp, k, t):
        return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vStartup[rp, k, t])

    lego.model.eUCMaxOut1_expr = pyo.Expression(lego.model.rp, lego.model.k, lego.model.thermalGenerators, rule=eUCMaxOut1_rule)
    lego.model.eUCMaxOut1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Maximum production for startup of thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, t: lego.model.eUCMaxOut1_expr[rp, k, t] <= 0)

    def eUCMaxOut2_rule(model, rp, k, t, transition_matrix):
        match lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
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
                raise ValueError(f"Period edge handling unit commitment '{lego.cs.dPower_Parameters['pReprPeriodEdgeHandlingUnitCommitment']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    lego.model.eUCMaxOut2_expr = pyo.Expression(lego.model.rp, lego.model.k, lego.model.thermalGenerators, rule=lambda m, rp, k, t: eUCMaxOut2_rule(m, rp, k, t, lego.cs.rpTransitionMatrixRelativeTo))
    lego.model.eUCMaxOut2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Maximum production for shutdown of thermal generators (from doi:10.1109/TPWRS.2013.2251373)',
                                           rule=lambda model, rp, k, t: lego.model.eUCMaxOut2_expr[rp, k, t] <= 0 if not ((lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "notEnforced") and (model.k.last() == k)) else pyo.Constraint.Skip)

    def eUCStrShut_rule(model, rp, k, t, transition_matrix):
        match lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
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
                raise ValueError(f"Period edge handling unit commitment '{lego.cs.dPower_Parameters['pReprPeriodEdgeHandlingUnitCommitment']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    lego.model.eUCStrShut = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Start-up and shut-down logic for thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, t: eUCStrShut_rule(model, rp, k, t, lego.cs.rpTransitionMatrixRelativeFrom))

    def eMinUpTime_rule(model, rp, k, t, transition_matrix):
        if model.pMinUpTime[t] == 0:
            raise ValueError("Minimum up time must be at least 1, got 0 instead")
        elif model.pMinUpTime[t] == 1:
            return pyo.Constraint.Skip
        else:
            match lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
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
                    raise ValueError(f"Invalid value for 'pReprPeriodEdgeHandlingUnitCommitment' in 'Global_Parameters.xlsx': {lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]} - please choose from 'notEnforced', 'cyclic' or 'markov'!")

    lego.model.eMinUpTime = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Minimum up time for thermal generators (from doi:10.1109/TPWRS.2013.2251373, adjusted to be cyclic)', rule=lambda m, rp, k, t: eMinUpTime_rule(m, rp, k, t, lego.cs.rpTransitionMatrixRelativeFrom))

    def eMinDownTime_rule(model, rp, k, t, transition_matrix):
        if model.pMinDownTime[t] == 0:
            raise ValueError("Minimum down time must be at least 1, got 0 instead")
        elif model.pMinDownTime[t] == 1:
            return pyo.Constraint.Skip
        else:
            match lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
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
                    raise ValueError(f"Invalid value for 'pReprPeriodEdgeHandlingUnitCommitment' in 'Global_Parameters.xlsx': {lego.cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]} - please choose from 'notEnforced', 'cyclic' or 'markov'!")

    lego.model.eMinDownTime = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Minimum down time for thermal generators (from doi:10.1109/TPWRS.2013.2251373, adjusted to be cyclic)', rule=lambda m, rp, k, t: eMinDownTime_rule(m, rp, k, t, lego.cs.rpTransitionMatrixRelativeFrom))

    # Objective function
    lego.model.objective = pyo.Objective(doc='Total production cost (Objective Function)', sense=pyo.minimize, expr=sum(sum(lego.model.vPNS[rp, k, :]) * lego.model.pWeight_rp[rp] * lego.model.pWeight_k[k] * lego.model.pENSCost for rp in lego.model.rp for k in lego.model.k) +  # Power not served
                                                                                                                    sum(sum(lego.model.vEPS[rp, k, :]) * lego.model.pWeight_rp[rp] * lego.model.pWeight_k[k] * lego.model.pENSCost * 2 for rp in lego.model.rp for k in lego.model.k) +  # Excess power served
                                                                                                                    sum(lego.model.vStartup[rp, k, t] * lego.model.pStartupCost[t] * lego.model.pWeight_rp[rp] * lego.model.pWeight_k[k] for rp in lego.model.rp for k in lego.model.k for t in lego.model.thermalGenerators) +  # Startup cost of thermal generators
                                                                                                                    sum(lego.model.vCommit[rp, k, t] * lego.model.pInterVarCost[t] * lego.model.pWeight_rp[rp] * lego.model.pWeight_k[k] for rp in lego.model.rp for k in lego.model.k for t in lego.model.thermalGenerators) +  # Commit cost of thermal generators
                                                                                                                    sum(lego.model.vGenP[rp, k, g] * lego.model.pOMVarCost[g] * lego.model.pWeight_rp[rp] * lego.model.pWeight_k[k] for rp in lego.model.rp for k in lego.model.k for g in lego.model.g) +  # Production cost of generators
                                                                                                                    sum(lego.model.pFixedCost[i, j, c] * lego.model.vLineInvest[i, j, c] for i, j, c in lego.model.lc) +  # Investment cost of transmission lines
                                                                                                                    sum(lego.model.pInvestCost[g] * lego.model.vGenInvest[g] for g in lego.model.g))  # Investment cost of generators
