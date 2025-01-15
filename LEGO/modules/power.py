import numpy as np
import pandas as pd
import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(lego: LEGO):
    # Sets
    lego.model.i = pyo.Set(doc='Buses', initialize=lego.cs.dPower_BusInfo.index.tolist())

    lego.model.la = pyo.Set(doc='All lines', initialize=lego.cs.dPower_Network[lego.cs.dPower_Network["InService"] == 1].index.tolist())
    lego.model.le = pyo.Set(doc='Existing lines', initialize=lego.cs.dPower_Network[(lego.cs.dPower_Network["InService"] == 1) & (lego.cs.dPower_Network["FixedCost"] == 0)].index.tolist(), within=lego.model.la)
    lego.model.lc = pyo.Set(doc='Candidate lines', initialize=lego.cs.dPower_Network[(lego.cs.dPower_Network["InService"] == 1) & (lego.cs.dPower_Network["FixedCost"] > 0)].index.tolist(), within=lego.model.la)

    lego.model.c = pyo.Set(doc='Circuits', initialize=lego.cs.dPower_Network["Circuit ID"].unique())
    lego.model.thermalGenerators = pyo.Set(doc='Thermal Generators', initialize=lego.cs.dPower_ThermalGen.index.tolist())
    lego.model.rorGenerators = pyo.Set(doc='Run-of-river generators', initialize=lego.cs.dPower_RoR.index.tolist())
    lego.model.vresGenerators = pyo.Set(doc='Variable renewable energy sources', initialize=lego.cs.dPower_VRES.index.tolist())
    lego.model.g = pyo.Set(doc='Generators', initialize=lego.model.thermalGenerators | lego.model.rorGenerators | lego.model.vresGenerators)

    lego.model.gi = pyo.Set(doc='Generator g connected to bus i', initialize=lego.cs.dPower_ThermalGen.reset_index().set_index(['g', 'i']).index, within=lego.model.g * lego.model.i)
    lego.addToSet("gi", lego.cs.dPower_RoR.reset_index().set_index(['g', 'i']).index)
    lego.addToSet("gi", lego.cs.dPower_VRES.reset_index().set_index(['g', 'i']).index)

    lego.model.p = pyo.Set(doc='Periods', initialize=lego.cs.dPower_Hindex.index.get_level_values('p').unique().tolist())
    lego.model.rp = pyo.Set(doc='Representative periods', initialize=lego.cs.dPower_Demand.index.get_level_values('rp').unique().tolist())
    lego.model.k = pyo.Set(doc='Timestep within representative period', initialize=lego.cs.dPower_Demand.index.get_level_values('k').unique().tolist())
    lego.model.hindex = lego.cs.dPower_Hindex.index

    # Parameters
    lego.model.pDemandP = pyo.Param(lego.model.rp, lego.model.k, lego.model.i, initialize=lego.cs.dPower_Demand['Demand'], doc='Demand at bus i in representative period rp and timestep k')
    lego.model.pMovWindow = lego.cs.dGlobal_Parameters['pMovWindow']

    # Helper for FuelCost that has dPower_ThermalGen['FuelCost'] for ThermalGen, and 0 for all gs in ror and vres
    hFuelCost = pd.concat([lego.cs.dPower_ThermalGen['FuelCost'].copy(), pd.Series(0, index=lego.model.rorGenerators), pd.Series(0, index=lego.model.vresGenerators)])
    lego.model.pProductionCost = pyo.Param(lego.model.g, initialize=hFuelCost, doc='Production cost of generator g')

    lego.model.pEnabInv = pyo.Param(lego.model.g, initialize=lego.cs.dPower_ThermalGen['EnableInvest'], doc='Enable investment in thermal generator g')
    lego.addToParameter("pEnabInv", lego.cs.dPower_RoR['EnableInvest'])
    lego.addToParameter("pEnabInv", lego.cs.dPower_VRES['EnableInvest'])

    lego.model.pMaxInvest = pyo.Param(lego.model.g, initialize=lego.cs.dPower_ThermalGen['MaxInvest'], doc='Maximum investment in thermal generator g')
    lego.addToParameter("pMaxInvest", lego.cs.dPower_RoR['MaxInvest'])
    lego.addToParameter("pMaxInvest", lego.cs.dPower_VRES['MaxInvest'])

    lego.model.pInvestCost = pyo.Param(lego.model.g, initialize=lego.cs.dPower_ThermalGen['InvestCostEUR'], doc='Investment cost for thermal generator g')
    lego.addToParameter("pInvestCost", lego.cs.dPower_RoR['InvestCostEUR'])
    lego.addToParameter("pInvestCost", lego.cs.dPower_VRES['InvestCostEUR'])

    lego.model.pInterVarCost = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['pInterVarCostEUR'], doc='Inter-variable cost of thermal generator g')
    lego.model.pSlopeVarCost = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['pSlopeVarCostEUR'], doc='Slope of variable cost of thermal generator g')
    lego.model.pStartupCost = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['pStartupCostEUR'], doc='Startup cost of thermal generator g')
    lego.model.pMinUpTime = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['MinUpTime'], doc='Minimum up time of thermal generator g')
    lego.model.pMinDownTime = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['MinDownTime'], doc='Minimum down time of thermal generator g')
    lego.model.pRampUp = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['RampUp'], doc='Ramp up of thermal generator g')
    lego.model.pRampDw = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['RampDw'], doc='Ramp down of thermal generator g')

    lego.model.pXline = pyo.Param(lego.model.la, lego.model.c, initialize=lego.cs.dPower_Network.reset_index().set_index(["i", "j", "Circuit ID"]).query("InService == 1")['X'], doc='Reactance of line la')
    lego.model.pAngle = pyo.Param(lego.model.la, lego.model.c, initialize=lego.cs.dPower_Network.reset_index().set_index(["i", "j", "Circuit ID"]).query("InService == 1")['TapAngle'] * np.pi / 180, doc='Transformer angle shift')
    lego.model.pRatio = pyo.Param(lego.model.la, lego.model.c, initialize=lego.cs.dPower_Network.reset_index().set_index(["i", "j", "Circuit ID"]).query("InService == 1")['TapRatio'], doc='Transformer ratio')
    lego.model.pPmax = pyo.Param(lego.model.la, lego.model.c, initialize=lego.cs.dPower_Network.reset_index().set_index(["i", "j", "Circuit ID"]).query("InService == 1")['Pmax'], doc='Maximum power flow on line la')
    lego.model.pFixedCost = pyo.Param(lego.model.la, lego.model.c, initialize=lego.cs.dPower_Network.reset_index().set_index(["i", "j", "Circuit ID"]).query("InService == 1")['FixedCostEUR'], doc='Fixed cost when investing in line la')
    lego.model.pSBase = pyo.Param(initialize=lego.cs.dPower_Parameters['pSBase'], doc='Base power')
    lego.model.pBigM_Flow = pyo.Param(initialize=1e3, doc="Big M for power flow")
    lego.model.pENSCost = pyo.Param(initialize=lego.cs.dPower_Parameters['pENSCost'], doc='Cost used for Power Not Served (PNS) and Excess Power Served (EPS)')

    lego.model.pWeight_rp = pyo.Param(lego.model.rp, initialize=lego.cs.dPower_WeightsRP, doc='Weight of representative period rp')
    lego.model.pWeight_k = pyo.Param(lego.model.k, initialize=lego.cs.dPower_WeightsK, doc='Weight of time step k')

    lego.addToParameter("pMaxProd", lego.cs.dPower_ThermalGen['MaxProd'], indices=[lego.model.g], doc='Maximum production of generator g')
    lego.addToParameter("pMaxProd", lego.cs.dPower_RoR['MaxProd'])
    lego.addToParameter("pMaxProd", lego.cs.dPower_VRES['MaxProd'])

    lego.addToParameter("pMinProd", lego.cs.dPower_ThermalGen['MinProd'], indices=[lego.model.g], doc='Minimum production of generator g')
    lego.addToParameter("pMinProd", lego.cs.dPower_RoR['MinProd'])
    lego.addToParameter("pMinProd", lego.cs.dPower_VRES['MinProd'])

    lego.addToParameter("pExisUnits", lego.cs.dPower_ThermalGen['ExisUnits'], indices=[lego.model.g], doc='Existing units of generator g')
    lego.addToParameter("pExisUnits", lego.cs.dPower_RoR['ExisUnits'])
    lego.addToParameter("pExisUnits", lego.cs.dPower_VRES['ExisUnits'])

    # Variables
    lego.model.vTheta = pyo.Var(lego.model.rp, lego.model.k, lego.model.i, doc='Angle of bus i', bounds=(-lego.cs.dPower_Parameters["pMaxAngleDCOPF"], lego.cs.dPower_Parameters["pMaxAngleDCOPF"]))  # TODO: Discuss impact on runtime etc.(based on discussion with Prof. Renner)
    lego.model.vAngle = pyo.Var(lego.model.rp, lego.model.k, lego.model.la, lego.model.c, doc='Angle phase shifting transformer')
    for i, j in lego.model.la:
        for c in lego.model.c:
            if lego.model.pAngle[i, j, c] == 0:
                lego.model.vAngle[:, :, i, j, c].fix(0)
            else:
                lego.model.vAngle[:, :, i, j, c].setub(lego.model.pAngle[i, j, c])
                lego.model.vAngle[:, :, i, j, c].setlb(-lego.model.pAngle[i, j, c])

    lego.model.vLineInvest = pyo.Var(lego.model.la, lego.model.c, doc='Transmission line investment', domain=pyo.Binary)
    for i, j in lego.model.le:
        lego.model.vLineInvest[i, j, :].fix(0)  # Set existing lines to not invest

    lego.model.vGenInvest = pyo.Var(lego.model.g, doc="Integer generation investment", bounds=lambda model, g: (0, model.pMaxInvest[g] * model.pEnabInv[g]))

    # For each DC-OPF "island", set node with highest demand as slack node
    dDCOPFIslands = pd.DataFrame(index=lego.cs.dPower_BusInfo.index, columns=[lego.cs.dPower_BusInfo.index], data=False)

    for index, entry in lego.cs.dPower_Network.iterrows():
        if lego.cs.dPower_Network.loc[(index[0], index[1])]["Technical Representation"] == "DC-OPF":
            dDCOPFIslands.loc[index[0], index[1]] = True
            dDCOPFIslands.loc[index[1], index[0]] = True

    completed_buses = set()  # Set of buses that have been looked at already
    i = 0
    for index, entry in dDCOPFIslands.iterrows():
        if index in completed_buses or entry[entry == True].empty:  # Skip if bus has already been looked at or has no connections
            continue

        connected_buses = lego.cs.get_connected_buses(dDCOPFIslands, str(index))

        for bus in connected_buses:
            completed_buses.add(bus)

        # Set slack node
        slack_node = lego.cs.dPower_Demand.loc[:, :, connected_buses].groupby('i').sum().idxmax().values[0]
        slack_node = lego.cs.dPower_Parameters["is"]  # TODO: Switch this again to be calculated (fixed to 'is' for compatibility)
        if i == 0: print("Setting slack nodes for DC-OPF zones:")
        print(f"DC-OPF Zone {i:>2} - Slack node: {slack_node}")
        i += 1
        lego.model.vTheta[:, :, slack_node].fix(0)

    lego.model.vPNS = pyo.Var(lego.model.rp, lego.model.k, lego.model.i, doc='Slack variable power not served', bounds=lambda model, rp, k, i: (0, model.pDemandP[rp, k, i]))
    lego.model.vEPS = pyo.Var(lego.model.rp, lego.model.k, lego.model.i, doc='Slack variable excess power served', bounds=(0, None))

    lego.model.vCommit = pyo.Var(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Unit commitment of generator g', domain=pyo.Binary)
    lego.model.vStartup = pyo.Var(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Start-up of thermal generator g', domain=pyo.Binary)
    lego.model.vShutdown = pyo.Var(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Shut-down of thermal generator g', domain=pyo.Binary)

    lego.model.vGenP = pyo.Var(lego.model.rp, lego.model.k, lego.model.g, doc='Power output of generator g', bounds=lambda model, rp, k, g: (0, lego.model.pMaxProd[g] * (lego.model.pExisUnits[g] + lego.model.pMaxInvest[g] * lego.model.pEnabInv[g])))
    lego.model.vGenP1 = pyo.Var(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Power output of generator g above minimum production', bounds=lambda model, rp, k, g: (0, (lego.model.pMaxProd[g] - lego.model.pMinProd[g]) * (lego.model.pExisUnits[g] + lego.model.pMaxInvest[g] * lego.model.pEnabInv[g])))

    for g in lego.model.rorGenerators:
        for rp in lego.model.rp:
            for k in lego.model.k:
                lego.model.vGenP[rp, k, g].setub(min(lego.model.pMaxProd[g], lego.cs.dPower_Inflows.loc[rp, g, k]['Inflow']))  # TODO: Check and adapt for storage

    for g in lego.model.vresGenerators:
        for rp in lego.model.rp:
            for k in lego.model.k:
                lego.model.vGenP[rp, k, g].setub((lego.model.pMaxProd[g] * (lego.model.pExisUnits[g] + (lego.model.pMaxInvest[g] * lego.model.pEnabInv[g])) * lego.cs.dPower_VRESProfiles.loc[rp, lego.cs.dPower_VRES.loc[g, 'i'], k, lego.cs.dPower_VRES.loc[g, 'tec']]['Capacity']))

    lego.model.vLineP = pyo.Var(lego.model.rp, lego.model.k, lego.model.la, lego.model.c, doc='Power flow from bus i to j', bounds=(None, None))
    for (i, j) in lego.model.la:
        match lego.cs.dPower_Network.loc[i, j]["Technical Representation"]:
            case "DC-OPF" | "TP":
                for c in lego.model.c:
                    lego.model.vLineP[:, :, (i, j), c].setlb(-lego.model.pPmax[i, j, c])
                    lego.model.vLineP[:, :, (i, j), c].setub(lego.model.pPmax[i, j, c])
            case "SN":
                assert False  # "SN" line found, although all "Single Node" buses should be merged
            case _:
                raise ValueError(f"Technical representation '{lego.cs.dPower_Network.loc[i, j]["Technical Representation"]}' "
                                 f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):
    # Power balance for nodes
    def eDC_BalanceP_rule(model, rp, k, i):
        return (sum(model.vGenP[rp, k, g] for g in model.g if (g, i) in model.gi) -  # Production of generators at bus i
                sum(model.vLineP[rp, k, e, c] for c in model.c for e in model.la if (e[0] == i)) +  # Power flow from bus i to bus j
                sum(model.vLineP[rp, k, e, c] for c in model.c for e in model.la if (e[1] == i)) -  # Power flow from bus j to bus i
                model.pDemandP[rp, k, i] +  # Demand at bus i
                model.vPNS[rp, k, i] -  # Slack variable for demand not served
                model.vEPS[rp, k, i])  # Slack variable for overproduction

    # Note: eDC_BalanceP_expr is defined as expression to enable later adding coefficients to the constraint (e.g., for import/export)
    lego.model.eDC_BalanceP_expr = pyo.Expression(lego.model.rp, lego.model.k, lego.model.i, rule=eDC_BalanceP_rule)
    lego.model.eDC_BalanceP = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.i, doc='Power balance constraint for each bus', rule=lambda model, rp, k, i: lego.model.eDC_BalanceP_expr[rp, k, i] == 0)

    def eDC_ExiLinePij_rule(model, rp, k, i, j, c):
        match lego.cs.dPower_Network.loc[i, j]["Technical Representation"]:
            case "DC-OPF":
                return model.vLineP[rp, k, i, j, c] == (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) * model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c])
            case "TP" | "SN":
                return pyo.Constraint.Skip
            case _:
                raise ValueError(f"Technical representation '{lego.cs.dPower_Network.loc[i, j]["Technical Representation"]}' "
                                 f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

    lego.model.eDC_ExiLinePij = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.le, lego.model.c, doc="Power flow existing lines (for DC-OPF)", rule=eDC_ExiLinePij_rule)

    def eDC_CanLinePij1_rule(model, rp, k, i, j, c):
        return model.vLineP[rp, k, i, j, c] / (model.pBigM_Flow * model.pPmax[i, j, c]) >= (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) * model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c]) / (model.pBigM_Flow * model.pPmax[i, j, c]) - 1 + model.vLineInvest[i, j, c]

    lego.model.eDC_CanLinePij1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, lego.model.c, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij1_rule)

    def eDC_CanLinePij2_rule(model, rp, k, i, j, c):
        return model.vLineP[rp, k, i, j, c] / (model.pBigM_Flow * model.pPmax[i, j, c]) <= (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) * model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c]) / (model.pBigM_Flow * model.pPmax[i, j, c]) + 1 - model.vLineInvest[i, j, c]

    lego.model.eDC_CanLinePij2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, lego.model.c, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij2_rule)

    def eDC_LimCanLine1_rule(model, rp, k, i, j, c):
        return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] + model.vLineInvest[i, j, c] >= 0

    lego.model.eDC_LimCanLine1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, lego.model.c, doc="Power flow limit reverse direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine1_rule)

    def eDC_LimCanLine2_rule(model, rp, k, i, j, c):
        return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] - model.vLineInvest[i, j, c] <= 0

    lego.model.eDC_LimCanLine2 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.lc, lego.model.c, doc="Power flow limit standard direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine2_rule)

    def eReMaxProd_rule(model, rp, k, r):
        capacity = lego.cs.dPower_VRESProfiles.loc[rp, lego.cs.dPower_VRES.loc[r, 'i'], k, lego.cs.dPower_VRES.loc[r, 'tec']]['Capacity']
        capacity = capacity.values[0] if isinstance(capacity, pd.Series) else capacity
        return model.vGenP[rp, k, r] <= model.pMaxProd[r] * (model.vGenInvest[r] + model.pExisUnits[r]) * capacity

    lego.model.eReMaxProd = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.vresGenerators, rule=eReMaxProd_rule)

    def eThRampUp_rule(model, rp, k, g):
        return model.vGenP1[rp, k, g] - model.vGenP1[rp, lego.model.k.prevw(k), g] - model.vCommit[rp, k, g] * model.pRampUp[g]

    lego.model.eThRampUp_expr = pyo.Expression(lego.model.rp, lego.model.k, lego.model.thermalGenerators, rule=eThRampUp_rule)
    lego.model.eThRampUp = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Ramp up for thermal generators', rule=lambda model, rp, k, t: lego.model.eThRampUp_expr[rp, k, t] <= 0)

    def eThRampDw_rule(model, rp, k, g):
        return model.vGenP1[rp, k, g] - model.vGenP1[rp, lego.model.k.prevw(k), g] + model.vCommit[rp, lego.model.k.prevw(k), g] * model.pRampDw[g]

    lego.model.eThRampDw_expr = pyo.Expression(lego.model.rp, lego.model.k, lego.model.thermalGenerators, rule=eThRampDw_rule)
    lego.model.eThRampDw = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Ramp down for thermal generators', rule=lambda model, rp, k, t: lego.model.eThRampDw_expr[rp, k, t] >= 0)

    # Todo: Required when we have circuits:
    #  eTranInves (i,j,c) $[lc(i,j,c) and pEnableTransNet and ord(c)>1]..
    #     vLineInvest(i,j,c) =l= vLineInvest(i,j,c-1) + sum[le(i,j,c-1),1];

    # Thermal Generator production with unit commitment & ramping constraints
    lego.model.eUCTotOut = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Total production of thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, g: model.vGenP[rp, k, g] == model.pMinProd[g] * model.vCommit[rp, k, g] + model.vGenP1[rp, k, g])

    def eThMaxUC_rule(model, rp, k, t):
        return model.vCommit[rp, k, t] <= model.vGenInvest[t] + model.pExisUnits[t]

    lego.model.eThMaxUC = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Maximum number of active units for thermal generators', rule=eThMaxUC_rule)

    def eUCMaxOut1_rule(model, rp, k, t):
        return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vStartup[rp, k, t])

    lego.model.eUCMaxOut1_expr = pyo.Expression(lego.model.rp, lego.model.k, lego.model.thermalGenerators, rule=eUCMaxOut1_rule)
    lego.model.eUCMaxOut1 = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Maximum production for startup of thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, t: lego.model.eUCMaxOut1_expr[rp, k, t] <= 0)

    lego.model.eUCStrShut = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Start-up and shut-down logic for thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, t: model.vCommit[rp, k, t] - model.vCommit[rp, model.k.prevw(k), t] == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t])

    def eMinUpTime_rule(model, rp, k, t):
        if model.pMinUpTime[t] == 0:
            raise ValueError("Minimum up time must be at least 1, got 0 instead")
        else:
            return sum(model.vStartup[rp, k2, t] for k2 in LEGOUtilities.set_range_cyclic(model.k, model.k.ord(k) - model.pMinUpTime[t] + 1, model.k.ord(k))) <= model.vCommit[rp, k, t]

    lego.model.eMinUpTime = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Minimum up time for thermal generators (from doi:10.1109/TPWRS.2013.2251373, adjusted to be cyclic)', rule=eMinUpTime_rule)

    def eMinDownTime_rule(model, rp, k, t):
        if model.pMinDownTime[t] == 0:
            raise ValueError("Minimum down time must be at least 1, got 0 instead")
        else:
            return sum(model.vShutdown[rp, k2, t] for k2 in LEGOUtilities.set_range_cyclic(model.k, model.k.ord(k) - model.pMinDownTime[t] + 1, model.k.ord(k))) <= 1 - model.vCommit[rp, k, t]

    lego.model.eMinDownTime = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.thermalGenerators, doc='Minimum down time for thermal generators (from doi:10.1109/TPWRS.2013.2251373, adjusted to be cyclic)', rule=eMinDownTime_rule)

    # Objective function
    lego.model.objective = pyo.Objective(doc='Total production cost (Objective Function)', sense=pyo.minimize, expr=sum(lego.model.pInterVarCost[g] * sum(lego.model.vCommit[g, :, :]) +  # Fixed cost of thermal generators
                                                                                                                        lego.model.pStartupCost[g] * sum(lego.model.vStartup[g, :, :]) +  # Startup cost of thermal generators
                                                                                                                        lego.model.pSlopeVarCost[g] * sum(lego.model.vGenP[g, :, :]) for g in lego.model.thermalGenerators) +  # Production cost of thermal generators
                                                                                                                    sum(lego.model.pProductionCost[g] * sum(lego.model.vGenP[g, :, :]) for g in lego.model.vresGenerators) +
                                                                                                                    sum(lego.model.pProductionCost[g] * sum(lego.model.vGenP[g, :, :]) for g in lego.model.rorGenerators) +
                                                                                                                    sum(lego.model.pOMVarCost[g] * sum(lego.model.vGenP[g, :, :]) for g in lego.model.storageUnits) +
                                                                                                                    sum(sum(lego.model.vPNS[rp, k, :]) * lego.model.pWeight_rp[rp] * lego.model.pWeight_k[k] * lego.model.pENSCost for rp in lego.model.rp for k in lego.model.k) +  # Power not served
                                                                                                                    sum(sum(lego.model.vEPS[rp, k, :]) * lego.model.pWeight_rp[rp] * lego.model.pWeight_k[k] * lego.model.pENSCost * 2 for rp in lego.model.rp for k in lego.model.k) +  # Excess power served
                                                                                                                    sum(lego.model.pFixedCost[i, j, c] * lego.model.vLineInvest[i, j, c] for i, j in lego.model.lc for c in lego.model.c) +  # Investment cost of transmission lines
                                                                                                                    sum(lego.model.pInvestCost[g] * lego.model.vGenInvest[g] for g in lego.model.g))  # Investment cost of thermal generators
