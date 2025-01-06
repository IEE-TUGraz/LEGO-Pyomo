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

    lego.model.p = pyo.Set(doc='Periods', initialize=lego.cs.dPower_Hindex.index.get_level_values('p').unique().tolist())
    lego.model.rp = pyo.Set(doc='Representative periods', initialize=lego.cs.dPower_Demand.index.get_level_values('rp').unique().tolist())
    lego.model.k = pyo.Set(doc='Timestep within representative period', initialize=lego.cs.dPower_Demand.index.get_level_values('k').unique().tolist())
    lego.model.hindex = lego.cs.dPower_Hindex.index

    # Parameters
    lego.model.pDemandP = pyo.Param(lego.model.rp, lego.model.i, lego.model.k, initialize=lego.cs.dPower_Demand['Demand'], doc='Demand at bus i in representative period rp and timestep k')
    lego.model.pMovWindow = lego.cs.dGlobal_Parameters['pMovWindow']

    # Helper for FuelCost that has dPower_ThermalGen['FuelCost'] for ThermalGen, and 0 for all gs in ror and vres
    hFuelCost = pd.concat([lego.cs.dPower_ThermalGen['FuelCost'].copy(), pd.Series(0, index=lego.model.rorGenerators), pd.Series(0, index=lego.model.vresGenerators)])
    lego.model.pProductionCost = pyo.Param(lego.model.g, initialize=hFuelCost, doc='Production cost of generator g')
    lego.model.pInterVarCost = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['pInterVarCostEUR'], doc='Inter-variable cost of thermal generator g')
    lego.model.pSlopeVarCost = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['pSlopeVarCostEUR'], doc='Slope of variable cost of thermal generator g')
    lego.model.pStartupCost = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['pStartupCostEUR'], doc='Startup cost of thermal generator g')
    lego.model.pMinUpTime = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['MinUpTime'], doc='Minimum up time of thermal generator g')
    lego.model.pMinDownTime = pyo.Param(lego.model.thermalGenerators, initialize=lego.cs.dPower_ThermalGen['MinDownTime'], doc='Minimum down time of thermal generator g')

    lego.model.pXline = pyo.Param(lego.model.la, lego.model.c, initialize=lego.cs.dPower_Network.reset_index().set_index(["i", "j", "Circuit ID"]).query("InService == 1")['X'], doc='Reactance of line la')
    lego.model.pAngle = pyo.Param(lego.model.la, lego.model.c, initialize=lego.cs.dPower_Network.reset_index().set_index(["i", "j", "Circuit ID"]).query("InService == 1")['TapAngle'] * np.pi / 180, doc='Transformer angle shift')
    lego.model.pRatio = pyo.Param(lego.model.la, lego.model.c, initialize=lego.cs.dPower_Network.reset_index().set_index(["i", "j", "Circuit ID"]).query("InService == 1")['TapRatio'], doc='Transformer ratio')
    lego.model.pPmax = pyo.Param(lego.model.la, lego.model.c, initialize=lego.cs.dPower_Network.reset_index().set_index(["i", "j", "Circuit ID"]).query("InService == 1")['Pmax'], doc='Maximum power flow on line la')
    lego.model.pFixedCost = pyo.Param(lego.model.la, lego.model.c, initialize=lego.cs.dPower_Network.reset_index().set_index(["i", "j", "Circuit ID"]).query("InService == 1")['FixedCost'], doc='Fixed cost when investing in line la')
    lego.model.pSBase = pyo.Param(initialize=lego.cs.dPower_Parameters['pSBase'], doc='Base power')
    lego.model.pBigM_Flow = pyo.Param(initialize=1e3, doc="Big M for power flow")
    #                                                                                                                                counter TODO: Discuss with Sonja & Diego
    lego.model.pSlackPrice = pyo.Param(lego.model.i, initialize=pd.DataFrame([(i, max(lego.model.pProductionCost.values()) * 100 + (0 * max(lego.model.pProductionCost.values()) / 10)) for counter, i in enumerate(lego.model.i)], columns=["i", "values"]).set_index("i"), doc='Price of slack variable')

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
        slack_node = lego.cs.dPower_Demand.loc[:, connected_buses, :].groupby('i').sum().idxmax().values[0]
        slack_node = lego.cs.dPower_Parameters["is"]  # TODO: Switch this again to be calculated (fixed to 'is' for compatibility)
        if i == 0: print("Setting slack nodes for DC-OPF zones:")
        print(f"DC-OPF Zone {i:>2} - Slack node: {slack_node}")
        i += 1
        lego.model.vTheta[:, :, slack_node].fix(0)

    lego.model.vPNS = pyo.Var(lego.model.rp, lego.model.k, lego.model.i, doc='Slack variable power not served', bounds=(0, None))
    lego.model.vEPS = pyo.Var(lego.model.rp, lego.model.k, lego.model.i, doc='Slack variable excess power served', bounds=(0, None))

    lego.model.bUC = pyo.Var(lego.model.thermalGenerators, lego.model.rp, lego.model.k, doc='Unit commitment of generator g', domain=pyo.Binary)
    lego.model.bStartup = pyo.Var(lego.model.thermalGenerators, lego.model.rp, lego.model.k, doc='Start-up of thermal generator g', domain=pyo.Binary)
    lego.model.bShutdown = pyo.Var(lego.model.thermalGenerators, lego.model.rp, lego.model.k, doc='Shut-down of thermal generator g', domain=pyo.Binary)

    lego.model.vGenP = pyo.Var(lego.model.g, lego.model.rp, lego.model.k, doc='Power output of generator g', bounds=(0, None))
    lego.model.vThermalOutput = pyo.Var(lego.model.thermalGenerators, lego.model.rp, lego.model.k, doc='Power output of thermal generator g', bounds=(0, None))
    for g in lego.model.thermalGenerators:
        lego.model.vGenP[g, :, :].setub(lego.model.pMaxProd[g] * lego.cs.dPower_ThermalGen.loc[g, 'ExisUnits'])
        lego.model.vThermalOutput[g, :, :].setub(lego.model.pMaxProd[g] * lego.cs.dPower_ThermalGen.loc[g, 'ExisUnits'] - lego.model.pMinProd[g] * lego.cs.dPower_ThermalGen.loc[g, 'ExisUnits'])
    for g in lego.model.rorGenerators:
        for rp in lego.model.rp:
            for k in lego.model.k:
                lego.model.vGenP[g, rp, k].setub(min(lego.model.pMaxProd[g], lego.cs.dPower_Inflows.loc[rp, g, k]['Inflow']))  # TODO: Check and adapt for storage
    for g in lego.model.vresGenerators:
        for rp in lego.model.rp:
            for k in lego.model.k:
                maxProd = lego.model.pMaxProd[g]
                capacity = lego.cs.dPower_VRESProfiles.loc[rp, lego.cs.dPower_VRES.loc[g, 'i'], k, lego.cs.dPower_VRES.loc[g, 'tec']]['Capacity']
                capacity = capacity.values[0] if isinstance(capacity, pd.Series) else capacity
                exisUnits = lego.cs.dPower_VRES.loc[g, 'ExisUnits']
                lego.model.vGenP[g, rp, k].setub(maxProd * capacity * exisUnits)

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
        return (sum(model.vGenP[g, rp, k] for g in model.g if lego.cs.hGenerators_to_Buses.loc[g]['i'] == i) -  # Production of generators at bus i
                sum(model.vLineP[rp, k, e, c] for c in model.c for e in model.la if (e[0] == i)) +  # Power flow from bus i to bus j
                sum(model.vLineP[rp, k, e, c] for c in model.c for e in model.la if (e[1] == i)) -  # Power flow from bus j to bus i
                model.pDemandP[rp, i, k] +  # Demand at bus i
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

    # Todo: Required when we have circuits:
    #  eTranInves (i,j,c) $[lc(i,j,c) and pEnableTransNet and ord(c)>1]..
    #     vLineInvest(i,j,c) =l= vLineInvest(i,j,c-1) + sum[le(i,j,c-1),1];

    # Thermal Generator production with unit commitment & ramping
    lego.model.cPowerOutput = pyo.ConstraintList(doc='Power output of thermal generators')
    lego.model.cPHatProduction = pyo.ConstraintList(doc='Production between min and max production of thermal generators')
    lego.model.cRampUp = pyo.ConstraintList(doc='Ramp-up constraint for thermal generators')
    lego.model.cRampDown = pyo.ConstraintList(doc='Ramp-down constraint for thermal generators')
    lego.model.cStartupLogic = pyo.ConstraintList(doc='Start-up and shut-down logic for thermal generators')
    lego.model.eMinUpTime = pyo.ConstraintList(doc='Minimum up time for thermal generators')
    lego.model.eMinDownTime = pyo.ConstraintList(doc='Minimum down time for thermal generators')

    for g in lego.model.thermalGenerators:
        for rp in lego.model.rp:
            for k in lego.model.k:
                lego.model.cPowerOutput.add(lego.model.vGenP[g, rp, k] == lego.cs.dPower_ThermalGen.loc[g, 'MinProd'] * lego.model.bUC[g, rp, k] + lego.model.vThermalOutput[g, rp, k])
                lego.model.cPHatProduction.add(lego.model.vThermalOutput[g, rp, k] <= (lego.cs.dPower_ThermalGen.loc[g, 'MaxProd'] - lego.cs.dPower_ThermalGen.loc[g, 'MinProd']) * lego.model.bUC[g, rp, k])
                lego.model.cStartupLogic.add(lego.model.bUC[g, rp, k] - lego.model.bUC[g, rp, lego.model.k.prevw(k)] == lego.model.bStartup[g, rp, k] - lego.model.bShutdown[g, rp, k])
                lego.model.cRampUp.add(lego.model.vThermalOutput[g, rp, k] - lego.model.vThermalOutput[g, rp, lego.model.k.prevw(k)] <= lego.cs.dPower_ThermalGen.loc[g, 'RampUp'] * lego.model.bUC[g, rp, k])
                lego.model.cRampDown.add(lego.model.vThermalOutput[g, rp, k] - lego.model.vThermalOutput[g, rp, lego.model.k.prevw(k)] >= lego.cs.dPower_ThermalGen.loc[g, 'RampDw'] * -lego.model.bUC[g, rp, lego.model.k.prevw(k)])

                # TODO: Check if implementation is correct
                # Only enforce MinUpTime and MinDownTime after the minimum time has passed
                if LEGOUtilities.k_to_int(k) >= max(lego.cs.dPower_ThermalGen.loc[g, 'MinUpTime'], lego.cs.dPower_ThermalGen.loc[g, 'MinDownTime']):
                    lego.model.eMinUpTime.add(sum(lego.model.bStartup[g, rp, LEGOUtilities.int_to_k(i)] for i in range(LEGOUtilities.k_to_int(k) - lego.model.pMinUpTime[g] + 1, LEGOUtilities.k_to_int(k))) <= lego.model.bUC[g, rp, k])  # Minimum Up-Time
                    lego.model.eMinDownTime.add(sum(lego.model.bShutdown[g, rp, LEGOUtilities.int_to_k(i)] for i in range(LEGOUtilities.k_to_int(k) - lego.model.pMinDownTime[g] + 1, LEGOUtilities.k_to_int(k))) <= 1 - lego.model.bUC[g, rp, k])  # Minimum Down-Time

    # Objective function
    lego.model.objective = pyo.Objective(doc='Total production cost (Objective Function)', sense=pyo.minimize, expr=sum(lego.model.pInterVarCost[g] * sum(lego.model.bUC[g, :, :]) +  # Fixed cost of thermal generators
                                                                                                                        lego.model.pStartupCost[g] * sum(lego.model.bStartup[g, :, :]) +  # Startup cost of thermal generators
                                                                                                                        lego.model.pSlopeVarCost[g] * sum(lego.model.vGenP[g, :, :]) for g in lego.model.thermalGenerators) +  # Production cost of thermal generators
                                                                                                                    sum(lego.model.pProductionCost[g] * sum(lego.model.vGenP[g, :, :]) for g in lego.model.vresGenerators) +
                                                                                                                    sum(lego.model.pProductionCost[g] * sum(lego.model.vGenP[g, :, :]) for g in lego.model.rorGenerators) +
                                                                                                                    sum(lego.model.pOMVarCost[g] * sum(lego.model.vGenP[g, :, :]) for g in lego.model.storageUnits) +
                                                                                                                    sum((sum(lego.model.vPNS[:, :, i]) + sum(lego.model.vEPS[:, :, i])) * lego.model.pSlackPrice[i] for i in lego.model.i) +  # Slack variables
                                                                                                                    sum(lego.model.pFixedCost[i, j, c] * lego.model.vLineInvest[i, j, c] for i, j in lego.model.lc for c in lego.model.c))  # Investment cost of transmission lines
