import pandas as pd
import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(lego: LEGO):
    # Sets
    lego.model.i = pyo.Set(doc='Buses', initialize=lego.cs.dPower_BusInfo.index.tolist())
    lego.model.e = pyo.Set(doc='Lines', initialize=lego.cs.dPower_Network.index.tolist())
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

    lego.model.pReactance = pyo.Param(lego.model.e, initialize=lego.cs.dPower_Network['X'], doc='Reactance of line e')
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
    lego.model.delta = pyo.Var(lego.model.i, lego.model.rp, lego.model.k, doc='Angle of bus i', bounds=(-lego.cs.dPower_Parameters["pMaxAngleDCOPF"], lego.cs.dPower_Parameters["pMaxAngleDCOPF"]))  # TODO: Discuss impact on runtime etc.(based on discussion with Prof. Renner)
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
        if i == 0: print("Setting slack nodes for DC-OPF zones:")
        print(f"DC-OPF Zone {i:>2} - Slack node: {slack_node}")
        i += 1
        lego.model.delta[slack_node, :, :].fix(0)

    lego.model.vSlack_DemandNotServed = pyo.Var(lego.model.rp, lego.model.k, lego.model.i, doc='Slack variable demand not served', bounds=(0, None))
    lego.model.vSlack_OverProduction = pyo.Var(lego.model.rp, lego.model.k, lego.model.i, doc='Slack variable overproduction', bounds=(0, None))

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

    lego.model.t = pyo.Var(lego.model.e, lego.model.rp, lego.model.k, doc='Power flow from bus i to j', bounds=(None, None))
    for (i, j) in lego.model.e:
        match lego.cs.dPower_Network.loc[i, j]["Technical Representation"]:
            case "DC-OPF" | "TP":
                lego.model.t[(i, j), :, :].setlb(-lego.cs.dPower_Network.loc[i, j]['Pmax'])
                lego.model.t[(i, j), :, :].setub(lego.cs.dPower_Network.loc[i, j]['Pmax'])
            case "SN":
                assert False  # "SN" line found, although all "Single Node" buses should be merged
            case _:
                raise ValueError(f"Technical representation '{lego.cs.dPower_Network.loc[i, j]["Technical Representation"]}' "
                                 f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):
    def eDC_BalanceP_rule(model, i, rp, k):
        return (sum(model.vGenP[g, rp, k] for g in model.g if lego.cs.hGenerators_to_Buses.loc[g]['i'] == i) -  # Production of generators at bus i
                sum(model.t[e, rp, k] for e in model.e if (e[0] == i)) +  # Power flow from bus i to bus j
                sum(model.t[e, rp, k] for e in model.e if (e[1] == i)) ==  # Power flow from bus j to bus i
                model.pDemandP[rp, i, k] -  # Demand at bus i
                model.vSlack_DemandNotServed[rp, k, i] +  # Slack variable for demand not served
                model.vSlack_OverProduction[rp, k, i])  # Slack variable for overproduction

    lego.model.eDC_BalanceP = pyo.Constraint(lego.model.i, lego.model.rp, lego.model.k, doc='Power balance constraint for each bus', rule=eDC_BalanceP_rule)

    lego.model.cReactance = pyo.ConstraintList(doc='Reactance constraint for each line (for DC-OPF)')
    for (i, j) in lego.model.e:
        match lego.cs.dPower_Network.loc[i, j]["Technical Representation"]:
            case "DC-OPF":
                for rp in lego.model.rp:
                    for k in lego.model.k:
                        lego.model.cReactance.add(lego.model.t[(i, j), rp, k] == (lego.model.delta[i, rp, k] - lego.model.delta[j, rp, k]) * lego.cs.dPower_Parameters["pSBase"] / lego.model.pReactance[(i, j)])
            case "TP" | "SN":
                continue
            case _:
                raise ValueError(f"Technical representation '{lego.cs.dPower_Network.loc[i, j]["Technical Representation"]}' "
                                 f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

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
