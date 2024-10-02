import typing

import pandas as pd
import pyomo.environ as pyo

import LEGOUtilities
from CaseStudy import CaseStudy


class LEGO:
    def __init__(self, cs: CaseStudy):
        self.cs = cs
        self.model: typing.Optional[pyo.Model] = None

    def build_model(self):
        model = pyo.ConcreteModel()

        # Sets
        model.i = pyo.Set(doc='Buses', initialize=self.cs.dPower_BusInfo.index.tolist())
        model.e = pyo.Set(doc='Lines', initialize=self.cs.dPower_Network.index.tolist())
        model.thermalGenerators = pyo.Set(doc='Thermal Generators', initialize=self.cs.dPower_ThermalGen.index.tolist())
        model.rorGenerators = pyo.Set(doc='Run-of-river generators', initialize=self.cs.dPower_RoR.index.tolist())
        model.vresGenerators = pyo.Set(doc='Variable renewable energy sources', initialize=self.cs.dPower_VRES.index.tolist())
        model.storageUnits = pyo.Set(doc='Storage units', initialize=self.cs.dPower_Storage.index.tolist())
        model.g = pyo.Set(doc='Generators', initialize=model.thermalGenerators | model.rorGenerators | model.vresGenerators | model.storageUnits)
        model.rp = pyo.Set(doc='Representative periods', initialize=self.cs.dPower_Demand.index.get_level_values('rp').unique().tolist(), ordered=True)
        model.k = pyo.Set(doc='Timestep within representative period', initialize=self.cs.dPower_Demand.index.get_level_values('k').unique().tolist(), ordered=True)

        # Helper Sets for zone of interest
        model.zoi_i = pyo.Set(doc="Buses in zone of interest", initialize=self.cs.dPower_BusInfo.loc[self.cs.dPower_BusInfo["ZoneOfInterest"] == "yes"].index.tolist(), within=model.i)
        model.zoi_g = pyo.Set(doc="Generators in zone of interest", initialize=self.cs.hGenerators_to_Buses.loc[self.cs.hGenerators_to_Buses["i"].isin(model.zoi_i)].index.tolist(), within=model.g)

        # Variables
        model.delta = pyo.Var(model.i, model.rp, model.k, doc='Angle of bus i', bounds=(-self.cs.pMaxAngleDCOPF, self.cs.pMaxAngleDCOPF))  # TODO: Discuss impact on runtime etc.(based on discussion with Prof. Renner)
        model.vSlack_DemandNotServed = pyo.Var(model.rp, model.k, model.i, doc='Slack variable demand not served', bounds=(0, None))
        model.vSlack_OverProduction = pyo.Var(model.rp, model.k, model.i, doc='Slack variable overproduction', bounds=(0, None))

        model.bUC = pyo.Var(model.thermalGenerators, model.rp, model.k, doc='Unit commitment of generator g', domain=pyo.Binary)
        model.bStartup = pyo.Var(model.thermalGenerators, model.rp, model.k, doc='Start-up of thermal generator g', domain=pyo.Binary)
        model.bShutdown = pyo.Var(model.thermalGenerators, model.rp, model.k, doc='Shut-down of thermal generator g', domain=pyo.Binary)
        model.vThermalOutput = pyo.Var(model.thermalGenerators, model.rp, model.k, doc='Power output of thermal generator g', bounds=(0, None))
        model.vCharge = pyo.Var(model.storageUnits, model.rp, model.k, doc='Charging of storage unit g', bounds=(0, None))
        model.vStIntraRes = pyo.Var(model.storageUnits, model.rp, model.k, doc='Intra-reserve of storage unit g', bounds=(0, None))
        model.bCharge = pyo.Var(model.storageUnits, model.rp, model.k, doc='Binary variable for charging of storage unit g', domain=pyo.Binary)

        model.p = pyo.Var(model.g, model.rp, model.k, doc='Power output of generator g', bounds=(0, None))
        for g in model.thermalGenerators:
            model.p[g, :, :].setub(self.cs.dPower_ThermalGen.loc[g, 'MaxProd'] * self.cs.dPower_ThermalGen.loc[g, 'ExisUnits'])
            model.vThermalOutput[g, :, :].setub(self.cs.dPower_ThermalGen.loc[g, 'MaxProd'] * self.cs.dPower_ThermalGen.loc[g, 'ExisUnits'] - self.cs.dPower_ThermalGen.loc[g, 'MinProd'] * self.cs.dPower_ThermalGen.loc[g, 'ExisUnits'])

        for g in model.rorGenerators:
            for rp in model.rp:
                for k in model.k:
                    model.p[g, rp, k].setub(min(self.cs.dPower_RoR.loc[g, 'MaxProd'], self.cs.dPower_Inflows.loc[rp, g, k]['Inflow']))  # TODO: Check and adapt for storage

        for g in model.vresGenerators:
            for rp in model.rp:
                for k in model.k:
                    maxProd = self.cs.dPower_VRES.loc[g, 'MaxProd']
                    capacity = self.cs.dPower_VRESProfiles.loc[rp, self.cs.dPower_VRES.loc[g, 'i'], k, self.cs.dPower_VRES.loc[g, 'tec']]['Capacity']
                    capacity = capacity.values[0] if isinstance(capacity, pd.Series) else capacity
                    exisUnits = self.cs.dPower_VRES.loc[g, 'ExisUnits']
                    model.p[g, rp, k].setub(maxProd * capacity * exisUnits)

        for g in model.storageUnits:
            for rp in model.rp:
                for k in model.k:
                    model.p[g, rp, k].setub(self.cs.dPower_Storage.loc[g, 'MaxProd'] * self.cs.dPower_Storage.loc[g, 'ExisUnits'])
                    model.vCharge[g, rp, k].setub(self.cs.dPower_Storage.loc[g, 'MaxProd'] * self.cs.dPower_Storage.loc[g, 'ExisUnits'])
                    model.vStIntraRes[g, rp, k].setub(self.cs.dPower_Storage.loc[g, 'MaxProd'] * self.cs.dPower_Storage.loc[g, 'ExisUnits'] * self.cs.dPower_Storage.loc[g, 'Ene2PowRatio'])

        model.t = pyo.Var(model.e, model.rp, model.k, doc='Power flow from bus i to j', bounds=(None, None))
        for (i, j) in model.e:
            match self.cs.dPower_Network.loc[i, j]["Technical Representation"]:
                case "DC-OPF" | "TP":
                    model.t[(i, j), :, :].setlb(-self.cs.dPower_Network.loc[i, j]['Pmax'])
                    model.t[(i, j), :, :].setub(self.cs.dPower_Network.loc[i, j]['Pmax'])
                case "SN":
                    assert False  # "SN" line found, although all "Single Node" buses should be merged
                case _:
                    raise ValueError(f"Technical representation '{self.cs.dPower_Network.loc[i, j]["Technical Representation"]}' "
                                     f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

        # Parameters
        model.pDemand = pyo.Param(model.rp, model.i, model.k, initialize=self.cs.dPower_Demand['Demand'], doc='Demand at bus i in representative period rp and timestep k')

        # Helper for FuelCost that has dPower_ThermalGen['FuelCost'] for ThermalGen, and 0 for all gs in ror and vres
        hFuelCost = pd.concat([self.cs.dPower_ThermalGen['FuelCost'].copy(), pd.Series(0, index=model.rorGenerators), pd.Series(0, index=model.vresGenerators)])
        model.pProductionCost = pyo.Param(model.g, initialize=hFuelCost, doc='Production cost of generator g')
        model.pInterVarCost = pyo.Param(model.thermalGenerators, initialize=self.cs.dPower_ThermalGen['pInterVarCostEUR'], doc='Inter-variable cost of thermal generator g')
        model.pSlopeVarCost = pyo.Param(model.thermalGenerators, initialize=self.cs.dPower_ThermalGen['pSlopeVarCostEUR'], doc='Slope of variable cost of thermal generator g')
        model.pStartupCost = pyo.Param(model.thermalGenerators, initialize=self.cs.dPower_ThermalGen['pStartupCostEUR'], doc='Startup cost of thermal generator g')
        model.pMinUpTime = pyo.Param(model.thermalGenerators, initialize=self.cs.dPower_ThermalGen['MinUpTime'], doc='Minimum up time of thermal generator g')
        model.pMinDownTime = pyo.Param(model.thermalGenerators, initialize=self.cs.dPower_ThermalGen['MinDownTime'], doc='Minimum down time of thermal generator g')

        model.pOMVarCost = pyo.Param(model.storageUnits, initialize=self.cs.dPower_Storage['pOMVarCostEUR'], doc='Variable O&M cost of storage unit g')

        model.pReactance = pyo.Param(model.e, initialize=self.cs.dPower_Network['X'], doc='Reactance of line e')
        model.pSlackPrice = pyo.Param(initialize=max(model.pProductionCost.values()) * 100, doc='Price of slack variable')

        # For each DC-OPF "island", set node with highest demand as slack node
        dDCOPFIslands = pd.DataFrame(index=self.cs.dPower_BusInfo.index, columns=[self.cs.dPower_BusInfo.index], data=False)

        for index, entry in self.cs.dPower_Network.iterrows():
            if self.cs.dPower_Network.loc[(index[0], index[1])]["Technical Representation"] == "DC-OPF":
                dDCOPFIslands.loc[index[0], index[1]] = True
                dDCOPFIslands.loc[index[1], index[0]] = True

        completed_buses = set()  # Set of buses that have been looked at already
        i = 0
        for index, entry in dDCOPFIslands.iterrows():
            if index in completed_buses or entry[entry == True].empty:  # Skip if bus has already been looked at or has no connections
                continue

            connected_buses = self.cs.get_connected_buses(dDCOPFIslands, str(index))

            for bus in connected_buses:
                completed_buses.add(bus)

            # Set slack node
            slack_node = self.cs.dPower_Demand.loc[:, connected_buses, :].groupby('i').sum().idxmax().values[0]
            if i == 0: print("Setting slack nodes for DC-OPF zones:")
            print(f"DC-OPF Zone {i:>2} - Slack node: {slack_node}")
            i += 1
            model.delta[slack_node, :, :].fix(0)

        # Constraint(s)
        model.cPower_Balance = pyo.ConstraintList(doc='Power balance constraint for each bus')
        for i in model.i:
            for rp in model.rp:
                for k in model.k:
                    model.cPower_Balance.add(
                        sum(model.p[g, rp, k] for g in model.g if self.cs.hGenerators_to_Buses.loc[g]['i'] == i) -  # Production of generators at bus i
                        sum(model.t[e, rp, k] for e in model.e if (e[0] == i)) +  # Power flow from bus i to bus j
                        sum(model.t[e, rp, k] for e in model.e if (e[1] == i)) ==  # Power flow from bus j to bus i
                        model.pDemand[rp, i, k] -  # Demand at bus i
                        model.vSlack_DemandNotServed[rp, k, i] +  # Slack variable for demand not served
                        model.vSlack_OverProduction[rp, k, i])  # Slack variable for overproduction

        model.cReactance = pyo.ConstraintList(doc='Reactance constraint for each line (for DC-OPF)')
        for (i, j) in model.e:
            match self.cs.dPower_Network.loc[i, j]["Technical Representation"]:
                case "DC-OPF":
                    for rp in model.rp:
                        for k in model.k:
                            model.cReactance.add(model.t[(i, j), rp, k] == (model.delta[i, rp, k] - model.delta[j, rp, k]) * self.cs.pSBase / model.pReactance[(i, j)])
                case "TP" | "SN":
                    continue
                case _:
                    raise ValueError(f"Technical representation '{self.cs.dPower_Network.loc[i, j]["Technical Representation"]}' "
                                     f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

        # Thermal Generator production with unit commitment & ramping
        model.cPowerOutput = pyo.ConstraintList(doc='Power output of thermal generators')
        model.cPHatProduction = pyo.ConstraintList(doc='Production between min and max production of thermal generators')
        model.cRampUp = pyo.ConstraintList(doc='Ramp-up constraint for thermal generators')
        model.cRampDown = pyo.ConstraintList(doc='Ramp-down constraint for thermal generators')
        model.cStartupLogic = pyo.ConstraintList(doc='Start-up and shut-down logic for thermal generators')
        model.cMinUpTime = pyo.ConstraintList(doc='Minimum up time for thermal generators')
        model.cMinDownTime = pyo.ConstraintList(doc='Minimum down time for thermal generators')

        for g in model.thermalGenerators:
            for rp in model.rp:
                for k in model.k:
                    model.cPowerOutput.add(model.p[g, rp, k] == self.cs.dPower_ThermalGen.loc[g, 'MinProd'] * model.bUC[g, rp, k] + model.vThermalOutput[g, rp, k])
                    model.cPHatProduction.add(model.vThermalOutput[g, rp, k] <= (self.cs.dPower_ThermalGen.loc[g, 'MaxProd'] - self.cs.dPower_ThermalGen.loc[g, 'MinProd']) * model.bUC[g, rp, k])
                    model.cStartupLogic.add(model.bUC[g, rp, k] - model.bUC[g, rp, model.k.prevw(k)] == model.bStartup[g, rp, k] - model.bShutdown[g, rp, k])
                    model.cRampUp.add(model.vThermalOutput[g, rp, k] - model.vThermalOutput[g, rp, model.k.prevw(k)] <= self.cs.dPower_ThermalGen.loc[g, 'RampUp'] * model.bUC[g, rp, k])
                    model.cRampDown.add(model.vThermalOutput[g, rp, k] - model.vThermalOutput[g, rp, model.k.prevw(k)] >= self.cs.dPower_ThermalGen.loc[g, 'RampDw'] * -model.bUC[g, rp, model.k.prevw(k)])

                    # TODO: Check if implementation is correct
                    # Only enforce MinUpTime and MinDownTime after the minimum time has passed
                    if LEGOUtilities.k_to_int(k) >= max(self.cs.dPower_ThermalGen.loc[g, 'MinUpTime'], self.cs.dPower_ThermalGen.loc[g, 'MinDownTime']):
                        model.cMinUpTime.add(sum(model.bStartup[g, rp, LEGOUtilities.int_to_k(i)] for i in range(LEGOUtilities.k_to_int(k) - model.pMinUpTime[g] + 1, LEGOUtilities.k_to_int(k))) <= model.bUC[g, rp, k])  # Minimum Up-Time
                        model.cMinDownTime.add(sum(model.bShutdown[g, rp, LEGOUtilities.int_to_k(i)] for i in range(LEGOUtilities.k_to_int(k) - model.pMinDownTime[g] + 1, LEGOUtilities.k_to_int(k))) <= 1 - model.bUC[g, rp, k])  # Minimum Down-Time

        # Storage unit charging and discharging
        model.cStIntraRes = pyo.ConstraintList(doc='Intra-reserve constraint for storage units')
        model.cExclusiveChargeDischarge = pyo.ConstraintList(doc='Enforce exclusive charge or discharge for storage units')
        for g in model.storageUnits:
            for rp in model.rp:
                for k in model.k:
                    if LEGOUtilities.rp_to_int(rp) == 1 and LEGOUtilities.k_to_int(k) != 1:  # Only cyclic if it has multiple representative periods (and skipping first timestep)
                        model.cStIntraRes.add(model.vStIntraRes[g, rp, k] == model.vStIntraRes[g, rp, model.k.prev(k)] - model.p[g, rp, k] / self.cs.dPower_Storage.loc[g, 'DisEffic'] + model.vCharge[g, rp, k] * self.cs.dPower_Storage.loc[g, 'ChEffic'])
                    elif LEGOUtilities.rp_to_int(rp) > 1:
                        model.cStIntraRes.add(model.vStIntraRes[g, rp, k] == model.vStIntraRes[g, rp, model.k.prevw(k)] - model.p[g, rp, k] / self.cs.dPower_Storage.loc[g, 'DisEffic'] + model.vCharge[g, rp, k] * self.cs.dPower_Storage.loc[g, 'ChEffic'])

                    # TODO: Check if we should rather do a +/- value and calculate charge/discharge ex-post
                    model.cExclusiveChargeDischarge.add(model.vCharge[g, rp, k] <= model.bCharge[g, rp, k] * self.cs.dPower_Storage.loc[g, 'MaxProd'] * self.cs.dPower_Storage.loc[g, 'ExisUnits'])
                    model.cExclusiveChargeDischarge.add(model.p[g, rp, k] <= (1 - model.bCharge[g, rp, k]) * self.cs.dPower_Storage.loc[g, 'MaxProd'] * self.cs.dPower_Storage.loc[g, 'ExisUnits'])

        # Objective function
        model.objective = pyo.Objective(doc='Total production cost (Objective Function)', sense=pyo.minimize, expr=sum(model.pInterVarCost[g] * sum(model.bUC[g, :, :]) +  # Fixed cost of thermal generators
                                                                                                                       model.pStartupCost[g] * sum(model.bStartup[g, :, :]) +  # Startup cost of thermal generators
                                                                                                                       model.pSlopeVarCost[g] * sum(model.p[g, :, :]) for g in model.thermalGenerators) +  # Production cost of thermal generators
                                                                                                                   sum(model.pProductionCost[g] * sum(model.p[g, :, :]) for g in model.vresGenerators) +
                                                                                                                   sum(model.pProductionCost[g] * sum(model.p[g, :, :]) for g in model.rorGenerators) +
                                                                                                                   sum(model.pOMVarCost[g] * sum(model.p[g, :, :]) for g in model.storageUnits) +
                                                                                                                   (sum(model.vSlack_DemandNotServed[:, :, :]) + sum(model.vSlack_OverProduction[:, :, :])) * model.pSlackPrice)
        self.model = model
        return model
