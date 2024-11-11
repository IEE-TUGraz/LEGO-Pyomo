import time
import typing

import pandas as pd
import pyomo.environ as pyo
import pyomo.opt.results.results_

import LEGOUtilities
from CaseStudy import CaseStudy
from LEGO import storage
from tools.printer import Printer

printer = Printer.getInstance()


class LEGO:
    def __init__(self, cs: CaseStudy = None, model: pyo.Model = None, results=None):
        self.cs: CaseStudy = cs
        self.model: typing.Optional[pyo.Model] = model
        self.results: typing.Optional[pyomo.opt.results.results_.SolverResults] = results
        self.timings = {"model_building": -1.0, "model_solving": -1.0}

    def build_model(self, already_existing_ok=False) -> (pyo.Model, float):
        if not already_existing_ok and self.model is not None:
            raise RuntimeError("Model already exists, please set already_existing_ok to True if that's intentional")

        start_time = time.time()
        model = pyo.ConcreteModel()
        self.model = model

        # Sets
        model.i = pyo.Set(doc='Buses', initialize=self.cs.dPower_BusInfo.index.tolist())
        model.e = pyo.Set(doc='Lines', initialize=self.cs.dPower_Network.index.tolist())
        model.thermalGenerators = pyo.Set(doc='Thermal Generators', initialize=self.cs.dPower_ThermalGen.index.tolist())
        model.rorGenerators = pyo.Set(doc='Run-of-river generators', initialize=self.cs.dPower_RoR.index.tolist())
        model.vresGenerators = pyo.Set(doc='Variable renewable energy sources', initialize=self.cs.dPower_VRES.index.tolist())
        model.g = pyo.Set(doc='Generators', initialize=model.thermalGenerators | model.rorGenerators | model.vresGenerators)
        model.rp = pyo.Set(doc='Representative periods', initialize=self.cs.dPower_Demand.index.get_level_values('rp').unique().tolist(), ordered=True)
        model.k = pyo.Set(doc='Timestep within representative period', initialize=self.cs.dPower_Demand.index.get_level_values('k').unique().tolist(), ordered=True)

        storage.add_variable_definitions(self)

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

        storage.add_variable_bounds(self)

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

        storage.add_constraints(self)

        # Objective function
        model.objective = get_objective(model)

        stop_time = time.time()
        self.timings["model_building"] = stop_time - start_time
        self.timings["model_solving"] = -1.0
        self.results = None

        return self.model, self.timings["model_building"]

    # Returns the objective value of this model (either overall or within the zone of interest)
    def get_objective_value(self, zoi: bool):
        return get_objective_value(self.model, zoi)

    def solve_model(self, optimizer, already_solved_ok=False) -> {pyomo.opt.results.results_.SolverResults, float}:
        if not already_solved_ok and self.results is not None:
            raise RuntimeError("Model already solved, please set already_solved_ok to True if that's intentional")

        start_time = time.time()
        results = optimizer.solve(self.model)
        stop_time = time.time()

        self.timings["model_solving"] = stop_time - start_time
        self.results = results

        return results, self.timings["model_solving"]

    def get_number_of_variables(self, dont_multiply_by_indices=False) -> int:
        if dont_multiply_by_indices:
            # Only count the number of variables, not multiplied by the number of indices
            return len(list(self.model.component_objects(pyo.Var, active=True)))
        else:
            # Iterate through variables and sum up each individual variable
            return sum([len(x) for x in self.model.component_objects(pyo.Var, active=True)])
        pass

    def get_number_of_constraints(self, dont_multiply_by_indices=False) -> int:
        if dont_multiply_by_indices:
            # Only count the number of constraints, not multiplied by the number of indices
            return len(list(self.model.component_objects(pyo.Constraint, active=True)))
        else:
            # Iterate through constraints and sum up each individual constraint
            return sum([len(x) for x in self.model.component_objects(pyo.Constraint, active=True)])

    # Update set 'g' to include given generators
    def update_generators(self, generators_to_be_added: list[str]):
        # Update set of generators
        if len(self.model.g) == 0:
            self.model.g = pyo.Set(doc='Generators', initialize=generators_to_be_added)
        else:
            for g in generators_to_be_added:
                self.model.g.add(g)


def get_objective(model: pyo.Model) -> pyo.Objective:
    result = pyo.Objective(doc='Total production cost (Objective Function)', sense=pyo.minimize, expr=sum(model.pInterVarCost[g] * sum(model.bUC[g, :, :]) +  # Fixed cost of thermal generators
                                                                                                          model.pStartupCost[g] * sum(model.bStartup[g, :, :]) +  # Startup cost of thermal generators
                                                                                                          model.pSlopeVarCost[g] * sum(model.p[g, :, :]) for g in model.thermalGenerators) +  # Production cost of thermal generators
                                                                                                      sum(model.pProductionCost[g] * sum(model.p[g, :, :]) for g in model.vresGenerators) +
                                                                                                      sum(model.pProductionCost[g] * sum(model.p[g, :, :]) for g in model.rorGenerators) +
                                                                                                      sum(model.pOMVarCost[g] * sum(model.p[g, :, :]) for g in model.storageUnits) +
                                                                                                      (sum(model.vSlack_DemandNotServed[:, :, :]) + sum(model.vSlack_OverProduction[:, :, :])) * model.pSlackPrice)

    return result


def get_objective_value(model: pyo.Model, zoi: bool):
    # This is calculated in any case to make sure that the objective is calculated correctly - please ALWAYS update this AND the calculation below whenever something changes in the objective function
    result_overall = (sum(pyo.value(model.pInterVarCost[g]) * sum(pyo.value(model.bUC[g, :, :])) +  # Fixed cost of thermal generators
                          pyo.value(model.pStartupCost[g]) * sum(pyo.value(model.bStartup[g, :, :])) +  # Startup cost of thermal generators
                          pyo.value(model.pSlopeVarCost[g]) * sum(pyo.value(model.p[g, :, :])) for g in model.thermalGenerators) +  # Production cost of thermal generators
                      sum(pyo.value(model.pProductionCost[g]) * sum(pyo.value(model.p[g, :, :])) for g in model.vresGenerators) +
                      sum(pyo.value(model.pProductionCost[g]) * sum(pyo.value(model.p[g, :, :])) for g in model.rorGenerators) +
                      sum(pyo.value(model.pOMVarCost[g]) * sum(pyo.value(model.p[g, :, :])) for g in model.storageUnits) +
                      (sum(pyo.value(model.vSlack_DemandNotServed[:, :, :])) + sum(pyo.value(model.vSlack_OverProduction[:, :, :]))) * pyo.value(model.pSlackPrice))

    if (abs(result_overall - pyo.value(model.objective)) / pyo.value(model.objective)) > 1e-12:
        raise RuntimeError(f"Check calculation of objective value, something is off: {result_overall} != {pyo.value(model.objective)}")
    if not zoi:
        return result_overall
    else:
        result_zoi = (sum(pyo.value(model.pInterVarCost[g]) * sum(pyo.value(model.bUC[g, :, :])) +  # Fixed cost of thermal generators
                          pyo.value(model.pStartupCost[g]) * sum(pyo.value(model.bStartup[g, :, :])) +  # Startup cost of thermal generators
                          pyo.value(model.pSlopeVarCost[g]) * sum(pyo.value(model.p[g, :, :])) for g in (model.thermalGenerators & model.zoi_g)) +  # Production cost of thermal generators
                      # sum(pyo.value(model.pProductionCost[g]) * sum(pyo.value(model.p[g, :, :])) for g in model.vresGenerators) +
                      # sum(pyo.value(model.pProductionCost[g]) * sum(pyo.value(model.p[g, :, :])) for g in model.rorGenerators) +
                      # sum(pyo.value(model.pOMVarCost[g]) * sum(pyo.value(model.p[g, :, :])) for g in model.storageUnits) +
                      sum(sum(pyo.value(model.vSlack_DemandNotServed[:, :, i])) + sum(pyo.value(model.vSlack_OverProduction[:, :, i])) for i in model.zoi_i) * pyo.value(model.pSlackPrice))

        return result_zoi


# Clone given model and fix specified variables to values from another model
def build_from_clone_with_fixed_results(model_to_be_cloned: pyo.Model, model_with_fixed_results: pyo.Model, variables_to_fix: list[str]) -> LEGO:
    model_new = model_to_be_cloned.clone()

    # Fix variables to values from model_with_fixed_results
    for var_name in variables_to_fix:
        var = getattr(model_with_fixed_results, var_name)
        new_var = getattr(model_new, var_name)
        for index in var:
            new_var[index].fix(pyo.value(var[index].value))

    return LEGO(model=model_new)
