import pyomo.environ as pyo
from pyomo.core import ConcreteModel

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGO, LEGOUtilities

@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Matthias Jannach Adaptations
    model = ConcreteModel()

    # Sets
    model.Hydroplants = pyo.Set(initialize=['P1', 'P2', 'P3', 'P4'], doc='Hydro plants')  # Example hydro plants, replace with actual data
    model.T = pyo.Set(initialize=[1, 2, 3])                                         # Example time steps, replace with actual data

    # Parameters
    model.pMaxInflow = pyo.Param(model.Hydroplants, initialize={'P1': 100, 'P2': 150, 'P3': 200, 'P4': 250}, doc='Maximum inflow rate for hydro plants')  # Example max inflow rates

    model.pInflowRiver = pyo.Param(model.Hydroplants, model.T, initialize={
        ('P1', 1): 50, ('P1', 2): 60, ('P1', 3): 70,
        ('P2', 1): 80, ('P2', 2): 90, ('P2', 3): 100,
        ('P3', 1): 110, ('P3', 2): 120, ('P3', 3): 130,
        ('P4', 1): 140, ('P4', 2): 150, ('P4', 3): 160
    }, doc='Flow of river for hydro plants at certain time steps')  # Example flow rates, replace with actual data

    model.pCapacityReservoir = pyo.Param(model.Hydroplants, initialize={
        'P1': 500, 'P2': 600, 'P3': 700, 'P4': 800}, doc='Capacity of reservoirs for hydro plants')  # not yet used, but can be added if needed

    model.pLowerLimitReservoir = pyo.Param(model.Hydroplants, initialize={
        'P1': 50, 'P2': 60, 'P3': 70, 'P4': 80}, doc='Lower limit of reservoir levels for hydro plants')  # not yet used, but can be added if needed

    model.pInitialStorage = pyo.Param(model.Hydroplants, initialize={
        'P1': 100, 'P2': 120, 'P3': 140, 'P4': 80}, doc='Initial storage levels for hydro plants')  # Initial storage levels, replace with

    model.pPowerFactor = pyo.Param(model.Hydroplants, initialize={ 'P1': 1.1, 'P2': 1.4, 'P3': 1.5, 'P4': 1.6
    }, doc='Power factor for hydro plants')  # Example power factors, replace with actual data

    model.pDemand = pyo.Param(model.T, initialize={1: 300, 2: 350, 3: 400}, doc='Demand for each time step')  # Example demand, replace with actual data

    model.pCost = pyo.Param(model.Hydroplants, initialize={'P1': 20, 'P2': 25, 'P3': 30, 'P4': 35}, doc='Cost of production for hydro plants')  # Example costs, replace with actual data

    # Variables
    model.vProd = pyo.Var(model.Hydroplants, model.T, domain=pyo.NonNegativeReals, doc='Production of hydro plants')  # Production of hydro plants; Domain cannot be changed?
    second_stage_variables += [model.vProd]
    model.vInflow = pyo.Var(model.Hydroplants, model.T, domain=pyo.NonNegativeReals, doc='Inflow rate into the hydro plants / actual intake of the hydro plant')  # Inflow rate for hydro plants
    second_stage_variables += [model.vInflow]
    model.vStorage = pyo.Var(model.Hydroplants, model.T, domain=pyo.NonNegativeReals, doc='Storage level of the reservoir at the hydro plants')  # Storage level for hydro plants at certain time steps
    second_stage_variables += [model.vStorage]

    model.vConsumption = pyo.Var(model.Hydroplants, model.T, domain=pyo.NonNegativeReals, doc='Consumption of the hydro plant from reservoir')  # Consumption from Reservoir
    second_stage_variables += [model.vConsumption]
    model.vSafe = pyo.Var(model.Hydroplants, model.T, domain=pyo.NonNegativeReals, doc='Safe to reservoir from river')     # Safe to Reservoir from River
    second_stage_variables += [model.vSafe]

    # Constraints

    def inflow_rule_1(model, i, t):
        return model.vInflow[i, t] <= model.pMaxInflow[i]
    model.eMaxInflow = pyo.Constraint(model.Hydroplants, model.T, rule=inflow_rule_1, doc='Inflow rate constraint for hydro plants')

    #def storage_rule_1(model, i, t):
    #    if t == 1:
    #        return model.vStorage[i, t] == model.pInitialStorage[i] + model.pInflowRiver[i, t] - model.vInflow[i, t]
    #    else:
    #        return model.vStorage[i, t] == model.vStorage[i, t-1] + model.pInflowRiver[i, t] - model.vInflow[i, t]
    # model.eStorage = pyo.Constraint(model.Hydroplants, model.T, rule=storage_rule_1, doc='Storage level constraint for hydro plants')

    def prod_def_rule(model, i, t):
        return model.vProd[i, t] == model.vInflow[i, t] * model.pPowerFactor[i]
    model.eProdDef = pyo.Constraint(model.Hydroplants, model.T, rule=prod_def_rule, doc='Production definition for hydro plants')  # Production definition constraint

    def demand_rule(model, t):
        return sum(model.vProd[i, t] for i in model.Hydroplants) == model.pDemand[t]
    model.eDemand_con = pyo.Constraint(model.T, rule=demand_rule)

    def inflow_rule_2(model, i, t):
        if t == 1:
            return model.vInflow[i, t] == model.pInflowRiver[i, t] + model.pInitialStorage[i] + model.pInflowRiver[i, t]
        else:
            return model.vInflow[i, t] == model.pInflowRiver[i, t] + model.vStorage[i, t - 1]

    #def storage_rule_2(model, i, t):
    #    if t == 1:
    #        return pyo.Constraint.Skip
    #    else:
    #        return model.vStorage[i, t] == model.vStorage[i, t-1] - model.vConsumption[i,t] + model.vSafe[i,t]
    #model.eStorage2 = pyo.Constraint(model.Hydroplants, model.T, rule=storage_rule_2)

    def min_capacity_rule(model, i, t):
       return model.pLowerLimitReservoir[i] <= model.vStorage[i, t]
    model.eMinCapacity = pyo.Constraint(model.Hydroplants, model.T, rule=min_capacity_rule, doc='Minimum capacity constraint for hydro plants')

    def max_capacity_rule(model, i, t):
        return model.vStorage[i, t] <= model.pCapacityReservoir[i]
    model.eMaxCapacity = pyo.Constraint(model.Hydroplants, model.T, rule=max_capacity_rule, doc='Maximum capacity constraint for hydro plants')

    # Cascade constraints
    # Cascade for linear set of HPP (P1 -> P2 -> P3)
    #def cascade_rule_1(model, i, t):
    #    hydro_list = list(model.Hydroplants)
    #    idx = hydro_list.index(i)           # Get index of current hydro plant in the list
    #    if idx < len(hydro_list) - 1:
    #        next_i = hydro_list[idx + 1]
    #        if t == 1:
     #           return model.vStorage[next_i, t] == model.pInitialStorage[next_i] + model.vInflow[i, t] + model.pInflowRiver[next_i, t]
     #       else:
     #           return model.vStorage[next_i, t] == model.vStorage[next_i, t - 1] + model.vInflow[i, t] + model.pInflowRiver[next_i, t] - model.vConsumption[i,t] + model.vSafe[i,t]
     #   else:
    #        return pyo.Constraint.Skip
    #model.eCascade1 = pyo.Constraint(model.Hydroplants, model.T, rule=cascade_rule_1, doc='First Cascade constraint for hydro plants')

    # Cascade for Network of HPP (P1 -> P2, P1 -> P3, P2 -> P4, P3 -> P4)
    #Pumps between hydro plants
    model.vCostPumps = pyo.Var(model.Hydroplants, domain=pyo.NonNegativeReals, doc='Cost for pumps at hydro plants')  # Cost for pumps at hydro plants
    second_stage_variables += [model.vCostPumps]
    allowed_pumps = [('P2', 'P1'), ('P3', 'P1'), ('P4', 'P2'), ('P4', 'P3')]  # Define allowed pump connections
    model.PumpPairs = pyo.Set(dimen=2, initialize=allowed_pumps, doc='Allowed pump connections between hydro plants')
    model.vPumpedWater = pyo.Var(model.PumpPairs, model.T, domain=pyo.NonNegativeReals, doc='Pumped water between hydro plants')  # Pumped water between hydro plants
    second_stage_variables += [model.vPumpedWater]


    model.CascadeNodes = pyo.Set(dimen=2, initialize=[('P1', 'P2'), ('P1', 'P3'), ('P2', 'P4'), ('P3', 'P4')], doc='Cascade nodes for hydro plants')
    model.pDistributionFactor = pyo.Param(model.CascadeNodes, initialize={
        ('P1', 'P2'): 0.5,  # 50% of the inflow from P1 goes to P2
        ('P1', 'P3'): 0.5,  # 50% of the inflow from P1 goes to P3
        ('P2', 'P4'): 1.0,  # 100% of the inflow from P2 goes to P4
        ('P3', 'P4'): 1.0  # 100% of the inflow from P3 goes to P4
    }, doc='Distribution factors for cascade nodes')
    # Find upstream plants for each hydro plant based on the cascade network structure
    def upstream_rule(model, i):
        return [u for (u, d) in model.CascadeNodes if d == i]
    model.UpstreamPlants = pyo.Set(model.Hydroplants, initialize=upstream_rule)

    def cascade_rule_graph(model, i, t):
        # Sum of all inflows from upstream plants
        inflow_from_upstream = sum(model.vInflow[u, t] * model.pDistributionFactor[u, i] for u in model.UpstreamPlants[i])
        #Pumpes water
        pumped_in = sum(model.vPumpedWater[j, i, t] for (j, k) in model.PumpPairs if k == i)
        pumped_out = sum(model.vPumpedWater[i, j, t] for (k, j) in model.PumpPairs if k == i)
        if t == 1:
            return model.vStorage[i, t] == model.pInitialStorage[i] + inflow_from_upstream + model.pInflowRiver[i, t] - model.vInflow[i, t] + pumped_in - pumped_out
        else:
            return model.vStorage[i, t] == model.vStorage[i, t - 1] + inflow_from_upstream + model.pInflowRiver[i, t] - model.vConsumption[i, t] + model.vSafe[i, t] - model.vInflow[i, t] + pumped_in - pumped_out

    model.eCascadeGraph = pyo.Constraint(model.Hydroplants, model.T, rule=cascade_rule_graph, doc='Cascade constraints for hydro plants based on graph structure')

    # Objectives
    def objective_rule(model):
        prod_cost = sum(model.vProd[i, t] * model.pCost[i] for i in model.Hydroplants for t in model.T)
        pump_cost = sum(model.vPumpedWater[i, j, t] * model.vCostPumps[i] for (i, j) in model.PumpPairs for t in model.T)
        return prod_cost + pump_cost
    model.obj_cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize, doc='Objective function for hydro plants including pump costs')

    print("Model created, trying to solve it...")
    optimizer = pyo.SolverFactory("highs")  # Use HiGHS solver for optimization
    results = optimizer.solve(model)
    model.pprint()
    exit(0)

    # Sets  # TODO: Add Hydro sets (e.g., hydro plants, reservoirs, etc.) if needed
    # storageUnits = ["S1, S2, S3"]
    # model.myVariable = pyo.Var(model.rp, model.k, doc='My variable for testing purposes', bounds=(0, 10))  # Example variable to show how to add variables
    #second_stage_variables += [model.myVariable]

    # model.storageUnits = pyo.Set(doc='Storage units', initialize=storageUnits)
    # LEGO.addToSet(model, "g", storageUnits)
    # LEGO.addToSet(model, "gi", cs.dPower_Storage.reset_index().set_index(['g', 'i']).index)  # Note: Add gi after g since it depends on g
    #
    # Parameters  # TODO: Add Hydro parameters (e.g., reservoir capacities, inflow rates, etc.) if needed
    # model.pEnableChDisPower = cs.dPower_Parameters['pEnableChDisPower']  # Avoid simultaneous charging and discharging
    # model.pE2PRatio = pyo.Param(model.storageUnits, initialize=cs.dPower_Storage['Ene2PowRatio'], doc='Energy to power ratio of storage unit g')
    # model.pMinReserve = pyo.Param(model.storageUnits, initialize=cs.dPower_Storage['MinReserve'], doc='Minimum reserve of storage unit g')
    #
    # LEGO.addToParameter(model, "pMaxCons", cs.dPower_Storage['MaxCons'], indices=[model.storageUnits], doc='Maximum consumption of storage unit')
    #
    # LEGO.addToParameter(model, "pMaxProd", cs.dPower_Storage['MaxProd'])
    # LEGO.addToParameter(model, "pMinProd", cs.dPower_Storage['MinProd'])
    # LEGO.addToParameter(model, "pExisUnits", cs.dPower_Storage['ExisUnits'])
    #
    # LEGO.addToParameter(model, "pOMVarCost", cs.dPower_Storage['pOMVarCostEUR'])
    # LEGO.addToParameter(model, "pMaxInvest", cs.dPower_Storage['MaxInvest'])
    # LEGO.addToParameter(model, "pEnabInv", cs.dPower_Storage['EnableInvest'])
    # LEGO.addToParameter(model, "pInvestCost", cs.dPower_Storage['InvestCostEUR'])
    #
    # Variables  # TODO: Add Hydro variables (e.g., reservoir levels, turbine outputs, etc.) if needed
    # model.bChargeDisCharge = pyo.Var(model.storageUnits, model.rp, model.k, doc='Binary variable for charging of storage unit g', domain=pyo.Binary)
    # second_stage_variables += [model.bChargeDisCharge]
    #
    # model.vConsump = pyo.Var(model.rp, model.k, model.storageUnits, doc='Charging of storage unit g', bounds=lambda model, rp, k, g: (0, model.pMaxCons[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g]))))
    # second_stage_variables += [model.vConsump]
    #
    # model.vStIntraRes = pyo.Var(model.rp, model.k, model.storageUnits, doc='Intra-reserve of storage unit g', bounds=(None, None))
    # second_stage_variables += [model.vStIntraRes]
    # for rp in model.rp:
    #     for k in model.k:
    #         for g in model.storageUnits:
    #             model.vStIntraRes[rp, k, g].setub(model.pE2PRatio[g] * model.pMaxProd[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g])))
    #             model.vStIntraRes[rp, k, g].setlb(model.pE2PRatio[g] * model.pMinReserve[g] * model.pMaxProd[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g])))
    #
    # model.vStInterRes = pyo.Var(model.p, model.storageUnits, doc='Inter-reserve of storage unit g', bounds=(None, None))
    # second_stage_variables += [model.vStInterRes]
    # for p in model.p:
    #     for g in model.storageUnits:
    #         if model.p.ord(p) == len(model.p):
    #             if cs.dPower_Parameters['pFixStInterResToIniReserve']:
    #                 model.vStInterRes[p, g].fix(cs.dPower_Storage.loc[g, 'IniReserve'])
    #             else:
    #                 model.vStInterRes[p, g].setlb(cs.dPower_Storage.loc[g, 'IniReserve'])
    #         elif model.p.ord(p) % model.pMovWindow == 0:
    #             model.vStInterRes[p, g].setub(model.pE2PRatio[g] * model.pMaxProd[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g])))
    #             model.vStInterRes[p, g].setlb(model.pE2PRatio[g] * model.pMinReserve[g] * model.pMaxProd[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g])))

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):  # TODO: Adapt to Hydro
    # # Constraint definitions
    # model.eStIntraRes = pyo.ConstraintList(doc='Intra-day reserve constraint for storage units')
    # model.eExclusiveChargeDischarge = pyo.ConstraintList(doc='Enforce exclusive charge or discharge for storage units')
    #
    # # Constraint implementations
    # for g in model.storageUnits:
    #     for rp in model.rp:
    #         for k in model.k:
    #             if len(model.rp) == 1:  # Only cyclic if it has multiple representative periods
    #                 if model.k.ord(k) == 1:  # Adding IniReserve if it is the first time step (instead of 'prev' value)
    #                     model.eStIntraRes.add(0 == cs.dPower_Storage.loc[g, 'IniReserve'] - model.vStIntraRes[rp, k, g] - model.vGenP[rp, k, g] * model.pWeight_k[k] / cs.dPower_Storage.loc[g, 'DisEffic'] + model.vConsump[rp, k, g] * model.pWeight_k[k] * cs.dPower_Storage.loc[g, 'ChEffic'])
    #                 else:
    #                     model.eStIntraRes.add(0 == model.vStIntraRes[rp, model.k.prev(k), g] - model.vStIntraRes[rp, k, g] - model.vGenP[rp, k, g] * model.pWeight_k[k] / cs.dPower_Storage.loc[g, 'DisEffic'] + model.vConsump[rp, k, g] * model.pWeight_k[k] * cs.dPower_Storage.loc[g, 'ChEffic'])
    #             elif len(model.rp) > 1:
    #                 model.eStIntraRes.add(0 == model.vStIntraRes[rp, model.k.prevw(k), g] - model.vStIntraRes[rp, k, g] - model.vGenP[rp, k, g] * model.pWeight_k[k] / cs.dPower_Storage.loc[g, 'DisEffic'] + model.vConsump[rp, k, g] * model.pWeight_k[k] * cs.dPower_Storage.loc[g, 'ChEffic'])
    #
    #             # TODO: Check if we should rather do a +/- value and calculate charge/discharge ex-post
    #             if model.pEnableChDisPower:
    #                 model.eExclusiveChargeDischarge.add(model.vConsump[rp, k, g] <= model.bChargeDisCharge[rp, k, g] * model.pMaxCons[g] * model.pExisUnits[g])
    #                 model.eExclusiveChargeDischarge.add(model.vGenP[rp, k, g] <= (1 - model.bChargeDisCharge[rp, k, g]) * model.pMaxProd[g] * model.pExisUnits[g])
    #
    # model.eStMaxProd_expr = pyo.Expression(model.rp, model.k, model.storageUnits, doc='Max production expression for storage units', rule=lambda model, rp, k, s: model.vGenP[rp, k, s] - model.vConsump[rp, k, s] - model.pMaxProd[s] * (model.vGenInvest[s] + model.pExisUnits[s]))
    # model.eStMaxProd = pyo.Constraint(model.rp, model.k, model.storageUnits, doc='Max production constraint for storage units', rule=lambda model, rp, k, s: model.eStMaxProd_expr[rp, k, s] <= 0)
    #
    # model.eStMaxCons_expr = pyo.Expression(model.rp, model.k, model.storageUnits, doc='Max consumption expression for storage units', rule=lambda model, rp, k, s: model.vGenP[rp, k, s] - model.vConsump[rp, k, s] + model.pMaxCons[s] * (model.vGenInvest[s] + model.pExisUnits[s]))
    # model.eStMaxCons = pyo.Constraint(model.rp, model.k, model.storageUnits, doc='Max consumption constraint for storage units', rule=lambda model, rp, k, s: model.eStMaxCons_expr[rp, k, s] >= 0)
    #
    # model.eStMaxIntraRes_expr = pyo.Expression(model.rp, model.k, model.storageUnits, doc='Max intra-reserve expression for storage units', rule=lambda model, rp, k, s: model.vStIntraRes[rp, k, s] - model.pMaxProd[s] * (model.vGenInvest[s] + model.pExisUnits[s]) * model.pE2PRatio[s])
    # model.eStMaxIntraRes = pyo.Constraint(model.rp, model.k, model.storageUnits, doc='Max intra-reserve constraint for storage units', rule=lambda model, rp, k, s: model.eStMaxIntraRes_expr[rp, k, s] <= 0)
    #
    # model.eStMinIntraRes_expr = pyo.Expression(model.rp, model.k, model.storageUnits, doc='Min intra-reserve expression for storage units', rule=lambda model, rp, k, s: model.vStIntraRes[rp, k, s] - model.pMaxProd[s] * (model.vGenInvest[s] + model.pExisUnits[s]) * model.pE2PRatio[s] * model.pMinReserve[s])
    # model.eStMinIntraRes = pyo.Constraint(model.rp, model.k, model.storageUnits, doc='Min intra-reserve constraint for storage units', rule=lambda model, rp, k, s: model.eStMinIntraRes_expr[rp, k, s] >= 0)
    #
    # def eStMaxInterRes_rule(model, p, s):
    #     # If current p is a multiple of moving window, add constraint
    #     if model.p.ord(p) % model.pMovWindow == 0:
    #         return 0 >= model.vStInterRes[p, s] - model.pMaxProd[s] * (model.vGenInvest[s] + model.pExisUnits[s]) * model.pE2PRatio[s]
    #     else:
    #         return pyo.Constraint.Skip
    #
    # model.eStMaxInterRes = pyo.Constraint(model.p, model.storageUnits, doc='Max inter-reserve constraint for storage units', rule=eStMaxInterRes_rule)
    #
    # def eStMinInterRes_rule(model, p, s):
    #     # If current p is a multiple of moving window, add constraint
    #     if model.p.ord(p) % model.pMovWindow == 0:
    #         return 0 <= model.vStInterRes[p, s] - model.pMaxProd[s] * (model.vGenInvest[s] + model.pExisUnits[s]) * model.pE2PRatio[s] * model.pMinReserve[s]
    #     else:
    #         return pyo.Constraint.Skip
    #
    # model.eStMinInterRes = pyo.Constraint(model.p, model.storageUnits, doc='Min inter-reserve constraint for storage units', rule=eStMinInterRes_rule)
    #
    # def eStInterRes_rule(model, p, storage_unit):
    #     # If current p is a multiple of moving window, add constraint
    #     if model.p.ord(p) % model.pMovWindow == 0:
    #         relevant_hindeces = model.hindex[model.p.ord(p) - model.pMovWindow:model.p.ord(p)]
    #         hindex_count = relevant_hindeces.to_frame(index=False).groupby(['rp', 'k']).size()
    #
    #         return (0 ==
    #                 (model.vStInterRes[model.p.at(model.p.ord(p) - model.pMovWindow), storage_unit] if model.p.ord(p) - model.pMovWindow > 0 else 0)
    #                 + (cs.dPower_Storage.loc[storage_unit, 'IniReserve'] if model.p.ord(p) == model.pMovWindow else 0)
    #                 - model.vStInterRes[p, storage_unit]
    #                 + sum(- model.vGenP[rp2, k2, storage_unit] * model.pWeight_k[k2] / cs.dPower_Storage.loc[storage_unit, 'DisEffic'] * hindex_count.loc[rp2, k2]
    #                       + model.vConsump[rp2, k2, storage_unit] * model.pWeight_k[k2] * cs.dPower_Storage.loc[storage_unit, 'ChEffic'] * hindex_count.loc[rp2, k2] for rp2, k2 in hindex_count.index))
    #
    #     else:  # Skip otherwise
    #         return pyo.Constraint.Skip
    #
    # if len(model.rp) > 1:
    #     model.eStInterRes = pyo.Constraint(model.p, model.storageUnits, doc='Inter-day reserve constraint for storage units', rule=eStInterRes_rule)
    #
    # # Add vConsump to eDC_BalanceP (vGenP should already be there, since it gets added for all generators)
    # for rp in model.rp:
    #     for k in model.k:
    #         for i in model.i:
    #             for g in model.storageUnits:
    #                 if (g, i) in model.gi:
    #                     model.eDC_BalanceP_expr[rp, k, i] -= model.vConsump[rp, k, g]

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = 0.0

    # No additional objective expressions for storage units (already handled in "power" module)

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
