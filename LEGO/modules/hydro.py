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
    # model.hydroPlants = pyo.Set(doc='Hydro plants', initialize=cs.dHydro_Reservoir.reset_index().set_index(['rp', 'k']).index)
    model.hydroplants = pyo.Set(initialize=['P1', 'P2', 'P3'], doc='Hydro plants')  # Example hydro plants, replace with actual data
    model.T = pyo.Set(initialize=[1, 2, 3])                                         # Example time steps, replace with actual data

    # Parameters
    model.pMaxInflow = pyo.Param(model.hydroplants, initialize={'P1': 100, 'P2': 150, 'P3': 200}, doc='Maximum inflow rate for hydro plants')  # Example max inflow rates
    model.pInflow = pyo.Param(model.hydroplants, model.T, initialize = {
        ('P1', 1): 80, ('P1', 2): 90, ('P1', 3): 100,
        ('P2', 1): 120, ('P2', 2): 130, ('P2', 3): 140,
        ('P3', 1): 160, ('P3', 2): 170, ('P3', 3): 180
    })  # Example inflow rates, replace with actual data (Values from Co-pilot)
    model.pCapacityReservoir = pyo.Param(model.hydroplants, initialize={}) # not yet defined or used, but can be added if needed

    # Variables
    model.vPowerFactor = pyo.Var(model.hydroplants)
    second_stage_variables += [model.vPowerFactor]

    # Constraints
    def max_inflow_rule(model, i, t):
        return model.pInflow[i, t] <= model.pMaxInflow[i]
    model.eInflowLimit = pyo.Constraint(model.hydroplants, model.T, rule=max_inflow_rule, doc='Maximum inflow limit for hydro plants')

    # Objectives
    def obj_rule(model):
        return sum(model.vPowerFactor[i] * model.pInflow[i, t] for i in model.hydroplants for t in model.T)

    model.objective = pyo.Objective(rule=obj_rule, sense=pyo.maximize, doc='Objective function for hydro plants') # function maximizes the total output of hydro plants


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
