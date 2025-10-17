import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGOUtilities, LEGO


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.thermalGenerators = pyo.Set(doc='Thermal Generators', initialize=cs.dPower_ThermalGen.index.tolist())
    LEGO.addToSet(model, "g", model.thermalGenerators)
    LEGO.addToSet(model, "gi", cs.dPower_ThermalGen.reset_index().set_index(['g', 'i']).index)

    # Parameters
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

    LEGO.addToParameter(model, 'pMaxGenQ', cs.dPower_ThermalGen['Qmax'])
    LEGO.addToParameter(model, 'pMinGenQ', cs.dPower_ThermalGen['Qmin'])

    # Variables
    # Used to relax vCommit, vStartup and vShutdown in the first timesteps of each representative period
    # Required when using Markov-Chains to connect the timesteps of the representative periods - since fractions of the binary variables (which are present due to the transition-probabilities) are otherwise not possible
    def vUC_domain(model, k, relax_duration_from_beginning):
        if model.k.ord(k) <= relax_duration_from_beginning:
            return pyo.PercentFraction  # PercentFraction = Floating point values in the interval [0,1]
        else:
            return pyo.Binary

    maxTime = {t: max(model.pMinUpTime[t], model.pMinDownTime[t]) for t in model.thermalGenerators}  # Pre-calculate maximum time for each generator to speed up the domain function
    model.vCommit = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Unit commitment of generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, maxTime[t]) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
    second_stage_variables += [model.vCommit]
    model.vStartup = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Start-up of thermal generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, model.pMinDownTime[t]) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
    second_stage_variables += [model.vStartup]
    model.vShutdown = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Shut-down of thermal generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, model.pMinUpTime[t]) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
    second_stage_variables += [model.vShutdown]
    model.vGenP1 = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Power output of generator g above minimum production', bounds=lambda model, rp, k, g: (0, (model.pMaxProd[g] - model.pMinProd[g]) * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g])))
    second_stage_variables += [model.vGenP1]

    # Fix vCommit, vStartup and vShutdown to 0 for generators that are not existing and cannot be invested in (will then be removed in pre-solve)
    # Note: The generators can not simply be removed overall, as this would break stochastic models with different existing generators in different scenarios
    for g in model.thermalGenerators:
        if cs.dPower_ThermalGen.loc[g, 'ExisUnits'] == 0 and cs.dPower_ThermalGen.loc[g, 'EnableInvest'] == 0:
            for rp in model.rp:
                for k in model.k:
                    model.vCommit[rp, k, g].fix(0)
                    model.vStartup[rp, k, g].fix(0)
                    model.vShutdown[rp, k, g].fix(0)

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    def eThRampUp_rule(model, rp, k, g, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"]:
            case "notEnforced":
                if model.constraintsActiveK.first() == k:
                    return None  # Is not enforced and should therefore be turned into pyo.Constraint.Skip in constraint construction
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.constraintsActiveK.prev(k), g] - model.vCommit[rp, k, g] * model.pRampUp[g]
            case "cyclic":
                return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.constraintsActiveK.prevw(k), g] - model.vCommit[rp, k, g] * model.pRampUp[g]
            case "markov":
                if model.constraintsActiveK.first() == k:
                    return model.vGenP1[rp, k, g] - LEGOUtilities.markov_summand(model.rp, rp, False, model.constraintsActiveK.prevw(k), model.vGenP1, transition_matrix, g) - model.vCommit[rp, k, g] * model.pRampUp[g]
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.constraintsActiveK.prev(k), g] - model.vCommit[rp, k, g] * model.pRampUp[g]
            case _:
                raise ValueError(f"Period edge handling ramping '{cs.dPower_Parameters['pReprPeriodEdgeHandlingRamping']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eThRampUp_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.thermalGenerators, rule=lambda m, rp, k, t: eThRampUp_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))
    model.eThRampUp = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc='Ramp up for thermal generators (based on doi:10.1007/s10107-015-0919-9)', rule=lambda model, rp, k, t: model.eThRampUp_expr[rp, k, t] <= 0 if not ((cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] == "notEnforced") and (model.constraintsActiveK.first() == k)) else pyo.Constraint.Skip)

    def eThRampDw_rule(model, rp, k, g, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"]:
            case "notEnforced":
                if model.constraintsActiveK.first() == k:
                    return None  # Is not enforced and should therefore be turned into pyo.Constraint.Skip in constraint construction
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.constraintsActiveK.prev(k), g] + model.vCommit[rp, model.constraintsActiveK.prevw(k), g] * model.pRampDw[g]
            case "cyclic":
                return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.constraintsActiveK.prevw(k), g] + model.vCommit[rp, model.constraintsActiveK.prevw(k), g] * model.pRampDw[g]
            case "markov":
                if model.constraintsActiveK.first() == k:
                    return model.vGenP1[rp, k, g] - LEGOUtilities.markov_summand(model.rp, rp, False, model.constraintsActiveK.prevw(k), model.vGenP1, transition_matrix, g) + LEGOUtilities.markov_summand(model.rp, rp, False, model.constraintsActiveK.prevw(k), model.vCommit, transition_matrix, g) * model.pRampDw[g]
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.constraintsActiveK.prev(k), g] + model.vCommit[rp, model.constraintsActiveK.prev(k), g] * model.pRampDw[g]
            case _:
                raise ValueError(f"Period edge handling ramping '{cs.dPower_Parameters['pReprPeriodEdgeHandlingRamping']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eThRampDw_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.thermalGenerators, rule=lambda m, rp, k, t: eThRampDw_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))
    model.eThRampDw = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc='Ramp down for thermal generators (based on doi:10.1007/s10107-015-0919-9)', rule=lambda model, rp, k, t: model.eThRampDw_expr[rp, k, t] >= 0 if not ((cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] == "notEnforced") and (model.constraintsActiveK.first() == k)) else pyo.Constraint.Skip)

    # Thermal Generator production with unit commitment & ramping constraints
    model.eUCTotOut = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc='Total production of thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, g: model.vGenP[rp, k, g] == model.pMinProd[g] * model.vCommit[rp, k, g] + model.vGenP1[rp, k, g])
    model.eThMaxUC = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc='Maximum number of active units for thermal generators', rule=lambda m, rp, k, t: m.vCommit[rp, k, t] <= m.vGenInvest[t] + m.pExisUnits[t])

    model.eUCMaxOut1_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.thermalGenerators, rule=lambda m, rp, k, t: m.vGenP1[rp, k, t] - (m.pMaxProd[t] - m.pMinProd[t]) * (m.vCommit[rp, k, t] - m.vStartup[rp, k, t]))
    model.eUCMaxOut1 = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc='Maximum production for startup of thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, t: model.eUCMaxOut1_expr[rp, k, t] <= 0)

    def eUCMaxOut2_rule(model, rp, k, t, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
            case "notEnforced":
                if model.constraintsActiveK.last() == k:
                    return None  # Is not enforced and should therefore be turned into pyo.Constraint.Skip in constraint construction
                else:
                    return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vShutdown[rp, model.constraintsActiveK.nextw(k), t])
            case "cyclic":
                return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vShutdown[rp, model.constraintsActiveK.nextw(k), t])
            case "markov":
                if model.constraintsActiveK.last() == k:  # Markov summand only required for very last timestep
                    return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - LEGOUtilities.markov_summand(model.rp, rp, True, model.constraintsActiveK.nextw(k), model.vShutdown, transition_matrix, t))
                else:
                    return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vShutdown[rp, model.constraintsActiveK.nextw(k), t])
            case _:
                raise ValueError(f"Period edge handling unit commitment '{cs.dPower_Parameters['pReprPeriodEdgeHandlingUnitCommitment']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eUCMaxOut2_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.thermalGenerators, rule=lambda m, rp, k, t: eUCMaxOut2_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeTo))
    model.eUCMaxOut2 = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc='Maximum production for shutdown of thermal generators (from doi:10.1109/TPWRS.2013.2251373)',
                                      rule=lambda model, rp, k, t: model.eUCMaxOut2_expr[rp, k, t] <= 0 if not ((cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "notEnforced") and (model.constraintsActiveK.last() == k)) else pyo.Constraint.Skip)

    def eUCStrShut_rule(model, rp, k, t, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
            case "notEnforced":
                if model.constraintsActiveK.ord(k) == 1:
                    return pyo.Constraint.Skip
                else:
                    return model.vCommit[rp, k, t] - model.vCommit[rp, model.constraintsActiveK.prevw(k), t] == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
            case "cyclic":
                return model.vCommit[rp, k, t] - model.vCommit[rp, model.constraintsActiveK.prevw(k), t] == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
            case "markov":
                if model.constraintsActiveK.ord(k) == 1:  # Markov summand only required for very first timestep
                    return model.vCommit[rp, k, t] - LEGOUtilities.markov_summand(model.rp, rp, False, model.constraintsActiveK.prevw(k), model.vCommit, transition_matrix, t) == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
                else:
                    return model.vCommit[rp, k, t] - model.vCommit[rp, model.constraintsActiveK.prevw(k), t] == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
            case _:
                raise ValueError(f"Period edge handling unit commitment '{cs.dPower_Parameters['pReprPeriodEdgeHandlingUnitCommitment']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eUCStrShut = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc='Start-up and shut-down logic for thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, t: eUCStrShut_rule(model, rp, k, t, cs.rpTransitionMatrixRelativeFrom))

    def eMinUpTime_rule(model, rp, k, t, transition_matrix):
        if model.pMinUpTime[t] == 0:
            raise ValueError("Minimum up time must be at least 1, got 0 instead")
        elif model.pMinUpTime[t] == 1:
            return pyo.Constraint.Skip
        else:
            match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
                case "notEnforced":
                    if model.constraintsActiveK.ord(k) < model.pMinUpTime[t]:
                        return pyo.Constraint.Skip  # Constraint is not active until the minimum up-time is reached
                    else:
                        return sum(model.vStartup[rp, k2, t] for k2 in LEGOUtilities.set_range_non_cyclic(model.constraintsActiveK, model.constraintsActiveK.ord(k) - model.pMinUpTime[t] + 1, model.constraintsActiveK.ord(k))) <= model.vCommit[rp, k, t]
                case "cyclic":
                    return sum(model.vStartup[rp, k2, t] for k2 in LEGOUtilities.set_range_cyclic(model.constraintsActiveK, model.constraintsActiveK.ord(k) - model.pMinUpTime[t] + 1, model.constraintsActiveK.ord(k))) <= model.vCommit[rp, k, t]
                case "markov":
                    return LEGOUtilities.markov_sum(model.rp, rp, model.constraintsActiveK, model.constraintsActiveK.ord(k) - model.pMinUpTime[t] + 1, model.constraintsActiveK.ord(k), model.vStartup, transition_matrix, t) <= model.vCommit[rp, k, t]
                case _:
                    raise ValueError(f"Invalid value for 'pReprPeriodEdgeHandlingUnitCommitment' in 'Global_Parameters.xlsx': {cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]} - please choose from 'notEnforced', 'cyclic' or 'markov'!")

    model.eMinUpTime = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc='Minimum up time for thermal generators (from doi:10.1109/TPWRS.2013.2251373, adjusted to be cyclic)', rule=lambda m, rp, k, t: eMinUpTime_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))

    def eMinDownTime_rule(model, rp, k, t, transition_matrix):
        if model.pMinDownTime[t] == 0:
            raise ValueError("Minimum down time must be at least 1, got 0 instead")
        elif model.pMinDownTime[t] == 1:
            return pyo.Constraint.Skip
        else:
            match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
                case "notEnforced":
                    if model.constraintsActiveK.ord(k) < model.pMinDownTime[t]:
                        return pyo.Constraint.Skip  # Constraint is not active until the minimum down-time is reached
                    else:
                        return sum(model.vShutdown[rp, k2, t] for k2 in LEGOUtilities.set_range_non_cyclic(model.constraintsActiveK, model.constraintsActiveK.ord(k) - model.pMinDownTime[t] + 1, model.constraintsActiveK.ord(k))) <= 1 - model.vCommit[rp, k, t]
                case "cyclic":
                    return sum(model.vShutdown[rp, k2, t] for k2 in LEGOUtilities.set_range_cyclic(model.constraintsActiveK, model.constraintsActiveK.ord(k) - model.pMinDownTime[t] + 1, model.constraintsActiveK.ord(k))) <= 1 - model.vCommit[rp, k, t]
                case "markov":
                    return LEGOUtilities.markov_sum(model.rp, rp, model.constraintsActiveK, model.constraintsActiveK.ord(k) - model.pMinDownTime[t] + 1, model.constraintsActiveK.ord(k), model.vShutdown, transition_matrix, t) <= 1 - model.vCommit[rp, k, t]
                case _:
                    raise ValueError(f"Invalid value for 'pReprPeriodEdgeHandlingUnitCommitment' in 'Global_Parameters.xlsx': {cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]} - please choose from 'notEnforced', 'cyclic' or 'markov'!")

    model.eMinDownTime = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc='Minimum down time for thermal generators (from doi:10.1109/TPWRS.2013.2251373, adjusted to be cyclic)', rule=lambda m, rp, k, t: eMinDownTime_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = sum(model.pWeight_rp[rp] *  # Weight of representative periods
                                 sum(model.pWeight_k[k] *  # Weight of time steps
                                     sum(+ model.vStartup[rp, k, t] * model.pStartupCost[t]  # Startup cost of thermal generators
                                         + model.vCommit[rp, k, t] * model.pInterVarCost[t]  # Commit cost of thermal generators)
                                         for t in model.thermalGenerators)
                                     for k in model.constraintsActiveK)
                                 for rp in model.rp)

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
