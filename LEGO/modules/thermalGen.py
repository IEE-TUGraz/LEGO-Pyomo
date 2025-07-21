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

    model.vCommit = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Unit commitment of generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, max(model.pMinUpTime[t], model.pMinDownTime[t])) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
    second_stage_variables += [model.vCommit]
    model.vStartup = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Start-up of thermal generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, model.pMinDownTime[t]) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
    second_stage_variables += [model.vStartup]
    model.vShutdown = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Shut-down of thermal generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, model.pMinUpTime[t]) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
    second_stage_variables += [model.vShutdown]
    model.vGenP1 = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Power output of generator g above minimum production', bounds=lambda model, rp, k, g: (0, (model.pMaxProd[g] - model.pMinProd[g]) * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g])))
    second_stage_variables += [model.vGenP1]

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = 0.0

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
