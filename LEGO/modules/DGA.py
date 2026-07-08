import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGO, LEGOUtilities

printer = Printer.getInstance()

@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    first_stage_variables = []
    second_stage_variables = []

    model.dummySet_DGA = pyo.Set(initialize=[None]) # Dummy set for scalar variable

    model.pDGAFactor = pyo.Param(model.rp, model.k, model.vresGenerators,initialize=cs.dPower_DGA['value'],default=0,doc="Curtailment factor of VRES generators")
    model.pDGACurtailShare = 0.3 # Maximum curtailment of the installed PV capacity

    model.vDGATestFirst= pyo.Var(model.rp, model.k,  doc="First Stage Test variable for DGA", bounds=(0, None))
    first_stage_variables.append(model.vDGATestFirst)

    model.vDGACurtailment = pyo.Var(model.rp, model.k, model.vresGenerators, doc="Curtailment per generator and time", bounds=(0, None))
    second_stage_variables.append(model.vDGACurtailment)

    model.vDGAGeneratorCurtailment = pyo.Var(model.vresGenerators, doc= "Total curtailment of each generator compared to the total energy produced", bounds=(0, None))
    second_stage_variables.append(model.vDGAGeneratorCurtailment)

    model.vDGATotalCurtailment = pyo.Var(model.dummySet_DGA,doc= "Total curtailed energy of VRES generators", bounds=(0, None))
    second_stage_variables.append(model.vDGATotalCurtailment)


    return first_stage_variables, second_stage_variables
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.

@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):

    def eDGA_MaxCurtailmentRule(model, rp, k, r):
        return model.vDGACurtailment[rp, k, r] <= model.pDGAFactor[rp, k, r]  # Example constraint for DGA test variable
    model.eDGA_MaxCurtailment = pyo.Constraint(model.rp, model.constraintsActiveK, model.vresGenerators, doc='Maximum curtailment constraint for DGA from parameters', rule=eDGA_MaxCurtailmentRule)

    def eMaxCPowerClipping_rule(model, rp, k, r):
        if model.pCapacityFactors[rp, k, r]  <= (1-model.pDGACurtailShare):
            return model.vDGACurtailment[rp, k, r] == 0
        else:
            return model.vDGACurtailment[rp, k, r] <= model.pMaxProd[r] * (model.pExisUnits[r] + model.vGenInvest[r]) * (model.pCapacityFactors[rp, k, r]  - (1-model.pDGACurtailShare))

    model.eMaxCPowerClipping = pyo.Constraint(model.rp, model.constraintsActiveK, model.vresGenerators, rule=eMaxCPowerClipping_rule , doc='Curtailment can only occur when the capacity factor exceeds the maximum allowed curtailment share')

    def eReMaxProdDGA_rule(model, rp, k, r):
        return model.vGenP[rp, k, r] + model.vDGACurtailment[rp, k, r] == model.pMaxProd[r] * (model.pExisUnits[r] + model.vGenInvest[r]) * model.pCapacityFactors[rp, k, r]
    model.eReMaxProdDGA = pyo.Constraint(model.rp, model.constraintsActiveK, model.vresGenerators, doc= 'Production constraint with curtailment', rule=eReMaxProdDGA_rule)

    # Result calculations for curtailed energy

    def eDGA_GeneratorCurtailment_rule(model, r):
        return model.vDGAGeneratorCurtailment[r] == sum(model.pWeight_rp[rp] * model.vDGACurtailment[rp, k, r] for rp in model.rp for k in model.constraintsActiveK)
    model.eDGA_GeneratorCurtailment = pyo.Constraint(model.vresGenerators, doc='Total curtailment for each generator compared to the maximum possible generation', rule=eDGA_GeneratorCurtailment_rule)

    def eDGA_TotalCurtailment_rule(model, d):
        return model.vDGATotalCurtailment[d] == sum(
            model.vDGACurtailment[rp, k, r]
            for rp in model.rp for k in model.constraintsActiveK for r in model.vresGenerators
        )

    model.eDGA_TotalCurtailment = pyo.Constraint(model.dummySet_DGA, rule=eDGA_TotalCurtailment_rule)

    def eDGA_Test_rule_first(model, rp, k):
        return model.vDGATestFirst[rp, k] == 33  # Example constraint for DGA test variable
    model.eDGA_Test_first = pyo.Constraint(model.rp, model.constraintsActiveK, doc='Test constraint for first stage varibale', rule=eDGA_Test_rule_first)


    first_stage_objective = 0.0
    second_stage_objective = sum(model.pWeight_rp[rp] *
                                 sum(model.pWeight_k[k] *
                                     sum(model.vDGACurtailment[rp, k, r]
                                         for r in model.vresGenerators)
                                     for k in model.constraintsActiveK)
                                 for rp in model.rp) * model.pLOLCost * 0.01

    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective