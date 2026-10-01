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

    model.vDGACurtailment = pyo.Var(model.rp, model.k, model.vresGenerators, doc="Curtailment per generator and time", bounds=(0, None))
    second_stage_variables.append(model.vDGACurtailment)

    # Result values only (not part of any constraint) - filled after solving by calculate_curtailment_results()
    model.vDGAGeneratorCurtailment = pyo.Var(model.vresGenerators, doc="Curtailed energy of each generator as share of its available energy [%]", bounds=(0, None))
    second_stage_variables.append(model.vDGAGeneratorCurtailment)

    model.vDGATotalCurtailment = pyo.Var(model.dummySet_DGA, doc="Total curtailed energy of VRES generators [p.u. x h]", bounds=(0, None))
    second_stage_variables.append(model.vDGATotalCurtailment)


    return first_stage_variables, second_stage_variables
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.

@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):

    def eMaxCPowerClipping_rule(model, rp, k, r):
        if r in model.vresGenerators:
            shaveable_share = max(0.0, pyo.value(model.pCapacityFactors[rp, k, r]) - (1 - pyo.value(model.pDGAFactor[rp, k, r])))
            installed_cap = model.pMaxProd[r] * (model.pExisUnits[r] + model.vGenInvest[r])
            if shaveable_share == 0:
                return model.vDGACurtailment[rp, k, r] == 0
            return model.vDGACurtailment[rp, k, r] <= installed_cap * shaveable_share
        return pyo.Constraint.Skip

    model.eMaxCPowerClipping = pyo.Constraint(model.rp, model.constraintsActiveK, model.vresGenerators, rule=eMaxCPowerClipping_rule , doc='Curtailment can only occur when the capacity factor exceeds the maximum allowed curtailment share')

    def eReMaxProdDGA_rule(model, rp, k, r):
        return model.vGenP[rp, k, r] + model.vDGACurtailment[rp, k, r] == model.pMaxProd[r] * (model.pExisUnits[r] + model.vGenInvest[r]) * model.pCapacityFactors[rp, k, r]
    model.eReMaxProdDGA = pyo.Constraint(model.rp, model.constraintsActiveK, model.vresGenerators, doc= 'Production constraint with curtailment', rule=eReMaxProdDGA_rule)

    first_stage_objective = 0.0
    second_stage_objective = sum(model.pWeight_rp[rp] *
                                 sum(model.pWeight_k[k] *
                                     sum(model.vDGACurtailment[rp, k, r]
                                         for r in model.vresGenerators)
                                     for k in model.constraintsActiveK)
                                 for rp in model.rp) * model.pLOLCost * 0.00001

    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective


def calculate_curtailment_results(model: pyo.ConcreteModel) -> None:
    """Fill vDGAGeneratorCurtailment and vDGATotalCurtailment after solving (call before writing results).

    Share per generator = weighted curtailed energy / weighted available energy (vGenP + vDGACurtailment) in %.
    This is a ratio of variables (vGenInvest is a decision), so it cannot be a linear constraint and is computed
    from the solution instead. Sums run over all k with a solution value, so moving-window runs are covered too.
    """
    total = 0.0
    for r in model.vresGenerators:
        curtailed = available = 0.0
        for rp in model.rp:
            for k in model.k:
                c = model.vDGACurtailment[rp, k, r].value
                p = model.vGenP[rp, k, r].value
                if c is None or p is None:
                    continue
                w = pyo.value(model.pWeight_rp[rp]) * pyo.value(model.pWeight_k[k])
                curtailed += w * c
                available += w * (p + c)
        share = 100 * curtailed / available if available > 1e-9 else 0.0
        model.vDGAGeneratorCurtailment[r].set_value(max(share, 0.0), skip_validation=True)
        total += curtailed
    for d in model.dummySet_DGA:
        model.vDGATotalCurtailment[d].set_value(max(total, 0.0), skip_validation=True)
