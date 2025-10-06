import numpy as np
import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGOUtilities
import RINGS_datahandler

printer = Printer.getInstance()

# This is intednded for small scale heat formulations, e.g. on a building or community level.

# read in custom data and assign it to the casestudy



@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # add data to the casestudy
    cs.dHeatDemand = RINGS_datahandler.get_dHeat_Demand(cs.data_folder)
    cs.dHeatDemand = cs.dHeatDemand.set_index(['rp', 'k', 'i'])


    # Sets


    # Parameters
    model.pDemandHeat = pyo.Param(model.rp, model.k, model.i, initialize=cs.dPower_Demand['value'], doc='Demand at bus i in representative period rp and timestep k')

    model.pCOP = pyo.Param(model.i, initialize=3, doc='Coefficient of performance of heat pump at bus i')

    # Variables
    model.vPowerforHeat = pyo.Var(model.rp, model.k, model.i, doc="Power required for heat generation", bounds=(0, None))
    second_stage_variables += [model.vPowerforHeat]

    return first_stage_variables, second_stage_variables

@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):


    # CONSTRAINTS
    def power_to_heat_rule(m,rp,k,i):
        return m.vPowerforHeat[rp,k,i] * model.pCOP[i] >= m.pDemandHeat[rp,k,i]
    #model.power_to_heat_expr = pyo.Expression(model.rp, model.k, model.i, rule=power_to_heat_rule)
    model.power_to_heat = pyo.Constraint(model.rp, model.k, model.i, rule=power_to_heat_rule, doc="Power required for heat generation constraint")

    # Add vPowerforHeat to eDC_BalanceP
    for rp in model.rp:
        for k in model.k:
            for i in model.i:
                model.eDC_BalanceP_expr[rp, k, i] -= model.vPowerforHeat[rp, k, i]


    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0
    second_stage_objective = 0

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective