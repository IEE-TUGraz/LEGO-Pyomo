import typing

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGOUtilities
from LEGO.LEGOUtilities import reset_execution_safety_dict, set_range_non_cyclic

printer = Printer.getInstance()

@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> typing.Tuple[list[pyo.Var], list[pyo.Var]]:
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets


    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables



@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):

    def eSelfSufficiency(m, rp, k, i, tbo, pvset, storage_set, thermal_set):
        if int(k[1:]) + tbo < len(model.k):
            set_t = set_range_non_cyclic(m.k, m.k.ord(k) + 1, m.k.ord(k) + 1 + tbo)
            return sum(m.pDemandP[rp, k, i] for k in set_t) <= (
                    sum((m.pCapacityFactors[rp, k, pv] * m.vGenP[rp, k, pv]) for pv in pvset for k in set_t) +
                    sum(m.vStIntraRes[rp, k, storage] for storage in storage_set) +
                    sum((m.pMaxProd[thermal] * m.vGenInvest[thermal] * tbo) for thermal in thermal_set) +
                    sum(m.vPNS[rp, k, i] for k in set_t)
            )
        else:
            return pyo.Constraint.Skip

    Tbo = cs.dGlobal_Parameters["pTBlackOut"]
    pvset = [pv for pv, tec in model.gtec if tec == "Solar"]
    thermal_set = [thermal for thermal, tec in model.gtec if tec == "FuelOilGas"]
    storage_set = [storage for storage, tec in model.gtec if tec == "BESS"]
    model.eSelfSufficiency = pyo.ConstraintList(doc='Self sufficiency constraint')
    for tbo in range(1, Tbo + 1):
        for rp in model.rp:
            for k in model.k:
                for i in model.i:
                    model.eSelfSufficiency.add(eSelfSufficiency(model, rp, k, i, tbo, pvset, storage_set, thermal_set))

    #model.eSelfSufficiency.pprint()




    #model.eSelfSufficiency = pyo.Constraint

    first_stage_objective = 0
    second_stage_objective = 0

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
