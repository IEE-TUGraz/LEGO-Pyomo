# import pandas as pd
# import pyomo.environ as pyo

# from InOutModule.CaseStudy import CaseStudy
# from InOutModule.printer import Printer
# from LEGO import LEGO, LEGOUtilities

# printer = Printer.getInstance()

# @LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
# def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
#     first_stage_variables = []
#     second_stage_variables = []

#     # Define variables and constraints for demand-side management
#     # ... (implementation details for DMSM)
#     model.vDSMTestSecond= pyo.Var(model.rp, model.k, doc="Second Stage Test variable for DSM", bounds=(0, None))
#     second_stage_variables.append(model.vDSMTestSecond)

#     model.vDSMTestFirst= pyo.Var(model.rp, model.k, doc="First Stage Test variable for DSM", bounds=(0, None))
#     first_stage_variables.append(model.vDSMTestFirst)

#     return first_stage_variables, second_stage_variables
#     # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.

# @LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
# def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
#     # Define constraints for demand-side management
#     # ... (implementation details for DMSM)

#     def eDSM_Test_rule_first(model, rp, k):
#         return model.vDSMTestFirst[rp, k] == 3  # Example constraint for DSM test variable

#     model.eDSM_Test_first = pyo.Constraint(model.rp, model.constraintsActiveK, rule=eDSM_Test_rule_first)

#     def eDSM_Test_rule_second(model, rp, k):
#         return model.vDSMTestSecond[rp, k] == 333  # Example constraint for DSM test variable

#     model.eDSM_Test_second = pyo.Constraint(model.rp, model.constraintsActiveK, rule=eDSM_Test_rule_second)

#     # Adjust objective function if necessary
#     # ... (implementation details for DMSM)
#     # OBJECTIVE FUNCTION ADJUSTMENT(S)
#     first_stage_objective = 0.0
#     second_stage_objective = 0.0

#     # Adjust objective and return first_stage_objective expression
#     model.objective.expr += first_stage_objective + second_stage_objective
#     return first_stage_objective

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

        
# Dynamische Regel für die Obergrenze, Eigene base_load Variable damit Wert übernommen werden kann
    def dsm_upper_bound_rule(model, rp, k, i):
        base_load = 0.0
        
        # Direkte Prüfung, ob die Kombination (Szenario, Zeitschritt, Knoten) existiert
        if (rp, k, i) in model.pDemandP:
            base_load = pyo.value(model.pDemandP[rp, k, i])
                    
        return (0.0, 0.50 * base_load)  # Reduktion Grenze je Knoten
              
                    
        
    
    # In pyo für Optimierungssolver, maximale reduktion um dsm_upper_bound_rule
    # Definition der DSM-Variable (Second-Stage: Reagiert flexibel je nach Szenario)
    # Dem Framework mitteilen, dass DSM eine betriebliche Reaktion (Second Stage) ist

    model.vDSM_Reduction = pyo.Var(model.rp, model.constraintsActiveK, model.i, bounds=dsm_upper_bound_rule)
    second_stage_variables.append(model.vDSM_Reduction)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    
    # 1. Variablen definieren für den Datenbank-Export
    model.eNetPowerDemandKnoten = pyo.Var(model.rp, model.constraintsActiveK, model.i)
    model.eTotalNetDemand = pyo.Var(model.rp, model.constraintsActiveK)

    # 2. Gleichung: Netto-Last pro Knoten = Last - DSM
    def net_power_demand_eq(m, rp, k, i):
        return m.eNetPowerDemandKnoten[rp, k, i] == m.pDemandP[rp, k, i] - m.vDSM_Reduction[rp, k, i]
    
    model.cNetPowerDemandKnoten = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, rule=net_power_demand_eq)

    # 3. Gleichung: Gesamtlast = Summe aller Knotenlasten
    def total_net_demand_eq(m, rp, k):
        return m.eTotalNetDemand[rp, k] == sum(m.eNetPowerDemandKnoten[rp, k, i] for i in m.i)
    
    model.cTotalNetDemand = pyo.Constraint(model.rp, model.constraintsActiveK, rule=total_net_demand_eq)

    # Zielfunktions-Rückgabe (Erwartet vom LEGO-Framework)
    first_stage_objective = 0.0
    model.objective.expr += first_stage_objective
    return first_stage_objective

#pDemandP wird derzeit noch nicht reduziert