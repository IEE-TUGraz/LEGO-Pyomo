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

    # Definition der DSM-Variable ohne feste Bounds beim Erstellen
    model.vDSM_Reduction = pyo.Var(model.rp, model.constraintsActiveK, model.i)
    
    second_stage_variables.append(model.vDSM_Reduction)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    
    # 1. Variablen definieren für den Datenbank-Export
    model.vNetPowerDemandKnoten = pyo.Var(model.rp, model.constraintsActiveK, model.i)
    model.vTotalNetDemand = pyo.Var(model.rp, model.constraintsActiveK)

    # 2. Gleichung: Netto-Last pro Knoten = Echte Excel-Last (pDemandP) - DSM
    def net_power_demand_eq(m, rp, k, i):
        if (rp, k, i) in m.pDemandP:
            return m.vNetPowerDemandKnoten[rp, k, i] == m.pDemandP[rp, k, i] - m.vDSM_Reduction[rp, k, i]
        return m.vNetPowerDemandKnoten[rp, k, i] == 0.0 - m.vDSM_Reduction[rp, k, i]
    
    model.cNetPowerDemandKnoten = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, rule=net_power_demand_eq)

    # 3. Gleichung: Gesamtlast = Summe aller Knotenlasten
    def total_net_demand_eq(m, rp, k):
        return m.vTotalNetDemand[rp, k] == sum(m.vNetPowerDemandKnoten[rp, k, i] for i in m.i)
    
    model.cTotalNetDemand = pyo.Constraint(model.rp, model.constraintsActiveK, rule=total_net_demand_eq)
    
# 4. HÄRTERE DSM-SCHRANKE ALS CONSTRAINT (Nutzt jetzt auch sicher pDemandP)
    def dsm_force_reduction_eq(m, rp, k, i):
        base_load = 0.0
        if (rp, k, i) in m.pDemandP:
            base_load = pyo.value(m.pDemandP[rp, k, i])
        
        # Bereich vorgeben: von Untergrenze bis Obergrenze
        lower_bound = 0.00 * base_load  # Mindestens 0% (keine Erhöhung)
        upper_bound = 0.20 * base_load  # Maximal 20% Reduktion
        
        return lower_bound <= m.vDSM_Reduction[rp, k, i] <= upper_bound

    model.cDsmBoundsEnforced = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, rule=dsm_force_reduction_eq)

# Zur Kontrolle ob DSM eine gesamt Reduktion beeinflusst

    # 1. Echte Variable für den Datenbank-Export definieren
    model.vErgebnisDC_BalanceP = pyo.Var(model.rp, model.constraintsActiveK, model.i)

    # 2. Gleichung: Die Variable spiegelt exakt den aktuellen Wert der Expression wider
    def export_balance_expr_eq(m, rp, k, i):
        # Wir weisen der Variable den mathematischen Ausdruck (Expression) direkt zu
        return m.vErgebnisDC_BalanceP[rp, k, i] == m.eDC_BalanceP_expr[rp, k, i]

    # WICHTIG: Damit Gurobi weiß, dass diese Variable nur ein "Spiegel" ist,
    # muss sie genau für die gleichen Indizes wie die Expression gebaut werden.
    model.cExportBalanceExpr = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, rule=export_balance_expr_eq)


    # Zielfunktions-Rückgabe (Erwartet vom LEGO-Framework)
    first_stage_objective = 0.0
    model.objective.expr += first_stage_objective
    return first_stage_objective
