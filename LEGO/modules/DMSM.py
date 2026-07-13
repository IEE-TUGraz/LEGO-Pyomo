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
    
# DSM Schranken
    for rp in model.rp:
        for k in model.constraintsActiveK:
            for i in model.i:
                base_load = 0.0
            if (rp, k, i) in model.pDemandP:
                base_load = pyo.value(model.pDemandP[rp, k, i])
            
            # Hier weisen wir die Schranken direkt der existierenden Variable zu
            model.vDSM_Reduction[rp, k, i].setlb(0.00 * base_load)  # Lower Bound
            model.vDSM_Reduction[rp, k, i].setub(0.20 * base_load)  # Upper Bound

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


    # Zielfunktions-Rückgabe + Kosten für DSM
    first_stage_objective = 0.0
    second_stage_objective = sum(model.pWeight_rp[rp] *
                                 sum(model.pWeight_k[k] *
                                     sum(model.vDSM_Reduction[rp, k, i]     
                                         for i in model.i)                   
                                     for k in model.constraintsActiveK)
                                 for rp in model.rp) * 0.01        #  DSM-Kostensatz, hardgecodete kosten

    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective

#doc mit übernehmen
#Check im AC eine If gleichung machen falls DSM ausgeschaltet ist sucht er die variable DSM reduction, eine If wenn das aktiv ist dann das mit sonst ohne, wie bei vres Zeile 69 pEnebleDSM
#Check Kosten übergeben damit vernünftigere Werte, minimal DGA Zeile 73 statt r für nodes i
#Datein einlesen DGA 17, 3 untermenüs immer kobieren und namen und indices ändern statt g hab ich i
#cs = cs.filter_timestamps kürzt Zeitabschnitte zusammen, deswegen in casestudy 22 meinen excel namen dazugeben
