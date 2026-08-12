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


from xml.parsers.expat import model

import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGO, LEGOUtilities

printer = Printer.getInstance()

@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    """
    Definiert den DSM-Parameter (maximales positives Reduktionspotenzial je rp/k/Knoten, aus Excel)
    sowie die DSM-Reduktionsvariable. Die oberen Schranken der Variable werden hier noch NICHT gesetzt,
    da sie von der Gesamtlast aller Knoten abhängen - das passiert dynamisch in add_constraints().
    """
    first_stage_variables = []
    second_stage_variables = []

    # Einlesen des DSM-Potenzials aus Excel (Power_DSM.xlsx -> cs.dPower_DSM_pos)
    model.pDSM_pos = pyo.Param(model.rp, model.constraintsActiveK, model.i, initialize=cs.dPower_DSM_pos['value'], default=0.0, doc="Maximum positive DSM reduction potential per node and timestep")

    # Einlesen von DSM_Ramping und DSM_activation_time aus Excel (Power_DSM_Ramping.xlsx -> cs.dPower_DSM_Ramping), statisch pro Knoten
    model.pDSM_Ramping = pyo.Param(model.i, initialize=cs.dPower_DSM_Ramping['DSM_Ramping'], default=0.0, doc="Maximum DSM ramping rate per node")
    model.pDSM_ActivationTime = pyo.Param(model.i, initialize=cs.dPower_DSM_Ramping['DSM_activation_time'], default=0.0, doc="DSM activation time per node")

    # Kontrollausgabe im Terminal, um die eingelesenen Werte gegen die Excel-Datei zu prüfen
    printer.information("DSM_Ramping / DSM_activation_time eingelesen (erste 10 Knoten):")
    for i in list(model.i)[:10]:
        printer.information(f"  {i}: DSM_Ramping={pyo.value(model.pDSM_Ramping[i])}, DSM_activation_time={pyo.value(model.pDSM_ActivationTime[i])}")

    # DSM-Variable ohne feste Bounds beim Erstellen (Bounds werden dynamisch in add_constraints gesetzt)
    model.vDSM_Reduction = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Power reduction at bus i through demand-side management")
    second_stage_variables.append(model.vDSM_Reduction)
    return first_stage_variables, second_stage_variables

    

@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    """
    Setzt die oberen Schranken von vDSM_Reduction dynamisch (pro rp/k/Knoten, siehe Berechnung unten)
    und fügt die DSM-Kosten der Zielfunktion hinzu.
    """


# # DSM Schranken fix
#     for rp in model.rp:
#         for k in model.constraintsActiveK:
#             for i in model.i:
#                 base_load = 0.0
#                 if (rp, k, i) in model.pDemandP:
#                     base_load = pyo.value(model.pDemandP[rp, k, i])
            
#                 # Hier weisen wir die Schranken direkt der existierenden Variable zu
#                 model.vDSM_Reduction[rp, k, i].setlb(0.00 * base_load)  # Lower Bound
#                 model.vDSM_Reduction[rp, k, i].setub(0.20 * base_load)  # Upper Bound

# Referenz-Knoten für DSM-Potenzial festlegen
    # -> pDSM_pos[rp, k, reference_node] wird als Gesamt-DSM-Potenzial des Zeitschritts interpretiert
    #    und weiter unten proportional zum Lastanteil jedes Knotens auf alle Knoten verteilt.
    reference_node = "kn001_IN"  # <-steht für kn001 = die maximale positive DSM-Reduktion, die in der Excel-Datei angegeben ist. Dies ist ein Beispiel und sollte entsprechend angepasst werden.

    # DSM Schranken dynamisch basierend auf dem Anteil der Last jedes Knotens an der Gesamtlast berechnen
    for rp in model.rp:
        for k in model.constraintsActiveK:
            # Gesamtlast aller Knoten in diesem Zeitschritt berechnen
            total_demand = sum(
                pyo.value(model.pDemandP[rp, k, j])
                for j in model.i
                if (rp, k, j) in model.pDemandP
            )

            # Maximales DSM-Potenzial für diesen Zeitschritt (von Knoten 1)
            dsm_potential_total = pyo.value(model.pDSM_pos[rp, k, reference_node])

            for idx, i in enumerate(model.i):
                base_load = 0.0
                if (rp, k, i) in model.pDemandP:
                    base_load = pyo.value(model.pDemandP[rp, k, i])

                # Anteil des Knotens an der Gesamtlast
                if total_demand > 0:
                    node_share = base_load / total_demand
                else:
                    node_share = 0.0

                dsm_potential_node = dsm_potential_total * node_share

                if rp == "rp01" and k == "k00001" and idx < 10: print(f"{i}: base_load={base_load:.4f}, node_share={node_share:.4f}, dsm_potential_node={dsm_potential_node:.4f}")  # Debug-Ausgabe für ersten Zeitschritt

                model.vDSM_Reduction[rp, k, i].setlb(0.0)
                model.vDSM_Reduction[rp, k, i].setub(dsm_potential_node)

    # Zielfunktions-Rückgabe + Kosten für DSM
    first_stage_objective = 0.0
    # DSM-Kosten: Summe aller Reduktionen über alle Knoten/Zeitschritte/rps, gewichtet mit pWeight_rp/pWeight_k,
    # multipliziert mit einem Kostensatz pro MW Reduktion (aktuell hardgecodet, siehe TODO unten)
    second_stage_objective = sum(model.pWeight_rp[rp] *
                                 sum(model.pWeight_k[k] *
                                     sum(model.vDSM_Reduction[rp, k, i]
                                         for i in model.i)
                                     for k in model.constraintsActiveK)
                                 for rp in model.rp) * 0.01        #  DSM-Kostensatz, hardgecodete kosten

    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective


#Datein einlesen DGA 17, 3 untermenüs immer kobieren und namen und indices ändern statt g hab ich i, auch bei Liste in CaseStudy, durchsuchen, dort mehr
#cs = cs.filter_timestamps kürzt Zeitabschnitte zusammen, deswegen in casestudy 22 meinen excel namen dazugeben
# wird im excel alles in MW angegeben
# RES haben auch energyrückgewinnung