

from xml.parsers.expat import model

import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGO, LEGOUtilities

printer = Printer.getInstance()


def _shift_window(model, k, activation_time_param, i):
    """
    Liefert die Liste der Zeitschritte kk, die bis zu ActivationTime[i] Schritte vor k liegen (inkl. k
    selbst), zyklisch innerhalb der rp. Wird sowohl für die Rückzahl-Deadline-Constraint als auch für die
    Schrankenberechnung der Rückzahlungsvariablen verwendet.
    """
    horizon = len(model.constraintsActiveK)
    activation_time = int(pyo.value(activation_time_param[i]))
    # Fenstergröße: ActivationTime Zeitschritte davor + k selbst, gedeckelt auf die volle Zykluslänge
    # (falls ActivationTime >= horizon, ist ohnehin der gesamte Zyklus erreichbar).
    window_size = min(activation_time + 1, horizon)

    # first_index (ggf. mehrfach um den Zyklus verschoben) in den gültigen Bereich [1, horizon]
    # normalisieren, damit set_range_cyclic auch bei ActivationTime-Werten nahe an oder über der
    # Fensterlänge nicht abbricht.
    first_index_raw = model.constraintsActiveK.ord(k) - window_size + 1
    first_index = ((first_index_raw - 1) % horizon) + 1
    last_index = model.constraintsActiveK.ord(k)
    if last_index < first_index:
        last_index += horizon

    return LEGOUtilities.set_range_cyclic(model.constraintsActiveK, first_index, last_index)


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    """
    Definiert die DSM-Parameter (maximaler Anteil des Knotenbedarfs, der je rp/k/Knoten in positiver
    Richtung reduziert bzw. in negativer Richtung erhöht werden darf, sowie Ramping-Verzögerung und
    maximale Aktivierungsdauer je Knoten und Richtung, aus Excel) sowie die DSM-Variablen: Aktivierung
    (Reduktion/Erhöhung), Rückzahlung (Erhöhung/Verringerung), den ausstehenden "Schulden"-Zustand
    (vDSM_Bank/vDSM_BankNeg, siehe add_constraints) je Richtung, und den binären Richtungsschalter
    (bDSM_NegMode), der die beiden Banken je Knoten/Zeitschritt gegenseitig ausschließt (siehe
    eDSM_ExclusivityPos/Neg in add_constraints). Die oberen Schranken werden direkt hier gesetzt, da sie
    nur von den bereits bekannten Excel-Werten (pDSM_pos, pDSM_neg, pDemandP) abhängen.
    """
    first_stage_variables = []
    second_stage_variables = []

    # Einlesen des DSM-Potenzials aus Excel, als Anteil [p.u.] des Knotenbedarfs
    model.pDSM_pos = pyo.Param(model.rp, model.constraintsActiveK, model.i, initialize=cs.dPower_DSM_pos['value'], default=0.0, doc="Maximum positive DSM reduction potential per node and timestep [p.u. of demand]")
    model.pDSM_neg = pyo.Param(model.rp, model.constraintsActiveK, model.i, initialize=cs.dPower_DSM_neg['value'], default=0.0, doc="Maximum negative DSM increase potential per node and timestep [p.u. of demand]")

    # Einlesen von Ramping und ActivationTime aus Excel (Power_DSM_Ramping.xlsx -> cs.dPower_DSM_Ramping), statisch pro Knoten, in Zeitschritten - getrennt für positive und negative Richtung
    model.pDSM_Ramping_pos = pyo.Param(model.i, initialize=cs.dPower_DSM_Ramping['DSM_Ramping_pos'], default=0.0, doc="Anlaufverzögerung (in Zeitschritten) nach Aktivierung, bevor DSM (positiv) überhaupt reduzieren kann")
    model.pDSM_ActivationTime_pos = pyo.Param(model.i, initialize=cs.dPower_DSM_Ramping['DSM_activation_time_pos'], default=0.0, doc="Maximum window (in timesteps) within which a positive DSM reduction must start being paid back")
    model.pDSM_Ramping_neg = pyo.Param(model.i, initialize=cs.dPower_DSM_Ramping['DSM_Ramping_neg'], default=0.0, doc="Anlaufverzögerung (in Zeitschritten) nach Aktivierung, bevor DSM (negativ) überhaupt erhöhen kann")
    model.pDSM_ActivationTime_neg = pyo.Param(model.i, initialize=cs.dPower_DSM_Ramping['DSM_activation_time_neg'], default=0.0, doc="Maximum window (in timesteps) within which a negative DSM increase must start being paid back")

    # Kontrollausgabe im Terminal, um die eingelesenen Werte gegen die Excel-Datei zu prüfen. pDSM_pos/pDSM_neg
    # sind zeitabhängig (rp/k/i) - hier wird exemplarisch der erste rp/k-Zeitpunkt gezeigt, damit die Zeile
    # kompakt bleibt. Für eine vollständige Prüfung über alle rp/k (und die tatsächlich gelösten
    # vDSM_*-Variablen, die es zum Zeitpunkt dieser Ausgabe noch gar nicht gibt) siehe die gleichnamigen
    # Tabellen in model.sqlite nach dem Solve.
    sample_rp = list(model.rp)[0]
    sample_k = list(model.constraintsActiveK)[0]
    printer.information("DSM_Ramping / DSM_activation_time eingelesen (erste 10 Knoten):")
    for i in list(model.i)[:10]:
        demand_sample = pyo.value(model.pDemandP[sample_rp, sample_k, i])
        dsm_pos_sample = pyo.value(model.pDSM_pos[sample_rp, sample_k, i])
        dsm_neg_sample = pyo.value(model.pDSM_neg[sample_rp, sample_k, i])
        printer.information(
            f"  {i}: pos: Ramping={pyo.value(model.pDSM_Ramping_pos[i])}, ActivationTime={pyo.value(model.pDSM_ActivationTime_pos[i])}, "
            f"pDSM_pos[{sample_rp},{sample_k}]={dsm_pos_sample:.4f} p.u. -> {dsm_pos_sample * demand_sample:.4f} MW | "
            f"neg: Ramping={pyo.value(model.pDSM_Ramping_neg[i])}, ActivationTime={pyo.value(model.pDSM_ActivationTime_neg[i])}, "
            f"pDSM_neg[{sample_rp},{sample_k}]={dsm_neg_sample:.4f} p.u. -> {dsm_neg_sample * demand_sample:.4f} MW "
            f"(pDemandP[{sample_rp},{sample_k}]={demand_sample:.4f} MW)")

    model.vDSM_pos = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Power reduction at bus i through demand-side management", bounds=(0, None))
    second_stage_variables.append(model.vDSM_pos)

    model.vDSM_pos_payback = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Power increase at bus i to pay back a prior DSM reduction", bounds=(0, None))
    second_stage_variables.append(model.vDSM_pos_payback)

    model.vDSM_neg = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Voluntary power increase at bus i through negative demand-side management", bounds=(0, None))
    second_stage_variables.append(model.vDSM_neg)

    model.vDSM_neg_payback = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Power decrease at bus i to pay back a prior negative DSM increase", bounds=(0, None))
    second_stage_variables.append(model.vDSM_neg_payback)

    model.vDSM_Bank = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Outstanding (not yet paid back) positive DSM reduction debt at bus i", bounds=(0, None))
    second_stage_variables.append(model.vDSM_Bank)

    model.vDSM_BankNeg = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Outstanding (not yet paid back) negative DSM increase debt at bus i", bounds=(0, None))
    second_stage_variables.append(model.vDSM_BankNeg)

    model.bDSM_NegMode = pyo.Var(model.rp, model.constraintsActiveK, model.i, domain=pyo.Binary,
        doc="1 falls Knoten i in Zeitschritt k eine offene negative DSM-Bank haben darf (positive Bank muss dann 0 sein), 0 falls eine offene positive Bank erlaubt ist (negative Bank muss dann 0 sein)")
    second_stage_variables.append(model.bDSM_NegMode)

    # Obere Schranke je Knoten/Zeitschritt für die Aktivierungsvariablen (vDSM_pos/vDSM_neg):
    # Anteil pDSM_pos/pDSM_neg des Knotenbedarfs im jeweils EIGENEN Zeitschritt. Innerhalb der jeweiligen
    # Ramping-Anlaufverzögerung (die ersten pDSM_Ramping_pos/neg[i] Zeitschritte je rp) kann DSM noch
    # nicht aktiviert werden - die Aktivierungsvariable bleibt dort auf 0 fixiert, danach springt die
    # Schranke auf den vollen Wert.
    #
    # Für die Rückzahlungsvariablen (vDSM_pos_payback/vDSM_neg_payback) ist die eigene Last des
    # Rückzahlungs-Zeitschritts NICHT die richtige Größe (das war der ursprüngliche Fehler) - die
    # Schranke muss stattdessen widerspiegeln, was in den Zeitschritten, in denen tatsächlich reduziert
    # bzw. erhöht wurde, überhaupt verfügbar ist. Dafür wird hier die Summe des Potenzials über das
    # gleiche ActivationTime-Fenster verwendet, das auch die eDSM_BankDeadline-Constraint (in
    # add_constraints) begrenzt - eine sichere, nie zu enge obere Schranke, deren eigentliche Präzision
    # aus der vDSM_Bank-Zustandsbilanz kommt, nicht aus dieser Schranke selbst.
    for rp in model.rp:
        for k in model.constraintsActiveK:
            for i in model.i:
                max_shift_pos = pyo.value(model.pDSM_pos[rp, k, i]) * pyo.value(model.pDemandP[rp, k, i])
                if model.constraintsActiveK.ord(k) <= pyo.value(model.pDSM_Ramping_pos[i]):
                    model.vDSM_pos[rp, k, i].setub(0)  # noch in der Anlaufverzögerung
                else:
                    model.vDSM_pos[rp, k, i].setub(max_shift_pos)

                window_pos = _shift_window(model, k, model.pDSM_ActivationTime_pos, i)
                max_payback_pos = sum(pyo.value(model.pDSM_pos[rp, kk, i]) * pyo.value(model.pDemandP[rp, kk, i]) for kk in window_pos)
                model.vDSM_pos_payback[rp, k, i].setub(max_payback_pos)
                model.vDSM_Bank[rp, k, i].setub(max_payback_pos)

                max_shift_neg = pyo.value(model.pDSM_neg[rp, k, i]) * pyo.value(model.pDemandP[rp, k, i])
                if model.constraintsActiveK.ord(k) <= pyo.value(model.pDSM_Ramping_neg[i]):
                    model.vDSM_neg[rp, k, i].setub(0)  # noch in der Anlaufverzögerung
                else:
                    model.vDSM_neg[rp, k, i].setub(max_shift_neg)

                window_neg = _shift_window(model, k, model.pDSM_ActivationTime_neg, i)
                max_payback_neg = sum(pyo.value(model.pDSM_neg[rp, kk, i]) * pyo.value(model.pDemandP[rp, kk, i]) for kk in window_neg)
                model.vDSM_neg_payback[rp, k, i].setub(max_payback_neg)
                model.vDSM_BankNeg[rp, k, i].setub(max_payback_neg)

    return first_stage_variables, second_stage_variables



@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    """
    Verfolgt DSM-Aktivierung und -Rückzahlung je Knoten und Richtung über einen Schulden-Zustand
    (vDSM_Bank/vDSM_BankNeg), analog zum Speicherstand in storage.py (eStIntraRes_rule): jede Aktivierung
    (Reduktion bzw. Erhöhung) erhöht den Schuldenstand, jede Rückzahlung senkt ihn, zyklisch fortgeführt
    über constraintsActiveK.prevw(k).

    Rückzahlung darf also sofort beginnen, muss aber spätestens in dem Zeitschritt begonnen haben, der
    ActivationTime Schritte nach der (durch Ramping bereits verzögerten) Aktivierung liegt - danach darf
    sie sich aber über mehrere weitere Zeitschritte hinziehen. Das wird dadurch erzwungen, dass der
    Schuldenstand zu keinem Zeitpunkt größer sein darf als das, was innerhalb der letzten ActivationTime+1
    Zeitschritte aktiviert wurde (eDSM_BankDeadline) - Schulden, die älter als ActivationTime sind, dürfen
    also nicht mehr offen sein, müssen also spätestens dann angefangen haben zurückgezahlt zu werden.

    Die Summe aller Aktivierungen entspricht dadurch automatisch der Summe aller Rückzahlungen je
    rp/Knoten/Richtung (DSM ist reine Lastverschiebung, keine Netto-Änderung) - eine explizite
    Gesamtbilanz-Constraint ist nicht mehr nötig, da sie sich aus der zyklischen Schuldenbilanz ergibt
    (Summe über alle k der Bilanzgleichung hebt die vDSM_Bank-Terme gegenseitig auf).

    Zusätzlich sind die beiden Richtungen je Knoten/Zeitschritt symmetrisch exklusiv (eDSM_ExclusivityPos/
    Neg, analog zu storage.py's eExclusiveChargeDischarge): eine offene negative Bank ist nur erlaubt,
    wenn keine positive Bank offen ist, und umgekehrt. Ein neuer Zyklus in der jeweils anderen Richtung
    kann also erst beginnen, nachdem die laufende Bank vollständig auf 0 zurückgezahlt wurde - beliebig
    viele Wechsel über den Horizont hinweg sind möglich, solange sich die beiden Zyklen nie zeitlich
    überlappen. Der binäre Schalter bDSM_NegMode entscheidet je Knoten/Zeitschritt, welche der beiden
    Banken offen sein darf; jede Seite wird dabei durch ihre eigene, bereits bekannte enge Schranke (.ub)
    begrenzt statt durch eine generische Big-M-Konstante.
    """

    def eDSM_BankBalance_rule(model, rp, k, i):
        prev_k = model.constraintsActiveK.prevw(k)
        return model.vDSM_Bank[rp, k, i] == model.vDSM_Bank[rp, prev_k, i] + model.vDSM_pos[rp, k, i] - model.vDSM_pos_payback[rp, k, i]

    model.eDSM_BankBalance = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Cyclic outstanding-debt balance for positive DSM", rule=eDSM_BankBalance_rule)

    def eDSM_BankDeadline_rule(model, rp, k, i):
        window = _shift_window(model, k, model.pDSM_ActivationTime_pos, i)
        return model.vDSM_Bank[rp, k, i] <= sum(model.vDSM_pos[rp, kk, i] for kk in window)

    model.eDSM_BankDeadline = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Outstanding positive DSM debt older than ActivationTime must already be paid back", rule=eDSM_BankDeadline_rule)

    def eDSM_BankBalanceNeg_rule(model, rp, k, i):
        prev_k = model.constraintsActiveK.prevw(k)
        return model.vDSM_BankNeg[rp, k, i] == model.vDSM_BankNeg[rp, prev_k, i] + model.vDSM_neg[rp, k, i] - model.vDSM_neg_payback[rp, k, i]

    model.eDSM_BankBalanceNeg = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Cyclic outstanding-debt balance for negative DSM", rule=eDSM_BankBalanceNeg_rule)

    def eDSM_BankDeadlineNeg_rule(model, rp, k, i):
        window = _shift_window(model, k, model.pDSM_ActivationTime_neg, i)
        return model.vDSM_BankNeg[rp, k, i] <= sum(model.vDSM_neg[rp, kk, i] for kk in window)

    model.eDSM_BankDeadlineNeg = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Outstanding negative DSM debt older than ActivationTime must already be paid back", rule=eDSM_BankDeadlineNeg_rule)

    # Symmetrische Exklusivität: pro Knoten und Zeitschritt darf höchstens eine der beiden Banken offen sein
    # (positiv ODER negativ, nie beide gleichzeitig) - ein neuer Zyklus in der jeweils anderen Richtung darf
    # also erst beginnen, wenn die laufende Bank vollständig auf 0 zurückgezahlt wurde. Analog zu
    # storage.py's eExclusiveChargeDischarge: bDSM_NegMode schaltet um, jede Seite wird durch ihre eigene,
    # bereits bekannte enge Schranke (.ub, siehe _shift_window-Summen oben) begrenzt statt durch eine
    # generische Big-M-Konstante.
    def eDSM_ExclusivityPos_rule(model, rp, k, i):
        return model.vDSM_Bank[rp, k, i] <= model.vDSM_Bank[rp, k, i].ub * (1 - model.bDSM_NegMode[rp, k, i])

    model.eDSM_ExclusivityPos = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Positive DSM-Bank darf nur offen sein, wenn keine negative DSM-Bank offen ist", rule=eDSM_ExclusivityPos_rule)

    def eDSM_ExclusivityNeg_rule(model, rp, k, i):
        return model.vDSM_BankNeg[rp, k, i] <= model.vDSM_BankNeg[rp, k, i].ub * model.bDSM_NegMode[rp, k, i]

    model.eDSM_ExclusivityNeg = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Negative DSM-Bank darf nur offen sein, wenn keine positive DSM-Bank offen ist", rule=eDSM_ExclusivityNeg_rule)

    # Zielfunktions-Rückgabe + Kosten für DSM
    first_stage_objective = 0.0
    # DSM-Kosten: nur auf die Aktivierung (Reduktion bzw. Erhöhung), nicht auf die jeweilige Rückzahlung -
    # gilt für beide Richtungen (positiv: vDSM_pos, negativ: vDSM_neg) - Summe aller
    # Aktivierungen über alle Knoten/Zeitschritte/rps, gewichtet mit pWeight_rp/pWeight_k, multipliziert
    # mit einem Kostensatz pro MW Aktivierung (bewusst hardgecodet)
    second_stage_objective = sum(model.pWeight_rp[rp] *
                                 sum(model.pWeight_k[k] *
                                     sum(model.vDSM_pos[rp, k, i] + model.vDSM_neg[rp, k, i]
                                         for i in model.i)
                                     for k in model.constraintsActiveK)
                                 for rp in model.rp) * 0.0        #  DSM-Kostensatz, hardgecodete kosten

    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective


#Datein einlesen DGA 17, 3 untermenüs immer kobieren und namen und indices ändern statt g hab ich i, auch bei Liste in CaseStudy, durchsuchen, dort mehr
#cs = cs.filter_timestamps kürzt Zeitabschnitte zusammen, deswegen in casestudy 22 meinen excel namen dazugeben
# wird im excel alles in MW angegeben
# RES haben auch energyrückgewinnung
