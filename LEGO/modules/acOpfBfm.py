import numpy as np
import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGOUtilities
from LEGO.modules.LoopDetectionBfm import detect_cycle_basis

printer = Printer.getInstance()


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> tuple[list[pyo.Var], list[pyo.Var]]:
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.hindex = cs.dPower_Hindex.index

    # Helper function for creating reverse and bidirectional sets
    def make_reverse_set(original_set):
        reverse = []
        for (i, j, c) in original_set:
            reverse.append((j, i, c))
        return reverse

    model.la_reverse = pyo.Set(doc='Reverse lines for la', initialize=lambda m: make_reverse_set(m.la), dimen=3)
    model.la_no_c = pyo.Set(doc='All lines without circuit dependency', initialize=lambda m: {(i, j) for (i, j, c) in m.la}, dimen=2)
    model.la_full = pyo.Set(doc='All lines incl. reverse lines', initialize=lambda m: set(m.la) | set(m.la_reverse), dimen=3)
    model.la_full_no_c = pyo.Set(doc='All lines incl. reverse lines without circuit dependency', initialize=lambda m: {(i, j) for (i, j, c) in m.la_full}, dimen=2)

    model.le_reverse = pyo.Set(doc='Reverse lines for le', initialize=lambda m: make_reverse_set(m.le), within=model.la_reverse, dimen=3)
    model.le_full = pyo.Set(doc='Existing lines incl. reverse lines', initialize=lambda m: set(m.le) | set(m.le_reverse), within=model.la_full, dimen=3)
    model.le_no_c = pyo.Set(doc='Existing lines without circuit dependency', initialize=lambda m: {(i, j) for (i, j, c) in m.le}, dimen=2)

    model.lc_reverse = pyo.Set(doc='Reverse lines for lc', initialize=lambda m: make_reverse_set(m.lc), within=model.la_reverse, dimen=3)
    model.lc_full = pyo.Set(doc='Candidate lines incl. reverse lines', initialize=lambda m: set(m.lc) | set(m.lc_reverse), within=model.la_full, dimen=3)
    model.lc_full_no_c = pyo.Set(doc='Candidate lines incl. reverse lines without circuit dependency', initialize=lambda m: {(i, j) for (i, j, c) in m.lc_full}, dimen=2)
    model.lc_no_c = pyo.Set(doc='Candidate lines without circuit dependency', initialize=lambda m: {(i, j) for (i, j, c) in m.lc}, dimen=2)

    # node -> list of directed lines where node is sending/receiving
    model.la_outflows = {node: [] for node in model.i}
    model.la_inflows = {node: [] for node in model.i}
    for (i, j, c) in model.la:
        model.la_outflows[i].append((i, j, c))
        model.la_inflows[j].append((i, j, c))

    # Parameters
    model.pBusG = pyo.Param(model.i, initialize=cs.dPower_BusInfo['pBusG'], doc='Conductance of bus i')
    model.pBusB = pyo.Param(model.i, initialize=cs.dPower_BusInfo['pBusB'], doc='Susceptance of bus i')
    model.pBus_pf = pyo.Param(model.i, initialize=cs.dPower_BusInfo['pBus_pf'], doc='PowerFactor of bus i')
    model.pRline = pyo.Param(model.la, initialize=cs.dPower_Network['pRline'], doc='Resistance of line la')
    model.pQmax = pyo.Param(model.la, initialize=lambda model, i, j, c: model.pPmax[i, j, c], doc='Maximum reactive power flow on line la')  # It is asumed that Qmax is ident to Pmax
    model.pBigM_SOCP = pyo.Param(initialize=1e3, doc="Big M for SOCP")
    model.pSOCPTighteningEpsilon = pyo.Param(initialize=1e-4, doc='Small penalty weight for line-current tightening in SOCP BFM')
    model.pVoltageSlackPenaltyFactor = pyo.Param(initialize=0.0001, mutable=True, doc='Voltage slack penalty relative to ENS cost')
    model.pMaxAngleDiff = pyo.Param(initialize=cs.dPower_Parameters["pMaxAngleDiff"] * np.pi / 180, doc='Maximum angle difference between two buses for the SOCP formulation')
    model.pBusMaxV = pyo.Param(model.i, initialize=cs.dPower_BusInfo['pBusMaxV'], doc='Maximum voltage at bus i')
    model.pBusMinV = pyo.Param(model.i, initialize=lambda model, i: max(cs.dPower_BusInfo['pBusMinV'][i], 0.1), doc='Minimum voltage at bus i (with a lower bound of 0.1)')

    if model.pEnableSoftVoltageLimits:
        model.pVoltageBoundsUp = pyo.Param(model.i, initialize=cs.dPower_BusInfo["pVoltageBoundsUp"], doc='Soft Limits for the upper voltage bounds')
        model.pVoltageBoundsLow = pyo.Param(model.i, initialize=cs.dPower_BusInfo["pVoltageBoundsLow"], doc='Soft Limits for the lower voltage bounds')
    else:
        model.pVoltageBoundsUp = pyo.Param(model.i, initialize=model.pBusMaxV, doc='Soft Limits for the upper voltage bounds (set to hard limits when soft voltage limits are disabled)')
        model.pVoltageBoundsLow = pyo.Param(model.i, initialize=model.pBusMinV, doc='Soft Limits for the lower voltage bounds (set to hard limits when soft voltage limits are disabled)')

    model.pRatioDemQP = pyo.Param(model.i, initialize=lambda model, i: pyo.tan(pyo.acos(model.pBus_pf[i])))
    model.pDemandQ = pyo.Param(model.rp, model.k, model.i, initialize=cs.dPowerQ_Demand['value'], doc='Reactive Demand at bus i in representative period rp and timestep k')

    # Todo: Line Investment not impelented in BFM yet
    model.vLineInvest = pyo.Var(model.la, doc='Transmission line investment', domain=pyo.Binary)
    for i, j, c in model.le:
        model.vLineInvest[i, j, c].fix(0)  # Set existing lines to not investable
    first_stage_variables += [model.vLineInvest]

    model.vGenInvest = pyo.Var(model.g, doc="Integer generation investment", bounds=lambda model, g: (0, model.pMaxInvest[g] * model.pEnabInv[g]))
    first_stage_variables += [model.vGenInvest]

    model.vPNS = pyo.Var(model.rp, model.k, model.i, doc='Slack variable power not served', bounds=lambda model, rp, k, i: (0, max(model.pDemandP[rp, k, i], 0)))
    second_stage_variables += [model.vPNS]
    model.vEPS = pyo.Var(model.rp, model.k, model.i, doc='Slack variable excess power served', bounds=(0, None))
    second_stage_variables += [model.vEPS]

    model.vGenP = pyo.Var(model.rp, model.k, model.g, doc='Power output of generator g', bounds=lambda model, rp, k, g: (0, model.pMaxProd[g] * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g])))
    second_stage_variables += [model.vGenP]

    model.vLineP = pyo.Var(model.rp, model.k, model.la, doc='Power flow from bus i to j', bounds=lambda m, rp, k, i, j, c: (-m.pPmax[i, j, c], m.pPmax[i, j, c]) if (i, j, c) in m.la else (-m.pPmax[j, i, c], m.pPmax[j, i, c]))
    second_stage_variables += [model.vLineP]

    model.vLineQ = pyo.Var(model.rp, model.k, model.la, doc="Reactive power flow from bus i to j", bounds=lambda m, rp, k, i, j, c: (-m.pQmax[i, j, c], m.pQmax[i, j, c]) if (i, j, c) in m.le else (-m.pQmax[j, i, c], m.pQmax[j, i, c]) if (i, j, c) in m.le_reverse else (None, None))
    second_stage_variables.append(model.vLineQ)

    model.vSOCP_ui = pyo.Var(model.rp, model.k, model.i, doc='Squared voltage magnitude at bus i', bounds=lambda m, rp, k, i: (m.pVoltageBoundsLow[i] ** 2, m.pVoltageBoundsUp[i] ** 2))
    second_stage_variables.append(model.vSOCP_ui)

    if model.pEnableSoftVoltageLimits:
        model.vSOCP_ui_slack_pos = pyo.Var(model.rp, model.k, model.i, doc='Slack variable to penalize voltage terms near the upper voltage limits', bounds=lambda m, rp, k, i: (0, m.pBusMaxV[i] **2 - m.pVoltageBoundsUp[i] ** 2))
        second_stage_variables.append(model.vSOCP_ui_slack_pos)

        model.vSOCP_ui_slack_neg = pyo.Var(model.rp, model.k, model.i, doc='Slack variable to penalize voltage terms near the lower voltage limits', bounds=lambda m, rp, k, i: (0, m.pVoltageBoundsLow[i] ** 2 - m.pBusMinV[i] ** 2))
        second_stage_variables.append(model.vSOCP_ui_slack_neg)
    else:
         model.vSOCP_ui_slack_pos = pyo.Param(model.rp, model.k, model.i, initialize=0, doc='Slack variable to penalize voltage terms near the upper voltage limits (set to 0 when soft voltage limits are disabled)')
         model.vSOCP_ui_slack_neg = pyo.Param(model.rp, model.k, model.i, initialize=0, doc='Slack variable to penalize voltage terms near the lower voltage limits (set to 0 when soft voltage limits are disabled)')

    model.vSOCP_lij = pyo.Var(model.rp, model.k, model.la, doc='Squared current magnitude on line ij', bounds=lambda m, rp, k, i, j, c: (0, None) if (i, j, c) in m.la else (None, None))  # m.pSijNom[i,j,c]/m.pBusMinV[i]
    second_stage_variables.append(model.vSOCP_lij)

    model.vGenQ = pyo.Var(model.rp, model.k, model.g, doc='Reactive power output of ge', bounds=lambda model, rp, k, g: (model.pMinGenQ[g] * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g]), model.pMaxGenQ[g] * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g])))
    second_stage_variables.append(model.vGenQ)

    model.vQNS = pyo.Var(model.rp, model.k, model.i, doc='Slack variable reactive power not served', bounds=lambda m, rp, k, i: (0, max(m.pDemandQ[rp, k, i], 0)))
    second_stage_variables += [model.vQNS]

    model.vEQS = pyo.Var(model.rp, model.k, model.i, doc='Slack variable excess reactive power served', bounds=(0, None))
    second_stage_variables += [model.vEQS]

    # For each DC-OPF/SOCP "island", set node with highest demand as slack node
    dTechnicalReprIslands = pd.DataFrame(index=cs.dPower_BusInfo.index, columns=[cs.dPower_BusInfo.index], data=False)

    for index, entry in cs.dPower_Network.iterrows():
        if cs.dPower_Network.loc[(index[0], index[1], index[2])]["pTecRepr"] in ["DC-OPF", "SOCP"]:
            dTechnicalReprIslands.loc[index[0], index[1]] = True
            dTechnicalReprIslands.loc[index[1], index[0]] = True
    completed_buses = set()  # Set of buses that have been looked at already

    i = 0
    for index, entry in dTechnicalReprIslands.iterrows():
        if index in completed_buses or entry[entry == True].empty:
            continue
        connected_buses = cs.get_connected_buses(dTechnicalReprIslands, str(index))
        for bus in connected_buses:
            completed_buses.add(bus)
        completed_buses.add(index)

        # Set slack node
        slack_node = cs.dPower_Demand.loc[:, :, connected_buses].groupby('i').sum().idxmax().values[0]
        slack_node = cs.dPower_Parameters["is"]  # TODO: Switch this again to be calculated (fixed to 'is' for compatibility)
        if i == 0: printer.information("Setting slack nodes for technical representation islands:")
        i += 1
        printer.information(f"Zone {i:>2} - Slack node: {slack_node}, other buses: {connected_buses}")
        slack_voltage_squared = cs.dPower_Parameters['pSlackVoltage'] ** 2
        printer.information(" Fixed voltage magnitude at slack node: ", pyo.value(slack_voltage_squared))

        for v in model.vSOCP_ui[:, :, slack_node]:  # set all slack nodes to fixed voltage
            v.setub(slack_voltage_squared)
            v.setlb(slack_voltage_squared)

        model.vSOCP_ui[:, :, slack_node].fix(slack_voltage_squared)
        if model.pEnableSoftVoltageLimits:
            model.vSOCP_ui_slack_neg[:, :, slack_node].fix(0)
            model.vSOCP_ui_slack_pos[:, :, slack_node].fix(0)

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    model.socp_cycle_basis = detect_cycle_basis(model)

    # Define active- and reactive power balance expressions
    def eActivePowerBalance_rule(m, rp, k, i):
        return (sum(m.vGenP[rp, k, g] for g in m.gi_node[i])
                - (m.pDemandP[rp, k, i])
                + m.vPNS[rp, k, i]
                - m.vEPS[rp, k, i]
                + sum(m.vLineP[rp, k, j, i, c] - m.pRline[j, i, c] * m.vSOCP_lij[rp, k, j, i, c] for (j, i2, c) in m.la_inflows[i] if i2 == i)
                - sum(m.vLineP[rp, k, i, j, c] for (i2, j, c) in m.la_outflows[i] if i2 == i)
                )

    def eReactivePowerBalance_rule(m, rp, k, i):
        return (sum(m.vGenQ[rp, k, g] for g in m.gi_node[i])
                - (m.pDemandQ[rp, k, i])
                + m.vQNS[rp, k, i]
                - m.vEQS[rp, k, i]
                + sum(m.vLineQ[rp, k, j, i, c] - m.pXline[j, i, c] * m.vSOCP_lij[rp, k, j, i, c] for (j, i2, c) in m.la_inflows[i] if i2 == i)
                - sum(m.vLineQ[rp, k, i, j, c] for (i2, j, c) in m.la_outflows[i] if i2 == i)
                )

    model.eDC_BalanceP_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.i, rule=eActivePowerBalance_rule)
    model.eSOCP_BalanceQ_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.i, rule=eReactivePowerBalance_rule)

    model.eSOCP_QMaxOut = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc="Max reactive power output of generator unit", rule=lambda m, rp, k, g: (m.vGenQ[rp, k, g] / m.pMaxGenQ[g] <= m.vCommit[rp, k, g]) if m.pMaxGenQ[g] != 0 and (m.pExisUnits[g] > 0 or m.pEnabInv[g] == 1) else pyo.Constraint.Skip)
    model.eSOCP_QMinOut1 = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc="Min positive reactive power output of generator unit", rule=lambda m, rp, k, g: (m.vGenQ[rp, k, g] / m.pMinGenQ[g] >= m.vCommit[rp, k, g]) if m.pMinGenQ[g] >= 0 and (m.pExisUnits[g] > 0 or m.pEnabInv[g] == 1) else pyo.Constraint.Skip)
    model.eSOCP_QMinOut2 = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc="Min negative reactive power output of generator unit", rule=lambda m, rp, k, g: (m.vGenQ[rp, k, g] / m.pMinGenQ[g] <= m.vCommit[rp, k, g]) if m.pMinGenQ[g] <= 0 and (m.pExisUnits[g] > 0 or m.pEnabInv[g] == 1) else pyo.Constraint.Skip)

    def eSOCP_VoltageDrop_rule(m, rp, k, i, j, c):
        return ((m.vSOCP_ui[rp, k, j] + model.vSOCP_ui_slack_pos[rp, k, j] - model.vSOCP_ui_slack_neg[rp, k, j]) ==
                (m.vSOCP_ui[rp, k, i] + model.vSOCP_ui_slack_pos[rp, k, i] - model.vSOCP_ui_slack_neg[rp, k, i]) -
                2 * (m.pRline[i, j, c] * m.vLineP[rp, k, i, j, c] + m.pXline[i, j, c] * m.vLineQ[rp, k, i, j, c]) +
                (m.pRline[i, j, c] ** 2 + m.pXline[i, j, c] ** 2) * m.vSOCP_lij[rp, k, i, j, c])

    model.eSOCP_VoltageDrop = pyo.Constraint(model.rp, model.constraintsActiveK, model.la, doc="SOCP constraints for voltage drop of line", rule=eSOCP_VoltageDrop_rule)

    def eSOCP_FlowDef_rule(m, rp, k, i, j, c):
        if any((i, j, c) in m.la for c in m.c):
            return m.vSOCP_lij[rp, k, i, j, c] * (m.vSOCP_ui[rp, k, i] + m.vSOCP_ui_slack_pos[rp, k, i] - m.vSOCP_ui_slack_neg[rp, k, i])>= m.vLineP[rp, k, i, j, c] ** 2 + m.vLineQ[rp, k, i, j, c] ** 2
        else:
            return pyo.Constraint.Skip

    model.eSOCP_FlowDef = pyo.Constraint(model.rp, model.constraintsActiveK, model.la, doc="SCOP constraints for existing lines (for AC-OPF) original set", rule=eSOCP_FlowDef_rule)

    model.eSOCP_CycleAngleLin = pyo.ConstraintList(doc='Linearized meshed-network cycle constraints for SOCP BFM')
    for cycle_edges in model.socp_cycle_basis:
        for rp in model.rp:
            for k in model.constraintsActiveK:
                model.eSOCP_CycleAngleLin.add(
                    sum(sign * (model.pXline[i, j, c] * model.vLineP[rp, k, i, j, c] - model.pRline[i, j, c] * model.vLineQ[rp, k, i, j, c])
                        for sign, i, j, c in cycle_edges) == 0
                )

    def eSOCP_VoltageLimitSlack_rule1(m, rp, k, i):
        return (m.vSOCP_ui[rp, k, i] - m.vSOCP_ui_slack_neg[rp, k, i]   >= m.pBusMinV[i] ** 2)

    def eSOCP_VoltageLimitSlack_rule2(m, rp, k, i):
        return (m.vSOCP_ui[rp, k, i] + m.vSOCP_ui_slack_pos[rp, k, i] <= m.pBusMaxV[i] ** 2)

    model.eSOCP_VoltageLimitSlack = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="SOCP constraints for voltage limits with slack variables", rule=eSOCP_VoltageLimitSlack_rule1)
    model.eSOCP_VoltageLimitSlack2 = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="SOCP constraints for voltage limits with slack variables", rule=eSOCP_VoltageLimitSlack_rule2)
    # FACTS (not yet Implemented) TODO: Add FACTS as a set, add FACTS parameters to nodes i
    if cs.dPower_Parameters["pEnableSOCP"] == 99999:
        model.eSOCP_QMinFACTS = pyo.Constraint(model.rp, model.constraintsActiveK, model.facts, doc='min reactive power output of FACTS unit', rule=lambda m, rp, k, i: m.vGenQ[rp, k, i] >= m.pMaxGenQ[i] * (m.pExisUnits[i] + m.vGenInvest[i]))
        model.eSOCP_QMaxFACTS = pyo.Constraint(model.rp, model.constraintsActiveK, model.facts, doc='max reactive power output of FACTS unit', rule=lambda m, rp, k, i: m.vGenQ[rp, k, i] <= m.pMaxGenQ[i] * (m.pExisUnits[i] + m.vGenInvest[i]))

    model.eDC_BalanceP = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc='Power balance constraint for each bus', rule=lambda m, rp, k, i: m.eDC_BalanceP_expr[rp, k, i] == 0)
    model.eSOCP_BalanceQ = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc='Reactive power balance constraint for each bus', rule=lambda m, rp, k, i: m.eSOCP_BalanceQ_expr[rp, k, i] == 0)

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = (sum(model.pFixedCost[i, j, c] * model.vLineInvest[i, j, c] for i, j, c in model.lc))  # Investment cost of transmission lines

    def reactive_slack_terms(rp, k):
        return sum(
            model.vQNS[rp, k, i] * model.pENSCost
            + model.vEQS[rp, k, i] * model.pENSCost * 2
            for i in model.i
        )

    model.eSOCPVoltageSlack = pyo.Expression(
        expr=(sum(model.pWeight_rp[rp] *
                  sum(model.pWeight_k[k] *
                      sum(model.vSOCP_ui_slack_pos[rp, k, i] + model.vSOCP_ui_slack_neg[rp, k, i]
                          for i in model.i)
                      for k in model.constraintsActiveK)
                  for rp in model.rp)
              if model.pEnableSoftVoltageLimits else 0.0),
        doc="Weighted voltage-limit violation in squared per-unit voltage",
    )
    model.eSOCPVoltageSlackPenalty = pyo.Expression(
        expr=model.pENSCost * model.pVoltageSlackPenaltyFactor * model.eSOCPVoltageSlack,
        doc="Voltage-limit violation contribution to the standard objective",
    )

    model.eSOCPTighteningLoss = pyo.Expression(
        expr=sum(model.pWeight_rp[rp] *
                 sum(model.pWeight_k[k] *
                     sum(model.pRline[i, j, c] * model.vSOCP_lij[rp, k, i, j, c] for i, j, c in model.la)
                     for k in model.constraintsActiveK)
                 for rp in model.rp),
        doc="Weighted active line losses used for SOCP tightening",
    )

    second_stage_objective = sum(model.pWeight_rp[rp] *
                                 sum(model.pWeight_k[k] * reactive_slack_terms(rp, k)
                                     for k in model.constraintsActiveK)
                                 for rp in model.rp)
    second_stage_objective += model.eSOCPVoltageSlackPenalty
    second_stage_objective += model.pSOCPTighteningEpsilon * model.eSOCPTighteningLoss

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
