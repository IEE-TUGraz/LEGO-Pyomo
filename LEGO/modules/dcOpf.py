import numpy as np
import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGOUtilities

printer = Printer.getInstance()


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []
  
    # Pre-partition line sets by technical representation (OPTIMIZATION: eliminates match/case in constraint rules)
    le_dcopf = [idx for idx in model.le if cs.dPower_Network.loc[idx]["pTecRepr"] == "DC-OPF"]
    lc_dcopf = [idx for idx in model.lc if cs.dPower_Network.loc[idx]["pTecRepr"] == "DC-OPF"]
    lc_dcopf_tp_sn = [idx for idx in model.lc if cs.dPower_Network.loc[idx]["pTecRepr"] in ["DC-OPF", "TP", "SN"]]
    model.le_dcopf = pyo.Set(doc='Existing DC-OPF lines', initialize=le_dcopf, within=model.le)
    model.lc_dcopf = pyo.Set(doc='Candidate DC-OPF lines', initialize=lc_dcopf, within=model.lc)
    model.lc_dcopf_tp_sn = pyo.Set(doc='Candidate DC-OPF/TP/SN lines', initialize=lc_dcopf_tp_sn, within=model.lc)
 
    model.p = pyo.Set(doc='Periods', initialize=cs.dPower_Hindex.index.get_level_values('p').unique().tolist())
    model.rp = pyo.Set(doc='Representative periods', initialize=cs.dPower_Demand.index.get_level_values('rp').unique().tolist())
    model.k = pyo.Set(doc='Timestep within representative period', initialize=cs.dPower_Demand.index.get_level_values('k').unique().tolist())

    model.hindex = cs.dPower_Hindex.index

    # Parameters
    model.coordsLat = pyo.Param(model.i, initialize=cs.dPower_BusInfo['lat'], doc='Latitude of bus i')
    model.coordsLon = pyo.Param(model.i, initialize=cs.dPower_BusInfo['lon'], doc='Longitude of bus i')

    # Variables
    model.vTheta = pyo.Var(model.rp, model.k, model.i, doc='Angle of bus i', bounds=(-cs.dPower_Parameters["pMaxAngleDCOPF"], cs.dPower_Parameters["pMaxAngleDCOPF"]))
    second_stage_variables += [model.vTheta]
    model.vAngle = pyo.Var(model.rp, model.k, model.la, doc='Angle phase shifting transformer', bounds=lambda m, rp, k, i, j, c: (-m.pAngle[i, j, c], m.pAngle[i, j, c]))
    second_stage_variables += [model.vAngle]

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

    model.vLineP = pyo.Var(model.rp, model.k, model.la, doc='Power flow from bus i to j', bounds=lambda m, rp, k, i, j, c: (-m.pPmax[i,j,c], m.pPmax[i, j, c]) if (i, j, c) in m.la else (-m.pPmax[j, i, c], m.pPmax[j, i, c]))
    second_stage_variables += [model.vLineP]

    # For each DC-OPF "island", set node with highest demand as slack node
    dTechnicalReprIslands = pd.DataFrame(index=cs.dPower_BusInfo.index, columns=[cs.dPower_BusInfo.index], data=False)

    for index, entry in cs.dPower_Network.iterrows():
        if cs.dPower_Network.loc[(index[0], index[1], index[2])]["pTecRepr"] in ["DC-OPF", ]:
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


    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    #Power balance for nodes DC
    def eDC_BalanceP_rule(m, rp, k, i):
        return (sum(m.vGenP[rp, k, g] for g in m.gi_node[i])  # Production of generators at bus i (O(1) lookup)
                    + sum(m.vLineP[rp, k, e] if (e[1] == i) else -m.vLineP[rp, k, e] for e in model.la_nodeRelevant[i])  # Add power flow from bus j to bus i and subtract from bus i to bus j
                    - m.pDemandP[rp, k, i]  # Demand at bus i
                    + m.vPNS[rp, k, i]  # Slack variable for demand not served
                    - m.vEPS[rp, k, i])  # Slack variable for overproduction
    
    model.eDC_BalanceP_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.i, rule=eDC_BalanceP_rule)
    model.eDC_BalanceP = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc='Power balance constraint for each bus', rule=lambda m, rp, k, i: m.eDC_BalanceP_expr[rp, k, i] == 0)


    # Use pre-partitioned sets instead of match/case
    def eDC_ExiLinePij_rule(m, rp, k, i, j, c):
        return m.vLineP[rp, k, i, j, c] == (m.vTheta[rp, k, i] - m.vTheta[rp, k, j] + m.vAngle[rp, k, i, j, c]) / (m.pXline[i, j, c] * m.pRatio[i, j, c])

    model.eDC_ExiLinePij = pyo.Constraint(model.rp, model.constraintsActiveK, model.le_dcopf, doc="Power flow existing lines (for DC-OPF)", rule=eDC_ExiLinePij_rule)

    def eDC_CanLinePij1_rule(m, rp, k, i, j, c):
        return (m.vLineP[rp, k, i, j, c] / (m.pBigM_Flow * m.pPmax[i, j, c]) >=
                        (m.vTheta[rp, k, i] - m.vTheta[rp, k, j] + m.vAngle[rp, k, i, j, c]) / (m.pXline[i, j, c] * m.pRatio[i, j, c]) /
                        (m.pBigM_Flow * m.pPmax[i, j, c]) - 1 + m.vLineInvest[i, j, c])

    model.eDC_CanLinePij1 = pyo.Constraint(model.rp, model.constraintsActiveK, model.lc_dcopf, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij1_rule)

    def eDC_CanLinePij2_rule(m, rp, k, i, j, c):
        return (m.vLineP[rp, k, i, j, c] / (m.pBigM_Flow * m.pPmax[i, j, c]) <=
                        (m.vTheta[rp, k, i] - m.vTheta[rp, k, j] + m.vAngle[rp, k, i, j, c]) / (m.pXline[i, j, c] * m.pRatio[i, j, c]) /
                        (m.pBigM_Flow * m.pPmax[i, j, c]) + 1 - m.vLineInvest[i, j, c])

    model.eDC_CanLinePij2 = pyo.Constraint(model.rp, model.constraintsActiveK, model.lc_dcopf, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij2_rule)

    def eDC_LimCanLine1_rule(m, rp, k, i, j, c):
        return m.vLineP[rp, k, i, j, c] / m.pPmax[i, j, c] + m.vLineInvest[i, j, c] >= 0

    model.eDC_LimCanLine1 = pyo.Constraint(model.rp, model.constraintsActiveK, model.lc_dcopf_tp_sn, doc="Power flow limit reverse direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine1_rule)

    def eDC_LimCanLine2_rule(m, rp, k, i, j, c):
        return m.vLineP[rp, k, i, j, c] / m.pPmax[i, j, c] - m.vLineInvest[i, j, c] <= 0

    model.eDC_LimCanLine2 = pyo.Constraint(model.rp, model.constraintsActiveK, model.lc_dcopf_tp_sn, doc="Power flow limit reverse direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine2_rule)


    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = (sum(model.pFixedCost[i, j, c] * model.vLineInvest[i, j, c] for i, j, c in model.lc))  # Investment cost of transmission lines
    

    # Adjust objective and return first_stage_objective expression
    # No changes to second stage objective needed for DC-OPF
    model.objective.expr += first_stage_objective 
    return first_stage_objective
