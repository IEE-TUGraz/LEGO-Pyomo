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

    # Sets
    model.i = pyo.Set(doc='Buses', initialize=cs.dPower_BusInfo.index.tolist())
    model.slack_node = pyo.Set(doc='Slack bus', initialize=[cs.dPower_Parameters['is']])

    model.c = pyo.Set(doc='Circuits', initialize=cs.dPower_Network.index.get_level_values('c').unique().tolist())
    model.la = pyo.Set(doc='All lines', initialize=cs.dPower_Network.index.tolist(), within=model.i * model.i * model.c)
    model.la_nodeRelevant = {node: [(i, j, c) for (i, j, c) in model.la if node == i or node == j] for node in model.i}
    model.le = pyo.Set(doc='Existing lines', initialize=cs.dPower_Network[(cs.dPower_Network["pEnableInvest"] == 0)].index.tolist(), within=model.la)
    model.le_nodeRelevant = {node: [(i, j, c) for (i, j, c) in model.le if node == i or node == j] for node in model.i}
    model.lc = pyo.Set(doc='Candidate lines', initialize=cs.dPower_Network[(cs.dPower_Network["pEnableInvest"] == 1)].index.tolist(), within=model.la)
    model.lc_nodeRelevant = {node: [(i, j, c) for (i, j, c) in model.lc if node == i or node == j] for node in model.i}
    model.g = pyo.Set(doc='Generators')
    model.gi = pyo.Set(doc='Generator g connected to bus i', within=model.g * model.i)

    model.p = pyo.Set(doc='Periods', initialize=cs.dPower_Hindex.index.get_level_values('p').unique().tolist())
    model.rp = pyo.Set(doc='Representative periods', initialize=cs.dPower_Demand.index.get_level_values('rp').unique().tolist())
    model.k = pyo.Set(doc='Timestep within representative period', initialize=cs.dPower_Demand.index.get_level_values('k').unique().tolist())

    if cs.dGlobal_Parameters["pMovingWindowLength"] > 0 and cs.dGlobal_Parameters["pMovingWindowOverlap"] >= 0:
        model.constraintsActiveK = pyo.Set(doc='Timesteps where constraints are active during the actual time window', initialize=cs.constraints_active_k)
    else:
        model.constraintsActiveK = pyo.Set(doc='Timesteps where constraints are active', initialize=model.k)

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

    # Helper to get the first circuit for each (i, j) pair
    df_circuits = cs.dPower_Network.reset_index()

    # Sort the DataFrame by the desired circuit order
    df_circuits["c_str"] = df_circuits["c"].astype(str)
    ordered_circuits = list(dict.fromkeys(df_circuits["c_str"].tolist()))
    circuit_order = {c: idx for idx, c in enumerate(ordered_circuits)}
    df_circuits["c_order"] = df_circuits["c_str"].map(circuit_order)

    # Get the first circuit per (i, j) based on this order
    first_circuit_map = df_circuits.sort_values("c_order").drop_duplicates(subset=["i", "j"]).set_index(["i", "j"])["c"].to_dict()
    # todo da kommt der fehler
    # DEPRECATED: Param 'first_circuit_map' declared with an implicit
    # domain of 'Any'. The default domain for Param objects is 'Any'.  However, we
    # will be changing that default to 'Reals' in the future.  If you really intend
    # the domain of this Paramto be 'Any', you can suppress this warning by
    # explicitly specifying 'within=Any' to the Param constructor.  (deprecated in
    # 5.6.9, will be removed in (or after) 6.0) (called from
    # C:\Users\Stephan\anaconda3\envs\LEGO-Pyomo_env\Lib\site-
    # packages\pyomo\core\base\indexed_component.py:718)
    model.first_circuit_map = pyo.Param(model.la_no_c, initialize=first_circuit_map, doc='First circuit for each line (i, j)')
    model.first_circuit_map_bidir = pyo.Param(model.la_full_no_c, initialize={(i, j): c for (i, j), c in model.first_circuit_map.items()} | {(j, i): c for (i, j), c in model.first_circuit_map.items()}, doc='First circuit for each line (i, j) bidirectional')

    # Parameters
    model.pDemandP = pyo.Param(model.rp, model.k, model.i, initialize=cs.dPower_Demand['value'], doc='Demand at bus i in representative period rp and timestep k')
    model.pMovWindowLDS = cs.dGlobal_Parameters['pMovWindowLDS']

    model.pOMVarCost = pyo.Param(model.g, doc='Production cost of generator g')
    model.pEnabInv = pyo.Param(model.g, doc='Enable investment in thermal generator g')
    model.pMaxInvest = pyo.Param(model.g, doc='Maximum investment in thermal generator g')
    model.pInvestCost = pyo.Param(model.g, doc='Investment cost for thermal generator g')
    model.pMaxProd = pyo.Param(model.g, doc='Maximum production of generator g')
    model.pMinProd = pyo.Param(model.g, doc='Minimum production of generator g')
    model.pExisUnits = pyo.Param(model.g, doc='Existing units of generator g')
    model.pMaxGenQ = pyo.Param(model.g, doc='Maximum reactive production of generator g')
    model.pMinGenQ = pyo.Param(model.g, doc='Minimum reactive production of generator g')

    model.pXline = pyo.Param(model.la, initialize=cs.dPower_Network['pXline'], doc='Reactance of line la')
    model.pAngle = pyo.Param(model.la, initialize=cs.dPower_Network['pAngle'] * np.pi / 180, doc='Transformer angle shift')
    model.pRatio = pyo.Param(model.la, initialize=cs.dPower_Network['pRatio'], doc='Transformer ratio')
    model.pPmax = pyo.Param(model.la, initialize=cs.dPower_Network['pPmax'], doc='Maximum power flow on line la')
    model.pFixedCost = pyo.Param(model.la, initialize=cs.dPower_Network['pInvestCost'], doc='Fixed cost when investing in line la')  # TODO: Think about renaming this parameter (something related to 'investment cost')
    model.pBigM_Flow = pyo.Param(initialize=1e3, doc="Big M for power flow")
    model.pENSCost = pyo.Param(initialize=cs.dPower_Parameters['pENSCost'], doc='Cost used for Power Not Served (PNS) and Excess Power Served (EPS)')
    model.pWeight_rp = pyo.Param(model.rp, initialize=cs.dPower_WeightsRP["pWeight_rp"], doc='Weight of representative period rp')
    model.pWeight_k = pyo.Param(model.k, initialize=cs.dPower_WeightsK["pWeight_k"], doc='Weight of time step k')

    model.pBigM = pyo.Param(doc="Big M for binary variables", initialize=1e3)
    model.eps = pyo.Param(doc="Very small number", initialize=1e-9)

    model.coordsLat = pyo.Param(model.i, initialize=cs.dPower_BusInfo['lat'], doc='Latitude of bus i')
    model.coordsLon = pyo.Param(model.i, initialize=cs.dPower_BusInfo['lon'], doc='Longitude of bus i')

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


    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    #Power balance for nodes DC ann SOCP
    def eDC_BalanceP_rule(m, rp, k, i):
        return (sum(m.vGenP[rp, k, g] for g in m.g if (g, i) in m.gi)  # Production of generators at bus i todo please also make quick
                    + sum(m.vLineP[rp, k, e] if (e[1] == i) else -m.vLineP[rp, k, e] for e in model.la_nodeRelevant[i])  # Add power flow from bus j to bus i and subtract from bus i to bus j
                    - m.pDemandP[rp, k, i]  # Demand at bus i
                    + m.vPNS[rp, k, i]  # Slack variable for demand not served
                    - m.vEPS[rp, k, i])  # Slack variable for overproduction
    
    model.eDC_BalanceP_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.i, rule=eDC_BalanceP_rule)
    model.eDC_BalanceP = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc='Power balance constraint for each bus', rule=lambda m, rp, k, i: m.eDC_BalanceP_expr[rp, k, i] == 0)


    def eDC_ExiLinePij_rule(m, rp, k, i, j, c):
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF":
                return m.vLineP[rp, k, i, j, c] == (m.vTheta[rp, k, i] - m.vTheta[rp, k, j] + m.vAngle[rp, k, i, j, c]) / (m.pXline[i, j, c] * m.pRatio[i, j, c])
            case "TP" | "SN" | "SOCP":
                return pyo.Constraint.Skip
            case _:
                raise ValueError(f"Technical representation '{cs.dPower_Network.loc[i, j]["pTecRepr"]}' "
                                    f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

    model.eDC_ExiLinePij = pyo.Constraint(model.rp, model.constraintsActiveK, model.le, doc="Power flow existing lines (for DC-OPF)", rule=eDC_ExiLinePij_rule)

    def eDC_CanLinePij1_rule(m, rp, k, i, j, c):
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF":
                return (m.vLineP[rp, k, i, j, c] / (m.pBigM_Flow * m.pPmax[i, j, c]) >=
                        (m.vTheta[rp, k, i] - m.vTheta[rp, k, j] + m.vAngle[rp, k, i, j, c]) / (m.pXline[i, j, c] * m.pRatio[i, j, c]) /
                        (m.pBigM_Flow * m.pPmax[i, j, c]) - 1 + m.vLineInvest[i, j, c])
            case "TP" | "SN" | "SOCP":
                return pyo.Constraint.Skip
            case _:
                raise ValueError(f"Unsupported pTecRepr: {cs.dPower_Network.loc[i, j, c]['pTecRepr']}")

    model.eDC_CanLinePij1 = pyo.Constraint(model.rp, model.constraintsActiveK, model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij1_rule)

    def eDC_CanLinePij2_rule(m, rp, k, i, j, c):
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF":
                return (m.vLineP[rp, k, i, j, c] / (m.pBigM_Flow * m.pPmax[i, j, c]) <=
                        (m.vTheta[rp, k, i] - m.vTheta[rp, k, j] + m.vAngle[rp, k, i, j, c]) / (m.pXline[i, j, c] * m.pRatio[i, j, c]) /
                        (m.pBigM_Flow * m.pPmax[i, j, c]) + 1 - m.vLineInvest[i, j, c])
            case "TP" | "SN" | "SOCP":
                return pyo.Constraint.Skip
            case _:
                raise ValueError(f"Unsupported pTecRepr: {cs.dPower_Network.loc[i, j, c]['pTecRepr']}")

    model.eDC_CanLinePij2 = pyo.Constraint(model.rp, model.constraintsActiveK, model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij2_rule)

    def eDC_LimCanLine1_rule(m, rp, k, i, j, c):
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF" | "TP" | "SN":
                return m.vLineP[rp, k, i, j, c] / m.pPmax[i, j, c] + m.vLineInvest[i, j, c] >= 0
            case 'SOCP':
                return pyo.Constraint.Skip
        return pyo.Constraint.Skip

    model.eDC_LimCanLine1 = pyo.Constraint(model.rp, model.constraintsActiveK, model.lc, doc="Power flow limit reverse direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine1_rule)

    def eDC_LimCanLine2_rule(m, rp, k, i, j, c):
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF" | "TP" | "SN":
                return m.vLineP[rp, k, i, j, c] / m.pPmax[i, j, c] - m.vLineInvest[i, j, c] <= 0
            case 'SOCP':
                return pyo.Constraint.Skip
        return pyo.Constraint.Skip

    model.eDC_LimCanLine2 = pyo.Constraint(model.rp, model.constraintsActiveK, model.lc, doc="Power flow limit reverse direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine2_rule)


    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = (sum(model.pFixedCost[i, j, c] * model.vLineInvest[i, j, c] for i, j, c in model.lc) +  # Investment cost of transmission lines
                             sum(model.pInvestCost[g] * model.vGenInvest[g] for g in model.g))  # Investment cost of generators
    
    def ens_terms(rp, k):
        return sum(
            model.vPNS[rp, k, i] * model.pENSCost
            + model.vEPS[rp, k, i] * model.pENSCost * 2
            for i in model.i
        )

    def line_losses_terms(rp, k):
        return 0

    second_stage_objective = sum(model.pWeight_rp[rp] *  # Weight of representative periods
                                 sum(model.pWeight_k[k] *  # Weight of time steps
                                     (ens_terms(rp, k)  # Power non supplied terms
                                      + sum(+ model.vGenP[rp, k, g] * model.pOMVarCost[g]  # Production cost of generators
                                            for g in model.g))
                                        + line_losses_terms(rp, k)  # Penalty for line losses
                                     for k in model.constraintsActiveK)
                                 for rp in model.rp)

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
