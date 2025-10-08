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

    model.c = pyo.Set(doc='Circuits', initialize=cs.dPower_Network.index.get_level_values('c').unique().tolist())
    model.la = pyo.Set(doc='All lines', initialize=cs.dPower_Network.index.tolist(), within=model.i * model.i * model.c)
    model.la_nodeRelevant = {node: [(i, j, c) for (i, j, c) in model.la if node == i or node == j] for node in model.i}
    model.le = pyo.Set(doc='Existing lines', initialize=cs.dPower_Network[(cs.dPower_Network["pEnableInvest"] == 0)].index.tolist(), within=model.la)
    model.lc = pyo.Set(doc='Candidate lines', initialize=cs.dPower_Network[(cs.dPower_Network["pEnableInvest"] == 1)].index.tolist(), within=model.la)

    model.g = pyo.Set(doc='Generators')
    model.gi = pyo.Set(doc='Generator g connected to bus i', within=model.g * model.i)

    model.p = pyo.Set(doc='Periods', initialize=cs.dPower_Hindex.index.get_level_values('p').unique().tolist())
    model.rp = pyo.Set(doc='Representative periods', initialize=cs.dPower_Demand.index.get_level_values('rp').unique().tolist())
    model.k = pyo.Set(doc='Timestep within representative period', initialize=cs.dPower_Demand.index.get_level_values('k').unique().tolist())
    model.hindex = cs.dPower_Hindex.index

    if cs.dPower_Parameters["pEnableSOCP"]:
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
    model.pSBase = pyo.Param(initialize=cs.dPower_Parameters['pSBase'], doc='Base power')
    model.pBigM_Flow = pyo.Param(initialize=1e3, doc="Big M for power flow")
    model.pENSCost = pyo.Param(initialize=cs.dPower_Parameters['pENSCost'], doc='Cost used for Power Not Served (PNS) and Excess Power Served (EPS)')
    model.pWeight_rp = pyo.Param(model.rp, initialize=cs.dPower_WeightsRP["pWeight_rp"], doc='Weight of representative period rp')
    model.pWeight_k = pyo.Param(model.k, initialize=cs.dPower_WeightsK["pWeight_k"], doc='Weight of time step k')

    model.pBigM = pyo.Param(doc="Big M for binary variables", initialize=1e3)
    model.eps = pyo.Param(doc="Very small number", initialize=1e-9)

    if cs.dPower_Parameters['pEnableSOCP']:
        model.pBusG = pyo.Param(model.i, initialize=cs.dPower_BusInfo['pBusG'], doc='Conductance of bus i')
        model.pBusB = pyo.Param(model.i, initialize=cs.dPower_BusInfo['pBusG'], doc='Susceptance of bus i')
        model.pBus_pf = pyo.Param(model.i, initialize=cs.dPower_BusInfo['pBus_pf'], doc='PowerFactor of bus i')
        model.pRline = pyo.Param(model.la, initialize=cs.dPower_Network['pRline'], doc='Resistance of line la')
        model.pBcline = pyo.Param(model.la, initialize=cs.dPower_Network['pBcline'], doc='Susceptance of line la')
        model.pQmax = pyo.Param(model.la, initialize=lambda model, i, j, c: model.pPmax[i, j, c], doc='Maximum reactive power flow on line la')  # It is asumed that Qmax is ident to Pmax
        model.pBigM_SOCP = pyo.Param(initialize=1e3, doc="Big M for SOCP")
        model.pMaxAngleDiff = pyo.Param(initialize=cs.dPower_Parameters["pMaxAngleDiff"] * np.pi / 180, doc='Maximum angle difference between two buses for the SOCP formulation')
        model.pBusMaxV = pyo.Param(model.i, initialize=cs.dPower_BusInfo['pBusMaxV'], doc='Maximum voltage at bus i')
        model.pBusMinV = pyo.Param(model.i, initialize=lambda model, i: max(cs.dPower_BusInfo['pBusMinV'][i], 0.1), doc='Minimum voltage at bus i (with a lower bound of 0.1)')
        model.pGline = pyo.Param(model.la, initialize=lambda model, i, j, c: model.pRline[i, j, c] / ((model.pRline[i, j, c] ** 2 + model.pXline[i, j, c] ** 2) if model.pRline[i, j, c] > 1e-6 else 1e-6), doc='Conductance of line la with lower bound')
        model.pBline = pyo.Param(model.la, initialize=lambda model, i, j, c: - model.pXline[i, j, c] / ((model.pRline[i, j, c] ** 2 + model.pXline[i, j, c] ** 2) if model.pRline[i, j, c] > 1e-6 else 1e-6), doc='Susceptance of line la with lower bound')
        model.pRatioDemQP = pyo.Param(model.i, initialize=lambda model, i: pyo.tan(pyo.acos(model.pBus_pf[i])))
        model.pDemandQ = pyo.Param(model.rp, model.k, model.i, initialize=lambda model, rp, k, i: model.pDemandP[rp, k, i] * model.pRatioDemQP[i], doc='Reactive demand at bus i in representative period rp and timestep k')

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

    model.vPNS = pyo.Var(model.rp, model.k, model.i, doc='Slack variable power not served', bounds=lambda model, rp, k, i: (0, model.pDemandP[rp, k, i]))
    second_stage_variables += [model.vPNS]
    model.vEPS = pyo.Var(model.rp, model.k, model.i, doc='Slack variable excess power served', bounds=(0, None))
    second_stage_variables += [model.vEPS]

    model.vGenP = pyo.Var(model.rp, model.k, model.g, doc='Power output of generator g', bounds=lambda model, rp, k, g: (0, model.pMaxProd[g] * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g])))
    second_stage_variables += [model.vGenP]

    model.vLineP = pyo.Var(model.rp, model.k, model.la_full if cs.dPower_Parameters["pEnableSOCP"] else model.la, doc='Power flow from bus i to j', bounds=lambda m, rp, k, i, j, c: (-model.pPmax[i, j, c], model.pPmax[i, j, c]) if (i, j, c) in m.la else (-model.pPmax[j, i, c], model.pPmax[j, i, c]))
    second_stage_variables += [model.vLineP]

    if cs.dPower_Parameters["pEnableSOCP"]:
        model.vLineQ = pyo.Var(model.rp, model.k, model.la_full, doc="Reactive power flow from bus i to j", bounds=lambda m, rp, k, i, j, c: (-m.pQmax[i, j, c], m.pQmax[i, j, c]) if (i, j, c) in m.le else (-m.pQmax[j, i, c], m.pQmax[j, i, c]) if (i, j, c) in m.le_reverse else (None, None))
        second_stage_variables.append(model.vLineQ)

        model.vSOCP_cii = pyo.Var(model.rp, model.k, model.i, bounds=lambda m, rp, k, i: (round(m.pBusMinV[i] ** 2, 4), round(m.pBusMaxV[i] ** 2, 4)))
        second_stage_variables.append(model.vSOCP_cii)

        # cij = (vi^real* vj^real) + vi^imag*vj^imag), Lower bounds for vSOCP_cij need to always be at least 0
        model.vSOCP_cij = pyo.Var(model.rp, model.k, model.la_no_c, bounds=lambda m, rp, k, i, j: (round(max(model.pBusMinV[i] ** 2, 0.1), 4), round(model.pBusMaxV[i] ** 2, 4)) if (i, j, c) in m.le else (0, None))
        second_stage_variables.append(model.vSOCP_cij)

        # sij = (vi^real* vj^imag) - vi^re*vj^imag))
        model.vSOCP_sij = pyo.Var(model.rp, model.k, model.la_no_c, bounds=lambda m, rp, k, i, j: (round(-model.pBusMaxV[i] ** 2, 4), round(model.pBusMaxV[i] ** 2, 4)) if (i, j, c) in m.le else (None, None))
        second_stage_variables.append(model.vSOCP_sij)

        model.vSOCP_IndicConnecNodes = pyo.Var({(i, j) for (i, j, c) in model.lc}, domain=pyo.Binary)
        second_stage_variables.append(model.vSOCP_IndicConnecNodes)

        model.vGenQ = pyo.Var(model.rp, model.k, model.g, doc='Reactive power output of ge', bounds=lambda model, rp, k, g: (model.pMinGenQ[g] * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g]), model.pMaxGenQ[g] * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g])))
        second_stage_variables.append(model.vGenQ)

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
        printer.information(f"Zone {i:>2} - Slack node: {slack_node}")
        model.vTheta[:, :, slack_node].fix(0)
        if cs.dPower_Parameters['pEnableSOCP']:
            printer.information("Fixed voltage magnitude at slack node: ", pyo.value(pyo.sqrt(cs.dPower_Parameters['pSlackVoltage'])))
            model.vSOCP_cii[:, :, slack_node].fix(pyo.sqrt(cs.dPower_Parameters['pSlackVoltage']))

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    # Power balance for nodes DC ann SOCP
    def eDC_BalanceP_rule(m, rp, k, i):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (sum(m.vGenP[rp, k, g] for g in m.g if (g, i) in m.gi)  # Gen at bus i
                    - sum(m.vLineP[rp, k, i, j, c] for (i2, j, c) in m.la_full if i2 == i)  # Only outflows from i
                    - m.vSOCP_cii[rp, k, i] * m.pBusG[i] * m.pSBase
                    - m.pDemandP[rp, k, i]
                    + m.vPNS[rp, k, i]
                    - m.vEPS[rp, k, i])
        else:
            return (sum(m.vGenP[rp, k, g] for g in m.g if (g, i) in m.gi)  # Production of generators at bus i
                    + sum(m.vLineP[rp, k, e] if (e[1] == i) else -m.vLineP[rp, k, e] for e in model.la_nodeRelevant[i])  # Add power flow from bus j to bus i and subtract from bus i to bus j
                    - m.pDemandP[rp, k, i]  # Demand at bus i
                    + m.vPNS[rp, k, i]  # Slack variable for demand not served
                    - m.vEPS[rp, k, i])  # Slack variable for overproduction

    model.eDC_BalanceP_expr = pyo.Expression(model.rp, model.k, model.i, rule=eDC_BalanceP_rule)
    model.eDC_BalanceP = pyo.Constraint(model.rp, model.k, model.i, doc='Power balance constraint for each bus', rule=lambda m, rp, k, i: m.eDC_BalanceP_expr[rp, k, i] == 0)

    if not cs.dPower_Parameters["pEnableSOCP"]:
        def eDC_ExiLinePij_rule(m, rp, k, i, j, c):
            match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
                case "DC-OPF":
                    return m.vLineP[rp, k, i, j, c] == (m.vTheta[rp, k, i] - m.vTheta[rp, k, j] + m.vAngle[rp, k, i, j, c]) * m.pSBase / (m.pXline[i, j, c] * m.pRatio[i, j, c])
                case "TP" | "SN" | "SOCP":
                    return pyo.Constraint.Skip
                case _:
                    raise ValueError(f"Technical representation '{cs.dPower_Network.loc[i, j]["pTecRepr"]}' "
                                     f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

        model.eDC_ExiLinePij = pyo.Constraint(model.rp, model.k, model.le, doc="Power flow existing lines (for DC-OPF)", rule=eDC_ExiLinePij_rule)

        def eDC_CanLinePij1_rule(m, rp, k, i, j, c):
            match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
                case "DC-OPF":
                    return (m.vLineP[rp, k, i, j, c] / (m.pBigM_Flow * m.pPmax[i, j, c]) >=
                            (m.vTheta[rp, k, i] - m.vTheta[rp, k, j] + m.vAngle[rp, k, i, j, c]) *
                            m.pSBase / (m.pXline[i, j, c] * m.pRatio[i, j, c]) /
                            (m.pBigM_Flow * m.pPmax[i, j, c]) - 1 + m.vLineInvest[i, j, c])
                case "TP" | "SN" | "SOCP":
                    return pyo.Constraint.Skip
                case _:
                    raise ValueError(f"Unsupported pTecRepr: {cs.dPower_Network.loc[i, j, c]['pTecRepr']}")

        model.eDC_CanLinePij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij1_rule)

        def eDC_CanLinePij2_rule(m, rp, k, i, j, c):
            match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
                case "DC-OPF":
                    return (m.vLineP[rp, k, i, j, c] / (m.pBigM_Flow * m.pPmax[i, j, c]) <=
                            (m.vTheta[rp, k, i] - m.vTheta[rp, k, j] + m.vAngle[rp, k, i, j, c]) *
                            m.pSBase / (m.pXline[i, j, c] * m.pRatio[i, j, c]) /
                            (m.pBigM_Flow * m.pPmax[i, j, c]) + 1 - m.vLineInvest[i, j, c])
                case "TP" | "SN" | "SOCP":
                    return pyo.Constraint.Skip
                case _:
                    raise ValueError(f"Unsupported pTecRepr: {cs.dPower_Network.loc[i, j, c]['pTecRepr']}")

        model.eDC_CanLinePij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij2_rule)

        def eDC_LimCanLine1_rule(m, rp, k, i, j, c):
            match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
                case "DC-OPF" | "TP" | "SN":
                    return m.vLineP[rp, k, i, j, c] / m.pPmax[i, j, c] + m.vLineInvest[i, j, c] >= 0
                case 'SOCP':
                    return pyo.Constraint.Skip
            return pyo.Constraint.Skip

        model.eDC_LimCanLine1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow limit reverse direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine1_rule)

        def eDC_LimCanLine2_rule(m, rp, k, i, j, c):
            match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
                case "DC-OPF" | "TP" | "SN":
                    return m.vLineP[rp, k, i, j, c] / m.pPmax[i, j, c] - m.vLineInvest[i, j, c] <= 0
                case 'SOCP':
                    return pyo.Constraint.Skip
            return pyo.Constraint.Skip

        model.eDC_LimCanLine2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow limit reverse direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine2_rule)

    else:  # SOCP formulation
        def eSOCP_BalanceQ_rule(m, rp, k, i):
            return (sum(m.vGenQ[rp, k, g] for g in m.g if (g, i) in m.gi)
                    # Only vLineQ where i is the sending end (i → j)
                    - sum(m.vLineQ[rp, k, i, j, c] for (i2, j, c) in m.la_full if i2 == i)
                    + m.vSOCP_cii[rp, k, i] * m.pBusB[i] * m.pSBase
                    - m.pDemandQ[rp, k, i]
                    + m.vPNS[rp, k, i] * m.pRatioDemQP[i]
                    - m.vEPS[rp, k, i] * m.pRatioDemQP[i])

        def eSOCP_ExiLinePij_rule(m, rp, k, i, j, c):
            return (m.vLineP[rp, k, i, j, c] == m.pSBase * (
                    + m.pGline[i, j, c] * m.vSOCP_cii[rp, k, i] / (m.pRatio[i, j, c] ** 2)
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j]
                    - (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * - m.vSOCP_sij[rp, k, i, j]))

        def eSOCP_ExiLinePji_rule(m, rp, k, i, j, c):
            return (m.vLineP[rp, k, j, i, c] == m.pSBase * (
                    + (m.pGline[i, j, c] * m.vSOCP_cii[rp, k, j])
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j]
                    - (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_sij[rp, k, i, j]))

        def eSOCP_ExiLineQij_rule(m, rp, k, i, j, c):
            return (m.vLineQ[rp, k, i, j, c] == m.pSBase * (
                    - m.vSOCP_cii[rp, k, i] * (m.pBline[i, j, c] + m.pBcline[i, j, c] / 2) / (m.pRatio[i, j, c] ** 2)
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * -m.vSOCP_sij[rp, k, i, j]
                    + (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j]))

        def eSOCP_ExiLineQji_rule(m, rp, k, i, j, c):
            return (m.vLineQ[rp, k, j, i, c] == m.pSBase * (
                    - m.vSOCP_cii[rp, k, j] * (m.pBline[i, j, c] + m.pBcline[i, j, c] / 2)
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_sij[rp, k, i, j]
                    + (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j]))

        def eSOCP_CanLinePij1_rule(m, rp, k, i, j, c):
            return (m.vLineP[rp, k, i, j, c] >= m.pSBase * (
                    + m.pGline[i, j, c] * m.vSOCP_cii[rp, k, i] / (m.pRatio[i, j, c] ** 2)
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j]
                    - (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * -m.vSOCP_sij[rp, k, i, j])
                    - m.pBigM_Flow * (1 - m.vLineInvest[i, j, c]))

        def eSOCP_CanLinePij2_rule(m, rp, k, i, j, c):
            return (m.vLineP[rp, k, i, j, c] <= m.pSBase * (
                    + m.pGline[i, j, c] * m.vSOCP_cii[rp, k, i] / (m.pRatio[i, j, c] ** 2)
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j]
                    - (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * -m.vSOCP_sij[rp, k, i, j])
                    + m.pBigM_Flow * (1 - m.vLineInvest[i, j, c]))

        def eSOCP_CanLinePji1_rule(m, rp, k, i, j, c):
            return (m.vLineP[rp, k, j, i, c] >= m.pSBase * (
                    + m.pGline[i, j, c] * m.vSOCP_cii[rp, k, j]
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j]
                    - (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_sij[rp, k, i, j])
                    - m.pBigM_Flow * (1 - m.vLineInvest[i, j, c]))

        def eSOCP_CanLinePji2_rule(m, rp, k, i, j, c):
            return (m.vLineP[rp, k, j, i, c] <= m.pSBase * (
                    + m.pGline[i, j, c] * m.vSOCP_cii[rp, k, j]
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j]
                    - (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_sij[rp, k, i, j])
                    + m.pBigM_Flow * (1 - m.vLineInvest[i, j, c]))

        def eSOCP_CanLineQij1_rule(m, rp, k, i, j, c):
            return (m.vLineQ[rp, k, i, j, c] >= m.pSBase * (
                    - m.vSOCP_cii[rp, k, i] * (m.pBline[i, j, c] + m.pBcline[i, j, c] / 2) / (m.pRatio[i, j, c] ** 2)
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * -m.vSOCP_sij[rp, k, i, j]
                    + (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j])
                    - m.pBigM_Flow * (1 - m.vLineInvest[i, j, c]))

        def eSOCP_CanLineQij2_rule(m, rp, k, i, j, c):
            return (m.vLineQ[rp, k, i, j, c] <= m.pSBase * (
                    - m.vSOCP_cii[rp, k, i] * (m.pBline[i, j, c] + m.pBcline[i, j, c] / 2) / (m.pRatio[i, j, c] ** 2)
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * -m.vSOCP_sij[rp, k, i, j]
                    + (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j])
                    + m.pBigM_Flow * (1 - m.vLineInvest[i, j, c]))

        def eSOCP_CanLineQji1_rule(m, rp, k, i, j, c):
            return (m.vLineQ[rp, k, j, i, c] >= m.pSBase * (
                    - m.vSOCP_cii[rp, k, j] * (m.pBline[i, j, c] + m.pBcline[i, j, c] / 2)
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_sij[rp, k, i, j]
                    + (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j])
                    - m.pBigM_Flow * (1 - m.vLineInvest[i, j, c]))

        def eSOCP_CanLineQji2_rule(m, rp, k, i, j, c):
            return (m.vLineQ[rp, k, j, i, c] <= m.pSBase * (
                    - m.vSOCP_cii[rp, k, j] * (m.pBline[i, j, c] + m.pBcline[i, j, c] / 2)
                    - (1 / m.pRatio[i, j, c]) * (m.pGline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) + m.pBline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_sij[rp, k, i, j]
                    + (1 / m.pRatio[i, j, c]) * (m.pBline[i, j, c] * pyo.cos(m.pAngle[i, j, c]) - m.pGline[i, j, c] * pyo.sin(m.pAngle[i, j, c])) * m.vSOCP_cij[rp, k, i, j])
                    + m.pBigM_Flow * (1 - m.vLineInvest[i, j, c]))

        model.eSOCP_QMaxOut = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc="Max reactive power output of generator unit", rule=lambda m, rp, k, g: (m.vGenQ[rp, k, g] / m.pMaxGenQ[g] <= m.vCommit[rp, k, g]) if m.pMaxGenQ[g] != 0 and (m.pExisUnits[g] > 0 or m.pEnabInv[g] == 1) else pyo.Constraint.Skip)
        model.eSOCP_QMinOut1 = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc="Min positive reactive power output of generator unit", rule=lambda m, rp, k, g: (m.vGenQ[rp, k, g] / m.pMinGenQ[g] >= m.vCommit[rp, k, g]) if m.pMinGenQ[g] >= 0 and (m.pExisUnits[g] > 0 or m.pEnabInv[g] == 1) else pyo.Constraint.Skip)
        model.eSOCP_QMinOut2 = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc="Min negative reactive power output of generator unit", rule=lambda m, rp, k, g: (m.vGenQ[rp, k, g] / m.pMinGenQ[g] <= m.vCommit[rp, k, g]) if m.pMinGenQ[g] <= 0 and (m.pExisUnits[g] > 0 or m.pEnabInv[g] == 1) else pyo.Constraint.Skip)

        model.eSOCP_BalanceQ_expr = pyo.Expression(model.rp, model.k, model.i, rule=eSOCP_BalanceQ_rule)
        model.eSOCP_BalanceQ = pyo.Constraint(model.rp, model.k, model.i, doc='Reactive power balance for each bus (SOCP)', rule=lambda m, rp, k, i: m.eSOCP_BalanceQ_expr[rp, k, i] == 0)

        model.eSOCP_ExiLinePij = pyo.Constraint(model.rp, model.k, model.le, doc=" Active power flow existing lines from i to j (for SOCP)", rule=eSOCP_ExiLinePij_rule)
        model.eSOCP_ExiLinePji = pyo.Constraint(model.rp, model.k, model.le, doc="Active power flow existing lines from j to i (for SOCP)", rule=eSOCP_ExiLinePji_rule)

        model.eSOCP_ExiLineQij = pyo.Constraint(model.rp, model.k, model.le, doc="Reactive power flow existing lines from i to j (for SOCP)", rule=eSOCP_ExiLineQij_rule)
        model.eSOCP_ExiLineQji = pyo.Constraint(model.rp, model.k, model.le, doc="Reactive power flow existing lines from j to i (for SOCP)", rule=eSOCP_ExiLineQji_rule)

        model.eSOCP_CanLinePij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power flow candidate lines from i to j (for SOCP) Big-M 1", rule=eSOCP_CanLinePij1_rule)
        model.eSOCP_CanLinePij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power flow candidate lines from i to j (for SOCP) Big-M 2", rule=eSOCP_CanLinePij2_rule)
        model.eSOCP_CanLinePji1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power flow candidate lines from j to i (for SOCP) Big-M 1", rule=eSOCP_CanLinePji1_rule)
        model.eSOCP_CanLinePji2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power flow candidate lines from j to i (for SOCP) Big-M 1", rule=eSOCP_CanLinePji2_rule)

        model.eSOCP_CanLineQij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power flow candidate lines from i to j (for SOCP) Big-M 1", rule=eSOCP_CanLineQij1_rule)
        model.eSOCP_CanLineQij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power flow candidate lines from i to j (for SOCP) Big-M 2", rule=eSOCP_CanLineQij2_rule)
        model.eSOCP_CanLineQji1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power flow candidate lines from j to i (for SOCP) Big-M 1", rule=eSOCP_CanLineQji1_rule)
        model.eSOCP_CanLineQji2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power flow candidate lines from j to i (for SOCP) Big-M 2", rule=eSOCP_CanLineQji2_rule)

        model.eSOCP_LimCanLinePij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power candidate lower limit i to j candidate lines (for SOCP)", rule=lambda m, rp, k, i, j, c: m.vLineP[rp, k, i, j, c] / m.pPmax[i, j, c] >= -m.vLineInvest[i, j, c])
        model.eSOCP_LimCanLinePij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power candidate upper limit i to j candidate lines (for SOCP)", rule=lambda m, rp, k, i, j, c: m.vLineP[rp, k, i, j, c] / m.pPmax[i, j, c] <= m.vLineInvest[i, j, c])
        model.eSOCP_LimCanLinePji1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power candidate lower limit j to i candidate lines (for SOCP)", rule=lambda m, rp, k, i, j, c: m.vLineP[rp, k, j, i, c] / m.pPmax[i, j, c] >= -m.vLineInvest[i, j, c])
        model.eSOCP_LimCanLinePji2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power candidate upper limit j to i candidate lines (for SOCP)", rule=lambda m, rp, k, i, j, c: m.vLineP[rp, k, j, i, c] / m.pPmax[i, j, c] <= m.vLineInvest[i, j, c])

        model.eSOCP_LimCanLineQij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power candidate lower limit i to j candidate lines (for SOCP)", rule=lambda m, rp, k, i, j, c: m.vLineQ[rp, k, i, j, c] / m.pQmax[i, j, c] >= -m.vLineInvest[i, j, c])
        model.eSOCP_LimCanLineQij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power candidate upper limit i to j candidate lines (for SOCP)", rule=lambda m, rp, k, i, j, c: m.vLineQ[rp, k, i, j, c] / m.pQmax[i, j, c] <= m.vLineInvest[i, j, c])
        model.eSOCP_LimCanLineQji1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power candidate lower limit j to i candidate lines (for SOCP)", rule=lambda m, rp, k, i, j, c: m.vLineQ[rp, k, j, i, c] / m.pQmax[i, j, c] >= -m.vLineInvest[i, j, c])
        model.eSOCP_LimCanLineQji2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power candidate upper limit j to i candidate lines (for SOCP)", rule=lambda m, rp, k, i, j, c: m.vLineQ[rp, k, j, i, c] / m.pQmax[i, j, c] <= m.vLineInvest[i, j, c])

        model.eSOCP_ExiLine = pyo.Constraint(model.rp, model.k, model.le_no_c, doc="SCOP constraints for existing lines (for AC-OPF) original set", rule=lambda m, rp, k, i, j: (model.vSOCP_cij[rp, k, i, j] ** 2 + model.vSOCP_sij[rp, k, i, j] ** 2 <= model.vSOCP_cii[rp, k, i] * model.vSOCP_cii[rp, k, j]) if any((i, j, c) in model.le for c in model.c) else pyo.Constraint.Skip)

        model.eSOCP_CanLine = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="SOCP constraint for candidate lines (only if not in le)", rule=lambda m, rp, k, i, j: m.vSOCP_cij[rp, k, i, j] ** 2 + m.vSOCP_sij[rp, k, i, j] ** 2 <= m.vSOCP_cii[rp, k, i] * m.vSOCP_cii[rp, k, j] if (i, j, m.first_circuit_map[i, j]) in m.lc and (i, j, m.first_circuit_map[i, j]) not in m.le else pyo.Constraint.Skip)
        model.eSOCP_CanLine_cij = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=lambda m, rp, k, i, j: m.vSOCP_cij[rp, k, i, j] <= m.pBigM_SOCP * m.vSOCP_IndicConnecNodes[i, j] if (i, j, m.first_circuit_map[i, j]) in m.lc and (i, j, m.first_circuit_map[i, j]) in m.le else pyo.Constraint.Skip)
        model.eSOCP_CanLine_sij1 = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=lambda m, rp, k, i, j: m.vSOCP_sij[rp, k, i, j] <= m.pBigM_SOCP * m.vSOCP_IndicConnecNodes[i, j] if (i, j, m.first_circuit_map[i, j]) in m.lc and (i, j, m.first_circuit_map[i, j]) in m.le else pyo.Constraint.Skip)
        model.eSOCP_CanLine_sij2 = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=lambda m, rp, k, i, j: m.vSOCP_sij[rp, k, i, j] >= -m.pBigM_SOCP * m.vSOCP_IndicConnecNodes[i, j] if (i, j, m.first_circuit_map[i, j]) in m.lc and (i, j, m.first_circuit_map[i, j]) in m.le else pyo.Constraint.Skip)

        model.eSOCP_IndicConnecNodes1 = pyo.Constraint(model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=lambda m, i, j: sum(m.vSOCP_IndicConnecNodes[i, j, c_] for (i_, j_, c_) in m.lc if i_ == i and j_ == j) == 1 if (i, j, m.first_circuit_map[i, j]) in m.lc and (i, j, m.first_circuit_map[i, j]) in m.le else pyo.Constraint.Skip)
        model.eSOCP_IndicConnecNodes2 = pyo.Constraint(model.lc_no_c, doc="SOCP constraint for candidate lines (only if not in le)", rule=lambda m, i, j: m.vSOCP_IndicConnecNodes[i, j] == m.vLineInvest[i, j, m.first_circuit_map[i, j]] if (i, j, m.first_circuit_map[i, j]) in m.lc and (i, j, m.first_circuit_map[i, j]) not in m.le else pyo.Constraint.Skip)

        model.eSOCP_CanLineCijUpLim = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Limits for SOCP variables lines (only if not in le)", rule=lambda m, rp, k, i, j: m.vSOCP_cij[rp, k, i, j] <= ((m.pBusMaxV[i] ** 2) + m.pBigM_SOCP * (1 - m.vSOCP_IndicConnecNodes[i, j])) if (i, j, m.first_circuit_map_bidir[i, j]) in m.lc and (i, j, m.first_circuit_map_bidir[i, j]) not in m.le else pyo.Constraint.Skip)
        model.eSOCP_CanLineCijLoLim = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Limits for SOCP variables (only if not in le)", rule=lambda m, rp, k, i, j: m.vSOCP_cij[rp, k, i, j] >= max(0.1, m.pBusMinV[i] ** 2) - m.pBigM_SOCP * (1 - m.vSOCP_IndicConnecNodes[i, j]) if (i, j, m.first_circuit_map_bidir[i, j]) in m.lc and (i, j, m.first_circuit_map_bidir[i, j]) not in m.le else pyo.Constraint.Skip)
        model.eSOCP_CanLineSijUpLim = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Limits for SOCP variables (only if not in le)", rule=lambda m, rp, k, i, j: m.vSOCP_sij[rp, k, i, j] <= (m.pBusMaxV[i] ** 2) + m.pBigM_SOCP * (1 - m.vSOCP_IndicConnecNodes[i, j]) if (i, j, m.first_circuit_map_bidir[i, j]) in m.lc and (i, j, m.first_circuit_map_bidir[i, j]) not in m.le else pyo.Constraint.Skip)
        model.eSOCP_CanLineSijLoLim = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Limits for SOCP variables (only if not in le)", rule=lambda m, rp, k, i, j: m.vSOCP_sij[rp, k, i, j] >= -(m.pBusMaxV[i] ** 2) - m.pBigM_SOCP * (1 - m.vSOCP_IndicConnecNodes[i, j]) if (i, j, m.first_circuit_map_bidir[i, j]) in m.lc and (i, j, m.first_circuit_map_bidir[i, j]) not in m.le else pyo.Constraint.Skip)

        model.eSOCP_ExiLineAngDif1 = pyo.Constraint(model.rp, model.k, model.le_no_c, doc="Angle difference upper bounds existing lines", rule=lambda m, rp, k, i, j: m.vSOCP_sij[rp, k, i, j] <= m.vSOCP_cij[rp, k, i, j] * pyo.tan(m.pMaxAngleDiff) if any((i, j, c) in m.le for c in m.c) else pyo.Constraint.Skip)
        model.eSOCP_ExiLineAngDif2 = pyo.Constraint(model.rp, model.k, model.le_no_c, doc="Angle difference lower bounds existing lines", rule=lambda m, rp, k, i, j: m.vSOCP_sij[rp, k, i, j] >= -m.vSOCP_cij[rp, k, i, j] * pyo.tan(m.pMaxAngleDiff) if any((i, j, c) in m.le for c in m.c) else pyo.Constraint.Skip)

        model.eSOCP_CanLineAngDif1 = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Angle difference upper bounds candidate lines", rule=lambda m, rp, k, i, j: m.vSOCP_sij[rp, k, i, j] <= m.vSOCP_cij[rp, k, i, j] * pyo.tan(m.pMaxAngleDiff) + m.pBigM_Flow * (1 - m.vSOCP_IndicConnecNodes[i, j]) if (i, j, m.first_circuit_map[i, j]) in m.lc and (i, j, m.first_circuit_map[i, j]) not in m.le else pyo.Constraint.Skip)
        model.eSOCP_CanLineAngDif2 = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Angle difference lowerf bounds candidate lines ", rule=lambda m, rp, k, i, j: m.vSOCP_sij[rp, k, i, j] >= - m.vSOCP_cij[rp, k, i, j] * pyo.tan(m.pMaxAngleDiff) - m.pBigM_Flow * (1 - m.vSOCP_IndicConnecNodes[i, j]) if (i, j, m.first_circuit_map[i, j]) in m.lc and (i, j, m.first_circuit_map[i, j]) not in m.le else pyo.Constraint.Skip)

        if cs.dPower_Parameters["pEnableSOCP"] == 99999:  # Deactivated
            # Apparent power constraints for existing and candidate lines (Disabled in the LEGO model due to increased solving time)
            # Constraints might need to be redefined for only the le set

            def eSOCP_ExiLineSLimit_rule(m, rp, k, i, j, c):
                if (i, j, c) in m.le:
                    return m.vLineP[rp, k, i, j, c] ** 2 + m.vLineQ[rp, k, i, j, c] ** 2 <= pyo.sqrt(m.pPmax[i, j, c] ** 2 + m.pQmax[i, j, c] ** 2)
                elif (j, i, c) in m.le:
                    # For the Reverse direction, the rhs should be 0 but this does not work with the solver. If this constraint is to be used, skip the reverse direction (will lead to error in the compare tool)
                    return m.vLineP[rp, k, i, j, c] ** 2 + m.vLineQ[rp, k, i, j, c] ** 2 <= 1
                else:
                    return pyo.Constraint.Skip  # Not in le or le_reverse

            model.eSOCP_ExiLineSLimit = pyo.Constraint(model.rp, model.k, model.le_full, doc="Apparent power constraints for existing lines ", rule=eSOCP_ExiLineSLimit_rule)

            def eSOCP_CanLineSLimit_rule(m, rp, k, i, j, c):
                if (i, j, c) in m.lc:
                    return m.vLineP[rp, k, i, j, c] ** 2 + m.vLineQ[rp, k, i, j, c] ** 2 <= pyo.sqrt(m.pPmax[i, j, c] ** 2 + m.pQmax[i, j, c] ** 2) * m.vLineInvest[i, j, c]
                elif (j, i, c) in m.lc:
                    # For the Reverse direction, the rhs should be 0 but this does not work with the solver. If this constraint is to be used skip the reverse direction (will lead to error in the compare tool)
                    return m.vLineP[rp, k, i, j, c] ** 2 + m.vLineQ[rp, k, i, j, c] ** 2 <= 1
                else:
                    return pyo.Constraint.Skip  # Not in le or le_reverse

            model.eSOCP_CanLineSLimit = pyo.Constraint(model.rp, model.k, model.lc_full, doc="Apparent power constraints for existing lines ", rule=eSOCP_CanLineSLimit_rule)

        # FACTS (not yet Implemented) TODO: Add FACTS as a set, add FACTS parameters to nodes i
        if cs.dPower_Parameters["pEnableSOCP"] == 99999:
            model.eSOCP_QMinFACTS = pyo.Constraint(model.rp, model.k, model.facts, doc='min reactive power output of FACTS unit', rule=lambda m, rp, k, i: m.vGenQ[rp, k, i] >= m.pMaxGenQ[i] * (m.pExisUnits[i] + m.vGenInvest[i]))
            model.eSOCP_QMaxFACTS = pyo.Constraint(model.rp, model.k, model.facts, doc='max reactive power output of FACTS unit', rule=lambda m, rp, k, i: m.vGenQ[rp, k, i] <= m.pMaxGenQ[i] * (m.pExisUnits[i] + m.vGenInvest[i]))

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = (sum(model.pFixedCost[i, j, c] * model.vLineInvest[i, j, c] for i, j, c in model.lc) +  # Investment cost of transmission lines
                             sum(model.pInvestCost[g] * model.vGenInvest[g] for g in model.g))  # Investment cost of generators
    second_stage_objective = sum(model.pWeight_rp[rp] *  # Weight of representative periods
                                 sum(model.pWeight_k[k] *  # Weight of time steps
                                     (+ sum(+ model.vPNS[rp, k, i] * model.pENSCost  # Power not served
                                            + model.vEPS[rp, k, i] * model.pENSCost * 2  # Excess power served
                                            for i in model.i)
                                      + sum(+ model.vGenP[rp, k, g] * model.pOMVarCost[g]  # Production cost of generators
                                            for g in model.g))
                                     for k in model.k)
                                 for rp in model.rp)

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
