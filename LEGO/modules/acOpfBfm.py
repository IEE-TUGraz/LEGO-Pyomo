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
    model.pDemandQ = pyo.Param(model.rp, model.k, model.i, initialize=cs.dPowerQ_Demand['value'], doc='Reactive Demand at bus i in representative period rp and timestep k')

    model.coordsLat = pyo.Param(model.i, initialize=cs.dPower_BusInfo['lat'], doc='Latitude of bus i')
    model.coordsLon = pyo.Param(model.i, initialize=cs.dPower_BusInfo['lon'], doc='Longitude of bus i')

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

    model.vLineP = pyo.Var(model.rp, model.k, model.la , doc='Power flow from bus i to j', bounds=lambda m, rp, k, i, j, c: (-m.pPmax[i,j,c], m.pPmax[i, j, c]) if (i, j, c) in m.la else (-m.pPmax[j, i, c], m.pPmax[j, i, c]))
    second_stage_variables += [model.vLineP]

    model.vLineQ = pyo.Var(model.rp, model.k, model.la, doc="Reactive power flow from bus i to j", bounds=lambda m, rp, k, i, j, c: (-m.pQmax[i,j,c], m.pQmax[i, j, c]) if (i, j, c) in m.le else (-m.pQmax[j, i, c], m.pQmax[j, i, c]) if (i, j, c) in m.le_reverse else (None, None))
    second_stage_variables.append(model.vLineQ)

    model.vSOCP_ui = pyo.Var(model.rp, model.k, model.i, doc='Squared voltage magnitude at bus i', bounds=lambda m, rp, k, i: (m.pBusMinV[i] ** 2, m.pBusMaxV[i] ** 2))
    second_stage_variables.append(model.vSOCP_ui)

    model.vSOCP_lij = pyo.Var(model.rp, model.k, model.la, doc='Squared current magnitude on line ij', bounds=lambda m, rp, k, i, j, c: (0, None) if (i, j, c) in m.la else (None, None)) #m.pSijNom[i,j,c]/m.pBusMinV[i]
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


    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):

    def eSOCP_ActivePowerFlow_rule(m, rp, k, i, j, c):
        return (- m.vLineP[rp, k, i, j, c] 
                + m.pRline[i, j, c] * m.vSOCP_lij[rp, k, i, j, c] -
                (sum(m.vGenP[rp, k, g] for g in m.g if (g, j) in m.gi)
                    - (m.pDemandP[rp, k, j])
                    + m.vPNS[rp, k, j]
                    - m.vEPS[rp, k, j]
                    ) +
                sum(m.vLineP[rp, k, j2, m_con, c] for (j2, m_con, c) in m.la if j2 == j)
                == 0
                ) 

    model.eSOCP_ActivePowerFlow = pyo.Constraint(model.rp, model.constraintsActiveK, model.la, doc='Active power flow on line ij', rule=eSOCP_ActivePowerFlow_rule)

    def eSOCP_ReactivePowerFlow_rule(m, rp, k, i, j, c):
        return (m.vLineQ[rp, k, i, j, c] == m.pXline[i, j, c] * m.vSOCP_lij[rp, k, i, j, c] -
            (sum(m.vGenQ[rp, k, g] for g in m.g if (g, j) in m.gi)
            - (m.pDemandQ[rp, k, j])
            + m.vQNS[rp, k, j]
            - m.vEQS[rp, k, j]
            ) +
            sum(m.vLineQ[rp, k, j2, m_con, c] for (j2, m_con, c) in m.la if j2 == j))  # Only outflows from i, Old indexing format

    model.eSOCP_ReactivePowerFlow = pyo.Constraint(model.rp, model.constraintsActiveK, model.la, doc='Reactive power flow over line ij (SOCP)', rule=eSOCP_ReactivePowerFlow_rule)

    model.eSOCP_QMaxOut = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc="Max reactive power output of generator unit", rule=lambda m, rp, k, g: (m.vGenQ[rp, k, g] / m.pMaxGenQ[g] <= m.vCommit[rp, k, g]) if m.pMaxGenQ[g] != 0 and (m.pExisUnits[g] > 0 or m.pEnabInv[g] == 1) else pyo.Constraint.Skip)
    model.eSOCP_QMinOut1 = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc="Min positive reactive power output of generator unit", rule=lambda m, rp, k, g: (m.vGenQ[rp, k, g] / m.pMinGenQ[g] >= m.vCommit[rp, k, g]) if m.pMinGenQ[g] >= 0 and (m.pExisUnits[g] > 0 or m.pEnabInv[g] == 1) else pyo.Constraint.Skip)
    model.eSOCP_QMinOut2 = pyo.Constraint(model.rp, model.constraintsActiveK, model.thermalGenerators, doc="Min negative reactive power output of generator unit", rule=lambda m, rp, k, g: (m.vGenQ[rp, k, g] / m.pMinGenQ[g] <= m.vCommit[rp, k, g]) if m.pMinGenQ[g] <= 0 and (m.pExisUnits[g] > 0 or m.pEnabInv[g] == 1) else pyo.Constraint.Skip)
    
    def eSOCP_VoltageDrop_rule(m, rp, k, i, j, c):
        return (m.vSOCP_ui[rp, k, j] ==
                m.vSOCP_ui[rp, k, i] -
                2 * (m.pRline[i,j,c] * m.vLineP[rp, k, i, j, c] + m.pXline[i,j,c] * m.vLineQ[rp, k, i, j, c]) +
                (m.pRline[i,j,c] ** 2 + m.pXline[i,j,c] ** 2) * m.vSOCP_lij[rp, k, i, j, c])

    model.eSOCP_VoltageDrop = pyo.Constraint(model.rp, model.constraintsActiveK, model.la, doc="SOCP constraints for voltage drop of line", rule=eSOCP_VoltageDrop_rule)


    def eSOCP_FlowDef_rule(m, rp, k, i, j, c):
        if any((i, j, c) in m.la for c in m.c):
            return m.vSOCP_lij[rp, k, i, j, c] * m.vSOCP_ui[rp, k, i] >= m.vLineP[rp, k, i, j, c] ** 2 + m.vLineQ[rp, k, i, j, c] ** 2
        else:
            return pyo.Constraint.Skip

    model.eSOCP_FlowDef = pyo.Constraint(model.rp, model.constraintsActiveK, model.la, doc="SCOP constraints for existing lines (for AC-OPF) original set", rule=eSOCP_FlowDef_rule)


    # FACTS (not yet Implemented) TODO: Add FACTS as a set, add FACTS parameters to nodes i
    if cs.dPower_Parameters["pEnableSOCP"] == 99999:
        model.eSOCP_QMinFACTS = pyo.Constraint(model.rp, model.constraintsActiveK, model.facts, doc='min reactive power output of FACTS unit', rule=lambda m, rp, k, i: m.vGenQ[rp, k, i] >= m.pMaxGenQ[i] * (m.pExisUnits[i] + m.vGenInvest[i]))
        model.eSOCP_QMaxFACTS = pyo.Constraint(model.rp, model.constraintsActiveK, model.facts, doc='max reactive power output of FACTS unit', rule=lambda m, rp, k, i: m.vGenQ[rp, k, i] <= m.pMaxGenQ[i] * (m.pExisUnits[i] + m.vGenInvest[i]))


    # define a active and reactive power balance constraint for the slack bus to use the ImExport implementation (only called DC to be consistent with the rest of the model)
    if cs.dPower_Parameters['pEnablePowerImportExport']:

        def eDC_BalanceP_rule(m, rp, k, i):
            return (- sum(m.vLineP[rp, k, j, m_con, c] for (j, m_con, c) in m.la if j == i))
        
        def eSOCP_ReactivePowerFlow_rule(m, rp, k, i):
            return (- sum(m.vLineQ[rp, k, j, m_con, c] for (j, m_con, c) in m.la if j == i))
        
        model.eDC_BalanceP_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.slack_node, rule=eDC_BalanceP_rule)
        model.eDC_BalanceP = pyo.Constraint(model.rp, model.constraintsActiveK, model.slack_node, doc='Power balance constraint for each bus', rule=lambda m, rp, k, i: m.eDC_BalanceP_expr[rp, k, i] == 0)

        model.eSOCP_BalanceQ_expr = pyo.Expression(model.rp, model.constraintsActiveK, model.slack_node, rule=eSOCP_ReactivePowerFlow_rule)
        model.eSOCP_BalanceQ = pyo.Constraint(model.rp, model.constraintsActiveK, model.slack_node, doc='Power balance constraint for each bus', rule=lambda m, rp, k, i: m.eSOCP_BalanceQ_expr[rp, k, i] == 0)

    else:
        model.vLineP[:, :, model.slack_node, :, :].fix(0)
        model.vLineQ[:, :, model.slack_node, :, :].fix(0)

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = (sum(model.pFixedCost[i, j, c] * model.vLineInvest[i, j, c] for i, j, c in model.lc) +  # Investment cost of transmission lines
                             sum(model.pInvestCost[g] * model.vGenInvest[g] for g in model.g))  # Investment cost of generators
    # Reactive slack node terms included when SOCP active
    def ens_terms(rp, k):
        return sum(
            model.vPNS[rp, k, i] * model.pENSCost
            + model.vEPS[rp, k, i] * model.pENSCost * 2
            + model.vQNS[rp, k, i] * model.pENSCost
            + model.vEQS[rp, k, i] * model.pENSCost * 2
            for i in model.i
        )

    second_stage_objective = sum(model.pWeight_rp[rp] *  # Weight of representative periods
                                 sum(model.pWeight_k[k] *  # Weight of time steps
                                     (ens_terms(rp, k)  # Power non supplied terms
                                      + sum(+ model.vGenP[rp, k, g] * model.pOMVarCost[g]  # Production cost of generators
                                            for g in model.g))
                                     for k in model.constraintsActiveK)
                                 for rp in model.rp)

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
