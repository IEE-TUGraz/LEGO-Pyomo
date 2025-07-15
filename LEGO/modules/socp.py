import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGOUtilities


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets

    # Parameters

    # Variables
    model.vLineQ = pyo.Var(model.rp, model.k, model.la_full, domain=pyo.Reals, doc="Reactive power flow from bus i to j")
    for (i, j, c) in model.le:
        for rp in model.rp:
            for k in model.k:
                model.vLineQ[rp, k, i, j, c].setlb(-model.pQmax[i, j, c])
                model.vLineQ[rp, k, i, j, c].setub(model.pQmax[i, j, c])
    second_stage_variables.append(model.vLineQ)

    model.vSOCP_cii = pyo.Var(model.rp, model.k, model.i, domain=pyo.Reals)
    for rp in model.rp:
        for k in model.k:
            for i in model.i:
                model.vSOCP_cii[rp, k, i].setub(round(model.pBusMaxV[i] ** 2, 4))  # Set upper bound for cii
                model.vSOCP_cii[rp, k, i].setlb(round(model.pBusMinV[i] ** 2, 4))
    second_stage_variables.append(model.vSOCP_cii)

    # For each DC-OPF "island", set node with highest demand as slack node
    dDCOPFIslands = pd.DataFrame(index=cs.dPower_BusInfo.index, columns=[cs.dPower_BusInfo.index], data=False)

    for index, entry in cs.dPower_Network.iterrows():
        if cs.dPower_Network.loc[(index[0], index[1], index[2])]["pTecRepr"] == "DC-OPF" or "SOCP":
            dDCOPFIslands.loc[index[0], index[1]] = True
            dDCOPFIslands.loc[index[1], index[0]] = True
    completed_buses = set()  # Set of buses that have been looked at already
    i = 0

    for index, entry in dDCOPFIslands.iterrows():
        if index in completed_buses or entry[entry == True].empty:
            continue
        connected_buses = cs.get_connected_buses(dDCOPFIslands, str(index))
        for bus in connected_buses:
            completed_buses.add(bus)
        completed_buses.add(index)

        # Set slack node
        slack_node = cs.dPower_Demand.loc[:, :, connected_buses].groupby('i').sum().idxmax().values[0]
        slack_node = cs.dPower_Parameters["is"]  # TODO: Switch this again to be calculated (fixed to 'is' for compatibility)

        if i == 0: print("Setting slack nodes for SOCP zones:")
        i += 1
        model.vSOCP_cii[:, :, slack_node].fix(pyo.sqrt(cs.dPower_Parameters['pSlackVoltage']))
        print(f"SOCP {i:>2} - Slack node: {slack_node}")
        print("Fixed voltage magnitude at slack node:", pyo.value(pyo.sqrt(cs.dPower_Parameters['pSlackVoltage'])))
        model.vTheta[:, :, slack_node].fix(0)

    model.vSOCP_cij = pyo.Var(model.rp, model.k, model.la_no_c, domain=pyo.Reals, bounds=(0, None))  # cij = (vi^real* vj^real) + vi^imag*vj^imag), Lower bounds for vSOCP_cij need to always be at least 0
    for (i, j, c) in model.le:
        for rp in model.rp:
            for k in model.k:
                if (rp, k, i, j) in model.vSOCP_cij:
                    model.vSOCP_cij[rp, k, i, j].setub(round(model.pBusMaxV[i] ** 2, 4))
                    model.vSOCP_cij[rp, k, i, j].setlb(round(max(model.pBusMinV[i] ** 2, 0.1), 4))
    second_stage_variables.append(model.vSOCP_cij)

    model.vSOCP_sij = pyo.Var(model.rp, model.k, model.la_no_c, domain=pyo.Reals)  # sij = (vi^real* vj^imag) - vi^re*vj^imag))
    for (i, j, c) in model.le:
        for rp in model.rp:
            for k in model.k:
                if (rp, k, i, j) in model.vSOCP_sij:
                    model.vSOCP_sij[rp, k, i, j].setub(round(model.pBusMaxV[i] ** 2, 4))
                    model.vSOCP_sij[rp, k, i, j].setlb(round(-model.pBusMaxV[i] ** 2, 4))
    second_stage_variables.append(model.vSOCP_sij)

    # Set bounds for reversed direction (la_reverse)
    for (j, i, c) in model.le_reverse:
        for rp in model.rp:
            for k in model.k:
                model.vLineQ[rp, k, j, i, c].setlb(-model.pQmax[i, j, c])
                model.vLineQ[rp, k, j, i, c].setub(model.pQmax[i, j, c])

    model.vSOCP_IndicConnecNodes = pyo.Var({(i, j) for (i, j, c) in model.lc}, domain=pyo.Binary)
    second_stage_variables.append(model.vSOCP_IndicConnecNodes)

    model.vGenQ = pyo.Var(model.rp, model.k, model.g, doc='Reactive power output of generator g', domain=pyo.Reals)
    second_stage_variables.append(model.vGenQ)

    if cs.dPower_Parameters["pEnableThermalGen"]:
        for g in model.thermalGenerators:
            for rp in model.rp:
                for k in model.k:
                    model.vGenQ[rp, k, g].setlb(model.pMinGenQ[g])
                    model.vGenQ[rp, k, g].setub(model.pMaxGenQ[g])

    if cs.dPower_Parameters["pEnableRoR"]:
        for g in model.rorGenerators:
            for rp in model.rp:
                for k in model.k:
                    model.vGenQ[rp, k, g].setlb(model.pMinGenQ[g])
                    model.vGenQ[rp, k, g].setub(model.pMaxGenQ[g])

    if cs.dPower_Parameters["pEnableVRES"]:
        for g in model.vresGenerators:
            for rp in model.rp:
                for k in model.k:
                    model.vGenQ[rp, k, g].setlb(model.pMinGenQ[g])
                    model.vGenQ[rp, k, g].setub(model.pMaxGenQ[g])

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    # Constraint definitions

    # Constraint implementations

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = 0.0

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
