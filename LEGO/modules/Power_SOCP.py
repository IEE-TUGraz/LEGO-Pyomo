import pyomo.environ as pyo
from LEGO import LEGO, LEGOUtilities
import pandas as pd

def add_element_definitions_and_bounds(lego: LEGO):

    # Helper function getting the first circuit for each (i, j) pair

    # Reset index to get (i, j, c) as columns
    df_circuits = lego.cs.dPower_Network.reset_index()

    # Sort the DataFrame by the desired circuit order
    df_circuits["c_str"] = df_circuits["c"].astype(str)
    ordered_circuits = list(dict.fromkeys(df_circuits["c_str"].tolist()))
    circuit_order = {c: idx for idx, c in enumerate(ordered_circuits)}
    df_circuits["c_order"] = df_circuits["c_str"].map(circuit_order)

    # Get the first circuit per (i, j) based on this order
    first_circuit_map = (
        df_circuits.sort_values("c_order")
        .drop_duplicates(subset=["i", "j"])
        .set_index(["i", "j"])["c"]
        .to_dict()
    )

    # Create a bidirectional version for bidirectional lines in the SOCP formulation
    first_circuit_map_bidir = {}
    for (i, j), c in first_circuit_map.items():
        first_circuit_map_bidir[(i, j)] = c
        first_circuit_map_bidir[(j, i)] = c  # Add reverse direction

    # Store in lego namespace
    lego.first_circuit_map = first_circuit_map
    lego.first_circuit_map_bidir = first_circuit_map_bidir

    # Helper function for creating reverse and bidirectional sets
    def make_reverse_set(original_set):
        reverse = []
        for (i, j, c) in original_set:
            reverse.append((j, i, c))
        return reverse


    # Create set of all reverse lines
    lego.model.la_reverse = pyo.Set(
        doc='Reverse lines (la)',
        initialize=lambda model: make_reverse_set(model.la),
        dimen=3
    )

    # Create set of all lines without the circuit dependency(needed for SOCP variables)
    lego.model.la_no_c = pyo.Set(
        doc='All lines without circuit dependency (la_no_c)',
        initialize=lambda model: {(i, j) for (i, j, c) in model.la},
        dimen=2
    )

    # Create set of all lines including reverse lines
    lego.model.la_full = pyo.Set(
        initialize=lambda m: set(m.la) | set(m.la_reverse),
        dimen=3
    )

    # Create sets for existing reverse lines
    lego.model.le_reverse = pyo.Set(
        doc='Reverse lines (le)',
        initialize=lambda model: make_reverse_set(model.le),
        within=lego.model.la_reverse,
        dimen=3
    )
    # Create set of all existing lines including reverse lines
    lego.model.le_full = pyo.Set(
        initialize=lambda m: set(m.le) | set(m.le_reverse),
        within=lego.model.la_full,
        dimen=3
    )
    # Create sets for existing lines without circuit dependency (needed for SOCP constraints)
    lego.model.le_no_c = pyo.Set(
        doc='Existing lines without circuit dependency (le_no_c)',
        initialize=lambda model: {(i, j) for (i, j, c) in model.le},
        dimen=2,
    )
    # Create sets for candidate reverse lines
    lego.model.lc_reverse = pyo.Set(
        doc='Reverse lines (lc)',
        initialize=lambda model: make_reverse_set(model.lc),
        within=lego.model.la_reverse,
        dimen=3
    )
    # Create set of all candidate lines including reverse lines
    lego.model.lc_full = pyo.Set(
        initialize=lambda m: set(m.lc) | set(m.lc_reverse),
        within=lego.model.la_full,
        dimen=3
    )
    # Create set of all candidate lines including reverse lines without circuit dependency (needed for SOCP constraints)
    lego.model.lc_full_no_c = pyo.Set(
        doc='All Candidate lines without circuit dependency (lc_no_c)',
        initialize=lambda model: {(i, j) for (i, j, c) in model.lc_full},
        dimen=2
    )
    lego.model.lc_no_c = pyo.Set(
        doc='Candidate lines without circuit dependency (lc_no_c)',
        initialize=lambda model: {(i, j) for (i, j, c) in model.lc},
        dimen=2
    )
    # SOCP Variables

    lego.model.vSOCP_cii = pyo.Var(lego.model.rp, lego.model.k, lego.model.i, domain=pyo.Reals)
    for rp in lego.model.rp:
        for k in lego.model.k:
            for i in lego.model.i:
                lego.model.vSOCP_cii[rp, k, i].setub(round(lego.model.pBusMaxV[i] ** 2,4))  # Set upper bound for cii
                lego.model.vSOCP_cii[rp, k, i].setlb(round(lego.model.pBusMinV[i] ** 2,4))


    lego.model.vSOCP_cij = pyo.Var(lego.model.rp, lego.model.k, lego.model.la_no_c, domain=pyo.Reals, bounds=(0,None))  # cij = (vi^real* vj^real) + vi^imag*vj^imag), Lower bounds for vSOCP_cij need to always be at least 0
    for (i, j, c) in lego.model.le:
        for rp in lego.model.rp:
            for k in lego.model.k:
                if (rp, k, i, j) in lego.model.vSOCP_cij:
                    lego.model.vSOCP_cij[rp, k, i, j].setub(round(lego.model.pBusMaxV[i] ** 2,4))
                    lego.model.vSOCP_cij[rp, k, i, j].setlb(round(max(lego.model.pBusMinV[i] ** 2, 0.1),4))

    lego.model.vSOCP_sij = pyo.Var(lego.model.rp, lego.model.k, lego.model.la_no_c, domain=pyo.Reals)  # sij = (vi^real* vj^imag) - vi^re*vj^imag))
    for (i, j, c) in lego.model.le:
        for rp in lego.model.rp:
            for k in lego.model.k:
                if (rp, k, i, j) in lego.model.vSOCP_sij:
                    lego.model.vSOCP_sij[rp, k, i, j].setub(round(lego.model.pBusMaxV[i] ** 2,4))
                    lego.model.vSOCP_sij[rp, k, i, j].setlb(round(-lego.model.pBusMaxV[i] ** 2,4))


    lego.model.vLineQ = pyo.Var(lego.model.rp, lego.model.k, lego.model.la_full, domain=pyo.Reals) # Reactive power flow from bus i to j
    for (i, j, c) in lego.model.le:
        for rp in lego.model.rp:
            for k in lego.model.k:
                lego.model.vLineQ[rp, k, i, j, c].setlb(-lego.model.pQmax[i, j, c])
                lego.model.vLineQ[rp, k, i, j, c].setub(lego.model.pQmax[i, j, c])

    # Set bounds for reversed direction (la_reverse)
    for (j, i, c) in lego.model.le_reverse:
        for rp in lego.model.rp:
            for k in lego.model.k:
                lego.model.vLineQ[rp, k, j, i, c].setlb(-lego.model.pQmax[i, j, c])
                lego.model.vLineQ[rp, k, j, i, c].setub(lego.model.pQmax[i, j, c])

    lego.model.vSOCP_IndicConnecNodes = pyo.Var({(i, j) for (i, j, c) in lego.model.lc}, domain=pyo.Binary)

    lego.model.vGenQ = pyo.Var(lego.model.rp, lego.model.k, lego.model.g, doc='Reactive power output of generator g', domain=pyo.Reals)

    if lego.cs.dPower_Parameters["pEnableThermalGen"]:
        for g in lego.model.thermalGenerators:
            for rp in lego.model.rp:
                for k in lego.model.k:
                    lego.model.vGenQ[rp, k, g].setlb(lego.model.pMinGenQ[g])
                    lego.model.vGenQ[rp, k, g].setub(lego.model.pMaxGenQ[g])

    if lego.cs.dPower_Parameters["pEnableRoR"]:
        for g in lego.model.rorGenerators:
            for rp in lego.model.rp:
                for k in lego.model.k:
                    lego.model.vGenQ[rp, k, g].setlb(lego.model.pMinGenQ[g])
                    lego.model.vGenQ[rp, k, g].setub(lego.model.pMaxGenQ[g])

    if lego.cs.dPower_Parameters["pEnableVRES"]:
        for g in lego.model.vresGenerators:
            for rp in lego.model.rp:
                for k in lego.model.k:
                    lego.model.vGenQ[rp, k, g].setlb(lego.model.pMinGenQ[g])
                    lego.model.vGenQ[rp, k, g].setub(lego.model.pMaxGenQ[g])

    dDCOPFIslands = pd.DataFrame(index=lego.cs.dPower_BusInfo.index, columns=[lego.cs.dPower_BusInfo.index], data=False)

    for index, entry in lego.cs.dPower_Network.iterrows():
        if lego.cs.dPower_Network.loc[(index[0], index[1], index[2])]["pTecRepr"] == "DC-OPF" or "SOCP":
            dDCOPFIslands.loc[index[0], index[1]] = True
            dDCOPFIslands.loc[index[1], index[0]] = True
    completed_buses = set()  # Set of buses that have been looked at already
    i = 0
    for index, entry in dDCOPFIslands.iterrows():
        if index in completed_buses or entry[entry == True].empty:
            continue
        connected_buses = lego.cs.get_connected_buses(dDCOPFIslands, str(index))
        for bus in connected_buses:
            completed_buses.add(bus)
        completed_buses.add(index)
    # Set slack node for SOCP zones
    slack_node = lego.cs.dPower_Demand.loc[:, :, connected_buses].groupby('i').sum().idxmax().values[0]
    slack_node = lego.cs.dPower_Parameters["is"]  # TODO: Switch this again to be calculated (fixed to 'is' for compatibility)
    if i == 0: print("Setting slack nodes for SOCP zones:")
    i += 1
    lego.model.vSOCP_cii[:, :, slack_node].fix(pyo.sqrt(lego.cs.dPower_Parameters['pSlackVoltage']))
    print(f"SOCP {i:>2} - Slack node: {slack_node}")
    print("Fixed voltage magnitude at slack node:", pyo.value(pyo.sqrt(lego.cs.dPower_Parameters['pSlackVoltage'])))
    lego.model.vTheta[:, :, slack_node].fix(0)

    lego.model.vLineP = pyo.Var(lego.model.rp, lego.model.k, lego.model.la_full, doc='Power flow from bus i to j', bounds=(None, None))
    for (i, j, c) in lego.model.la:
        for rp in lego.model.rp:
            for k in lego.model.k:
                lego.model.vLineP[rp, k, i, j, c].setlb(-lego.model.pPmax[i, j, c])
                lego.model.vLineP[rp, k, i, j, c].setub(lego.model.pPmax[i, j, c])


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):

     def eDC_BalanceP_rule(model, rp, k, i):
         return (
                 sum(model.vGenP[rp, k, g] for g in model.g if (g, i) in model.gi)  # Gen at bus i
                 - sum(model.vLineP[rp, k, i, j, c] for (i2, j, c) in model.la_full if i2 == i)  # Only outflows from i
                 - model.vSOCP_cii[rp, k, i] * model.pBusG[i] * model.pSBase
                 - model.pDemandP[rp, k, i]
                 + model.vPNS[rp, k, i]
                 - model.vEPS[rp, k, i]
         )

     lego.model.eDC_BalanceP_expr = pyo.Expression(lego.model.rp, lego.model.k, lego.model.i, rule=eDC_BalanceP_rule)
     lego.model.eDC_BalanceP = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.i, doc='Power balance constraint for each bus', rule=lambda model, rp, k, i: lego.model.eDC_BalanceP_expr[rp, k, i] == 0)