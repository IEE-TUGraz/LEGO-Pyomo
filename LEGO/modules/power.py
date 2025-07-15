import numpy as np
import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGOUtilities, LEGO

global first_circuit_map, first_circuit_map_bidir  # TODO: Solve this differently


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    global first_circuit_map, first_circuit_map_bidir  # TODO: Solve this differently
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.i = pyo.Set(doc='Buses', initialize=cs.dPower_BusInfo.index.tolist())

    model.c = pyo.Set(doc='Circuits', initialize=cs.dPower_Network.index.get_level_values('c').unique().tolist())
    model.la = pyo.Set(doc='All lines', initialize=cs.dPower_Network.index.tolist(), within=model.i * model.i * model.c)
    model.le = pyo.Set(doc='Existing lines', initialize=cs.dPower_Network[(cs.dPower_Network["pEnableInvest"] == 0)].index.tolist(), within=model.la)
    model.lc = pyo.Set(doc='Candidate lines', initialize=cs.dPower_Network[(cs.dPower_Network["pEnableInvest"] == 1)].index.tolist(), within=model.la)

    if cs.dPower_Parameters["pEnableSOCP"]:
        # Helper function getting the first circuit for each (i, j) pair

        # Reset index to get (i, j, c) as columns
        df_circuits = cs.dPower_Network.reset_index()

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

    if cs.dPower_Parameters["pEnableSOCP"]:
        # Helper function for creating reverse and bidirectional sets
        def make_reverse_set(original_set):
            reverse = []
            for (i, j, c) in original_set:
                reverse.append((j, i, c))
            return reverse

        # Create set of all reverse lines
        model.la_reverse = pyo.Set(
            doc='Reverse lines (la)',
            initialize=lambda model: make_reverse_set(model.la),
            dimen=3
        )

        # Create set of all lines without the circuit dependency(needed for SOCP variables)
        model.la_no_c = pyo.Set(
            doc='All lines without circuit dependency (la_no_c)',
            initialize=lambda model: {(i, j) for (i, j, c) in model.la},
            dimen=2
        )

        # Create set of all lines including reverse lines
        model.la_full = pyo.Set(
            initialize=lambda m: set(m.la) | set(m.la_reverse),
            dimen=3
        )

        # Create sets for existing reverse lines
        model.le_reverse = pyo.Set(
            doc='Reverse lines (le)',
            initialize=lambda model: make_reverse_set(model.le),
            within=model.la_reverse,
            dimen=3
        )
        # Create set of all existing lines including reverse lines
        model.le_full = pyo.Set(
            initialize=lambda m: set(m.le) | set(m.le_reverse),
            within=model.la_full,
            dimen=3
        )
        # Create sets for existing lines without circuit dependency (needed for SOCP constraints)
        model.le_no_c = pyo.Set(
            doc='Existing lines without circuit dependency (le_no_c)',
            initialize=lambda model: {(i, j) for (i, j, c) in model.le},
            dimen=2,
        )
        # Create sets for candidate reverse lines
        model.lc_reverse = pyo.Set(
            doc='Reverse lines (lc)',
            initialize=lambda model: make_reverse_set(model.lc),
            within=model.la_reverse,
            dimen=3
        )
        # Create set of all candidate lines including reverse lines
        model.lc_full = pyo.Set(
            initialize=lambda m: set(m.lc) | set(m.lc_reverse),
            within=model.la_full,
            dimen=3
        )
        # Create set of all candidate lines including reverse lines without circuit dependency (needed for SOCP constraints)
        model.lc_full_no_c = pyo.Set(
            doc='All Candidate lines without circuit dependency (lc_no_c)',
            initialize=lambda model: {(i, j) for (i, j, c) in model.lc_full},
            dimen=2
        )
        model.lc_no_c = pyo.Set(
            doc='Candidate lines without circuit dependency (lc_no_c)',
            initialize=lambda model: {(i, j) for (i, j, c) in model.lc},
            dimen=2
        )

    model.g = pyo.Set(doc='Generators')
    model.gi = pyo.Set(doc='Generator g connected to bus i', within=model.g * model.i)

    if cs.dPower_Parameters["pEnableThermalGen"]:
        model.thermalGenerators = pyo.Set(doc='Thermal Generators', initialize=cs.dPower_ThermalGen.index.tolist())
        LEGO.addToSet(model, "g", model.thermalGenerators)
        LEGO.addToSet(model, "gi", cs.dPower_ThermalGen.reset_index().set_index(['g', 'i']).index)

    if cs.dPower_Parameters["pEnableRoR"]:
        model.rorGenerators = pyo.Set(doc='Run-of-river generators', initialize=cs.dPower_RoR.index.tolist())
        LEGO.addToSet(model, "g", model.rorGenerators)
        LEGO.addToSet(model, "gi", cs.dPower_RoR.reset_index().set_index(['g', 'i']).index)

    if cs.dPower_Parameters["pEnableVRES"]:
        model.vresGenerators = pyo.Set(doc='Variable renewable energy sources', initialize=cs.dPower_VRES.index.tolist())
        LEGO.addToSet(model, "g", model.vresGenerators)
        LEGO.addToSet(model, "gi", cs.dPower_VRES.reset_index().set_index(['g', 'i']).index)

    model.p = pyo.Set(doc='Periods', initialize=cs.dPower_Hindex.index.get_level_values('p').unique().tolist())
    model.rp = pyo.Set(doc='Representative periods', initialize=cs.dPower_Demand.index.get_level_values('rp').unique().tolist())
    model.k = pyo.Set(doc='Timestep within representative period', initialize=cs.dPower_Demand.index.get_level_values('k').unique().tolist())
    model.hindex = cs.dPower_Hindex.index

    # Parameters
    model.pDemandP = pyo.Param(model.rp, model.k, model.i, initialize=cs.dPower_Demand['value'], doc='Demand at bus i in representative period rp and timestep k')
    model.pMovWindow = cs.dGlobal_Parameters['pMovWindow']

    model.pOMVarCost = pyo.Param(model.g, doc='Production cost of generator g')
    model.pEnabInv = pyo.Param(model.g, doc='Enable investment in thermal generator g')
    model.pMaxInvest = pyo.Param(model.g, doc='Maximum investment in thermal generator g')
    model.pInvestCost = pyo.Param(model.g, doc='Investment cost for thermal generator g')
    model.pMaxProd = pyo.Param(model.g, doc='Maximum production of generator g')
    model.pMinProd = pyo.Param(model.g, doc='Minimum production of generator g')
    model.pExisUnits = pyo.Param(model.g, doc='Existing units of generator g')
    model.pMaxGenQ = pyo.Param(model.g, doc='Maximum reactive production of generator g')
    model.pMinGenQ = pyo.Param(model.g, doc='Minimum reactive production of generator g')

    if cs.dPower_Parameters["pEnableThermalGen"]:
        LEGO.addToParameter(model, "pOMVarCost", cs.dPower_ThermalGen['pSlopeVarCostEUR'])
        LEGO.addToParameter(model, "pEnabInv", cs.dPower_ThermalGen['EnableInvest'])
        LEGO.addToParameter(model, "pMaxInvest", cs.dPower_ThermalGen['MaxInvest'])
        LEGO.addToParameter(model, "pInvestCost", cs.dPower_ThermalGen['InvestCostEUR'])
        LEGO.addToParameter(model, "pMaxProd", cs.dPower_ThermalGen['MaxProd'])
        LEGO.addToParameter(model, "pMinProd", cs.dPower_ThermalGen['MinProd'])
        LEGO.addToParameter(model, "pExisUnits", cs.dPower_ThermalGen['ExisUnits'])

        # Reactive generator power limits for SOCP formulation
        LEGO.addToParameter(model, 'pMaxGenQ', 1e-3 * cs.dPower_ThermalGen['Qmax'].fillna(0))  # Convert from MVar to kVar
        LEGO.addToParameter(model, 'pMinGenQ', 1e-3 * cs.dPower_ThermalGen['Qmin'].fillna(0))  # Convert from MVar to kVar

        model.pInterVarCost = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['pInterVarCostEUR'], doc='Inter-variable cost of thermal generator g')
        model.pStartupCost = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['pStartupCostEUR'], doc='Startup cost of thermal generator g')
        model.pMinUpTime = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['MinUpTime'], doc='Minimum up time of thermal generator g')
        model.pMinDownTime = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['MinDownTime'], doc='Minimum down time of thermal generator g')
        model.pRampUp = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['RampUp'], doc='Ramp up of thermal generator g')
        model.pRampDw = pyo.Param(model.thermalGenerators, initialize=cs.dPower_ThermalGen['RampDw'], doc='Ramp down of thermal generator g')

    if cs.dPower_Parameters["pEnableRoR"]:
        LEGO.addToParameter(model, "pOMVarCost", cs.dPower_RoR['OMVarCost'])
        LEGO.addToParameter(model, "pEnabInv", cs.dPower_RoR['EnableInvest'])
        LEGO.addToParameter(model, "pMaxInvest", cs.dPower_RoR['MaxInvest'])
        LEGO.addToParameter(model, "pInvestCost", cs.dPower_RoR['InvestCostEUR'])
        LEGO.addToParameter(model, "pMaxProd", cs.dPower_RoR['MaxProd'])
        LEGO.addToParameter(model, "pMinProd", cs.dPower_RoR['MinProd'])
        LEGO.addToParameter(model, "pExisUnits", cs.dPower_RoR['ExisUnits'])

        # Reactive ROR power limits for SOCP formulation
        LEGO.addToParameter(model, 'pMaxGenQ', 1e-3 * cs.dPower_RoR['Qmax'].fillna(0))  # Convert from MVar to kVar
        LEGO.addToParameter(model, 'pMinGenQ', 1e-3 * cs.dPower_RoR['Qmin'].fillna(0))  # Convert from MVar to kVar

    if cs.dPower_Parameters["pEnableVRES"]:
        LEGO.addToParameter(model, "pOMVarCost", cs.dPower_VRES['OMVarCost'])
        LEGO.addToParameter(model, "pEnabInv", cs.dPower_VRES['EnableInvest'])
        LEGO.addToParameter(model, "pMaxInvest", cs.dPower_VRES['MaxInvest'])
        LEGO.addToParameter(model, "pInvestCost", cs.dPower_VRES['InvestCostEUR'])
        LEGO.addToParameter(model, "pMaxProd", cs.dPower_VRES['MaxProd'])
        LEGO.addToParameter(model, "pMinProd", cs.dPower_VRES['MinProd'])
        LEGO.addToParameter(model, "pExisUnits", cs.dPower_VRES['ExisUnits'])

        # Reactive VRES power limits for SOCP formulation
        LEGO.addToParameter(model, 'pMaxGenQ', 1e-3 * cs.dPower_VRES['Qmax'].fillna(0))  # Convert from MVar to kVar
        LEGO.addToParameter(model, 'pMinGenQ', 1e-3 * cs.dPower_VRES['Qmin'].fillna(0))  # Convert from MVar to kVar

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

    # SOCP Parameters
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
    model.vTheta = pyo.Var(model.rp, model.k, model.i, doc='Angle of bus i', bounds=(-cs.dPower_Parameters["pMaxAngleDCOPF"], cs.dPower_Parameters["pMaxAngleDCOPF"]))  # TODO: Discuss impact on runtime etc.(based on discussion with Prof. Renner)
    second_stage_variables += [model.vTheta]
    model.vAngle = pyo.Var(model.rp, model.k, model.la, doc='Angle phase shifting transformer')
    second_stage_variables += [model.vAngle]
    for i, j, c in model.la:
        if model.pAngle[i, j, c] == 0:
            model.vAngle[:, :, i, j, c].fix(0)
        else:
            model.vAngle[:, :, i, j, c].setub(model.pAngle[i, j, c])
            model.vAngle[:, :, i, j, c].setlb(-model.pAngle[i, j, c])

    if cs.dPower_Parameters["pEnableSOCP"]:
        model.vLineInvest = pyo.Var(model.la, doc='Transmission line investment', domain=pyo.Binary)
        for i, j, c in model.le:
            model.vLineInvest[i, j, c].fix(0)  # Set existing lines to not investable
    else:
        model.vLineInvest = pyo.Var(model.la, doc='Transmission line investment', domain=pyo.Binary)
        for i, j, c in model.le:
            model.vLineInvest[i, j, c].fix(0)  # Set existing lines to not investable
    first_stage_variables += [model.vLineInvest]

    model.vGenInvest = pyo.Var(model.g, doc="Integer generation investment", bounds=lambda model, g: (0, model.pMaxInvest[g] * model.pEnabInv[g]))
    first_stage_variables += [model.vGenInvest]

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
        if not cs.dPower_Parameters["pEnableSOCP"]:
            slack_node = cs.dPower_Demand.loc[:, :, connected_buses].groupby('i').sum().idxmax().values[0]
            slack_node = cs.dPower_Parameters["is"]  # TODO: Switch this again to be calculated (fixed to 'is' for compatibility)
            if i == 0: print("Setting slack nodes for DC-OPF zones:")
            print(f"DC-OPF Zone {i:>2} - Slack node: {slack_node}")
            i += 1
            model.vTheta[:, :, slack_node].fix(0)

    model.vPNS = pyo.Var(model.rp, model.k, model.i, doc='Slack variable power not served', bounds=lambda model, rp, k, i: (0, model.pDemandP[rp, k, i]))
    second_stage_variables += [model.vPNS]
    model.vEPS = pyo.Var(model.rp, model.k, model.i, doc='Slack variable excess power served', bounds=(0, None))
    second_stage_variables += [model.vEPS]

    # Used to relax vCommit, vStartup and vShutdown in the first timesteps of each representative period
    # Required when using Markov-Chains to connect the timesteps of the representative periods - since fractions of the binary variables (which are present due to the transition-probabilities) are otherwise not possible
    def vUC_domain(model, k, relax_duration_from_beginning):
        if model.k.ord(k) <= relax_duration_from_beginning:
            return pyo.PercentFraction  # PercentFraction = Floating point values in the interval [0,1]
        else:
            return pyo.Binary

    model.vGenP = pyo.Var(model.rp, model.k, model.g, doc='Power output of generator g', bounds=lambda model, rp, k, g: (0, model.pMaxProd[g] * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g])))
    second_stage_variables += [model.vGenP]

    if cs.dPower_Parameters["pEnableThermalGen"]:
        model.vCommit = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Unit commitment of generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, max(model.pMinUpTime[t], model.pMinDownTime[t])) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
        second_stage_variables += [model.vCommit]
        model.vStartup = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Start-up of thermal generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, model.pMinDownTime[t]) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
        second_stage_variables += [model.vStartup]
        model.vShutdown = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Shut-down of thermal generator g', domain=lambda model, rp, k, t: vUC_domain(model, k, model.pMinUpTime[t]) if cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "markov" else pyo.Binary)
        second_stage_variables += [model.vShutdown]
        model.vGenP1 = pyo.Var(model.rp, model.k, model.thermalGenerators, doc='Power output of generator g above minimum production', bounds=lambda model, rp, k, g: (0, (model.pMaxProd[g] - model.pMinProd[g]) * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g])))
        second_stage_variables += [model.vGenP1]

    if cs.dPower_Parameters["pEnableRoR"]:
        for g in model.rorGenerators:
            for rp in model.rp:
                for k in model.k:
                    model.vGenP[rp, k, g].setub(min(model.pMaxProd[g], cs.dPower_Inflows.loc[rp, g, k]['Inflow']))  # TODO: Check and adapt for storage

    if cs.dPower_Parameters["pEnableVRES"]:
        for g in model.vresGenerators:
            for rp in model.rp:
                for k in model.k:
                    model.vGenP[rp, k, g].setub((model.pMaxProd[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g])) * cs.dPower_VRESProfiles.loc[rp, k, g]['value']))

    model.vLineP = pyo.Var(model.rp, model.k, model.la, doc='Power flow from bus i to j', bounds=(None, None))
    if cs.dPower_Parameters["pEnableSOCP"]:  # Bound sonly apply in forward direction for existing and candidate lines
        model.vLineP = pyo.Var(model.rp, model.k, model.la_full, doc='Power flow from bus i to j', bounds=(None, None))
        for (i, j, c) in model.la:
            for rp in model.rp:
                for k in model.k:
                    model.vLineP[rp, k, i, j, c].setlb(-model.pPmax[i, j, c])
                    model.vLineP[rp, k, i, j, c].setub(model.pPmax[i, j, c])
    else:
        model.vLineP = pyo.Var(model.rp, model.k, model.la, doc='Power flow from bus i to j', bounds=(None, None))
        for (i, j, c) in model.la:
            match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
                case "DC-OPF" | "TP":
                    model.vLineP[:, :, (i, j), c].setlb(-model.pPmax[i, j, c])
                    model.vLineP[:, :, (i, j), c].setub(model.pPmax[i, j, c])
                case "SN":
                    assert False  # "SN" line found, although all "Single Node" buses should be merged
                case _:
                    raise ValueError(f"Technical representation '{cs.dPower_Network.loc[i, j]["pTecRepr"]}' "
                                     f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")
    second_stage_variables += [model.vLineP]

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    global first_circuit_map, first_circuit_map_bidir  # TODO: Solve this differently

    # Power balance for nodes DC ann SOCP
    def eDC_BalanceP_rule(model, rp, k, i):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (
                    sum(model.vGenP[rp, k, g] for g in model.g if (g, i) in model.gi)  # Gen at bus i
                    - sum(model.vLineP[rp, k, i, j, c] for (i2, j, c) in model.la_full if i2 == i)  # Only outflows from i
                    - model.vSOCP_cii[rp, k, i] * model.pBusG[i] * model.pSBase
                    - model.pDemandP[rp, k, i]
                    + model.vPNS[rp, k, i]
                    - model.vEPS[rp, k, i]
            )
        else:
            return (sum(model.vGenP[rp, k, g] for g in model.g if (g, i) in model.gi) -  # Production of generators at bus i
                    sum(model.vLineP[rp, k, e] for e in model.la if (e[0] == i)) +  # Power flow from bus i to bus j
                    sum(model.vLineP[rp, k, e] for e in model.la if (e[1] == i)) -  # Power flow from bus j to bus i
                    model.pDemandP[rp, k, i] +  # Demand at bus i
                    model.vPNS[rp, k, i] -  # Slack variable for demand not served
                    model.vEPS[rp, k, i])  # Slack variable for overproduction

    # Note: eDC_BalanceP_expr is defined as expression to enable later adding coefficients to the constraint (e.g., for import/export)
    model.eDC_BalanceP_expr = pyo.Expression(model.rp, model.k, model.i, rule=eDC_BalanceP_rule)
    model.eDC_BalanceP = pyo.Constraint(model.rp, model.k, model.i, doc='Power balance constraint for each bus', rule=lambda model, rp, k, i: model.eDC_BalanceP_expr[rp, k, i] == 0)

    # Note: eDC_BalanceP_expr is defined as expression to enable later adding coefficients to the constraint (e.g., for import/export)

    def eSOCP_BalanceQ_rule(model, rp, k, i):
        return (
                sum(model.vGenQ[rp, k, g] for g in model.g if (g, i) in model.gi)
                # Only vLineQ where i is the sending end (i → j)
                - sum(model.vLineQ[rp, k, i, j, c] for (i2, j, c) in model.la_full if i2 == i)
                + model.vSOCP_cii[rp, k, i] * model.pBusB[i] * model.pSBase
                - model.pDemandQ[rp, k, i]
                + model.vPNS[rp, k, i] * model.pRatioDemQP[i]
                - model.vEPS[rp, k, i] * model.pRatioDemQP[i]
        )

    def eDC_ExiLinePij_rule(model, rp, k, i, j, c):
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF":
                return model.vLineP[rp, k, i, j, c] == (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) * model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c])
            case "TP" | "SN" | "SOCP":
                return pyo.Constraint.Skip
            case _:
                raise ValueError(f"Technical representation '{cs.dPower_Network.loc[i, j]["pTecRepr"]}' "
                                 f"for line ({i}, {j}) not recognized - please check input file 'Power_Network.xlsx'!")

    if not cs.dPower_Parameters["pEnableSOCP"]:
        model.eDC_ExiLinePij = pyo.Constraint(model.rp, model.k, model.le, doc="Power flow existing lines (for DC-OPF)", rule=eDC_ExiLinePij_rule)

    model.eDC_ExiLinePij = pyo.Constraint(model.rp, model.k, model.le, doc="Power flow existing lines (for DC-OPF)", rule=eDC_ExiLinePij_rule)

    def eDC_CanLinePij1_rule(model, rp, k, i, j, c):
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF":
                return (
                        model.vLineP[rp, k, i, j, c] / (model.pBigM_Flow * model.pPmax[i, j, c]) >=
                        (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) *
                        model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c]) /
                        (model.pBigM_Flow * model.pPmax[i, j, c]) - 1 + model.vLineInvest[i, j, c]
                )
            case "TP" | "SN" | "SOCP":
                return pyo.Constraint.Skip
            case _:
                raise ValueError(f"Unsupported pTecRepr: {cs.dPower_Network.loc[i, j, c]['pTecRepr']}")

    if not cs.dPower_Parameters["pEnableSOCP"]:
        model.eDC_CanLinePij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij1_rule)

    model.eDC_CanLinePij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij1_rule)

    def eDC_CanLinePij2_rule(model, rp, k, i, j, c):
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF":
                return (
                        model.vLineP[rp, k, i, j, c] / (model.pBigM_Flow * model.pPmax[i, j, c]) <=
                        (model.vTheta[rp, k, i] - model.vTheta[rp, k, j] + model.vAngle[rp, k, i, j, c]) *
                        model.pSBase / (model.pXline[i, j, c] * model.pRatio[i, j, c]) /
                        (model.pBigM_Flow * model.pPmax[i, j, c]) + 1 - model.vLineInvest[i, j, c]
                )
            case "TP" | "SN" | "SOCP":
                return pyo.Constraint.Skip
            case _:
                raise ValueError(f"Unsupported pTecRepr: {cs.dPower_Network.loc[i, j, c]['pTecRepr']}")

    if not cs.dPower_Parameters["pEnableSOCP"]:
        model.eDC_CanLinePij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij2_rule)

    model.eDC_CanLinePij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow candidate lines (for DC-OPF)", rule=eDC_CanLinePij2_rule)

    def eDC_LimCanLine1_rule(model, rp, k, i, j, c):
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF" | "TP" | "SN":
                return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] + model.vLineInvest[i, j, c] >= 0
            case 'SOCP':
                return pyo.Constraint.Skip
        return pyo.Constraint.Skip

    if not cs.dPower_Parameters["pEnableSOCP"]:
        model.eDC_LimCanLine1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow limit standart direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine1_rule)

    model.eDC_LimCanLine1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow limit reverse direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine1_rule)

    def eDC_LimCanLine2_rule(model, rp, k, i, j, c):
        match cs.dPower_Network.loc[i, j, c]["pTecRepr"]:
            case "DC-OPF" | "TP" | "SN":
                return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] - model.vLineInvest[i, j, c] <= 0
            case 'SOCP':
                return pyo.Constraint.Skip
        return pyo.Constraint.Skip

    if not cs.dPower_Parameters["pEnableSOCP"]:
        model.eDC_LimCanLine2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Power flow limit reverse direction for candidate lines (for DC-OPF)", rule=eDC_LimCanLine2_rule)

    # Reactive power limits

    def eSOCP_QMaxOut_rule(model, rp, k, g):
        if cs.dPower_Parameters["pEnableSOCP"]:
            if model.pMaxGenQ[g] != 0:
                return model.vGenQ[rp, k, g] / model.pMaxGenQ[g] <= model.vCommit[rp, k, g]
            else:
                return pyo.Constraint.Skip
        else:
            return pyo.Constraint.Skip

    def eSOCP_QMinOut1_rule(model, rp, k, g):
        if cs.dPower_Parameters["pEnableSOCP"]:
            if model.pMinGenQ[g] >= 0:
                return model.vGenQ[rp, k, g] / model.pMinGenQ[g] >= model.vCommit[rp, k, g]
            else:
                return pyo.Constraint.Skip
        else:
            return pyo.Constraint.Skip

    def eSOCP_QMinOut2_rule(model, rp, k, g):
        if cs.dPower_Parameters["pEnableSOCP"]:
            if model.pMinGenQ[g] <= 0:
                return model.vGenQ[rp, k, g] / model.pMinGenQ[g] <= model.vCommit[rp, k, g]
            else:
                return pyo.Constraint.Skip
        else:
            return pyo.Constraint.Skip

    # FACTS (not yet Implemented)
    # TODO: Add FACTS as a set, add FACTS parameters to nodes i
    def eSOCP_QMaxFACTS_rule(model, rp, k, i):
        if cs.dPower_Parameters["pEnableSOCP"] == 99999:
            return (model.vGenQ[rp, k, i]) <= model.pMaxGenQ[i] * (model.pExisUnits[i] + model.vGenInvest[i])
        else:
            return pyo.Constraint.Skip

    def eSOCP_QMinFACTS_rule(model, rp, k, i):
        if cs.dPower_Parameters["pEnableSOCP"] == 99999:
            return (model.vGenQ[rp, k, i]) >= model.pMaxGenQ[i] * (model.pExisUnits[i] + model.vGenInvest[i])
        else:
            return pyo.Constraint.Skip

    # Active and reactive power flow on existing lines SOCP
    # Active power flow on existing lines
    def eSOCP_ExiLinePij_rule(model, rp, k, i, j, c):  # Fertig
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineP[rp, k, i, j, c] == model.pSBase * (
                    + model.pGline[i, j, c] * model.vSOCP_cii[rp, k, i] / (model.pRatio[i, j, c] ** 2)
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                    - (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * - model.vSOCP_sij[rp, k, i, j])
                    )
        else:
            return pyo.Constraint.Skip

    def eSOCP_ExiLinePji_rule(model, rp, k, i, j, c):  # Fertig
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineP[rp, k, j, i, c] == model.pSBase * (
                    + (model.pGline[i, j, c] * model.vSOCP_cii[rp, k, j])
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                    - (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j])
                    )
        else:
            return pyo.Constraint.Skip

    # Reactive power flow on existing lines
    def eSOCP_ExiLineQij_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineQ[rp, k, i, j, c] == model.pSBase * (
                    - model.vSOCP_cii[rp, k, i] * (model.pBline[i, j, c] + model.pBcline[i, j, c] / 2) / (model.pRatio[i, j, c] ** 2)
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * -model.vSOCP_sij[rp, k, i, j]
                    + (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
                    )
        else:
            return pyo.Constraint.Skip

    def eSOCP_ExiLineQji_rule(model, rp, k, i, j, c):  # Fertig
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineQ[rp, k, j, i, c] == model.pSBase * (
                    - model.vSOCP_cii[rp, k, j] * (model.pBline[i, j, c] + model.pBcline[i, j, c] / 2)
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j]
                    + (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
                    )
        else:
            return pyo.Constraint.Skip

    # Active Power flow limits for candidate lines c

    # Active and reactive power flow on candidte lines SOCP
    # Active power flow on candidate lines

    def eSOCP_CanLinePij1_rule(model, rp, k, i, j, c):  # Fertig
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineP[rp, k, i, j, c] >= model.pSBase * (
                    + model.pGline[i, j, c] * model.vSOCP_cii[rp, k, i] / (model.pRatio[i, j, c] ** 2)
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                    - (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * -model.vSOCP_sij[rp, k, i, j])
                    - model.pBigM_Flow * (1 - model.vLineInvest[i, j, c])
                    )
        else:
            return pyo.Constraint.Skip

    def eSOCP_CanLinePij2_rule(model, rp, k, i, j, c):  # Fertig...
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineP[rp, k, i, j, c] <= model.pSBase * (
                    + model.pGline[i, j, c] * model.vSOCP_cii[rp, k, i] / (model.pRatio[i, j, c] ** 2)
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                    - (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * -model.vSOCP_sij[rp, k, i, j])
                    + model.pBigM_Flow * (1 - model.vLineInvest[i, j, c])
                    )
        else:
            return pyo.Constraint.Skip

    def eSOCP_CanLinePji1_rule(model, rp, k, i, j, c):  # Fertig
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineP[rp, k, j, i, c] >= model.pSBase * (
                    + model.pGline[i, j, c] * model.vSOCP_cii[rp, k, j]
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                    - (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j])
                    - model.pBigM_Flow * (1 - model.vLineInvest[i, j, c])
                    )
        else:
            return pyo.Constraint.Skip

    def eSOCP_CanLinePji2_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineP[rp, k, j, i, c] <= model.pSBase * (
                    + model.pGline[i, j, c] * model.vSOCP_cii[rp, k, j]
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j]
                    - (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j])
                    + model.pBigM_Flow * (1 - model.vLineInvest[i, j, c])
                    )
        else:
            return pyo.Constraint.Skip

    # Reactive power flow on candidate lines
    def eSOCP_CanLineQij1_rule(model, rp, k, i, j, c):  # Fertig
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineQ[rp, k, i, j, c] >= model.pSBase * (
                    - model.vSOCP_cii[rp, k, i] * (model.pBline[i, j, c] + model.pBcline[i, j, c] / 2) / (model.pRatio[i, j, c] ** 2)
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * -model.vSOCP_sij[rp, k, i, j]
                    + (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
                    - model.pBigM_Flow * (1 - model.vLineInvest[i, j, c])
                    )
        else:
            return pyo.Constraint.Skip

    def eSOCP_CanLineQij2_rule(model, rp, k, i, j, c):  # Fertig
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineQ[rp, k, i, j, c] <= model.pSBase * (
                    - model.vSOCP_cii[rp, k, i] * (model.pBline[i, j, c] + model.pBcline[i, j, c] / 2) / (model.pRatio[i, j, c] ** 2)
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * -model.vSOCP_sij[rp, k, i, j]
                    + (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
                    + model.pBigM_Flow * (1 - model.vLineInvest[i, j, c])
                    )
        else:
            return pyo.Constraint.Skip

    def eSOCP_CanLineQji1_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineQ[rp, k, j, i, c] >= model.pSBase * (
                    - model.vSOCP_cii[rp, k, j] * (model.pBline[i, j, c] + model.pBcline[i, j, c] / 2)
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j]
                    + (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
                    - model.pBigM_Flow * (1 - model.vLineInvest[i, j, c])
                    )
        else:
            return pyo.Constraint.Skip

    def eSOCP_CanLineQji2_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return (model.vLineQ[rp, k, j, i, c] <= model.pSBase * (
                    - model.vSOCP_cii[rp, k, j] * (model.pBline[i, j, c] + model.pBcline[i, j, c] / 2)
                    - (1 / model.pRatio[i, j, c]) * (model.pGline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) + model.pBline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_sij[rp, k, i, j]
                    + (1 / model.pRatio[i, j, c]) * (model.pBline[i, j, c] * pyo.cos(model.pAngle[i, j, c]) - model.pGline[i, j, c] * pyo.sin(model.pAngle[i, j, c])) * model.vSOCP_cij[rp, k, i, j])
                    + model.pBigM_Flow * (1 - model.vLineInvest[i, j, c])
                    )
        else:
            return pyo.Constraint.Skip

    # Active and reactive power flow limits for candidates lines
    # Active power flow limits for candidate lines

    def eSOCP_LimCanLinePij1_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] >= - model.vLineInvest[i, j, c]
        else:
            return pyo.Constraint.Skip

    def eSOCP_LimCanLinePij2_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineP[rp, k, i, j, c] / model.pPmax[i, j, c] <= model.vLineInvest[i, j, c]
        else:
            return pyo.Constraint.Skip

    def eSOCP_LimCanLinePji1_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineP[rp, k, j, i, c] / model.pPmax[i, j, c] >= - model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip

    def eSOCP_LimCanLinePji2_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineP[rp, k, j, i, c] / model.pPmax[i, j, c] <= model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip

    # Reactive power flow limits for candidate lines

    def eSOCP_LimCanLineQij1_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineQ[rp, k, i, j, c] / model.pQmax[i, j, c] >= - model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip

    def eSOCP_LimCanLineQij2_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineQ[rp, k, i, j, c] / model.pQmax[i, j, c] <= model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip

    def eSOCP_LimCanLineQji1_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineQ[rp, k, j, i, c] / model.pQmax[i, j, c] >= - model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip

    def eSOCP_LimCanLineQji2_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"]:
            return model.vLineQ[rp, k, j, i, c] / model.pQmax[i, j, c] <= model.vLineInvest[i, j, c]
        return pyo.Constraint.Skip

    # SCOP constraints for existing and candidate lines
    # SOCP constraints for existing lines

    def eSOCP_ExiLine_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            if any((i, j, c) in model.le for c in model.c):
                return (model.vSOCP_cij[rp, k, i, j] ** 2 + model.vSOCP_sij[rp, k, i, j] ** 2 <= model.vSOCP_cii[rp, k, i] * model.vSOCP_cii[rp, k, j])
        return pyo.Constraint.Skip

    # SOCP constraints for candidate lines
    # Does only apply if the line is not in le (existing lines set) for the first circuit and is a candidate line (lc), therefore is not a candidate line in a different circuit while one already exists(parallel lines)   

    def eSOCP_CanLine_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            c = first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLine '.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in model.lc:
                if (i, j, c) in model.le:
                    return pyo.Constraint.Skip
                return (
                        model.vSOCP_cij[rp, k, i, j] ** 2 + model.vSOCP_sij[rp, k, i, j] ** 2 <= model.vSOCP_cii[rp, k, i] * model.vSOCP_cii[rp, k, j])
        return pyo.Constraint.Skip

    # Does only apply if the line is in le (existing lines set) for the first circuit and is a candidate line (lc), therefore is a candidate line in a different circuit while one already exists (parallel lines)   

    def eSOCP_CanLine_cij_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            c = first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLine_cij'.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in model.lc:
                if (i, j, c) not in model.le:
                    return pyo.Constraint.Skip
                return (
                        model.vSOCP_cij[rp, k, i, j] <= model.pBigM_SOCP * model.vSOCP_IndicConnecNodes[i, j])
        return pyo.Constraint.Skip

    def eSOCP_CanLine_sij1_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            c = first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line for eSOCP_CanLine_sij1 ({}, {})'.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in model.lc:
                if (i, j, c) not in model.le:
                    return pyo.Constraint.Skip
                return (
                        model.vSOCP_sij[rp, k, i, j] <= model.pBigM_SOCP * model.vSOCP_IndicConnecNodes[i, j])
        return pyo.Constraint.Skip

    def eSOCP_CanLine_sij2_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            c = first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLine_sij2'.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in model.lc:
                if (i, j, c) not in model.le:
                    return pyo.Constraint.Skip
                return (
                        model.vSOCP_sij[rp, k, i, j] >= -model.pBigM_SOCP * model.vSOCP_IndicConnecNodes[i, j])

        return pyo.Constraint.Skip

    def eSOCP_IndicConnecNodes1_rule(model, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            c = first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_IndicConnecNodes1'.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in model.lc:
                if (i, j, c) not in model.le:
                    return pyo.Constraint.Skip
                return (sum(
                    model.vSOCP_IndicConnecNodes[i, j, c_]
                    for (i_, j_, c_) in model.lc
                    if i_ == i and j_ == j
                ) == 1)
        return pyo.Constraint.Skip

    def eSOCP_IndicConnecNodes2_rule(model, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            c = first_circuit_map.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_IndicConnecNodes2'.format(i, j))
                return pyo.Constraint.Skip

            if (i, j, c) in model.lc:
                if (i, j, c) in model.le:
                    return pyo.Constraint.Skip
                return (
                        model.vSOCP_IndicConnecNodes[i, j] == model.vLineInvest[i, j, c])
        return pyo.Constraint.Skip

    # Limits for SOCP variables of candidates lines

    def eSOCP_CanLineCijUpLim_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            # Look up the first circuit 'c' for this line (i, j)
            c = first_circuit_map_bidir.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLineCijUpLim'.format(i, j))
                return pyo.Constraint.Skip

            # Check forward direction
            if (i, j, c) in model.lc:
                if (i, j, c) in model.le:
                    return pyo.Constraint.Skip
                return (
                        model.vSOCP_cij[rp, k, i, j]
                        <= ((model.pBusMaxV[i] ** 2) + model.pBigM_SOCP * (1 - model.vSOCP_IndicConnecNodes[i, j]))
                )

    def eSOCP_CanLineCijLoLim_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            if cs.dPower_Parameters["pEnableSOCP"]:
                # Look up the first circuit 'c' for this line (i, j)
                c = first_circuit_map_bidir.get((i, j))
                if c is None:
                    # No first circuit info -> skip
                    print('No first circuit info for line ({}, {}) for eSOCP_CanLineCijLoLim'.format(i, j))
                    return pyo.Constraint.Skip
                if (i, j, c) in model.lc:
                    if (i, j, c) in model.le:
                        return pyo.Constraint.Skip
                    expr = max(0.1, model.pBusMinV[i] ** 2)
                    return model.vSOCP_cij[rp, k, i, j] >= expr - model.pBigM_SOCP * (1 - model.vSOCP_IndicConnecNodes[i, j])

        return pyo.Constraint.Skip

    def eSOCP_CanLineSijUpLim_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            c = first_circuit_map_bidir.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {} for eSOCP_CanLineSijUpLim)'.format(i, j))
                return pyo.Constraint.Skip
            if (i, j, c) in model.lc:
                if (i, j, c) in model.le:
                    return pyo.Constraint.Skip
                return (
                        model.vSOCP_sij[rp, k, i, j] <= (model.pBusMaxV[i] ** 2) + model.pBigM_SOCP * (1 - model.vSOCP_IndicConnecNodes[i, j]))

        return pyo.Constraint.Skip

    def eSOCP_CanLineSijLoLim_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            c = first_circuit_map_bidir.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLineSijLoLim'.format(i, j))
                return pyo.Constraint.Skip
            if (i, j, c) in model.lc:
                if (i, j, c) in model.le:
                    return pyo.Constraint.Skip
                return (
                        model.vSOCP_sij[rp, k, i, j] >= -(model.pBusMaxV[i] ** 2) - model.pBigM_SOCP * (1 - model.vSOCP_IndicConnecNodes[i, j]))
        return pyo.Constraint.Skip

    # Angle difference constraints for lines
    # Angle difference constraints for existing lines

    def eSOCP_ExiLineAngDif1_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            if any((i, j, c) in model.le for c in model.c):
                return model.vSOCP_sij[rp, k, i, j] <= model.vSOCP_cij[rp, k, i, j] * pyo.tan(model.pMaxAngleDiff)
        return pyo.Constraint.Skip

    def eSOCP_ExiLineAngDif2_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            if any((i, j, c) in model.le for c in model.c):
                return model.vSOCP_sij[rp, k, i, j] >= -model.vSOCP_cij[rp, k, i, j] * pyo.tan(model.pMaxAngleDiff)
        return pyo.Constraint.Skip

    # Angle difference constraints for candidate lines

    def eSOCP_CanLineAngDif1_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            c = first_circuit_map_bidir.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLineAngDif1'.format(i, j))
                return pyo.Constraint.Skip
            if (i, j, c) in model.lc:
                if (i, j, c) in model.le:
                    return pyo.Constraint.Skip
                return model.vSOCP_sij[rp, k, i, j] <= model.vSOCP_cij[rp, k, i, j] * pyo.tan(model.pMaxAngleDiff) + model.pBigM_Flow * (1 - model.vSOCP_IndicConnecNodes[i, j])
        return pyo.Constraint.Skip

    def eSOCP_CanLineAngDif2_rule(model, rp, k, i, j):
        if cs.dPower_Parameters["pEnableSOCP"]:
            c = first_circuit_map_bidir.get((i, j))
            if c is None:
                # No first circuit info -> skip
                print('No first circuit info for line ({}, {}) for eSOCP_CanLineAngDif2'.format(i, j))
                return pyo.Constraint.Skip
            if (i, j, c) in model.lc:
                if (i, j, c) in model.le:
                    return pyo.Constraint.Skip
            return model.vSOCP_sij[rp, k, i, j] >= - model.vSOCP_cij[rp, k, i, j] * pyo.tan(model.pMaxAngleDiff) - model.pBigM_Flow * (1 - model.vSOCP_IndicConnecNodes[i, j])
        return pyo.Constraint.Skip

    # Apparent power constraints for existing and candidate lines (Disabled in the LEGO model due to increased solving time) T
    # Constraints might need to be redefined for only the le set

    def eSOCP_ExiLineSLimit_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"] == 9999:
            if (i, j, c) in model.le:
                return (model.vLineP[rp, k, i, j, c] ** 2
                        + model.vLineQ[rp, k, i, j, c] ** 2
                        <= pyo.sqrt(model.pPmax[i, j, c] ** 2
                                    + model.pQmax[i, j, c] ** 2))
            elif (j, i, c) in model.le:
                # For the Reverse direction, the rhs should be 0 but this does not work with the solver. If this constraint is to be used, skip the reverse direction (will lead to error in the compare tool)
                rhs = 1
            else:
                return pyo.Constraint.Skip  # Not in le or le_reverse

            return (
                    model.vLineP[rp, k, i, j, c] ** 2
                    + model.vLineQ[rp, k, i, j, c] ** 2
                    <= rhs
            )
        return pyo.Constraint.Skip

    def eSOCP_CanLineSLimit_rule(model, rp, k, i, j, c):
        if cs.dPower_Parameters["pEnableSOCP"] == 9999:
            if (i, j, c) in model.lc:
                return (
                        model.vLineP[rp, k, i, j, c] ** 2
                        + model.vLineQ[rp, k, i, j, c] ** 2
                        <= pyo.sqrt(model.pPmax[i, j, c] ** 2
                                    + model.pQmax[i, j, c] ** 2) * model.vLineInvest[i, j, c])
            elif (j, i, c) in model.lc:
                # For the Reverse direction, the rhs should be 0 but this does not work with the solver. If this constraint is to be used skip the reverse direction (will lead to error in the compare tool)
                rhs = 1
            else:
                return pyo.Constraint.Skip  # Not in le or le_reverse

            return (model.vLineP[rp, k, i, j, c] ** 2
                    + model.vLineQ[rp, k, i, j, c] ** 2
                    <= rhs)
        return pyo.Constraint.Skip

    if cs.dPower_Parameters["pEnableSOCP"]:
        model.eSOCP_QMaxOut = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc=" max reactive power output of generator unit", rule=eSOCP_QMaxOut_rule)
        model.eSOCP_QMinOut1 = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc=" min postive reactive power output of generator unit", rule=eSOCP_QMinOut1_rule)
        model.eSOCP_QMinOut2 = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc=" min negative reactive power output of generator unit ", rule=eSOCP_QMinOut2_rule)
        model.eSOCP_BalanceQ_expr = pyo.Expression(model.rp, model.k, model.i, rule=eSOCP_BalanceQ_rule)
        model.eSOCP_BalanceQ = pyo.Constraint(model.rp, model.k, model.i, doc='Reactive power balance for each bus (SOCP)', rule=lambda model, rp, k, i: model.eSOCP_BalanceQ_expr[rp, k, i] == 0)
        model.eSOCP_ExiLinePij = pyo.Constraint(model.rp, model.k, model.le, doc=" Active power flow existing lines from i to j (for SOCP)", rule=eSOCP_ExiLinePij_rule)
        model.eSOCP_ExiLinePji = pyo.Constraint(model.rp, model.k, model.le, doc="Active power flow existing lines from j to i (for SOCP)", rule=eSOCP_ExiLinePji_rule)
        model.eSOCP_ExiLineQij = pyo.Constraint(model.rp, model.k, model.le, rule=eSOCP_ExiLineQij_rule, doc="Reactive power flow existing lines from i to j (for SOCP)")
        model.eSOCP_ExiLineQji = pyo.Constraint(model.rp, model.k, model.le, rule=eSOCP_ExiLineQji_rule, doc="Reactive power flow existing lines from j to i (for SOCP)")
        model.eSOCP_CanLinePij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power flow candidate lines from i to j (for SOCP) Big-M 1", rule=eSOCP_CanLinePij1_rule)
        model.eSOCP_CanLinePij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power flow candidate lines from i to j (for SOCP) Big-M 2", rule=eSOCP_CanLinePij2_rule)
        model.eSOCP_CanLinePji1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power flow candidate lines from j to i (for SOCP) Big-M 1", rule=eSOCP_CanLinePji1_rule)
        model.eSOCP_CanLinePji2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power flow candidate lines from j to i (for SOCP) Big-M 1", rule=eSOCP_CanLinePji2_rule)
        model.eSOCP_CanLineQij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power flow candidate lines from i to j (for SOCP) Big-M 1", rule=eSOCP_CanLineQij1_rule)
        model.eSOCP_CanLineQij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power flow candidate lines from i to j (for SOCP) Big-M 2", rule=eSOCP_CanLineQij2_rule)
        model.eSOCP_CanLineQji1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power flow candidate lines from j to i (for SOCP) Big-M 1", rule=eSOCP_CanLineQji1_rule)
        model.eSOCP_CanLineQji2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power flow candidate lines from j to i (for SOCP) Big-M 2", rule=eSOCP_CanLineQji2_rule)
        model.eSOCP_LimCanLinePij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power candidate lower limit i to j candidate lines (for SOCP)", rule=eSOCP_LimCanLinePij1_rule)
        model.eSOCP_LimCanLinePij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power candidate upper limit i to j candidate lines (for SOCP)", rule=eSOCP_LimCanLinePij2_rule)
        model.eSOCP_LimCanLinePji1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power candidate lower limit j to i candidate lines (for SOCP)", rule=eSOCP_LimCanLinePji1_rule)
        model.eSOCP_LimCanLinePji2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Active power candidate upper limit j to i candidate lines (for SOCP)", rule=eSOCP_LimCanLinePji2_rule)
        model.eSOCP_LimCanLineQij1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power candidate lower limit i to j candidate lines (for SOCP)", rule=eSOCP_LimCanLineQij1_rule)
        model.eSOCP_LimCanLineQij2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power candidate upper limit i to j candidate lines (for SOCP)", rule=eSOCP_LimCanLineQij2_rule)
        model.eSOCP_LimCanLineQji1 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power candidate lower limit j to i candidate lines (for SOCP)", rule=eSOCP_LimCanLineQji1_rule)
        model.eSOCP_LimCanLineQji2 = pyo.Constraint(model.rp, model.k, model.lc, doc="Reactive power candidate upper limit j to i candidate lines (for SOCP)", rule=eSOCP_LimCanLineQji2_rule)
        model.eSOCP_ExiLine = pyo.Constraint(model.rp, model.k, model.le_no_c, doc="SCOP constraints for existing lines (for AC-OPF) original set", rule=eSOCP_ExiLine_rule)
        model.eSOCP_CanLine = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="SOCP constraint for candidate lines (only if not in le)", rule=eSOCP_CanLine_rule)
        model.eSOCP_CanLine_cij = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=eSOCP_CanLine_cij_rule)
        model.eSOCP_CanLine_sij1 = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=eSOCP_CanLine_sij1_rule)
        model.eSOCP_CanLine_sij2 = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=eSOCP_CanLine_sij2_rule)
        model.eSOCP_IndicConnecNodes1 = pyo.Constraint(model.lc_no_c, doc="SOCP constraint for candidate lines (only if in le)", rule=eSOCP_IndicConnecNodes1_rule)
        model.eSOCP_IndicConnecNodes2 = pyo.Constraint(model.lc_no_c, doc="SOCP constraint for candidate lines (only if not in le)", rule=eSOCP_IndicConnecNodes2_rule)
        model.eSOCP_CanLineCijUpLim = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Limits for SOCP variables lines (only if not in le)", rule=eSOCP_CanLineCijUpLim_rule)
        model.eSOCP_CanLineCijLoLim = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Limits for SOCP variables (only if not in le)", rule=eSOCP_CanLineCijLoLim_rule)
        model.eSOCP_CanLineSijUpLim = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Limits for SOCP variables (only if not in le)", rule=eSOCP_CanLineSijUpLim_rule)
        model.eSOCP_CanLineSijLoLim = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Limits for SOCP variables (only if not in le)", rule=eSOCP_CanLineSijLoLim_rule)
        model.eSOCP_ExiLineAngDif1 = pyo.Constraint(model.rp, model.k, model.le_no_c, doc="Angle difference upper bounds existing lines", rule=eSOCP_ExiLineAngDif1_rule)
        model.eSOCP_ExiLineAngDif2 = pyo.Constraint(model.rp, model.k, model.le_no_c, doc="Angle difference lower bounds existing lines", rule=eSOCP_ExiLineAngDif2_rule)
        model.eSOCP_CanLineAngDif1 = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Angle difference upper bounds candidate lines", rule=eSOCP_CanLineAngDif1_rule)
        model.eSOCP_CanLineAngDif2 = pyo.Constraint(model.rp, model.k, model.lc_no_c, doc="Angle difference lowerf bounds candidate lines ", rule=eSOCP_CanLineAngDif2_rule)

        if cs.dPower_Parameters["pEnableSOCP"] == 99999:  # Not used in the GAMS model as well
            model.eSOCP_ExiLineSLimit = pyo.Constraint(model.rp, model.k, model.le_full, doc="Apparent power constraints for existing lines ", rule=eSOCP_ExiLineSLimit_rule)
            model.eSOCP_CanLineSLimit = pyo.Constraint(model.rp, model.k, model.lc_full, doc="Apparent power constraints for existing lines ", rule=eSOCP_CanLineSLimit_rule)

        if cs.dPower_Parameters["pEnableSOCP"] == 99999:  # FACTS are not implemented yet
            model.eSOCP_QMinFACTS = pyo.Constraint(model.rp, model.k, model.facts, doc='min reactive power output of FACTS unit', rule=eSOCP_QMinFACTS_rule)
            model.eSOCP_QMaxFACTS = pyo.Constraint(model.rp, model.k, model.facts, doc='max reactive power output of FACTS unit', rule=eSOCP_QMaxFACTS_rule)

    # Production constraints

    def eReMaxProd_rule(model, rp, k, r):
        capacity = cs.dPower_VRESProfiles.loc[rp, k, r]['value']
        capacity = capacity.values[0] if isinstance(capacity, pd.Series) else capacity
        return model.vGenP[rp, k, r] <= model.pMaxProd[r] * (model.vGenInvest[r] + model.pExisUnits[r]) * capacity

    if cs.dPower_Parameters["pEnableVRES"]:
        model.eReMaxProd = pyo.Constraint(model.rp, model.k, model.vresGenerators, rule=eReMaxProd_rule)

    def eThRampUp_rule(model, rp, k, g, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"]:
            case "notEnforced":
                if model.k.first() == k:
                    return None  # Is not enforced and should therefore be turned into pyo.Constraint.Skip in constraint construction
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prev(k), g] - model.vCommit[rp, k, g] * model.pRampUp[g]
            case "cyclic":
                return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prevw(k), g] - model.vCommit[rp, k, g] * model.pRampUp[g]
            case "markov":
                if model.k.first() == k:
                    return model.vGenP1[rp, k, g] - LEGOUtilities.markov_summand(model.rp, rp, False, model.k.prevw(k), model.vGenP1, transition_matrix, g) - model.vCommit[rp, k, g] * model.pRampUp[g]
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prev(k), g] - model.vCommit[rp, k, g] * model.pRampUp[g]
            case _:
                raise ValueError(f"Period edge handling ramping '{cs.dPower_Parameters['pReprPeriodEdgeHandlingRamping']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eThRampUp_expr = pyo.Expression(model.rp, model.k, model.thermalGenerators, rule=lambda m, rp, k, t: eThRampUp_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))
    model.eThRampUp = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Ramp up for thermal generators (based on doi:10.1007/s10107-015-0919-9)', rule=lambda model, rp, k, t: model.eThRampUp_expr[rp, k, t] <= 0 if not ((cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] == "notEnforced") and (model.k.first() == k)) else pyo.Constraint.Skip)

    def eThRampDw_rule(model, rp, k, g, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"]:
            case "notEnforced":
                if model.k.first() == k:
                    return None  # Is not enforced and should therefore be turned into pyo.Constraint.Skip in constraint construction
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prev(k), g] + model.vCommit[rp, model.k.prevw(k), g] * model.pRampDw[g]
            case "cyclic":
                return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prevw(k), g] + model.vCommit[rp, model.k.prevw(k), g] * model.pRampDw[g]
            case "markov":
                if model.k.first() == k:
                    return model.vGenP1[rp, k, g] - LEGOUtilities.markov_summand(model.rp, rp, False, model.k.prevw(k), model.vGenP1, transition_matrix, g) + LEGOUtilities.markov_summand(model.rp, rp, False, model.k.prevw(k), model.vCommit, transition_matrix, g) * model.pRampDw[g]
                else:
                    return model.vGenP1[rp, k, g] - model.vGenP1[rp, model.k.prev(k), g] + model.vCommit[rp, model.k.prev(k), g] * model.pRampDw[g]
            case _:
                raise ValueError(f"Period edge handling ramping '{cs.dPower_Parameters['pReprPeriodEdgeHandlingRamping']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eThRampDw_expr = pyo.Expression(model.rp, model.k, model.thermalGenerators, rule=lambda m, rp, k, t: eThRampDw_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))
    model.eThRampDw = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Ramp down for thermal generators (based on doi:10.1007/s10107-015-0919-9)', rule=lambda model, rp, k, t: model.eThRampDw_expr[rp, k, t] >= 0 if not ((cs.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] == "notEnforced") and (model.k.first() == k)) else pyo.Constraint.Skip)

    # Thermal Generator production with unit commitment & ramping constraints
    model.eUCTotOut = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Total production of thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, g: model.vGenP[rp, k, g] == model.pMinProd[g] * model.vCommit[rp, k, g] + model.vGenP1[rp, k, g])

    def eThMaxUC_rule(model, rp, k, t):
        return model.vCommit[rp, k, t] <= model.vGenInvest[t] + model.pExisUnits[t]

    model.eThMaxUC = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Maximum number of active units for thermal generators', rule=eThMaxUC_rule)

    def eUCMaxOut1_rule(model, rp, k, t):
        return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vStartup[rp, k, t])

    model.eUCMaxOut1_expr = pyo.Expression(model.rp, model.k, model.thermalGenerators, rule=eUCMaxOut1_rule)
    model.eUCMaxOut1 = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Maximum production for startup of thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, t: model.eUCMaxOut1_expr[rp, k, t] <= 0)

    def eUCMaxOut2_rule(model, rp, k, t, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
            case "notEnforced":
                if model.k.last() == k:
                    return None  # Is not enforced and should therefore be turned into pyo.Constraint.Skip in constraint construction
                else:
                    return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vShutdown[rp, model.k.nextw(k), t])
            case "cyclic":
                return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vShutdown[rp, model.k.nextw(k), t])
            case "markov":
                if model.k.last() == k:  # Markov summand only required for very last timestep
                    return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - LEGOUtilities.markov_summand(model.rp, rp, True, model.k.nextw(k), model.vShutdown, transition_matrix, t))
                else:
                    return model.vGenP1[rp, k, t] - (model.pMaxProd[t] - model.pMinProd[t]) * (model.vCommit[rp, k, t] - model.vShutdown[rp, model.k.nextw(k), t])
            case _:
                raise ValueError(f"Period edge handling unit commitment '{cs.dPower_Parameters['pReprPeriodEdgeHandlingUnitCommitment']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eUCMaxOut2_expr = pyo.Expression(model.rp, model.k, model.thermalGenerators, rule=lambda m, rp, k, t: eUCMaxOut2_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeTo))
    model.eUCMaxOut2 = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Maximum production for shutdown of thermal generators (from doi:10.1109/TPWRS.2013.2251373)',
                                      rule=lambda model, rp, k, t: model.eUCMaxOut2_expr[rp, k, t] <= 0 if not ((cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] == "notEnforced") and (model.k.last() == k)) else pyo.Constraint.Skip)

    def eUCStrShut_rule(model, rp, k, t, transition_matrix):
        match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
            case "notEnforced":
                if model.k.ord(k) == 1:
                    return pyo.Constraint.Skip
                else:
                    return model.vCommit[rp, k, t] - model.vCommit[rp, model.k.prevw(k), t] == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
            case "cyclic":
                return model.vCommit[rp, k, t] - model.vCommit[rp, model.k.prevw(k), t] == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
            case "markov":
                if model.k.ord(k) == 1:  # Markov summand only required for very first timestep
                    return model.vCommit[rp, k, t] - LEGOUtilities.markov_summand(model.rp, rp, False, model.k.prevw(k), model.vCommit, transition_matrix, t) == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
                else:
                    return model.vCommit[rp, k, t] - model.vCommit[rp, model.k.prevw(k), t] == model.vStartup[rp, k, t] - model.vShutdown[rp, k, t]
            case _:
                raise ValueError(f"Period edge handling unit commitment '{cs.dPower_Parameters['pReprPeriodEdgeHandlingUnitCommitment']}' not recognized - please check input file 'Power_Parameters.xlsx'!")

    model.eUCStrShut = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Start-up and shut-down logic for thermal generators (from doi:10.1109/TPWRS.2013.2251373)', rule=lambda model, rp, k, t: eUCStrShut_rule(model, rp, k, t, cs.rpTransitionMatrixRelativeFrom))

    def eMinUpTime_rule(model, rp, k, t, transition_matrix):
        if model.pMinUpTime[t] == 0:
            raise ValueError("Minimum up time must be at least 1, got 0 instead")
        elif model.pMinUpTime[t] == 1:
            return pyo.Constraint.Skip
        else:
            match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
                case "notEnforced":
                    if model.k.ord(k) < model.pMinUpTime[t]:
                        return pyo.Constraint.Skip  # Constraint is not active until the minimum up-time is reached
                    else:
                        return sum(model.vStartup[rp, k2, t] for k2 in LEGOUtilities.set_range_non_cyclic(model.k, model.k.ord(k) - model.pMinUpTime[t] + 1, model.k.ord(k))) <= model.vCommit[rp, k, t]
                case "cyclic":
                    return sum(model.vStartup[rp, k2, t] for k2 in LEGOUtilities.set_range_cyclic(model.k, model.k.ord(k) - model.pMinUpTime[t] + 1, model.k.ord(k))) <= model.vCommit[rp, k, t]
                case "markov":
                    return LEGOUtilities.markov_sum(model.rp, rp, model.k, model.k.ord(k) - model.pMinUpTime[t] + 1, model.k.ord(k), model.vStartup, transition_matrix, t) <= model.vCommit[rp, k, t]
                case _:
                    raise ValueError(f"Invalid value for 'pReprPeriodEdgeHandlingUnitCommitment' in 'Global_Parameters.xlsx': {cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]} - please choose from 'notEnforced', 'cyclic' or 'markov'!")

    model.eMinUpTime = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Minimum up time for thermal generators (from doi:10.1109/TPWRS.2013.2251373, adjusted to be cyclic)', rule=lambda m, rp, k, t: eMinUpTime_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))

    def eMinDownTime_rule(model, rp, k, t, transition_matrix):
        if model.pMinDownTime[t] == 0:
            raise ValueError("Minimum down time must be at least 1, got 0 instead")
        elif model.pMinDownTime[t] == 1:
            return pyo.Constraint.Skip
        else:
            match cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]:
                case "notEnforced":
                    if model.k.ord(k) < model.pMinDownTime[t]:
                        return pyo.Constraint.Skip  # Constraint is not active until the minimum down-time is reached
                    else:
                        return sum(model.vShutdown[rp, k2, t] for k2 in LEGOUtilities.set_range_non_cyclic(model.k, model.k.ord(k) - model.pMinDownTime[t] + 1, model.k.ord(k))) <= 1 - model.vCommit[rp, k, t]
                case "cyclic":
                    return sum(model.vShutdown[rp, k2, t] for k2 in LEGOUtilities.set_range_cyclic(model.k, model.k.ord(k) - model.pMinDownTime[t] + 1, model.k.ord(k))) <= 1 - model.vCommit[rp, k, t]
                case "markov":
                    return LEGOUtilities.markov_sum(model.rp, rp, model.k, model.k.ord(k) - model.pMinDownTime[t] + 1, model.k.ord(k), model.vShutdown, transition_matrix, t) <= 1 - model.vCommit[rp, k, t]
                case _:
                    raise ValueError(f"Invalid value for 'pReprPeriodEdgeHandlingUnitCommitment' in 'Global_Parameters.xlsx': {cs.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"]} - please choose from 'notEnforced', 'cyclic' or 'markov'!")

    model.eMinDownTime = pyo.Constraint(model.rp, model.k, model.thermalGenerators, doc='Minimum down time for thermal generators (from doi:10.1109/TPWRS.2013.2251373, adjusted to be cyclic)', rule=lambda m, rp, k, t: eMinDownTime_rule(m, rp, k, t, cs.rpTransitionMatrixRelativeFrom))

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = (sum(model.pFixedCost[i, j, c] * model.vLineInvest[i, j, c] for i, j, c in model.lc) +  # Investment cost of transmission lines
                             sum(model.pInvestCost[g] * model.vGenInvest[g] for g in model.g))  # Investment cost of generators
    second_stage_objective = (sum(sum(model.vPNS[rp, k, :]) * model.pWeight_rp[rp] * model.pWeight_k[k] * model.pENSCost for rp in model.rp for k in model.k) +  # Power not served
                              sum(sum(model.vEPS[rp, k, :]) * model.pWeight_rp[rp] * model.pWeight_k[k] * model.pENSCost * 2 for rp in model.rp for k in model.k) +  # Excess power served
                              sum(model.vStartup[rp, k, t] * model.pStartupCost[t] * model.pWeight_rp[rp] * model.pWeight_k[k] for rp in model.rp for k in model.k for t in model.thermalGenerators) +  # Startup cost of thermal generators
                              sum(model.vCommit[rp, k, t] * model.pInterVarCost[t] * model.pWeight_rp[rp] * model.pWeight_k[k] for rp in model.rp for k in model.k for t in model.thermalGenerators) +  # Commit cost of thermal generators
                              sum(model.vGenP[rp, k, g] * model.pOMVarCost[g] * model.pWeight_rp[rp] * model.pWeight_k[k] for rp in model.rp for k in model.k for g in model.g))  # Production cost of generators

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
