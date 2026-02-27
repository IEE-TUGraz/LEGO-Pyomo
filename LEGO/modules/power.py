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
    if any(cs.dPower_Network["pTecRepr"] == "SN"):
        printer.warning("Technical representation 'SN' (Single Node) detected in power network - please execute 'merge_single_node_buses' again if you've adjusted the case-study in code! Continuing anyway...")
    model.la_nodeRelevant = {node: [(i, j, c) for (i, j, c) in model.la if node == i or node == j] for node in model.i}
    model.le = pyo.Set(doc='Existing lines', initialize=cs.dPower_Network[(cs.dPower_Network["pEnableInvest"] == 0)].index.tolist(), within=model.la)
    model.le_nodeRelevant = {node: [(i, j, c) for (i, j, c) in model.le if node == i or node == j] for node in model.i}
    model.lc = pyo.Set(doc='Candidate lines', initialize=cs.dPower_Network[(cs.dPower_Network["pEnableInvest"] == 1)].index.tolist(), within=model.la)
    model.lc_nodeRelevant = {node: [(i, j, c) for (i, j, c) in model.lc if node == i or node == j] for node in model.i}

    model.g = pyo.Set(doc='Generators')
    model.gi = pyo.Set(doc='Generator g connected to bus i', within=model.g * model.i)

    # NOTE: We initialize it here and update it in add_constraints when gi is complete
    model.gi_node = {}  # bus i -> list of generators at that bus (for O(1) lookups)

    model.p = pyo.Set(doc='Periods', initialize=cs.dPower_Hindex.index.get_level_values('p').unique().tolist())
    model.rp = pyo.Set(doc='Representative periods', initialize=cs.dPower_Demand.index.get_level_values('rp').unique().tolist())
    model.k = pyo.Set(doc='Timestep within representative period', initialize=cs.dPower_Demand.index.get_level_values('k').unique().tolist())

    if cs.dGlobal_Parameters["pMovingWindowLength"] > 0 and cs.dGlobal_Parameters["pMovingWindowOverlap"] >= 0:
        model.constraintsActiveK = pyo.Set(doc='Timesteps where constraints are active during the actual time window', initialize=cs.constraints_active_k)
    else:
        model.constraintsActiveK = pyo.Set(doc='Timesteps where constraints are active', initialize=model.k)

    model.hindex = cs.dPower_Hindex.index

    # General Parameters
    model.pEnableSoftVoltageLimits = pyo.Param(doc='Whether to enable soft voltage limits', initialize=cs.dPower_Parameters['pEnableSoftVoltageLimits'])
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

    # Parameters used in all OPF formulations
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

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


def _build_gi_node_helper(model: pyo.ConcreteModel) -> dict:
    """
    Build helper dictionary mapping each bus to its connected generators.

    This provides O(1) lookup instead of O(n) iteration over all generators
    when building power balance constraints.

    Args:
        model: Pyomo model with gi set populated.

    Returns:
        Dictionary mapping bus i -> list of generators at that bus.
    """
    gi_node = {i: [] for i in model.i}
    for (g, i) in model.gi:
        gi_node[i].append(g)
    return gi_node


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    # Build helper dictionary for generator-to-bus mapping (now that gi is complete)
    model.gi_node = _build_gi_node_helper(model)

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = (sum(model.pInvestCost[g] * model.vGenInvest[g] for g in model.g))  # Investment cost of generators

    def ens_terms(rp, k):
        return sum(
            model.vPNS[rp, k, i] * model.pENSCost
            + model.vEPS[rp, k, i] * model.pENSCost * 2
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
