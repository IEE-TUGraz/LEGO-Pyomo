import typing

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGOUtilities
from LEGO.LEGOUtilities import reset_execution_safety_dict, set_range_non_cyclic

printer = Printer.getInstance()

@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> typing.Tuple[list[pyo.Var], list[pyo.Var]]:
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Parameters
    # Cast to int to avoid range() receiving a Pyomo Param object
    T_outage_value = int(cs.dCustom_Parameters['T_grid_outage'])
    model.T_outage = pyo.Param(initialize=T_outage_value, doc="Duration of power outage in hours")

    model.full_load_period = pyo.Param(initialize=cs.dGlobal_Parameters['full_load_period'])
    model.critial_load_factor = pyo.Param(initialize=cs.dGlobal_Parameters['critial_load_factor'])

    model.Building_MaxTempOutage = pyo.Param(initialize=cs.dCustom_Parameters['Building_MaxTempOutage'], doc="Maximum allowed temperature during power outage in K")
    model.Building_MinTempOutage = pyo.Param(initialize=cs.dCustom_Parameters['Building_MinTempOutage'], doc="Minimum allowed temperature during power outage in K")

    model.DiselStorageTankCost = pyo.Param(initialize=cs.dCustom_Parameters['DiselStorageTankCost'], doc="Cost per unit of fuel storage capacity for backup generator in €/MWh")

    # Sets
    # a set over the grid outage time steps (empty if T_outage == 0)
    model.tau = pyo.Set(initialize=range(1, T_outage_value + 1), doc="Time steps during power outage", ordered=True)

    # Only create the fuel tank set when there is an outage to plan against
    if T_outage_value > 0:
        model.tanks = pyo.Set(initialize=["T1"])
    else:
        model.tanks = pyo.Set(initialize=[])

    model.node_subset = pyo.Set(
        initialize=["Node_1"], within=model.i,
        doc="Single-node subset for heat resilience constraint"
    )

    print(f"The outage duration is {T_outage_value} hours")
    print(f"The outage set therefore is {list(model.tau)}")

    # variables
    # Variable: explicit triple temporal index (k, tau, k_prime)
    k_list = list(model.k)
    n_k = len(k_list)

    # Only generate triples whose full window fits within the horizon.
    # i = "last hour before outage", window is k_list[i+1 .. i+tau].
    # Require i + tau <= n_k - 1, i.e. i + tau < n_k.
    valid_triples = [
        (k_list[i], tau, k_list[j])
        for i in range(n_k)
        for tau in model.tau
        if i + tau < n_k
        for j in range(i + 1, i + tau + 1)
    ]

    model.valid_kp = pyo.Set(
        initialize=valid_triples,
        within=model.k * model.tau * model.k,
        doc="Valid (k, tau, k') triples where k' falls inside the outage window"
    )

    model.vOutage_P2H = pyo.Var(
        model.rp, model.node_subset, model.valid_kp,
        bounds=(0, None),
    )

    model.availabeBESS = pyo.Var(model.rp, model.k, model.tau, bounds=(0, None), doc="Available BESS capacity during outage window")
    model.availableBackupGen = pyo.Var(model.rp, model.k, model.tau, bounds=(0, None), doc="Available backup generator capacity during outage window")

    # Tank investment variable: only meaningful when an outage is modelled
    if T_outage_value > 0:
        model.DieselStorageTankInvest = pyo.Var(model.tanks, bounds=(0, None), doc="Investment in fuel storage capacity for backup generator in MWh")
        first_stage_variables += [model.DieselStorageTankInvest]

    second_stage_variables += [model.vOutage_P2H, model.availabeBESS, model.availableBackupGen]
    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables



@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):

    # Build window_map only for (k, tau) combinations whose full window fits within the horizon.
    # This keeps it consistent with valid_kp (no truncated windows).
    window_map = {}
    k_list = list(model.k)
    n_k = len(k_list)
    for ki, k in enumerate(k_list):
        for tau in model.tau:
            if ki + tau < n_k:
                window_map[(k, tau)] = tuple(k_list[ki + 1: ki + tau + 1])


    def eHeatSelfSufficiency_LB(m, rp, k, node_subset, tau, hn, dt, htec):
        if (k, tau) not in window_map:
            return pyo.Constraint.Skip

        set_t = window_map[(k, tau)]

        lhs_demand = sum(m.pHeatDemandPerTechnology[rp, kp, hn, dt, htec] for kp in set_t)

        # Efficiency indexed at kp (the hour during the outage), not at k.
        rhs_p2h = sum(
            m.vOutage_P2H[rp, node_subset, k, tau, kp] * m.pP2HConversionEfficiency[rp, kp, hn, dt, htec]
            for kp in set_t
        )
        rhs_buffer = m.vHeatStorageLevel[rp, k, hn, dt, htec] - (m.Building_ThermalMass[hn] * m.Building_MinTempOutage)

        return lhs_demand <= rhs_p2h + rhs_buffer

    def eHeatSelfSufficiency_UB(m, rp, k, node_subset, tau, hn, dt, htec):
        if (k, tau) not in window_map:
            return pyo.Constraint.Skip

        set_t = window_map[(k, tau)]

        # LHS: P2H heat production minus heat demand over the window
        # Efficiency indexed at kp (the hour during the outage), not at k.
        lhs_p2h = sum(
            m.vOutage_P2H[rp, node_subset, k, tau, kp] * m.pP2HConversionEfficiency[rp, kp, hn, dt, htec]
            for kp in set_t
        )
        lhs_demand = sum(m.pHeatDemandPerTechnology[rp, kp, hn, dt, htec] for kp in set_t)

        # RHS: storage level + max thermal headroom
        rhs_buffer = m.vHeatStorageLevel[rp, k, hn, dt, htec] + (m.Building_ThermalMass[hn] * m.Building_MaxTempOutage)

        return lhs_p2h - lhs_demand <= rhs_buffer


    model.eHeatSelfSufficiency_LB = pyo.Constraint(
        model.rp, model.k, model.node_subset, model.tau, model.hn, model.dt, model.htec,
        rule=eHeatSelfSufficiency_LB,
        doc="Heat resilience lower bound: sufficient heat over any outage window"
    )
    model.eHeatSelfSufficiency_UB = pyo.Constraint(
        model.rp, model.k, model.node_subset, model.tau, model.hn, model.dt, model.htec,
        rule=eHeatSelfSufficiency_UB,
        doc="Heat resilience upper bound: heat production does not exceed comfort ceiling"
    )


    def eP2HCapacity(m, rp, node_subset, k, tau, kp, hn, dt, htec):
        # Efficiency indexed at kp (the actual hour of P2H operation), not at k.
        return m.vOutage_P2H[rp, node_subset, k, tau, kp] <= (
                m.pHeatInstalledCapacity[hn, dt, htec] / m.pP2HConversionEfficiency[rp, kp, hn, dt, htec]
        )

    # restrict the P2H to installed capacity
    model.eP2HCapacity = pyo.Constraint(
        model.rp, model.node_subset, model.valid_kp, model.hn, model.dt, model.htec,
        rule=eP2HCapacity,
        doc="P2H allocation bounded by installed heat capacity per technology"
    )


    ## Power self-sufficiency constraint: ensure that for any outage duration, the demand can be met by local generation and storage without imports

    # Technology subsets — defined once, outside the constraint
    pvset = [pv for pv, tec in model.gtec if tec == "Solar"]
    thermal_set = [thermal for thermal, tec in model.gtec if tec == "FuelOilGas"]  # TODO: swap to backup generator
    storage_set = [storage for storage, tec in model.gtec if tec == "BESS"]

    def ePowerSelfSufficiency(m, rp, k, node_subset, tau):
        if (k, tau) not in window_map:
            return pyo.Constraint.Skip

        set_t = window_map[(k, tau)]
        full_load_steps = set_t[:pyo.value(m.full_load_period)]
        reduced_load_steps = set_t[pyo.value(m.full_load_period):]

        # LHS: electricity demand + P2H load drawn during the outage window
        lhs = (
                sum(m.pDemandP[rp, kp, node_subset] for kp in full_load_steps)
                + m.critial_load_factor * sum(m.pDemandP[rp, kp, node_subset] for kp in reduced_load_steps)
                + sum(m.vOutage_P2H[rp, node_subset, k, tau, kp] for kp in set_t)  # additional P2H demand
        )

        # RHS: solar generation + BESS discharge + thermal backup capacity
        rhs = (
                sum(m.pCapacityFactors[rp, kp, pv] * m.vGenP[rp, kp, pv]
                    for kp in set_t for pv in pvset)
                + m.availabeBESS[rp, k, tau]
                + m.availableBackupGen[rp, k, tau]
        )

        return lhs <= rhs

    model.ePowerSelfSufficiency = pyo.Constraint(
        model.rp, model.k, model.node_subset, model.tau,
        rule=ePowerSelfSufficiency,
        doc="Power self-sufficiency: local supply covers demand + P2H load over any outage window"
    )

    def eOutageBESSAvailablity_investment(m, rp, k, tau):
        return m.availabeBESS[rp, k, tau] <= sum(m.vGenInvest[storage] * m.pMaxProd[storage] for storage in storage_set) * tau

    model.eOutageBESSAvailablity_investment = pyo.Constraint(
        model.rp, model.k, model.tau,
        rule=eOutageBESSAvailablity_investment,
        doc="Available BESS during outage limited by investment decisions"
    )

    def eOutageBESSAvailability_level(m, rp, k, tau):
        if (k, tau) not in window_map:
            return pyo.Constraint.Skip

        # Storage level at end of hour k = level "just before" the outage begins.
        return m.availabeBESS[rp, k, tau] <= sum(m.vStIntraRes[rp, k, storage] for storage in storage_set)

    model.eOutageBESSAvailability_level = pyo.Constraint(
        model.rp, model.k, model.tau,
        rule=eOutageBESSAvailability_level,
        doc="Available BESS during outage limited by storage levels"
    )


    def eBackupGen_investment(m, rp, k, tau):
        return m.availableBackupGen[rp, k, tau] <= sum(m.vGenInvest[thermal] * m.pMaxProd[thermal] for thermal in thermal_set) * tau

    model.eBackupGen_investment = pyo.Constraint(
        model.rp, model.k, model.tau,
        rule=eBackupGen_investment,
        doc="Available backup generator capacity during outage limited by investment decisions"
    )

    def eBackupGen_fuel_storage(m, rp, k, tau):
        return m.availableBackupGen[rp, k, tau] <= sum(m.DieselStorageTankInvest[tank] for tank in m.tanks)

    model.eBackupGen_fuel_storage = pyo.Constraint(
        model.rp, model.k, model.tau,
        rule=eBackupGen_fuel_storage,
        doc="Available backup generator capacity during outage limited by fuel storage capacity"
    )


    # Objective: tank investment cost is only relevant when tanks exist (T_outage > 0)
    if len(model.tanks) > 0:
        first_stage_objective = sum(model.DieselStorageTankInvest[tank] * model.DiselStorageTankCost for tank in model.tanks)
    else:
        first_stage_objective = 0
    second_stage_objective = 0

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective