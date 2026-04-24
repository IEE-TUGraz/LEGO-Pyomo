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



    #parameter
    model.T_outage = pyo.Param(initialize=cs.dGlobal_Parameters['pTOutage'], doc="Duration of power outage in hours")

    model.Building_MaxTempOutage = pyo.Param(initialize=cs.dCustom_Parameters['Building_MaxTempOutage'], doc="Maximum allowed temperature during power outage in K")
    model.Building_MinTempOutage = pyo.Param(initialize=cs.dCustom_Parameters['Building_MinTempOutage'], doc="Minimum allowed temperature during power outage in K")

    model.DiselStorageTankCost = pyo.Param(initialize=cs.dCustom_Parameters['DiselStorageTankCost'], doc="Cost per unit of fuel storage capacity for backup generator in €/MWh")

    # Sets
    # a set over the grid outage time steps
    model.tau = pyo.Set(initialize=range(1, cs.dGlobal_Parameters['pTOutage'] + 1), doc="Time steps during power outage", ordered=True)
    model.tanks = pyo.Set(initialize=["T1"])



    # variables
    # Variable: explicit triple temporal index (k, tau, k_prime)
    valid_triples = [
        (k, tau, kp)
        for k in model.k
        for tau in model.tau
        for kp in model.k
        if model.k.ord(k) + 1 <= model.k.ord(kp) <= model.k.ord(k) + tau
    ]

    model.valid_kp = pyo.Set(
        initialize=valid_triples,
        within=model.k * model.tau * model.k,
        doc="Valid (k, tau, k') triples where t' falls inside the outage window"
    )

    model.vOutage_P2H = pyo.Var(
        model.rp, model.i, model.valid_kp,
        bounds=(0, None),
        doc="P2H allocation — only defined for valid (k, tau, t') triples"
    )

    model.availabeBESS = pyo.Var(model.rp, model.k, model.tau, bounds=(0, None), doc="Available BESS capacity during outage window")
    model.availableBackupGen = pyo.Var(model.rp, model.k, model.tau, bounds=(0, None), doc="Available backup generator capacity during outage window")

    model.DieselStorageTankInvest = pyo.Var(model.tanks, bounds=(0, None), doc="Investment in fuel storage capacity for backup generator in MWh")

    first_stage_variables += [model.DieselStorageTankInvest]
    second_stage_variables += [model.vOutage_P2H, model.availabeBESS, model.availableBackupGen]
    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables



@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    def eHeatSelfSufficiency_LB(m, rp, k, i, tau, hn, dt, htec):
        k_ord = m.k.ord(k)
        if k_ord + tau > len(m.k):
            return pyo.Constraint.Skip

        set_t = set_range_non_cyclic(m.k, k_ord + 1, k_ord + tau)

        lhs_demand = sum(m.pHeatDemandPerTechnology[rp, kp, hn, dt, htec] for kp in set_t for hn in m.hn for dt in m.dt for htec in m.htec)

        # hn, dt, htec already bound from constraint index — no inner loop
        rhs_p2h = sum(
            m.vOutage_P2H[rp, i, k, tau, kp] * m.pP2HConversionEfficiency[rp, kp, hn, dt, htec]
            for kp in set_t
        )
        rhs_buffer = m.vHeatStorageLevel[rp, k, hn, dt, htec] - (m.Building_ThermalMass[hn] * m.Building_MinTempOutage)

        return lhs_demand <= rhs_p2h + rhs_buffer

    def eHeatSelfSufficiency_UB(m, rp, k, i, tau, hn, dt, htec):
        k_ord = m.k.ord(k)
        if k_ord + tau > len(m.k):
            return pyo.Constraint.Skip

        set_t = set_range_non_cyclic(m.k, k_ord + 1, k_ord + tau)

        # LHS: P2H heat production minus heat demand over the window
        lhs_p2h = sum(
            m.vOutage_P2H[rp, i, k, tau, kp] * m.pP2HConversionEfficiency[rp, kp, hn, dt, htec]
            for kp in set_t
        )
        lhs_demand = sum(m.pHeatDemandPerTechnology[rp, kp, hn, dt, htec] for kp in set_t for hn in m.hn for dt in m.dt for htec in m.htec)

        # RHS: storage level + max thermal headroom
        rhs_buffer = m.vHeatStorageLevel[rp, k, hn, dt, htec] + (m.Building_ThermalMass[hn] * m.Building_MaxTempOutage)

        return lhs_p2h - lhs_demand <= rhs_buffer

    node_subset = pyo.Set(
        initialize=["Node_1"], within=model.i,
        doc="Single-node subset for heat resilience constraint"
    )

    model.eHeatSelfSufficiency_LB = pyo.Constraint(
        model.rp, model.k, node_subset, model.tau, model.hn, model.dt, model.htec,
        rule=eHeatSelfSufficiency_LB,
        doc="Heat resilience lower bound: sufficient heat over any outage window"
    )
    model.eHeatSelfSufficiency_UB = pyo.Constraint(
        model.rp, model.k, node_subset, model.tau, model.hn, model.dt, model.htec,
        rule=eHeatSelfSufficiency_UB,
        doc="Heat resilience upper bound: heat production does not exceed comfort ceiling"
    )


    def eP2HCapacity(m, rp, i, k, tau, kp, hn, dt, htec):
        return m.vOutage_P2H[rp, i, k, tau, kp] <= (
                m.pHeatInstalledCapacity[hn, dt, htec] / m.pP2HConversionEfficiency[rp, k, hn, dt, htec]
        )

    # restricht the P2H to installed capacity
    model.eP2HCapacity = pyo.Constraint(
        model.rp, model.i, model.valid_kp, model.hn, model.dt, model.htec,
        rule=eP2HCapacity,
        doc="P2H allocation bounded by installed heat capacity per technology"
    )


    #model.eHeatSelfSufficiency.pprint()

    ## Power self-sufficiency constraint: ensure that for any outage duration, the demand can be met by local generation and storage without imports

    # Technology subsets — defined once, outside the constraint
    pvset = [pv for pv, tec in model.gtec if tec == "Solar"]
    thermal_set = [thermal for thermal, tec in model.gtec if tec == "FuelOilGas"]  # TODO: swap to backup generator
    storage_set = [storage for storage, tec in model.gtec if tec == "BESS"]


    def ePowerSelfSufficiency(m, rp, k, i, tau):
        k_ord = m.k.ord(k)
        if k_ord + tau > len(m.k):
            return pyo.Constraint.Skip

        set_t = set_range_non_cyclic(m.k, k_ord + 1, k_ord + tau)

        # LHS: electricity demand + P2H load drawn during the outage window
        lhs = (
                sum(m.pDemandP[rp, kp, i] for kp in set_t)
                + sum(m.vOutage_P2H[rp, i, k, tau, kp] for kp in set_t)  # additional P2H demand
        )

        # RHS: solar generation + BESS discharge + thermal backup capacity
        rhs = (
                sum(m.pCapacityFactors[rp, kp, pv] * m.vGenP[rp, kp, pv]
                    for kp in set_t for pv in pvset)
                + m.availabeBESS[rp,k,tau]
                + m.availableBackupGen[rp,k,tau]
        )

        return lhs <= rhs

    model.ePowerSelfSufficiency = pyo.Constraint(
        model.rp, model.k, model.i, model.tau,
        rule=ePowerSelfSufficiency,
        doc="Power self-sufficiency: local supply covers demand + P2H load over any outage window"
    )

    def eOutageBESSAvailablity_investment(m, rp, k, tau):
        return m.availabeBESS[rp,k,tau] <= sum(m.vGenInvest[storage] * m.pMaxProd[storage] for storage in storage_set) * (tau)


    model.eOutageBESSAvailablity_investment = pyo.Constraint(model.rp, model.k, model.tau, rule=eOutageBESSAvailablity_investment, doc="Available BESS during outage limited by investment decisions")

    def eOutageBESSAvailability_level(m, rp, k, tau):
        k_ord = m.k.ord(k)
        if k_ord + tau > len(m.k):
            return pyo.Constraint.Skip

        set_t = set_range_non_cyclic(m.k, k_ord + 1, k_ord + tau)

        return m.availabeBESS[rp, k, tau] <= sum(m.vStIntraRes[rp, k, storage] for storage in storage_set)
    model.eOutageBESSAvailability_level = pyo.Constraint(model.rp, model.k, model.tau, rule=eOutageBESSAvailability_level, doc="Available BESS during outage limited by storage levels")



    def eBackupGen_investment(m, rp, k, tau):
        return m.availableBackupGen[rp, k, tau] <= sum(m.vGenInvest[thermal] * m.pMaxProd[thermal] for thermal in thermal_set) * (tau)
    model.eBackupGen_investment = pyo.Constraint(model.rp, model.k, model.tau, rule=eBackupGen_investment, doc="Available backup generator capacity during outage limited by investment decisions")

    def eBackupGen_fuel_storage(m, rp, k, tau):
        return m.availableBackupGen[rp, k, tau] <= sum((m.DieselStorageTankInvest[tank] for tank in m.tanks))
    model.eBackupGen_fuel_storage = pyo.Constraint(model.rp, model.k, model.tau, rule=eBackupGen_fuel_storage, doc="Available backup generator capacity during outage limited by fuel storage capacity")


    first_stage_objective = sum(model.DieselStorageTankInvest[tank] * model.DiselStorageTankCost for tank in model.tanks)
    second_stage_objective = 0

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
