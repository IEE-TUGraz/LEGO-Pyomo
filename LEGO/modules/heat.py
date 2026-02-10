import typing

import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGOUtilities

import pandas as pd

printer = Printer.getInstance()

from dataclasses import dataclass

@dataclass(frozen=True)
class HeatScenarioConfig:
    heat_storage_formulation: str
    heat_conversion_formulation: str


SCENARIOS = {
    "SC1": HeatScenarioConfig(
        heat_storage_formulation="no_storage",
        heat_conversion_formulation="linear",
    ),
    "SC2": HeatScenarioConfig(
        heat_storage_formulation="simple_storage",
        heat_conversion_formulation="linear",
    ),
    "SC3": HeatScenarioConfig(
        heat_storage_formulation="advanced_storage",
        heat_conversion_formulation="linear",
    ),
    "SC4": HeatScenarioConfig(
        heat_storage_formulation="advanced_storage",
        heat_conversion_formulation="conic",
    ),
}

scenario = "SC1"   # <- change here only
config = SCENARIOS[scenario]



@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> typing.Tuple[list[pyo.Var], list[pyo.Var]]:
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.hn = pyo.Set(doc='Heat Nodes', initialize=cs.dHeat_Demand.index.get_level_values("hn").unique().tolist())
    model.dt = pyo.Set(doc='Demand Types', initialize=cs.dHeat_Demand.index.get_level_values("dt").unique().tolist())
    model.htec = pyo.Set(doc='Conversion Technologies', initialize=cs.dHeat_P2H_Technologies.index.get_level_values("htec").unique().tolist())

    # define subsets (2D sets)
    model.i_hn = pyo.Set(doc='Set of Heat Nodes connected to power node', dimen=2, within=model.i * model.hn, initialize=cs.dHeat_Nodes.index.tolist())
    model.hn_dt = pyo.Set(doc='Set of Demand Types per Heat Node', dimen=2, within=model.hn * model.dt, initialize=cs.dHeat_Demand.index.to_frame(index=False)[["hn", "dt"]].itertuples(index=False, name=None))
    model.hn_dt_htec = pyo.Set(doc='Set of Conversion Technologies per Heat Node and Demand Type', dimen=3, within=model.hn * model.dt * model.htec, initialize=cs.dHeat_P2H_Technologies.index.to_frame(index=False)[["hn", "dt", "htec"]].itertuples(index=False, name=None))

    # parameters
    model.pHeatDemand = pyo.Param(model.rp, model.k, model.hn, model.dt, initialize=cs.dHeat_Demand['value'], doc='Heat Demand')
    model.pP2HConversionEfficiency = pyo.Param(model.rp, model.k, model.hn, model.dt, model.htec, initialize=cs.dHeat_P2H_Conversion_Factors['value'], doc='Power to Heat Conversion Efficiency')

    # storage parameters
    model.pHeatStorageCapacity = pyo.Param(model.hn, model.dt, model.htec, initialize=cs.dHeat_P2H_Technologies['ImpStoCap'], doc='(Implicit) Heat Storage Capacity in *Wh_therm')
    model.pHeatStorageChEfficiency = pyo.Param(model.hn, model.dt, model.htec, initialize=cs.dHeat_P2H_Technologies['ChEffic'], doc='Heat Storage Charging Efficiency')
    model.pHeatStorageDischEfficiency = pyo.Param(model.hn, model.dt, model.htec, initialize=cs.dHeat_P2H_Technologies['DisEffic'], doc='Heat Storage Discharging Efficiency')
    model.pHeatStorageSelfDischarge = pyo.Param(model.hn, model.dt, model.htec, initialize=cs.dHeat_P2H_Technologies['SelfDisch'], doc='Heat Storage Self Discharge Rate')



    # specific parameters for testcase study
    # calculate max heat demand per heat node and demand type across representative periods and time steps, to set an upper bound for heat production variables
    df_heat_demand = cs.dHeat_Demand.reset_index()
    df_heat_demand['HeatDemand'] = df_heat_demand['value']
    df_max_heat_demand = df_heat_demand.groupby(['hn', 'dt'])['HeatDemand'].max().reset_index()


    # set the Innstalled capacity to the peak x 1.5 x technology share; in GWthermal
    model.pHeatInstalledCapacity = pyo.Param(model.hn, model.dt, model.htec, initialize=lambda m, hn, dt, htec: df_max_heat_demand.loc[(df_max_heat_demand['hn'] == hn) & (df_max_heat_demand['dt'] == dt), 'HeatDemand'].values[0] * 1.5 * cs.dHeat_P2H_Technologies.loc[(hn, dt, htec), 'TecShare'], doc='Installed Heat Capacity in *W_therm')

    # calculate the annual heat demand
    model.pAnnualHeatDemand = pyo.Param(model.hn, model.dt, initialize=lambda m, hn, dt: df_heat_demand.loc[(df_heat_demand['hn'] == hn) & (df_heat_demand['dt'] == dt), 'HeatDemand'].sum() * 120, doc='Annual Heat Demand in *Wh_therm')
    printer.warning("REMOVE the 120 factor for the whole year!")

    average_specific_heat_demand = 80 # kWh/(m2*year)

    # calculate the average heated area from
    model.pHeatedArea = pyo.Param(model.hn, model.dt, initialize=lambda m, hn, dt: model.pAnnualHeatDemand[hn, dt] / (average_specific_heat_demand * 1e-6), doc='Average Heated Area in m2')

    # speficif heat capacity
    cp_room = 0.001 # kWh/(m2*K)
    cp_floor = 0.044 # kWh/(m2*K)

    # calculate the total heat capacity of the room and floor
    model.C_room = pyo.Param(model.hn, model.dt, initialize=lambda m, hn, dt: model.pHeatedArea[hn, dt] * cp_room * 1e-6, doc='Total Heat Capacity of the Room in kWh/K')
    model.C_floor = pyo.Param(model.hn, model.dt, initialize=lambda m, hn, dt: model.pHeatedArea[hn, dt] * cp_floor * 1e-6, doc='Total Heat Capacity of the Floor in kWh/K')
    # Parameters
    model.C_building = pyo.Param(model.hn, model.dt, initialize=lambda m, hn, dt: model.C_room[hn, dt] + model.C_floor[hn, dt], doc='Total Heat Capacity of the Building in kWh/K')
    model.T_base = pyo.Param(initialize=295)
    model.T_max_floor = pyo.Param(initialize=308)

    model.alpha = pyo.Param(initialize=9e-9) #GW/(m2*K)

    model.OverTempCost = pyo.Param(initialize=100.0, doc='Cost of exceeding set temperature in the room, in M€/K')
    model.UnderTempCost = pyo.Param(initialize=100.0, doc='Cost of undergoing the temp, in M€/K')



    model.Cost_Pos_Temp_Dev = pyo.Param(initialize=1e-6) # M€/K
    model.Cost_Neg_Temp_Dev = pyo.Param(initialize=1e-6) # M€/K

    # installed capacity
    #model.pHeatInstalledCapacity = pyo.Param(model.hn, model.dt, model.htec, initialize=cs.dHeat_P2H_Technologies['InstCap'], doc='Installed Heat Capacity in *W_therm')

    # write the tecnical values (heated_area, C_room, C_floor, C_building) to a Excel
    df_technical_values = pd.DataFrame({
        'hn': model.hn,
        'dt': model.dt,
        'HeatedArea': [model.pHeatedArea[hn, dt] for hn in model.hn for dt in model.dt],
        'C_room': [model.C_room[hn, dt] for hn in model.hn for dt in model.dt],
        'C_floor': [model.C_floor[hn, dt] for hn in model.hn for dt in model.dt],
        'C_building': [model.C_building[hn, dt] for hn in model.hn for dt in model.dt],
    })
    df_technical_values.to_excel("technical_values.xlsx", index=False)


    # technology share by demand type
    model.pHeatTechShareByDemandType = pyo.Param(model.hn, model.dt, model.htec, initialize=cs.dHeat_P2H_Technologies['TecShare'], doc='Technology Share by Demand Type')
    # calc the sum of technology shares per demand type
    df_sum_shares = cs.dHeat_P2H_Technologies.groupby(level=["hn", "dt"])['TecShare'].sum().reset_index()
    for idx, row in df_sum_shares.iterrows():
        if abs(row['TecShare'] - 1.0) > 1e-5:
            printer.warning(f"Sum of technology shares for heat node {row['hn']} and demand type {row['dt']} is {row['TecShare']:.4f}, which deviates from 1. Please check the input data.")

    # calculate the heat demand per technology
    def calc_heat_demand_per_technology(model, rp, k, hn, dt, htec):
        return model.pHeatDemand[rp, k, hn, dt] * model.pHeatTechShareByDemandType[hn, dt, htec]

    model.pHeatDemandPerTechnology = pyo.Param(model.rp, model.k, model.hn_dt_htec, initialize=calc_heat_demand_per_technology, doc='Heat Demand per Technology in *W_therm')

    # variables
    # heat production variables
    model.vHeatProduction = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc='Heat Production in *W_therm')
    model.vHeatNotServed = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc='Heat Not Served in *W_therm')
    model.vExcessHeatServed = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc='Excess Heat Served in *W_therm')
    second_stage_variables += [model.vHeatProduction, model.vHeatNotServed, model.vExcessHeatServed]

    # electricity consumption variables
    model.vPower2Heat = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc='Power to Heat Electricity Consumption in W_el')
    model.vPower2HeatDemand = pyo.Var(model.rp, model.k, model.i, within=pyo.NonNegativeReals, doc='Aggregated Electricity Demand W_el')
    second_stage_variables += [model.vPower2Heat, model.vPower2HeatDemand]

    # thermal storage varaibles
    model.vHeatStorageLevel = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc='Heat Storage Level in *Wh_therm')
    model.vHeatStorageCharge = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc='Heat Storage Charging in *W_therm')
    model.vHeatStorageDischarge = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc='Heat Storage Discharging in *W_therm')
    second_stage_variables += [model.vHeatStorageLevel, model.vHeatStorageCharge, model.vHeatStorageDischarge]

    # additional ones
    model.q_floor = pyo.Var(
        model.rp, model.k, model.hn, within=pyo.NonNegativeReals
    )
    second_stage_variables += [model.q_floor]

    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    if config.heat_storage_formulation == "no_storage":
        add_no_storage_constraints(model)
    elif config.heat_storage_formulation == "simple_storage":
        add_simple_storage_constraints(model)
    elif config.heat_storage_formulation == "advanced_storage":
        add_advanced_storage_constraints(model)

    add_heat_conversion_constraints(model, config)

    # max heat production rule
    def max_heat_production_rule(m, rp, k, hn, dt, htec):
        return m.vHeatProduction[rp, k, hn, dt, htec] <= m.pHeatInstalledCapacity[hn, dt, htec]
    model.MaxHeatProductionConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=max_heat_production_rule)

    # power demand per power node
    def power2heat_demand_rule(m, rp, k, i):
        return m.vPower2HeatDemand[rp, k, i] == sum(m.vPower2Heat[rp, k, hn, dt, htec] for (hn, dt, htec) in m.hn_dt_htec if (i, hn) in m.i_hn)

    model.Power2HeatDemandConstr = pyo.Constraint(model.rp, model.k, model.i, rule=power2heat_demand_rule)

    # add power2heat demand to the overall power balance
    for rp in model.rp:
        for k in model.k:
            for i in model.i:
                model.eDC_BalanceP_expr[rp, k, i] -= model.vPower2HeatDemand[rp, k, i]

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = sum(model.pWeight_rp[rp] *  # Weight of representative periods
                                 sum(model.pWeight_k[k] *  # Weight of time steps
                                     sum(+ model.vExcessHeatServed[rp, k, hn, dt, htec] * model.pENSCost  # Excess Heat Serves
                                         + model.vHeatNotServed[rp, k, hn, dt, htec] * model.pENSCost  # Not Served Heat cost
                                         for (hn, dt, htec) in model.hn_dt_htec)
                                     for k in model.k)
                                 for rp in model.rp)

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective

    return first_stage_objective



## helper functions

def add_heat_conversion_constraints(model, config):
    if config.heat_conversion_formulation == "linear":
        printer.information("Using linear heat conversion formulation")

        def heat_conversion_rule(m, rp, k, hn, dt, htec):
            return (
                    m.vHeatProduction[rp, k, hn, dt, htec]
                    == m.pP2HConversionEfficiency[rp, k, hn, dt, htec]
                    * m.vPower2Heat[rp, k, hn, dt, htec]
            )

        model.HeatConversionConstr = pyo.Constraint(
            model.rp, model.k, model.hn_dt_htec, rule=heat_conversion_rule
        )

    elif config.heat_conversion_formulation == "conic":
        printer.information("Using conic heat conversion formulation")

        model.s_conic_relaxation = pyo.Var(
            model.rp, model.k, model.hn, model.dt, model.htec,
            within=pyo.NonNegativeReals
        )

        model.A = pyo.Param(initialize=-0.001)
        model.B = pyo.Param(initialize=1.0)
        model.M = pyo.Param(initialize=-50.0)

        def heat_conversion_rule(m, rp, k, hn, dt, htec):
            return (
                    m.vHeatProduction[rp, k, hn, dt, htec]
                    == m.pP2HConversionEfficiency[rp, k, hn, dt, htec]
                    * (m.A/m.pHeatInstalledCapacity[hn, dt, htec]**2 * m.s_conic_relaxation[rp, k, hn, dt, htec] + m.B/m.pHeatInstalledCapacity[hn, dt, htec] * m.vPower2Heat[rp, k, hn, dt, htec])
            )

        model.HeatConversionConstr = pyo.Constraint(
            model.rp, model.k, model.hn_dt_htec, rule=heat_conversion_rule
        )

        def conic_relaxation_rule(m, rp, k, hn, dt, htec):
            return (
                    m.s_conic_relaxation[rp, k, hn, dt, htec]
                    * (m.q_floor[rp, k, hn] / m.C_floor[hn, dt] + m.M)
                    >= m.vPower2Heat[rp, k, hn, dt, htec] ** 2
            )

        model.ConicRelaxationConstr = pyo.Constraint(
            model.rp, model.k, model.hn_dt_htec, rule=conic_relaxation_rule
        )


def add_no_storage_constraints(model):
    printer.information("Using no storage formulation")

    # Heat balance
    def heat_balance_rule(m, rp, k, hn, dt, htec):
        return (
                m.vHeatProduction[rp, k, hn, dt, htec]
                + m.vHeatStorageDischarge[rp, k, hn, dt, htec]
                + m.vHeatNotServed[rp, k, hn, dt, htec]
                ==
                m.pHeatDemandPerTechnology[rp, k, hn, dt, htec]
                + m.vHeatStorageCharge[rp, k, hn, dt, htec]
                + m.vExcessHeatServed[rp, k, hn, dt, htec]
        )

    model.HeatBalanceConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec, rule=heat_balance_rule
    )

    # Storage fixed to zero
    model.HeatStorageBalanceConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec,
        rule=lambda m, rp, k, hn, dt, htec: m.vHeatStorageLevel[rp, k, hn, dt, htec] == 0
    )

    model.HeatStorageChargeConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec,
        rule=lambda m, rp, k, hn, dt, htec: m.vHeatStorageCharge[rp, k, hn, dt, htec] == 0
    )

    model.HeatStorageDischargeConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec,
        rule=lambda m, rp, k, hn, dt, htec: m.vHeatStorageDischarge[rp, k, hn, dt, htec] == 0
    )

def add_simple_storage_constraints(model):
    printer.information("Using simple storage formulation")


    model.q_room_pos_dev = pyo.Var(
        model.rp, model.k, model.hn, model.dt, model.htec,
        within=pyo.NonNegativeReals
    )
    model.q_room_neg_dev = pyo.Var(
        model.rp, model.k, model.hn, model.dt, model.htec,
        within=pyo.NonNegativeReals
    )

    # Heat balance
    def heat_balance_rule(m, rp, k, hn, dt, htec):
        return (
                m.vHeatProduction[rp, k, hn, dt, htec]
                + m.vHeatStorageDischarge[rp, k, hn, dt, htec]
                + m.vHeatNotServed[rp, k, hn, dt, htec]
                ==
                m.pHeatDemandPerTechnology[rp, k, hn, dt, htec]
                + m.vHeatStorageCharge[rp, k, hn, dt, htec]
                + m.vExcessHeatServed[rp, k, hn, dt, htec]
        )

    model.HeatBalanceConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec, rule=heat_balance_rule
    )

    # Initial storage level
    def initial_storage_level_rule(m, rp, hn, dt, htec):
        return m.vHeatStorageLevel[rp, m.k.first(), hn, dt, htec] == m.C_building[hn, dt] * m.T_base

    model.InitialStorageLevelConstr = pyo.Constraint(
        model.rp, model.hn_dt_htec, rule=initial_storage_level_rule
    )

    # Storage balance (cyclic)
    def heat_storage_balance_rule(m, rp, k, hn, dt, htec):
        k_prev = m.k.last() if k == m.k.first() else m.k.prev(k)
        return (
                m.vHeatStorageLevel[rp, k, hn, dt, htec]
                ==
                m.vHeatStorageLevel[rp, k_prev, hn, dt, htec]
                + m.vHeatStorageCharge[rp, k_prev, hn, dt, htec]
                - m.vHeatStorageDischarge[rp, k_prev, hn, dt, htec]
        )

    model.HeatStorageBalanceConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec, rule=heat_storage_balance_rule
    )


    # Temperature deviation
    def temp_dev_rule(m, rp, k, hn, dt, htec):
        return (
                m.vHeatStorageLevel[rp, k, hn, dt, htec]
                - m.C_building[hn, dt] * m.T_base
                ==
                m.q_room_pos_dev[rp, k, hn, dt, htec]
                - m.q_room_neg_dev[rp, k, hn, dt, htec]
        )

    model.TempDevConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec, rule=temp_dev_rule
    )

    # Objective adjustment
    model.objective.expr += sum(
        model.pWeight_rp[rp]
        * sum(
            model.pWeight_k[k]
            * sum(
                model.Cost_Pos_Temp_Dev * model.q_room_pos_dev[rp, k, hn, dt, htec] / model.C_building[hn, dt]
                + model.Cost_Neg_Temp_Dev * model.q_room_neg_dev[rp, k, hn, dt, htec] / model.C_building[hn, dt]
                for (hn, dt, htec) in model.hn_dt_htec
            )
            for k in model.k
        )
        for rp in model.rp
    )

def add_advanced_storage_constraints(model):
    printer.information("Using advanced storage formulation")


    # Variables
    model.q_floor_charge = pyo.Var(model.rp, model.k, model.hn, within=pyo.NonNegativeReals)
    model.q_floor_discharge = pyo.Var(model.rp, model.k, model.hn, within=pyo.NonNegativeReals)

    model.q_transfer = pyo.Var(
        model.rp, model.k, model.hn, model.dt, model.htec,
        within=pyo.NonNegativeReals
    )

    model.q_room_pos_dev = pyo.Var(
        model.rp, model.k, model.hn, model.dt, model.htec,
        within=pyo.NonNegativeReals
    )
    model.q_room_neg_dev = pyo.Var(
        model.rp, model.k, model.hn, model.dt, model.htec,
        within=pyo.NonNegativeReals
    )


    # Room heat balance
    def heat_balance_rule(m, rp, k, hn, dt, htec):
        return (
                m.q_transfer[rp, k, hn, dt, htec]
                + m.vHeatStorageDischarge[rp, k, hn, dt, htec]
                + m.vHeatNotServed[rp, k, hn, dt, htec]
                ==
                m.pHeatDemandPerTechnology[rp, k, hn, dt, htec]
                + m.vHeatStorageCharge[rp, k, hn, dt, htec]
                + m.vExcessHeatServed[rp, k, hn, dt, htec]
        )

    model.HeatBalanceConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec, rule=heat_balance_rule
    )

    # Floor heat balance
    def floor_heat_balance_rule(m, rp, k, hn, dt, htec):
        return (
                m.q_transfer[rp, k, hn, dt, htec]
                ==
                m.vHeatProduction[rp, k, hn, dt, htec]
                + m.q_floor_discharge[rp, k, hn]
                - m.q_floor_charge[rp, k, hn]
        )

    model.FloorHeatBalanceConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec, rule=floor_heat_balance_rule
    )

    # Initial conditions
    def initial_room_storage_rule(m, rp, hn, dt, htec):
        return m.vHeatStorageLevel[rp, m.k.first(), hn, dt, htec] == m.C_room[hn, dt] * m.T_base

    model.InitialRoomStorageConstr = pyo.Constraint(
        model.rp, model.hn_dt_htec, rule=initial_room_storage_rule
    )

    # Storage dynamics
    def room_storage_balance_rule(m, rp, k, hn, dt, htec):
        k_prev = m.k.last() if k == m.k.first() else m.k.prev(k)
        return (
                m.vHeatStorageLevel[rp, k, hn, dt, htec]
                ==
                m.vHeatStorageLevel[rp, k_prev, hn, dt, htec]
                + m.vHeatStorageCharge[rp, k_prev, hn, dt, htec]
                - m.vHeatStorageDischarge[rp, k_prev, hn, dt, htec]
        )

    model.RoomStorageBalanceConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec, rule=room_storage_balance_rule
    )

    def floor_storage_balance_rule(m, rp, k, hn):
        k_prev = m.k.last() if k == m.k.first() else m.k.prev(k)
        return (
                m.q_floor[rp, k, hn]
                ==
                m.q_floor[rp, k_prev, hn]
                + m.q_floor_charge[rp, k_prev, hn]
                - m.q_floor_discharge[rp, k_prev, hn]
        )

    model.FloorStorageBalanceConstr = pyo.Constraint(
        model.rp, model.k, model.hn, rule=floor_storage_balance_rule
    )

    # Limits and transfer
    model.MaxFloorStorageConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec,
        rule=lambda m, rp, k, hn, dt, htec:
        m.q_floor[rp, k, hn] <= m.C_floor[hn, dt] * m.T_max_floor
    )

    model.FloorRoomHeatTransferConstr = pyo.Constraint(
        model.rp, model.k, model.hn_dt_htec,
        rule=lambda m, rp, k, hn, dt, htec:
        m.q_transfer[rp, k, hn, dt, htec]
        == m.alpha * m.pHeatedArea[hn, dt]
        * (m.q_floor[rp, k, hn] / m.C_floor[hn, dt]
           - m.vHeatStorageLevel[rp, k, hn, dt, htec] / m.C_room[hn, dt])
    )

    # Objective adjustment
    model.objective.expr += sum(
        model.pWeight_rp[rp]
        * sum(
            model.pWeight_k[k]
            * sum(
                model.Cost_Pos_Temp_Dev * model.q_room_pos_dev[rp, k, hn, dt, htec] / model.C_building[hn, dt]
                + model.Cost_Neg_Temp_Dev * model.q_room_neg_dev[rp, k, hn, dt, htec] / model.C_building[hn, dt]
                for (hn, dt, htec) in model.hn_dt_htec
            )
            for k in model.k
        )
        for rp in model.rp
    )

