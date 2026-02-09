import typing

import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGOUtilities

printer = Printer.getInstance()

# for reseach different heat formulations, select here
heat_storage_formulation = "advanced_storage"  # options: "no_storage", "simple_storage", "advanced_storage"
heat_conversion_formulation = "linear" # options: "linear", "conic" (works only with advanced storage formulation)

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

    # installed capacity
    model.pHeatInstalledCapacity = pyo.Param(model.hn, model.dt, model.htec, initialize=cs.dHeat_P2H_Technologies['InstCap'], doc='Installed Heat Capacity in *W_therm')

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

    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):


    # heat conversion rule --> will be replaced by special formulation!
    if heat_conversion_formulation == "linear":
        printer.information("Using linear heat conversion formulation")
        def heat_conversion_rule(m, rp, k, hn, dt, htec):
            return m.vHeatProduction[rp, k, hn, dt, htec] == m.pP2HConversionEfficiency[rp, k, hn, dt, htec] * m.vPower2Heat[rp, k, hn, dt, htec]

        model.HeatConversionConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_conversion_rule)

    elif heat_conversion_formulation == "conic":
        printer.information("Using conic heat conversion formulation")

        # define additional parameters and variables
        model.s_conic_relaxation = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec ,within=pyo.NonNegativeReals, doc='Auxiliary variable for conic relaxation of power to heat conversion')
        model.A = pyo.Param(initialize=1.0, doc='Scaling parameter for conic relaxation')
        model.B = pyo.Param(initialize=1.0, doc='Scaling parameter for conic relaxation')
        model.C = pyo.Param(initialize=1.0, doc='Scaling parameter for conic relaxation')

        def heat_conversion_rule(m, rp, k, hn, dt, htec):
            return m.vHeatProduction[rp, k, hn, dt, htec] == m.pP2HConversionEfficiency[rp, k, hn, dt, htec] * (m.vPower2Heat[rp, k, hn, dt, htec] + m.A * m.s_conic_relaxation[rp, k, hn, dt, htec])
        model.HeatConversionConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_conversion_rule)

        def conic_relaxation_rule(m, rp, k, hn, dt, htec):
            return m.s_conic_relaxation[rp, k, hn, dt, htec] * (m.q_floor[rp, k, hn, dt, htec] / m.C_floor[hn] + m.B) >= m.vPower2Heat[rp, k, hn, dt, htec]**2
        model.ConicRelaxationConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=conic_relaxation_rule)



    # max heat production rule
    def max_heat_production_rule(m, rp, k, hn, dt, htec):
        return m.vHeatProduction[rp, k, hn, dt, htec] <= m.pHeatInstalledCapacity[hn, dt, htec]

    model.MaxHeatProductionConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=max_heat_production_rule)


    if heat_storage_formulation == "no_storage":
        printer.information("Using no storage formulation")

        # Heat Balance Constraint
        def heat_balance_rule(m, rp, k, hn, dt, htec):
            return (m.vHeatProduction[rp, k, hn, dt, htec]
                    + m.vHeatStorageDischarge[rp, k, hn, dt, htec]
                    + m.vHeatNotServed[rp, k, hn, dt, htec]
                    ) == (m.pHeatDemandPerTechnology[rp, k, hn, dt, htec]
                          + m.vHeatStorageCharge[rp, k, hn, dt, htec]
                          + m.vExcessHeatServed[rp, k, hn, dt, htec]
                          )
        model.HeatBalanceConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_balance_rule)
        def heat_storage_balance_rule(m, rp, k, hn, dt, htec):
            return m.vHeatStorageLevel[rp, k, hn, dt, htec] == 0
        model.HeatStorageBalanceConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_balance_rule)

        def heat_storage_charge_rule(m, rp, k, hn, dt, htec):
            return m.vHeatStorageCharge[rp, k, hn, dt, htec] == 0
        model.HeatStorageChargeConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_storage_charge_rule)

        def heat_storage_discharge_rule(m, rp, k, hn, dt, htec):
            return m.vHeatStorageDischarge[rp, k, hn, dt, htec] == 0
        model.HeatStorageDischargeConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_storage_discharge_rule)

    elif heat_storage_formulation == "simple_storage":
        # heat storage balance rule: cyclic over each rp for short term storage
        printer.information("Using simple storage formulation")
        model.C_building = pyo.Param(model.hn, initialize=1.0, doc="Building heat storage capacity for cyclic storage balance rule")
        model.T_base = pyo.Param(initialize=305, doc="Room base temperatre")
        model.T_max = pyo.Param(initialize=320, doc="Maximum room temperature")

        model.q_room_pos_dev = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc='Positive deviation of room temperature from base temperature')
        model.q_room_neg_dev = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc='Negative deviation of room temperature from base temperature')
        model.Cost_Pos_Temp_Dev = pyo.Param(initialize=1.0, doc="Cost for positive deviation of room temperature from base temperature")
        model.Cost_Neg_Temp_Dev = pyo.Param(initialize=1.0, doc="Cost for negative deviation of room temperature from base temperature")

        # Heat Balance Constraint
        def heat_balance_rule(m, rp, k, hn, dt, htec):
            return (m.vHeatProduction[rp, k, hn, dt, htec]
                    + m.vHeatStorageDischarge[rp, k, hn, dt, htec]
                    + m.vHeatNotServed[rp, k, hn, dt, htec]
                    ) == (m.pHeatDemandPerTechnology[rp, k, hn, dt, htec]
                          + m.vHeatStorageCharge[rp, k, hn, dt, htec]
                          + m.vExcessHeatServed[rp, k, hn, dt, htec]
                          )

        model.HeatBalanceConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_balance_rule)

        # fix initial storage level to room base temperature
        def initial_storage_level_rule(m, rp, hn, dt, htec):
            return m.vHeatStorageLevel[rp, m.k.first(), hn, dt, htec] == m.C_building[hn] * m.T_base
        model.InitialStorageLevelConstr = pyo.Constraint(model.rp, model.hn_dt_htec, rule=initial_storage_level_rule)

        def heat_storage_balance_rule(m, rp, k, hn, dt, htec):
            # predecessor hour (cyclic)
            if k == m.k.first():
                k_prev = m.k.last()
            else:
                k_prev = m.k.prev(k)

            return (m.vHeatStorageLevel[rp, k, hn, dt, htec]
                    ==
                    m.vHeatStorageLevel[rp, k_prev, hn, dt, htec]
                    + m.vHeatStorageCharge[rp, k_prev, hn, dt, htec]
                    - m.vHeatStorageDischarge[rp, k_prev, hn, dt, htec]
            )
        model.HeatStorageBalanceConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_storage_balance_rule)

        # max storage level value
        def max_storage_level_rule(m, rp, k, hn, dt, htec):
            return m.vHeatStorageLevel[rp, k, hn, dt, htec] <= m.C_building[hn] * m.T_max
        model.MaxHeatStorageLevelConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=max_storage_level_rule)

        # temperature deviation rules
        def temp_dev_pos_rule(m, rp, k, hn, dt, htec):
            return m.vHeatStorageLevel[rp, k, hn, dt, htec] - m.C_building[hn] * m.T_base == m.q_room_pos_dev[rp, k, hn, dt, htec] - m.q_room_neg_dev[rp, k, hn, dt, htec]
        model.TempDevPosConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=temp_dev_pos_rule)

        # objective function adjustment for temperature deviation costs
        def temp_dev_cost_rule(m):
            return sum(m.pWeight_rp[rp] *
                       sum(m.pWeight_k[k] *
                           sum(m.Cost_Pos_Temp_Dev * m.q_room_pos_dev[rp, k, hn, dt, htec] + m.Cost_Neg_Temp_Dev * m.q_room_neg_dev[rp, k, hn, dt, htec]
                               for (hn, dt, htec) in m.hn_dt_htec)
                           for k in m.k)
                       for rp in m.rp)
        model.objective.expr += temp_dev_cost_rule(model)

    elif heat_storage_formulation == "advanced_storage":
        # tbd: add more complex storage formulation, e.g. with state of charge variables, or with storage level variables that are decoupled from charge/discharge variables
        printer.information(f"Using heat storage formulation: {heat_storage_formulation}")

        # define additional varialbes and parameters for advanced storage formulation
        model.q_floor = pyo.Var(model.rp, model.k, model.hn, within=pyo.NonNegativeReals, doc="Storage level of the floor")
        model.q_floor_charge = pyo.Var(model.rp, model.k, model.hn, within=pyo.NonNegativeReals, doc="Heat storage charging to the floor")
        model.q_floor_discharge = pyo.Var(model.rp, model.k, model.hn, within=pyo.NonNegativeReals, doc="Heat storage discharging from the floor")
        model.q_transfer = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc="Heat transfer between floor and room")
        model.C_floor = pyo.Param(model.hn, within=pyo.NonNegativeReals, initialize=1, doc="Storage capacity of the floor")
        model.C_room = pyo.Param(model.hn, within=pyo.NonNegativeReals, initialize=1, doc="Storage capacity of the room")
        model.alpha = pyo.Param(model.hn, within=pyo.NonNegativeReals, initialize=0.5, doc="Heat transfer coefficient between floor and room")

        model.T_base = pyo.Param(initialize=305, doc="Room base temperatre")
        model.T_max = pyo.Param(initialize=320, doc="Maximum room temperature")

        model.q_room_pos_dev = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc='Positive deviation of room temperature from base temperature')
        model.q_room_neg_dev = pyo.Var(model.rp, model.k, model.hn, model.dt, model.htec, within=pyo.NonNegativeReals, doc='Negative deviation of room temperature from base temperature')
        model.Cost_Pos_Temp_Dev = pyo.Param(initialize=1.0, doc="Cost for positive deviation of room temperature from base temperature")
        model.Cost_Neg_Temp_Dev = pyo.Param(initialize=1.0, doc="Cost for negative deviation of room temperature from base temperature")

        # Heat Balance Constraint room
        def heat_balance_rule(m, rp, k, hn, dt, htec):
            return (m.q_transfer[rp, k, hn, dt, htec]
                    + m.vHeatStorageDischarge[rp, k, hn, dt, htec]
                    + m.vHeatNotServed[rp, k, hn, dt, htec]
                    ) == (m.pHeatDemandPerTechnology[rp, k, hn, dt, htec]
                          + m.vHeatStorageCharge[rp, k, hn, dt, htec]
                          + m.vExcessHeatServed[rp, k, hn, dt, htec]
                          )
        model.HeatBalanceConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_balance_rule)

        # heat balance rule for the floor
        def floor_heat_balance_rule(m, rp, k, hn, dt, htec):
            return m.q_transfer[rp, k, hn, dt, htec] == m.vHeatProduction[rp, k, hn, dt, htec] + m.q_floor_discharge[rp, k, hn] - m.q_floor_charge[rp, k, hn]
        model.FloorHeatBalanceConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=floor_heat_balance_rule)

        # fix initial storage level to room base temperature
        def initial_storage_level_rule(m, rp, hn, dt, htec):
            return m.vHeatStorageLevel[rp, m.k.first(), hn, dt, htec] == m.C_room[hn] * m.T_base
        model.InitialStorageLevelConstr = pyo.Constraint(model.rp, model.hn_dt_htec, rule=initial_storage_level_rule)

        # fix floor temperature
        def initial_floor_storage_level_rule(m, rp, hn):
            return m.q_floor[rp, m.k.first(), hn] == m.C_floor[hn] * m.T_base
        model.InitialFloorStorageLevelConstr = pyo.Constraint(model.rp, model.hn, rule=initial_floor_storage_level_rule)

        def heat_storage_balance_rule(m, rp, k, hn, dt, htec):
            # predecessor hour (cyclic)
            if k == m.k.first():
                k_prev = m.k.last()
            else:
                k_prev = m.k.prev(k)

            return (m.vHeatStorageLevel[rp, k, hn, dt, htec]
                    ==
                    m.vHeatStorageLevel[rp, k_prev, hn, dt, htec]
                    + m.vHeatStorageCharge[rp, k_prev, hn, dt, htec]
                    - m.vHeatStorageDischarge[rp, k_prev, hn, dt, htec]
                    )
        model.HeatStorageBalanceConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_storage_balance_rule)

        # max storage level value
        def max_storage_level_rule(m, rp, k, hn, dt, htec):
            return m.vHeatStorageLevel[rp, k, hn, dt, htec] <= m.C_room[hn] * m.T_max
        model.MaxHeatStorageLevelConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=max_storage_level_rule)

        # temperature deviation rules
        def temp_dev_pos_rule(m, rp, k, hn, dt, htec):
            return m.vHeatStorageLevel[rp, k, hn, dt, htec] - m.C_room[hn] * m.T_base == m.q_room_pos_dev[rp, k, hn, dt, htec] - m.q_room_neg_dev[rp, k, hn, dt, htec]
        model.TempDevPosConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=temp_dev_pos_rule)

        # objective function adjustment for temperature deviation costs
        def temp_dev_cost_rule(m):
            return sum(m.pWeight_rp[rp] *
                       sum(m.pWeight_k[k] *
                           sum(m.Cost_Pos_Temp_Dev * m.q_room_pos_dev[rp, k, hn, dt, htec] + m.Cost_Neg_Temp_Dev * m.q_room_neg_dev[rp, k, hn, dt, htec]
                               for (hn, dt, htec) in m.hn_dt_htec)
                           for k in m.k)
                       for rp in m.rp)

        model.objective.expr += temp_dev_cost_rule(model)

        # storage constraint for the floor
        def floor_storage_balance_rule(m, rp, k, hn):
            # predecessor hour (cyclic)
            if k == m.k.first():
                k_prev = m.k.last()
            else:
                k_prev = m.k.prev(k)

            return (m.q_floor[rp, k, hn]
                    ==
                    m.q_floor[rp, k_prev, hn]
                    + m.q_floor_charge[rp, k_prev, hn]
                    - m.q_floor_discharge[rp, k_prev, hn]
            )
        model.FloorStorageBalanceConstr = pyo.Constraint(model.rp, model.k, model.hn, rule=floor_storage_balance_rule)

        # heat transfer between floor and room
        def floor_room_heat_transfer_rule(m, rp, k, hn, dt, htec):
            return m.q_transfer[rp, k, hn, dt, htec] == m.alpha[hn] * (m.q_floor[rp, k, hn] / m.C_floor[hn] - m.vHeatStorageLevel[rp, k, hn, dt, htec] / m.C_room[hn])
        model.FloorRoomHeatTransferConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=floor_room_heat_transfer_rule)



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
