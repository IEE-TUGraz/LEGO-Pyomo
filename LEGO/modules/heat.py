import typing

import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGOUtilities

printer = Printer.getInstance()


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
    # Heat Balance Constraint
    def heat_balance_rule(m, rp, k, hn, dt, htec):
        return (m.vHeatProduction[rp, k, hn, dt, htec]
                + m.vHeatStorageDischarge[rp, k, hn, dt, htec]
                + m.vExcessHeatServed[rp, k, hn, dt, htec]
                ) == (m.pHeatDemandPerTechnology[rp, k, hn, dt, htec]
                      + m.vHeatStorageCharge[rp, k, hn, dt, htec]
                      + m.vHeatNotServed[rp, k, hn, dt, htec]
                      )

    model.HeatBalanceConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_balance_rule)

    # heat conversion rule
    def heat_conversion_rule(m, rp, k, hn, dt, htec):
        return m.vHeatProduction[rp, k, hn, dt, htec] == m.pP2HConversionEfficiency[rp, k, hn, dt, htec] * m.vPower2Heat[rp, k, hn, dt, htec]

    model.HeatConversionConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_conversion_rule)

    # max heat production rule
    def max_heat_production_rule(m, rp, k, hn, dt, htec):
        return m.vHeatProduction[rp, k, hn, dt, htec] <= m.pHeatInstalledCapacity[hn, dt, htec]

    model.MaxHeatProductionConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=max_heat_production_rule)

    # heat storage balance rule
    def heat_storage_balance_rule(m, rp, k, hn, dt, htec):
        if k == m.k.first():
            return m.vHeatStorageLevel[rp, k, hn, dt, htec] == 0.0
        else:
            k_prev = m.k.prev(k)
            return m.vHeatStorageLevel[rp, k, hn, dt, htec] == (m.vHeatStorageLevel[rp, k_prev, hn, dt, htec] * (1 - m.pHeatStorageSelfDischarge[hn, dt, htec])
                                                                + m.vHeatStorageCharge[rp, k_prev, hn, dt, htec] * m.pHeatStorageChEfficiency[hn, dt, htec]
                                                                - m.vHeatStorageDischarge[rp, k_prev, hn, dt, htec] / m.pHeatStorageDischEfficiency[hn, dt, htec])

    model.HeatStorageBalanceConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=heat_storage_balance_rule)

    # max storage level value
    def max_storage_level_rule(m, rp, k, hn, dt, htec):
        return m.vHeatStorageLevel[rp, k, hn, dt, htec] <= m.pHeatStorageCapacity[hn, dt, htec]

    model.MaxHeatStorageLevelConstr = pyo.Constraint(model.rp, model.k, model.hn_dt_htec, rule=max_storage_level_rule)

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
