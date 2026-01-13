import argparse
import logging
import os
import time

import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from rich_argparse import RichHelpFormatter

from InOutModule import SQLiteWriter
from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO

printer = Printer.getInstance()

# Set up logging so that infeasible constraints are logged by pyomo
logger = logging.getLogger("pyomo")
logger.setLevel("INFO")


def main(caseStudyDirectory, numberOfRPs, lengthOfRPs, scaleDemand, scaleInflows, scaleVRESMaxProd, clusterOnOriginalData, scaleRoRToInflowScaling):
    caseStudyName = caseStudyDirectory.replace("/", "_").replace("\\", "_")

    printer.information(f"Loading original case study from '{caseStudyDirectory}'")
    start_time = time.time()
    cs_inflow_hourly = CaseStudy(caseStudyDirectory)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    if scaleDemand != 1.0:
        printer.information(f"Scaling demand by factor {scaleDemand}")
        cs_inflow_hourly.dPower_Demand['value'] *= scaleDemand

    if scaleVRESMaxProd != 1.0:
        printer.information(f"Scaling VRES maximum production by factor {scaleVRESMaxProd}")
        cs_inflow_hourly.dPower_VRES['MaxProd'] *= scaleVRESMaxProd

    if scaleInflows != 1.0:
        printer.information(f"Scaling inflows by factor {scaleInflows}")
        cs_inflow_hourly.dPower_Inflows['value'] *= scaleInflows

    if scaleRoRToInflowScaling:
        printer.information(f"Scaling run-of-river generation capacity by inflow scaling factor {scaleInflows} to prevent inflows that exceed maximum production of RoR plants")
        ror_gens = cs_inflow_hourly.dPower_VRES[cs_inflow_hourly.dPower_VRES['tec'].str.lower() == 'ror'].index
        cs_inflow_hourly.dPower_VRES.loc[ror_gens, 'MaxProd'] *= scaleInflows

    printer.information("Creating copies of case study with different levels of aggregation for inflow data")
    cs_inflow_yearly_aggregated = cs_inflow_hourly.copy()
    cs_inflow_monthly_aggregated = cs_inflow_hourly.copy()
    cs_inflow_weekly_aggregated = cs_inflow_hourly.copy()
    cs_inflow_daily_aggregated = cs_inflow_hourly.copy()

    printer.information("Aggregating inflow data based on yearly, monthly, weekly, and daily averages")
    generators = cs_inflow_monthly_aggregated.dPower_Inflows.index.get_level_values("g").unique()

    # Yearly
    mean_inflows = cs_inflow_yearly_aggregated.dPower_Inflows["value"].groupby(["g"]).mean()  # Calculate mean inflow per generator
    cs_inflow_yearly_aggregated.dPower_Inflows.loc[:, "value"] = mean_inflows[cs_inflow_yearly_aggregated.dPower_Inflows.index.droplevel(["rp", "k"])].values  # Set all inflows to mean inflow of respective generator

    # Monthly
    cs_inflow_monthly_aggregated.dPower_Inflows["k_int"] = cs_inflow_monthly_aggregated.dPower_Inflows.index.get_level_values("k").str[1:].astype(int)
    cs_inflow_monthly_aggregated.dPower_Inflows["month"] = (cs_inflow_monthly_aggregated.dPower_Inflows["k_int"] - 1) // 720
    cs_inflow_monthly_aggregated.dPower_Inflows["value"] = cs_inflow_monthly_aggregated.dPower_Inflows.groupby(["g", "month"])["value"].transform("mean")
    cs_inflow_monthly_aggregated.dPower_Inflows.drop(columns=["k_int", "month"], inplace=True)

    # Weekly
    cs_inflow_weekly_aggregated.dPower_Inflows["k_int"] = cs_inflow_weekly_aggregated.dPower_Inflows.index.get_level_values("k").str[1:].astype(int)
    cs_inflow_weekly_aggregated.dPower_Inflows["week"] = (cs_inflow_weekly_aggregated.dPower_Inflows["k_int"] - 1) // 168
    cs_inflow_weekly_aggregated.dPower_Inflows["value"] = cs_inflow_weekly_aggregated.dPower_Inflows.groupby(["g", "week"])["value"].transform("mean")
    cs_inflow_weekly_aggregated.dPower_Inflows.drop(columns=["k_int", "week"], inplace=True)

    # Daily
    cs_inflow_daily_aggregated.dPower_Inflows["k_int"] = cs_inflow_daily_aggregated.dPower_Inflows.index.get_level_values("k").str[1:].astype(int)
    cs_inflow_daily_aggregated.dPower_Inflows["day"] = (cs_inflow_daily_aggregated.dPower_Inflows["k_int"] - 1) // 24
    cs_inflow_daily_aggregated.dPower_Inflows["value"] = cs_inflow_daily_aggregated.dPower_Inflows.groupby(["g", "day"])["value"].transform("mean")
    cs_inflow_daily_aggregated.dPower_Inflows.drop(columns=["k_int", "day"], inplace=True)

    caseStudy_objects = {
        "hourly": cs_inflow_hourly,
        "yearly": cs_inflow_yearly_aggregated,
        "monthly": cs_inflow_monthly_aggregated,
        "weekly": cs_inflow_weekly_aggregated,
        "daily": cs_inflow_daily_aggregated}
    printer.information("Aggregation of inflow data completed")

    if numberOfRPs > 0:
        printer.information(f"Clustering data into {numberOfRPs} representative periods of length {lengthOfRPs}")

        if clusterOnOriginalData:
            printer.information("Calculating representative periods based on original hourly data")

            aggregation_results = caseStudy_objects["hourly"].get_kmedoids_representative_periods(numberOfRPs, lengthOfRPs)

            for name, cs in caseStudy_objects.items():
                printer.information(f"Clustering inflow data for case study with {name} inflows based on original hourly data")
                cs.apply_representative_periods(aggregation_results, lengthOfRPs)
        else:
            for name, cs in caseStudy_objects.items():
                printer.information(f"Clustering inflow data for case study with {name} inflows")
                cs.apply_kmedoids_aggregation(numberOfRPs, lengthOfRPs)
        printer.information("Clustering of simplified case study completed")
    elif numberOfRPs == 0:
        printer.information("Skipping clustering of case studies since numberOfRPs == 0")

    printer.information("Building LEGO models")
    legos = {}
    for name, cs in caseStudy_objects.items():
        printer.information(f"Building LEGO model for case study with {name} inflows")
        lego = LEGO(cs)
        model, timing = lego.build_model()
        printer.information(f"Building LEGO model for case study with {name} inflows took {timing:.2f} seconds")
        legos[name] = (lego, model)

    hourly_objective = None
    printer.information("Solving LEGO models and calculating regret")
    for name, (lego, model) in legos.items():
        printer.information(f"Solving LEGO model for case study with {name} inflows")
        results, timing, objective_value = lego.solve_model()
        printer.information(f"Solving LEGO model for case study with {name} inflows took {timing:.2f} seconds")

        match results.solver.termination_condition:
            case pyo.TerminationCondition.optimal:
                printer.success(f"Optimal solution: {pyo.value(model.objective):.4f}")
            case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                printer.error(f"Model returned as {results.solver.termination_condition}, logging infeasible constraints:")
                log_infeasible_constraints(model, log_expression=False)
            case _:
                printer.warning(f"Solver terminated with condition: {results.solver.termination_condition}")

        SQLiteWriter.model_to_sqlite(model, f"model_{name}-{caseStudyName}-rps{numberOfRPs}-ks{lengthOfRPs}-demand{scaleDemand}-inflows{scaleInflows}-vresMaxProd{scaleVRESMaxProd}-clusteredOnOriginal{clusterOnOriginalData}.sqlite")

        if name == "hourly":
            hourly_objective = objective_value
            continue  # No regret calculation for hourly inflows (since this is the reference case)
        else:
            printer.information(f"Calculating regret for case study with {name} inflows")
            cs = cs_inflow_hourly.copy()
            lego_regret = LEGO(cs)
            lego_regret_model, timing = lego_regret.build_model()
            for var in lego_regret_model.component_objects(pyo.Var, active=True):
                if var.name == "vGenInvest":
                    for index in var:
                        var[index].value = model.vGenInvest[index].value
                        var[index].fixed = True
            results_regret, timing, objective_value_regret = lego_regret.solve_model()
            printer.information(f"Solving regret model for case study with {name} inflows took {timing:.2f} seconds")

            match results.solver.termination_condition:
                case pyo.TerminationCondition.optimal:
                    printer.success(f"Optimal solution: {pyo.value(model.objective):.4f}")
                case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                    printer.error(f"Model returned as {results.solver.termination_condition}, logging infeasible constraints:")
                    log_infeasible_constraints(model, log_expression=False)
                case _:
                    printer.warning(f"Solver terminated with condition: {results.solver.termination_condition}")

            if hourly_objective is not None:
                regret = objective_value_regret - objective_value
                printer.information(f"Regret for case study with {name} inflows: {regret:.4f}")
            else:
                printer.warning("Hourly objective is None, cannot calculate regret (yet?)")

        SQLiteWriter.model_to_sqlite(lego_regret_model, f"model_{name}-regret-{caseStudyName}-rps{numberOfRPs}-ks{lengthOfRPs}-demand{scaleDemand}-inflows{scaleInflows}-vresMaxProd{scaleVRESMaxProd}-clusteredOnOriginal{clusterOnOriginalData}.sqlite")


if __name__ == "__main__":
    # Parse command line arguments and automatically check for correct usage
    parser = argparse.ArgumentParser(description="Shows difference of detailed and averaged inflow data for given case study", formatter_class=RichHelpFormatter)


    # Check if given string path is a directory
    def directory_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise argparse.ArgumentTypeError(f"Directory path not valid: '{string}'")


    def check_NumberOfRPs(value):
        float_check = float(value)
        if not float_check.is_integer():
            raise argparse.ArgumentTypeError(f"numberOfRPs needs to be an integer, but got: '{value}'")
        if value < 0:
            raise argparse.ArgumentTypeError(f"numberOfRPs needs to be positive, but got: '{value}'")
        return value


    def check_lengthOfRPs(value):
        float_check = float(value)
        if not float_check.is_integer():
            raise argparse.ArgumentTypeError(f"lengthOfRPs needs to be an integer, but got: '{value}'")
        if value <= 0:
            raise argparse.ArgumentTypeError(f"lengthOfRPs needs to be greater than zero, but got: '{value}'")
        return value


    parser.add_argument("caseStudyDirectory", type=directory_path, help="Path to folder containing data for LEGO model")
    parser.add_argument("--numberOfRPs", type=check_NumberOfRPs, default=0, help="Number of representative periods to cluster data into")
    parser.add_argument("--lengthOfRPs", type=check_lengthOfRPs, default=24, help="Length of representative periods (in number of time steps)")
    parser.add_argument("--scaleDemand", type=float, default=1.0, help="Scaling factor for demand (default: 1.0 = no scaling)")
    parser.add_argument("--scaleVRESMaxProd", type=float, default=1.0, help="Scaling factor for VRES maximum production (default: 1.0 = no scaling)")
    parser.add_argument("--scaleInflows", type=float, default=1.0, help="Scaling factor for inflows (default: 1.0 = no scaling)")
    parser.add_argument("--clusterOnOriginalData", action="store_true", help="Cluster data on original data (instead of after aggregation)")
    parser.add_argument("--scaleRoRToInflowScaling", action="store_true", help="Scale run-of-river generation capacity according to inflow scaling factor to prevent inflows that exceed maximum production of RoR plants. Note: This happens ON-TOP of the VRES scaling.")
    args = parser.parse_args()

    if args.numberOfRPs < 1:
        printer.error("numberOfRPs must be at least 1")
        exit(1)
    if args.lengthOfRPs < 1:
        printer.error("lengthOfRPs must be at least 1")
        exit(1)

    main(args.caseStudyDirectory, args.numberOfRPs, args.lengthOfRPs, args.scaleDemand, args.scaleInflows, args.scaleVRESMaxProd, args.clusterOnOriginalData, args.scaleRoRToInflowScaling)
