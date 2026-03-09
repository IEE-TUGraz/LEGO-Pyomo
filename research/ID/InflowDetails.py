import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

SCALE_DEFAULT = 1.0  # Scaling default should always be 1.0 (no scaling)
NUMBER_OF_RPS_DEFAULT = 0
LENGTH_OF_RPS_DEFAULT = 24


class _AggregationProxy:
    """Lightweight proxy that provides the tsam attributes needed by apply_representative_periods."""

    def __init__(self, clusterCenterIndices, clusterOrder, clusterPeriodNoOccur):
        self.clusterCenterIndices = clusterCenterIndices
        self._clusterOrder = clusterOrder
        self._clusterPeriodNoOccur = clusterPeriodNoOccur


printer = Printer.getInstance()

# Set up logging so that infeasible constraints are logged by pyomo
logger = logging.getLogger("pyomo")
logger.setLevel("INFO")


def build_run_parameters(case_study_directory, inflow_aggregation, number_of_rps, length_of_rps,
                         scale_demand, scale_inflows, scale_vres_max_prod, scale_pmax,
                         cluster_on_original_data, scale_ror_to_inflow_scaling, rmip, single_node,
                         limit_k, tp, is_regret=False):
    """Build a dict of run parameters for storage in sqlite."""
    params = {
        'case_study_directory': case_study_directory,
        'inflow_aggregation': inflow_aggregation,
        'is_regret': is_regret,
    }
    if number_of_rps != NUMBER_OF_RPS_DEFAULT:
        params['number_of_rps'] = number_of_rps
    if length_of_rps != LENGTH_OF_RPS_DEFAULT:
        params['length_of_rps'] = length_of_rps
    if scale_demand != SCALE_DEFAULT:
        params['scale_demand'] = scale_demand
    if scale_inflows != SCALE_DEFAULT:
        params['scale_inflows'] = scale_inflows
    if scale_vres_max_prod != SCALE_DEFAULT:
        params['scale_vres_max_prod'] = scale_vres_max_prod
    if scale_pmax != SCALE_DEFAULT:
        params['scale_pmax'] = scale_pmax
    if cluster_on_original_data:
        params['cluster_on_original_data'] = True
    if scale_ror_to_inflow_scaling:
        params['scale_ror_to_inflow_scaling'] = True
    if rmip:
        params['rmip'] = True
    if single_node:
        params['single_node'] = True
    if limit_k is not None:
        params['limit_k'] = limit_k
    if tp:
        params['tp'] = True
    return params


AGGREGATION_LEVELS = ["daily", "weekly", "monthly", "yearly"]


def main(caseStudyDirectory, numberOfRPs, lengthOfRPs, scaleDemand, scaleInflows, scaleVRESMaxProd,
         clusterOnOriginalData, scaleRoRToInflowScaling, rMIP, singleNode, scalePMax, limitK, noOverwrite, rerunWithRPLength, aggregations, tp=False, aggregationToBeApplied=None):
    if rerunWithRPLength is not None:
        if rerunWithRPLength >= lengthOfRPs:
            raise ValueError(f"rerunWithRPLength ({rerunWithRPLength}) must be smaller than original lengthOfRPs ({lengthOfRPs})")
        if lengthOfRPs % rerunWithRPLength != 0:
            raise ValueError(f"lengthOfRPs ({lengthOfRPs}) must be divisible by rerunWithRPLength ({rerunWithRPLength})")
        if not clusterOnOriginalData and aggregationToBeApplied is None:
            raise ValueError("rerunWithRPLength requires --clusterOnOriginalData (aggregation results must exist)")
        if numberOfRPs == 0:
            raise ValueError("rerunWithRPLength requires --numberOfRPs > 0")

    if singleNode and tp:
        raise ValueError("Options --singleNode and --TP cannot be combined since both modify the technical representation of lines")

    if scaleRoRToInflowScaling and scaleInflows == SCALE_DEFAULT:
        printer.warning("--scaleRoRToInflowScaling is set but --scaleInflows is 1.0 (no scaling), so RoR scaling has no effect")

    caseStudyName = caseStudyDirectory.replace("/", "_").replace("\\", "_")

    printer.information(f"Loading original case study from '{caseStudyDirectory}'")
    identifier_parts_base = [f"data{caseStudyName}"]  # Build identifier with only non-default parameters
    start_time = time.time()
    cs_inflow_hourly = CaseStudy(caseStudyDirectory)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    if limitK is not None:
        printer.information(f"Limiting K values to '{limitK}'")
        identifier_parts_base.append(f"limitK{limitK}")
        start, end = limitK.split("-")
        cs_inflow_hourly.filter_timesteps(start, end, inplace=True)

    if scaleDemand != SCALE_DEFAULT:
        printer.information(f"Scaling demand by factor {scaleDemand}")
        identifier_parts_base.append(f"demand{scaleDemand:g}")
        cs_inflow_hourly.dPower_Demand['value'] *= scaleDemand

    if scaleVRESMaxProd != SCALE_DEFAULT:
        printer.information(f"Scaling VRES maximum production by factor {scaleVRESMaxProd}")
        identifier_parts_base.append(f"vresMaxProd{scaleVRESMaxProd:g}")
        cs_inflow_hourly.dPower_VRES['MaxProd'] *= scaleVRESMaxProd

    if scalePMax != SCALE_DEFAULT:
        printer.information(f"Scaling pPmax (line capacity) by factor {scalePMax}")
        identifier_parts_base.append(f"pmax{scalePMax:g}")
        cs_inflow_hourly.dPower_Network['pPmax'] *= scalePMax

    if scaleRoRToInflowScaling:
        printer.information(f"Scaling run-of-river generation capacity by inflow scaling factor {scaleInflows} to prevent inflows that exceed maximum production of RoR plants")
        identifier_parts_base.append("rorScaled")
        ror_gens = cs_inflow_hourly.dPower_VRES[cs_inflow_hourly.dPower_VRES['tec'].str.lower() == 'ror'].index
        cs_inflow_hourly.dPower_VRES.loc[ror_gens, 'MaxProd'] *= scaleInflows

    if rMIP:
        printer.information("Setting up case study as rMIP")
        identifier_parts_base.append("rMIP")
        cs_inflow_hourly.dGlobal_Parameters["pEnableRMIP"] = True

    if singleNode:
        printer.information("Setting up case study as single node (no network constraints)")
        identifier_parts_base.append("SN")
        cs_inflow_hourly.dPower_Network["pTecRepr"] = "SN"
        cs_inflow_hourly.merge_single_node_buses()

    if tp:
        printer.information("Setting technical representation of all lines to TP (Transport Problem)")
        identifier_parts_base.append("TP")
        cs_inflow_hourly.dPower_Network["pTecRepr"] = "TP"

    if numberOfRPs != NUMBER_OF_RPS_DEFAULT:
        identifier_parts_base.append(f"rps{numberOfRPs}")
    if lengthOfRPs != LENGTH_OF_RPS_DEFAULT:
        identifier_parts_base.append(f"ks{lengthOfRPs}")
    if clusterOnOriginalData:
        identifier_parts_base.append("clustOriginal")

    if scaleInflows != SCALE_DEFAULT:
        printer.information(f"Scaling inflows by factor {scaleInflows}")
        identifier_parts_base.append(f"inflows{scaleInflows:g}")
        cs_inflow_hourly.dPower_Inflows['value'] *= scaleInflows

    active_aggregations = set(aggregations)

    printer.information(f"Creating copies of case study for aggregation levels: {', '.join(sorted(active_aggregations))}")
    caseStudy_objects = {"hourly": cs_inflow_hourly}

    printer.information("Aggregating inflow data based on selected levels")

    if "yearly" in active_aggregations:
        cs_inflow_yearly_aggregated = cs_inflow_hourly.copy()
        mean_inflows = cs_inflow_yearly_aggregated.dPower_Inflows["value"].groupby(["g"]).mean()  # Calculate mean inflow per generator
        cs_inflow_yearly_aggregated.dPower_Inflows.loc[:, "value"] = mean_inflows[cs_inflow_yearly_aggregated.dPower_Inflows.index.droplevel(["rp", "k"])].values  # Set all inflows to mean inflow of respective generator
        caseStudy_objects["yearly"] = cs_inflow_yearly_aggregated

    if "monthly" in active_aggregations:
        cs_inflow_monthly_aggregated = cs_inflow_hourly.copy()
        cs_inflow_monthly_aggregated.dPower_Inflows["k_int"] = cs_inflow_monthly_aggregated.dPower_Inflows.index.get_level_values("k").str[1:].astype(int)
        cs_inflow_monthly_aggregated.dPower_Inflows["month"] = (cs_inflow_monthly_aggregated.dPower_Inflows["k_int"] - 1) // 720
        cs_inflow_monthly_aggregated.dPower_Inflows["value"] = cs_inflow_monthly_aggregated.dPower_Inflows.groupby(["g", "month"])["value"].transform("mean")
        cs_inflow_monthly_aggregated.dPower_Inflows.drop(columns=["k_int", "month"], inplace=True)
        caseStudy_objects["monthly"] = cs_inflow_monthly_aggregated

    if "weekly" in active_aggregations:
        cs_inflow_weekly_aggregated = cs_inflow_hourly.copy()
        cs_inflow_weekly_aggregated.dPower_Inflows["k_int"] = cs_inflow_weekly_aggregated.dPower_Inflows.index.get_level_values("k").str[1:].astype(int)
        cs_inflow_weekly_aggregated.dPower_Inflows["week"] = (cs_inflow_weekly_aggregated.dPower_Inflows["k_int"] - 1) // 168
        cs_inflow_weekly_aggregated.dPower_Inflows["value"] = cs_inflow_weekly_aggregated.dPower_Inflows.groupby(["g", "week"])["value"].transform("mean")
        cs_inflow_weekly_aggregated.dPower_Inflows.drop(columns=["k_int", "week"], inplace=True)
        caseStudy_objects["weekly"] = cs_inflow_weekly_aggregated

    if "daily" in active_aggregations:
        cs_inflow_daily_aggregated = cs_inflow_hourly.copy()
        cs_inflow_daily_aggregated.dPower_Inflows["k_int"] = cs_inflow_daily_aggregated.dPower_Inflows.index.get_level_values("k").str[1:].astype(int)
        cs_inflow_daily_aggregated.dPower_Inflows["day"] = (cs_inflow_daily_aggregated.dPower_Inflows["k_int"] - 1) // 24
        cs_inflow_daily_aggregated.dPower_Inflows["value"] = cs_inflow_daily_aggregated.dPower_Inflows.groupby(["g", "day"])["value"].transform("mean")
        cs_inflow_daily_aggregated.dPower_Inflows.drop(columns=["k_int", "day"], inplace=True)
        caseStudy_objects["daily"] = cs_inflow_daily_aggregated
    printer.information("Aggregation of inflow data completed")

    aggregation_results = None
    if numberOfRPs > 0:
        printer.information(f"Clustering data into {numberOfRPs} representative periods of length {lengthOfRPs}")

        if aggregationToBeApplied is not None:
            printer.information("Using pre-computed aggregation results (from rerun)")
            aggregation_results = aggregationToBeApplied
            for name, cs in caseStudy_objects.items():
                printer.information(f"Applying pre-computed clustering to case study with {name} inflows")
                cs.apply_representative_periods(aggregation_results, lengthOfRPs)
        elif clusterOnOriginalData:
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
        identifier_parts = identifier_parts_base.copy() + [name]
        sqlite_filename = f"ID-{'-'.join(identifier_parts)}.sqlite"

        if noOverwrite and os.path.exists(sqlite_filename):
            printer.information(f"  File '{sqlite_filename}' already exists, skipping (--noOverwrite)")
            continue

        printer.information(f"\n{'=' * 60}")
        printer.information(f"  {name} inflows")
        printer.information(f"{'=' * 60}\n")

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

        # --- Export ---
        SQLiteWriter.model_to_sqlite(model, sqlite_filename)
        SQLiteWriter.add_solver_statistics_to_sqlite(sqlite_filename, results, work_units=lego.work_units)
        SQLiteWriter.add_run_parameters_to_sqlite(
            sqlite_filename,
            **build_run_parameters(
                caseStudyDirectory, name, numberOfRPs, lengthOfRPs,
                scaleDemand, scaleInflows, scaleVRESMaxProd, scalePMax,
                clusterOnOriginalData, scaleRoRToInflowScaling, rMIP, singleNode, limitK, tp,
                is_regret=False
            )
        )
        printer.information(f"  Saved to '{sqlite_filename}'")

        if name == "hourly":
            hourly_objective = objective_value
            continue  # No regret calculation for hourly inflows (since this is the reference case)

        # --- Regret ---
        regret_identifier_parts = identifier_parts + ["regret"]
        regret_sqlite_filename = f"ID-{'-'.join(regret_identifier_parts)}.sqlite"

        if noOverwrite and os.path.exists(regret_sqlite_filename):
            printer.information(f"  File '{regret_sqlite_filename}' already exists, skipping (--noOverwrite)")
            continue

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

        match results_regret.solver.termination_condition:
            case pyo.TerminationCondition.optimal:
                printer.success(f"Optimal regret solution: {pyo.value(lego_regret_model.objective):.4f}")
            case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                printer.error(f"Regret model returned as {results_regret.solver.termination_condition}, logging infeasible constraints:")
                log_infeasible_constraints(lego_regret_model, log_expression=False)
            case _:
                printer.warning(f"Regret solver terminated with condition: {results_regret.solver.termination_condition}")

        if hourly_objective is not None:
            regret = objective_value_regret - hourly_objective
            printer.information(f"Regret for case study with {name} inflows: {regret:.4f}")
        else:
            printer.warning("Hourly objective is None, cannot calculate regret (yet?)")

        # --- Export regret ---
        SQLiteWriter.model_to_sqlite(lego_regret_model, regret_sqlite_filename)
        SQLiteWriter.add_solver_statistics_to_sqlite(regret_sqlite_filename, results_regret, work_units=lego_regret.work_units)
        SQLiteWriter.add_run_parameters_to_sqlite(
            regret_sqlite_filename,
            **build_run_parameters(
                caseStudyDirectory, name, numberOfRPs, lengthOfRPs,
                scaleDemand, scaleInflows, scaleVRESMaxProd, scalePMax,
                clusterOnOriginalData, scaleRoRToInflowScaling, rMIP, singleNode, limitK, tp,
                is_regret=True
            )
        )
        printer.information(f"  Saved regret to '{regret_sqlite_filename}'")

    if rerunWithRPLength is not None:
        printer.information(f"\n{'=' * 60}")
        printer.information(f"Re-running with lengthOfRPs={rerunWithRPLength}")
        printer.information(f"{'=' * 60}\n")

        multiplier = lengthOfRPs // rerunWithRPLength  # sub-periods per original RP (e.g. 168/24 = 7)
        newNumberOfRPs = numberOfRPs * multiplier  # e.g. 2 * 7 = 14

        printer.information(f"Splitting {numberOfRPs} RPs of length {lengthOfRPs} into {newNumberOfRPs} RPs of length {rerunWithRPLength} (multiplier={multiplier})")

        newAggregationResults = {}
        for scenario, tsa in aggregation_results.items():
            # clusterCenterIndices: each original medoid period spawns `multiplier` new sub-period medoids
            newClusterCenterIndices = []
            for center in tsa.clusterCenterIndices:
                for i in range(multiplier):
                    newClusterCenterIndices.append(center * multiplier + i)

            # _clusterOrder: each original period maps to a cluster; now each original period
            # splits into `multiplier` sub-periods, each mapping to a new sub-cluster
            # Original period p in cluster c -> sub-periods map to clusters c*multiplier+0, c*multiplier+1, ...
            newClusterOrder = []
            for old_cluster in tsa._clusterOrder:
                for i in range(multiplier):
                    newClusterOrder.append(old_cluster * multiplier + i)

            # _clusterPeriodNoOccur: each old cluster spawns `multiplier` new clusters,
            # each with the same occurrence count
            newClusterPeriodNoOccur = {}
            for old_cluster, count in tsa._clusterPeriodNoOccur.items():
                for i in range(multiplier):
                    newClusterPeriodNoOccur[old_cluster * multiplier + i] = count

            newAggregationResults[scenario] = _AggregationProxy(
                newClusterCenterIndices, newClusterOrder, newClusterPeriodNoOccur)

        main(caseStudyDirectory, newNumberOfRPs, rerunWithRPLength, scaleDemand, scaleInflows, scaleVRESMaxProd,
             clusterOnOriginalData, scaleRoRToInflowScaling, rMIP, singleNode, scalePMax, limitK, noOverwrite, None, aggregations, tp, newAggregationResults)


if __name__ == "__main__":
    # Parse command line arguments and automatically check for correct usage
    parser = argparse.ArgumentParser(description="Shows difference of detailed and averaged inflow data for given case study", formatter_class=RichHelpFormatter, fromfile_prefix_chars='@')


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
        if float_check < 0:
            raise argparse.ArgumentTypeError(f"numberOfRPs needs to be positive, but got: '{value}'")
        return int(float_check)


    def check_lengthOfRPs(value):
        float_check = float(value)
        if not float_check.is_integer():
            raise argparse.ArgumentTypeError(f"lengthOfRPs needs to be an integer, but got: '{value}'")
        if float_check <= 0:
            raise argparse.ArgumentTypeError(f"lengthOfRPs needs to be greater than zero, but got: '{value}'")
        return int(float_check)


    parser.add_argument("caseStudyDirectory", type=directory_path, help="Path to folder containing data for LEGO model")
    parser.add_argument("--numberOfRPs", type=check_NumberOfRPs, default=NUMBER_OF_RPS_DEFAULT, help=f"Number of representative periods to cluster data into (default: {NUMBER_OF_RPS_DEFAULT})")
    parser.add_argument("--lengthOfRPs", type=check_lengthOfRPs, default=LENGTH_OF_RPS_DEFAULT, help=f"Length of representative periods (in number of time steps) (default: {LENGTH_OF_RPS_DEFAULT})")
    parser.add_argument("--limitK", type=str, help="Limit the ks, format: 'k0025-k0048'", nargs="?", default=None)
    parser.add_argument("--scaleDemand", type=float, help=f"Scaling factor for demand (default: {SCALE_DEFAULT} = no scaling)", default=SCALE_DEFAULT)
    parser.add_argument("--scaleVRESMaxProd", type=float, help=f"Scaling factor for VRES maximum production (default: {SCALE_DEFAULT} = no scaling)", default=SCALE_DEFAULT)
    parser.add_argument("--scaleInflows", type=float, help=f"Scaling factor for inflows (default: {SCALE_DEFAULT} = no scaling)", default=SCALE_DEFAULT)
    parser.add_argument("--scalePMax", type=float, help=f"Scaling factor for pPmax (line capacity) (default: {SCALE_DEFAULT} = no scaling)", nargs="?", default=SCALE_DEFAULT)
    parser.add_argument("--clusterOnOriginalData", action="store_true", help="Cluster data on original data (instead of after aggregation)")
    parser.add_argument("--scaleRoRToInflowScaling", action="store_true", help="Scale run-of-river generation capacity according to inflow scaling factor to prevent inflows that exceed maximum production of RoR plants. Note: This happens ON-TOP of the VRES scaling.")
    parser.add_argument("--rMIP", action="store_true", help="Solve model as rMIP")
    parser.add_argument("--singleNode", action="store_true", help="Solve model as single node (no network constraints)")
    parser.add_argument("--TP", action="store_true", help="Set technical representation of all lines to TP (Transport Problem)")
    parser.add_argument("--noOverwrite", action="store_true", help="Skip cases where the output .sqlite file already exists")
    parser.add_argument("--rerunWithRPLength", type=check_lengthOfRPs, help="Re-run with given length of RPs", nargs="?", default=None)
    parser.add_argument("--aggregations", nargs="+", choices=AGGREGATION_LEVELS, default=AGGREGATION_LEVELS, metavar="LEVEL", help=f"Aggregation levels to run, choose from {AGGREGATION_LEVELS} (default: all). hourly is always included.")
    args = parser.parse_args()

    main(args.caseStudyDirectory, args.numberOfRPs, args.lengthOfRPs, args.scaleDemand, args.scaleInflows, args.scaleVRESMaxProd, args.clusterOnOriginalData, args.scaleRoRToInflowScaling, args.rMIP, args.singleNode, args.scalePMax, args.limitK, args.noOverwrite, args.rerunWithRPLength, args.aggregations, args.TP)
