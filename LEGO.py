import argparse
import logging
import os
import time

import pyomo.environ as pyo
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from pyomo.core import NameLabeler
from pyomo.core.base.var import IndexedVar
from rich_argparse import RichHelpFormatter

from InOutModule import SQLiteWriter, ExcelWriter
from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO, ModelType
from LEGO.LEGOUtilities import analyze_infeasible_constraints

from tools.checkSocpExactness import check_exactness_of_socp_solution

printer = Printer.getInstance()

# Set up logging so that infeasible constraints are logged by pyomo
logger = logging.getLogger("pyomo")
logger.setLevel("INFO")

# Parse command line arguments and automatically check for correct usage
parser = argparse.ArgumentParser(description="Starts LEGO for given case study", formatter_class=RichHelpFormatter)


# Check if given string path is a directory
def directory_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise argparse.ArgumentTypeError(f"Directory path not valid: '{string}'")


def process_results(model_results):
    logger = logging.getLogger('pyomo.util.infeasible')
    logger.setLevel(logging.INFO)

    # Ensure there is a handler attached
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)

    match model_results.solver.termination_condition:
        case pyo.TerminationCondition.optimal:
            match args.modelType:
                case ModelType.DETERMINISTIC:
                    printer.success(f"Optimal solution: {pyo.value(model.objective):.4f}\n")
                case ModelType.EXTENSIVE_FORM:
                    printer.success(f"Optimal solution: {lego._extensive_form.get_objective_value():.4f}\n")
                case _:
                    printer.warning(f"Model type {args.modelType} not fully tested yet, no objective value reported.\n")
        case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
            printer.error(f"Model returned as {model_results.solver.termination_condition}")
            analyze_infeasible_constraints(model)
        case _:
            printer.warning(f"Solver terminated with condition: {model_results.solver.termination_condition}")


def report_socp_metrics(model, label):
    def weighted_sum(component):
        return sum(
            pyo.value(model.pWeight_rp[rp]) * pyo.value(model.pWeight_k[k]) * pyo.value(component[rp, k, i])
            for rp in model.rp
            for k in model.constraintsActiveK
            for i in model.i
        )

    pns = weighted_sum(model.vPNS)
    eps = weighted_sum(model.vEPS)
    voltage_slack = 0.0
    if model.pEnableSoftVoltageLimits:
        voltage_slack = weighted_sum(model.vSOCP_ui_slack_pos) + weighted_sum(model.vSOCP_ui_slack_neg)

    storage_charge = 0.0
    if hasattr(model, "vConsump"):
        storage_charge = sum(
            pyo.value(model.pWeight_rp[rp]) * pyo.value(model.pWeight_k[k]) * pyo.value(model.vConsump[rp, k, g])
            for rp in model.rp
            for k in model.constraintsActiveK
            for g in model.storageUnits
        )

    printer.information(
        f"{label}: PNS={pns:.8g}, EPS={eps:.8g}, voltage slack={voltage_slack:.8g}, "
        f"storage charge={storage_charge:.8g}, weighted losses={pyo.value(model.eSOCPTighteningLoss):.8g}"
    )


def solve_with_optional_socp_tightening(lego, model):
    if not args.socp_two_stage:
        return lego.solve_model(model_type=args.modelType)

    if args.modelType != ModelType.DETERMINISTIC or not hasattr(model, "eSOCPTighteningLoss"):
        printer.warning("Hierarchical SOCP tightening is only available for the deterministic BFM model; using the normal solve.")
        return lego.solve_model(model_type=args.modelType)

    if args.socp_voltage_slack_factor is not None:
        model.pVoltageSlackPenaltyFactor.set_value(args.socp_voltage_slack_factor)

    tightening_term = model.pSOCPTighteningEpsilon * model.eSOCPTighteningLoss
    voltage_slack_penalty = model.eSOCPVoltageSlackPenalty
    primary_objective = model.objective.expr - tightening_term - voltage_slack_penalty
    reporting_objective = primary_objective + voltage_slack_penalty
    model.objective.set_value(primary_objective)

    printer.information("Phase 1/3: solving supply, investment and operating costs")
    primary_results, primary_timing, primary_value = lego.solve_model(model_type=args.modelType)
    if primary_results.solver.termination_condition != pyo.TerminationCondition.optimal:
        return primary_results, primary_timing, primary_value
    report_socp_metrics(model, "Phase 1")
    check_exactness_of_socp_solution(lego)

    allowed_degradation = max(
        args.socp_primary_absolute_tolerance,
        abs(primary_value) * args.socp_primary_relative_tolerance,
    )
    model.eSOCPPrimaryObjectiveLimit = pyo.Constraint(
        expr=primary_objective <= primary_value + allowed_degradation
    )
    model.objective.set_value(model.eSOCPTighteningLoss)

    printer.information(
        f"Phase 2/3: minimizing weighted line losses with primary objective <= "
        f"{primary_value + allowed_degradation:.8g}"
    )
    loss_results, loss_timing, loss_value = lego.solve_model(
        model_type=args.modelType,
        already_solved_ok=True,
    )
    if loss_results.solver.termination_condition != pyo.TerminationCondition.optimal:
        model.objective.set_value(reporting_objective)
        return loss_results, primary_timing + loss_timing, pyo.value(reporting_objective)
    report_socp_metrics(model, "Phase 2")
    check_exactness_of_socp_solution(lego)

    allowed_loss_degradation = max(
        args.socp_loss_absolute_tolerance,
        abs(loss_value) * args.socp_loss_relative_tolerance,
    )
    model.eSOCPLossLimit = pyo.Constraint(
        expr=model.eSOCPTighteningLoss <= loss_value + allowed_loss_degradation
    )
    model.objective.set_value(model.eSOCPVoltageSlack)

    printer.information(
        f"Phase 3/3: minimizing voltage soft-limit violations with weighted losses <= "
        f"{loss_value + allowed_loss_degradation:.8g}"
    )
    voltage_results, voltage_timing, voltage_value = lego.solve_model(
        model_type=args.modelType,
        already_solved_ok=True,
    )

    model.objective.set_value(reporting_objective)
    if voltage_results.solver.termination_condition == pyo.TerminationCondition.optimal:
        report_socp_metrics(model, "Phase 3")
        printer.information(
            f"Hierarchical result: primary objective={pyo.value(primary_objective):.8g}, "
            f"weighted losses={pyo.value(model.eSOCPTighteningLoss):.8g}, "
            f"voltage slack={voltage_value:.8g}"
        )

    return voltage_results, primary_timing + loss_timing + voltage_timing, pyo.value(reporting_objective)


parser.add_argument("caseStudyDirectory", type=directory_path, help="Path to folder containing data for LEGO model")
parser.add_argument("modelType", default=ModelType.DETERMINISTIC, type=lambda s: ModelType[s], choices=list(ModelType), nargs="?", help="ModelType of first model")
parser.add_argument("--socp-hierarchical", "--socp-two-stage", dest="socp_two_stage", action="store_true", help="Solve BFM hierarchically: system objective, losses, then voltage soft-limit violations")
parser.add_argument("--socp-primary-relative-tolerance", type=float, default=1e-6, help="Relative primary-objective degradation allowed after phase 1 (default: 1e-6)")
parser.add_argument("--socp-primary-absolute-tolerance", type=float, default=1e-8, help="Absolute primary-objective degradation allowed after phase 1 (default: 1e-8)")
parser.add_argument("--socp-loss-relative-tolerance", type=float, default=1e-6, help="Relative loss degradation allowed in phase 3 (default: 1e-6)")
parser.add_argument("--socp-loss-absolute-tolerance", type=float, default=1e-8, help="Absolute loss degradation allowed in phase 3 (default: 1e-8)")
parser.add_argument("--socp-voltage-slack-factor", type=float, default=None, help="Voltage-slack penalty as a factor of ENS cost; default keeps the model value 0.0001")
args = parser.parse_args()

# Load case study
printer.information(f"Loading case study from '{args.caseStudyDirectory}'\n")
start_time = time.time()
cs = CaseStudy(args.caseStudyDirectory)
#cs = cs.filter_timesteps('k00001','k01000')


rh_length = cs.dGlobal_Parameters["pMovingWindowLength"]
rh_overlap = cs.dGlobal_Parameters["pMovingWindowOverlap"]

# Check if moving window is disabled (both parameters are 0)
use_moving_window = rh_length > 0 and rh_overlap >= 0

if not use_moving_window:
    printer.information("Moving window disabled - running entire problem at once\n")

    lego = LEGO(cs)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    # Build LEGO model
    printer.information("Building LEGO model")
    model, timing = lego.build_model(model_type=args.modelType)
    # with open("pprint.txt", "w") as f:
    #     model.objective.pprint(f)
    printer.information(f"Building LEGO model took {timing:.2f} seconds")

    # Solve LEGO model
    printer.information("Solving LEGO model")
    try:
        results, timing, objective_value = solve_with_optional_socp_tightening(lego, model)
        printer.information(f"Solving LEGO model took {timing:.2f} seconds\n")
        process_results(results)
        check_exactness_of_socp_solution(lego)
    except NoFeasibleSolutionError:
        printer.error("No feasible solution found!")
        analyze_infeasible_constraints(model)
        exit(1)

else:
    printer.information(f"Using moving window: length={rh_length}, overlap={rh_overlap}\n")

    model_old = None
    total_timesteps = len(cs.dPower_WeightsK.index.unique())
    k_padding = len(cs.dPower_WeightsK.index.unique()[0]) - 1

    start_timestep = 1
    while start_timestep <= total_timesteps:
        start_time_iteration = time.time()

        # Calculate the end of the window
        end_timestep = min(start_timestep + rh_length - 1, total_timesteps)

        # Format timestep strings for filtering and printing
        start_k = f"k{start_timestep:0{k_padding}}"
        end_k = f"k{end_timestep:0{k_padding}}"
        print(f"Start k: {start_k}, End k: {end_k}")

        cs.constraints_active_k = [f"k{i:0{k_padding}}" for i in range(start_timestep, end_timestep + 1)]
        printer.information(f"Processing window from {start_k} to {end_k}...")

        cut_cs = cs.filter_timesteps(cs.dPower_WeightsK.index.unique()[0], end_k)

        lego = LEGO(cut_cs)
        printer.information(f"Loading case study took {time.time() - start_time_iteration:.2f} seconds")

        # Build LEGO model
        printer.information("Building LEGO model")
        model, timing = lego.build_model(model_type=args.modelType)


        printer.information(f"Building LEGO model took {timing:.2f} seconds")

        if model_old is not None:
            new_end = f"k{start_timestep-1:05}"
            print(f"New end: {new_end}")
            for component in list(model_old.component_objects()):
                if isinstance(component, IndexedVar):
                    indices = [str(i) for i in component.index_set().subsets()]

                    if "k" in indices:
                        new_component = getattr(model, str(component))
                        for n, v in list(component.items()):
                            if n[(indices.index('k'))] <= new_end:
                                if v.value is not None:
                                    new_component[n].fix(pyo.value(v))  # TODO skip validation

        # Solve LEGO model
        printer.information("Solving LEGO model")
        try:
            results, timing, objective_value = solve_with_optional_socp_tightening(lego, model)
            printer.information(f"Solving LEGO model took {timing:.2f} seconds")
            process_results(results)
            check_exactness_of_socp_solution(lego)
        except NoFeasibleSolutionError:
            printer.error(f"No feasible solution found for window {start_k} to {end_k}!")
            analyze_infeasible_constraints(model)
            exit(1)

        if start_timestep + rh_length >= total_timesteps:
            break
        model_old = model
        start_timestep += rh_length - rh_overlap

printer.information(f"Finished in {time.time() - start_time:.2f} seconds")

SQLiteWriter.model_to_sqlite(model, "model.sqlite")
#ExcelWriter.model_to_excel(model, "model.xlsx")
model.write("model.mps", io_options={'labeler': NameLabeler()})
