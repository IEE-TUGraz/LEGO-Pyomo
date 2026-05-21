import argparse
import json
import logging
import os
import time
from pathlib import Path

import pyomo.environ as pyo

from InOutModule import SQLiteWriter
from InOutModule.ExcelWriter import ExcelWriter
from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from pyomo.core import NameLabeler
from pyomo.util.infeasible import log_infeasible_constraints
from rich_argparse import RichHelpFormatter
from LEGO.LEGO import LEGO, ModelType

printer = Printer.getInstance()

# Set up logging so that infeasible constraints are logged by pyomo
logger = logging.getLogger("pyomo")
logger.setLevel("INFO")


# Check if given string path is a directory
def directory_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise argparse.ArgumentTypeError(f"Directory path not valid: '{string}'")


def ensure_dir(path):
    """Create directory if it doesn't exist, return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(case_study_directory, model_type, scenario_params, output_dir):
    # Load case study
    printer.information(f"Loading case study from '{case_study_directory}'")
    start_time = time.time()

    def K(celsius):
        return 273.15 + celsius

    myCustomParameters = {
        "Building_ThermalMass": 3.5,  # MWh/K (example order of magnitude)

        # --- normal operation temperatures (K) ---
        "Building_MinTemp": K(20),  # 20 °C
        "Building_MaxTemp": K(23),  # 24 °C
        "Building_SetTemp": K(21),  # 21 °C

        # --- comfort penalties ---
        "UnderTempPenaltyCost": 0.175,  # k€/K deviation (high to enforce comfort); €/kWh outage * thermal mass for estimation
        "OverTempPenaltyCost": 0.010,  # k€/K deviation
        "PenaltyFreeTemperatureDeviation": 0.8,  # ±0.5 K deadband

        # --- outage conditions (K) ---
        "Building_MaxTempOutage": K(scenario_params['Building_MaxTempOutage']),  # upper safety limit
        "Building_MinTempOutage": K(scenario_params['Building_MinTempOutage']),  # lower safety limit
        "T_grid_outage": scenario_params['T_outage'],  # hours of grid outage

        # --- Storage min level ---
        "MinSorLevel": 0.15,  # 15 % min SOC

        # --- costs ---
        "DiselStorageTankCost": 0.200,  # k€/MWh = €/kWh (includinc conversion allready)
    }

    cs = CaseStudy(case_study_directory, dCustom_Parameters=myCustomParameters)

    # filter only a small time slice
    # cs.filter_timesteps('k0001','k0012',inplace=True)

    # change the value of T_bo in dGlobalParameter
    cs.dGlobal_Parameters['pTOutage'] = cs.dCustom_Parameters['T_grid_outage']

    lego = LEGO(cs)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    # Build LEGO model
    printer.information("Building LEGO model")
    model, timing = lego.build_model(model_type=model_type)
    printer.information(f"Building LEGO model took {timing:.2f} seconds")

    # Solve LEGO model
    printer.information("Solving LEGO model")
    results, timing, objective_value = lego.solve_model(model_type=model_type)
    printer.information(f"Solving LEGO model took {timing:.2f} seconds")

    infeasible_logger = logging.getLogger('pyomo.util.infeasible')
    infeasible_logger.setLevel(logging.INFO)

    # Ensure there is a handler attached
    if not infeasible_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        infeasible_logger.addHandler(handler)

    match results.solver.termination_condition:
        case pyo.TerminationCondition.optimal:
            match model_type:
                case ModelType.DETERMINISTIC:
                    printer.success(f"Optimal solution: {pyo.value(model.objective):.4f}")
                case ModelType.EXTENSIVE_FORM:
                    printer.success(f"Optimal solution: {lego._extensive_form.get_objective_value():.4f}")
                case _:
                    printer.warning(f"Model type {model_type} not fully tested yet, no objective value reported.")
        case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
            printer.error(f"Model returned as {results.solver.termination_condition}, logging infeasible constraints:")
            log_infeasible_constraints(model, log_expression=False)
        case _:
            printer.warning(f"Solver terminated with condition: {results.solver.termination_condition}")

    SQLiteWriter.model_to_sqlite(model, str(output_dir / "model.sqlite"))
    # ExcelWriter.model_to_excel(model, str(output_dir / "model.xlsx"))
    # model.write(str(output_dir / "model.mps"), io_options={'labeler': NameLabeler()})

    # with open(output_dir / "model_structure.txt", "w") as f:
    #     model.pprint(ostream=f)

    printer.success(f"Scenario results written to '{output_dir}'")


def load_scenario_params(params_path):
    """Load the per-scenario parameter dict from a JSON file."""
    params_path = Path(params_path)
    if not params_path.is_file():
        raise FileNotFoundError(f"Params file not found: {params_path}")
    with open(params_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs LEGO for a single scenario (parameters supplied via a JSON file)",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("caseStudyDirectory", type=directory_path,
                        help="Path to folder containing data for LEGO model")
    parser.add_argument("modelType", default=ModelType.DETERMINISTIC,
                        type=lambda s: ModelType[s], choices=list(ModelType),
                        nargs="?", help="ModelType of the model")
    parser.add_argument("--params", required=True,
                        help="Path to the JSON file with this scenario's parameters")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where this scenario's results are written")
    parser.add_argument("--scenario-name", default=None,
                        help="Optional scenario name (for nicer logging)")
    args = parser.parse_args()

    scenario_params = load_scenario_params(args.params)
    output_dir = ensure_dir(args.output_dir)

    name = args.scenario_name or output_dir.name
    printer.information(f"===== Running scenario: {name} =====")
    printer.information(f"Parameters: {scenario_params}")

    main(
        case_study_directory=args.caseStudyDirectory,
        model_type=args.modelType,
        scenario_params=scenario_params,
        output_dir=output_dir,
    )
