import argparse
import logging
import os
import time
from pathlib import Path
import shutil

import pandas as pd
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
        "Building_ThermalMass": 1.6,  # MWh/K (example order of magnitude)

        # --- normal operation temperatures (K) ---
        "Building_MinTemp": K(20),  # 20 °C
        "Building_MaxTemp": K(22),  # 24 °C
        "Building_SetTemp": K(21),  # 21 °C

        # --- comfort penalties ---
        "UnderTempPenaltyCost": 1000,  # €/K deviation (high to enforce comfort)
        "OverTempPenaltyCost": 500,  # €/K deviation
        "PenaltyFreeTemperatureDeviation": 0.5,  # ±0.5 K deadband

        # --- outage conditions (K) ---
        "Building_MaxTempOutage": K(scenario_params['Building_MaxTempOutage']),  # upper safety limit
        "Building_MinTempOutage": K(scenario_params['Building_MinTempOutage']),  # lower safety limit
        "T_grid_outage": scenario_params['T_outage'],  # hours of grid outage

        # --- costs ---
        "DiselStorageTankCost": 200,  # k€/MWh(includinc conversion allready)
    }

    cs = CaseStudy(case_study_directory, dCustom_Parameters=myCustomParameters)

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
    #ExcelWriter.model_to_excel(model, str(output_dir / "model.xlsx"))
    #model.write(str(output_dir / "model.mps"), io_options={'labeler': NameLabeler()})

    #with open(output_dir / "model_structure.txt", "w") as f:
    #    model.pprint(ostream=f)

    printer.success(f"Scenario results written to '{output_dir}'")

    #with open("model_structure.txt", "w") as f:
    #    model.pprint(ostream=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs LEGO for every scenario in an Excel sheet",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("caseStudyDirectory", type=directory_path,
                        help="Path to folder containing data for LEGO model")
    parser.add_argument("modelType", default=ModelType.DETERMINISTIC,
                        type=lambda s: ModelType[s], choices=list(ModelType),
                        nargs="?", help="ModelType of the model")
    args = parser.parse_args()

    # Hardcoded scenario file path (raw string → backslashes are safe)
    scenario_file = Path(r"C:\Users\Simon Malacek\Nextcloud\A_PhD-IEE\2026-04_ResearchStay_SelfSufficiency\data\benders_test\Scenario_Input.xlsx")

    if not scenario_file.is_file():
        raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

    # Output root = "results" folder next to the scenario file
    output_root = ensure_dir(scenario_file.parent / "results")
    printer.information(f"Results will be written to '{output_root}'")

    # Load scenarios
    printer.information(f"Loading scenarios from '{scenario_file}'")
    df_scenarios = pd.read_excel(scenario_file, skiprows=[1])
    printer.information(f"Found {len(df_scenarios)} scenario(s)")


    # Run each scenario
    for idx, row in df_scenarios.iterrows():
        scenario_name = (
            str(row["ScenarioName"])
            if "ScenarioName" in df_scenarios.columns
            else f"scenario_{idx:03d}"
        )
        printer.information(f"\n===== Running scenario {idx + 1}/{len(df_scenarios)}: {scenario_name} =====")

        # Build param dict from the row, skip the name column and any NaNs
        scenario_params = {
            col: row[col]
            for col in df_scenarios.columns
            if col != "ScenarioName" and pd.notna(row[col])
        }

        scenario_output_dir = output_root / scenario_name

        try:
            main(
                case_study_directory=args.caseStudyDirectory,
                model_type=args.modelType,
                scenario_params=scenario_params,
                output_dir=scenario_output_dir,
            )
        except Exception as e:
            printer.error(f"Scenario '{scenario_name}' failed: {e}")
            # Continue with the next scenario instead of aborting the whole batch
            continue

    printer.success(f"\nAll scenarios finished. Results in '{output_root}'")