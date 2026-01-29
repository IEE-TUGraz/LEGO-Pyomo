import argparse

from rich_argparse import RichHelpFormatter

from CaseStudy import CaseStudy
from InOutModule import SQLiteWriter
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO

printer = Printer.getInstance()


def main(case_study_directory: str, limit_k: str):
    SCENARIO_MAIN = "Scenario_2016"
    SCENARIO_SPIN = "Scenario_2015"
    SPINUP = 168
    HOURS = 8760

    cs = CaseStudy(case_study_directory)
    cs.filter_scenario(SCENARIO_MAIN, inplace=True)

    cs_old_formulation = cs.copy()
    cs_old_formulation.add_input_for_old_hydro_formulation(SCENARIO_MAIN, SCENARIO_SPIN, SPINUP, HOURS)

    if limit_k is not None:
        printer.information(f"Limiting K values to '{limit_k}'")
        start, end = limit_k.split("-")
        cs.filter_timesteps(start, end, inplace=True)
        cs_old_formulation.filter_timesteps(start, end, inplace=True)

    cs.dPower_Parameters["pEnableAdvancedHydro"] = True
    cs_old_formulation.dPower_Parameters["pEnableAdvancedHydro"] = False

    printer.information(f"Building LEGO model for case study with old formulation")
    lego_old = LEGO(cs_old_formulation)
    model_old, timing_old = lego_old.build_model()
    printer.information(f"Building LEGO model for case study with old formulation took {timing_old:.2f} seconds")
    printer.information(f"Solving LEGO model for case study with old formulation")
    results_old, timing_old, objective_value_old = lego_old.solve_model()
    printer.information(f"Solving LEGO model for case study with old formulation took {timing_old:.2f} seconds")
    SQLiteWriter.model_to_sqlite(lego_old.model, f"formulation_old.sqlite")
    printer.information(f"Saved old formulation model to formulation_old.sqlite")

    printer.information(f"Building LEGO model for case study with new formulation")
    lego_new = LEGO(cs)
    model, timing = lego_new.build_model()
    printer.information(f"Building LEGO model for case study with new formulation took {timing:.2f} seconds")
    printer.information(f"Solving LEGO model for case study with new formulation")
    results, timing, objective_value = lego_new.solve_model()
    printer.information(f"Solving LEGO model for case study with new formulation took {timing:.2f} seconds")
    SQLiteWriter.model_to_sqlite(lego_new.model, f"formulation_new.sqlite")
    printer.information(f"Saved new formulation model to formulation_new.sqlite")


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Tests different formulations for Hydro Assets", formatter_class=RichHelpFormatter)
        parser.add_argument("caseStudyDirectory", type=str, help="Path to folder containing data for LEGO model")
        parser.add_argument("--limitK", type=str, help="Limit the ks, format: 'k0025-k0048'", nargs="?", default=None)

        args = parser.parse_args()

        main(args.caseStudyDirectory, args.limitK)
