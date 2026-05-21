import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rich_argparse import RichHelpFormatter

from InOutModule.printer import Printer

printer = Printer.getInstance()


def ensure_dir(path):
    """Create directory if it doesn't exist, return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value):
    """Convert pandas/numpy scalars into plain Python types for JSON."""
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def build_jobs(scenario_file, case_study_directory, model_type, lego_script, python_exe):
    """Read the scenario Excel, write per-scenario params JSON files, and
    return (output_root, list_of_jobs) where each job is a dict describing
    one scenario invocation."""
    scenario_file = Path(scenario_file)
    if not scenario_file.is_file():
        raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

    # Output root = "results" folder next to the scenario file
    output_root = ensure_dir(scenario_file.parent / "results")
    printer.information(f"Results will be written to '{output_root}'")

    printer.information(f"Loading scenarios from '{scenario_file}'")
    df_scenarios = pd.read_excel(scenario_file, skiprows=[1])
    printer.information(f"Found {len(df_scenarios)} scenario(s)")

    jobs = []
    for idx, row in df_scenarios.iterrows():
        scenario_name = (
            str(row["ScenarioName"])
            if "ScenarioName" in df_scenarios.columns
            else f"scenario_{idx:03d}"
        )

        # Build param dict from the row, skip the name column and any NaNs
        scenario_params = {
            col: _json_safe(row[col])
            for col in df_scenarios.columns
            if col != "ScenarioName" and pd.notna(row[col])
        }

        scenario_output_dir = ensure_dir(output_root / scenario_name)
        params_path = scenario_output_dir / "params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(scenario_params, f, indent=2)

        # Argument list (NOT a shell string) -> no quoting problems with spaces
        cmd = [
            python_exe, str(lego_script),
            str(case_study_directory), str(model_type),
            "--params", str(params_path),
            "--output-dir", str(scenario_output_dir),
            "--scenario-name", scenario_name,
        ]
        jobs.append({"name": scenario_name, "cmd": cmd, "log": scenario_output_dir / "run.log"})
        printer.information(f"Prepared scenario {idx + 1}/{len(df_scenarios)}: {scenario_name}")

    return output_root, jobs


def run_job(job):
    """Run a single scenario as a subprocess. Returns (name, returncode, seconds)."""
    start = time.time()
    with open(job["log"], "w", encoding="utf-8") as logf:
        proc = subprocess.run(job["cmd"], stdout=logf, stderr=subprocess.STDOUT)
    return job["name"], proc.returncode, time.time() - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reads scenarios from an Excel sheet, writes a params.json per "
                    "scenario, and runs them in parallel (one subprocess each, capped "
                    "at --workers).",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("caseStudyDirectory", type=str,
                        help="Path to folder containing data for LEGO model")
    parser.add_argument("scenarioFile", type=str,
                        help="Path to the Excel file containing the scenarios")
    parser.add_argument("modelType", default="DETERMINISTIC", nargs="?",
                        help="ModelType of the model (default: DETERMINISTIC)")
    parser.add_argument("--lego-script", default="LEGO.py",
                        help="Path to the LEGO.py worker script (default: LEGO.py)")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to use for each job (default: current)")
    parser.add_argument("--workers", type=int, default=12,
                        help="Maximum number of scenarios to run in parallel (default: 12). "
                             "Automatically capped at the number of scenarios.")
    args = parser.parse_args()

    output_root, jobs = build_jobs(
        scenario_file=args.scenarioFile,
        case_study_directory=args.caseStudyDirectory,
        model_type=args.modelType,
        lego_script=args.lego_script,
        python_exe=args.python,
    )

    if not jobs:
        printer.warning("No scenarios found, nothing to run.")
        sys.exit(0)

    # Never spawn more workers than there are scenarios
    n_workers = max(1, min(args.workers, len(jobs)))
    printer.information(
        f"Running {len(jobs)} scenario(s) with up to {n_workers} in parallel"
    )

    overall_start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_name = {executor.submit(run_job, job): job["name"] for job in jobs}
        for future in as_completed(future_to_name):
            name, rc, seconds = future.result()
            results.append((name, rc, seconds))
            if rc == 0:
                printer.success(f"Scenario '{name}' finished in {seconds:.1f}s")
            else:
                printer.error(
                    f"Scenario '{name}' FAILED (exit {rc}) after {seconds:.1f}s "
                    f"- see log in results/{name}/run.log"
                )

    total = time.time() - overall_start
    n_ok = sum(1 for _, rc, _ in results if rc == 0)
    n_fail = len(results) - n_ok
    printer.information(f"\n===== Summary ({total:.1f}s total) =====")
    printer.success(f"{n_ok} succeeded")
    if n_fail:
        printer.error(f"{n_fail} failed")
    printer.success(f"All results in '{output_root}'")

    # Non-zero exit if anything failed, so the .bat / shell can detect it
    sys.exit(1 if n_fail else 0)