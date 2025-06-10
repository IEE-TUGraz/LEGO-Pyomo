import logging
import time

import pyomo.environ as pyo
from pyomo.core import NameLabeler
from pyomo.util.infeasible import log_infeasible_constraints

from InOutModule.CaseStudy import CaseStudy
from LEGO.LEGO import LEGO
from tools.mpsCompare import compare_mps
from tools.printer import Printer

########################################################################################################################
# Setup
########################################################################################################################

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)
printer = Printer.getInstance()

scenario_folder = "data/example/"

# Select which parts are executed
execute_gams = True
execute_pyomo = True
solve_pyomo = True # Note: GAMS always solves if it's executed in the current setup as otherwise it won't create an MPS file
comparison_mps = True  # Compare MPS files?
check_vars = False
check_constraints = True
print_additional_information = False

constraints_to_skip_from1 = []
constraints_to_keep_from1 = []
coefficients_to_skip_from1 = ['vGenQ','vLineQ']

constraints_to_skip_from2 = []
constraints_to_keep_from2 = []
coefficients_to_skip_from2 = []
constraints_to_enforce_from2 = []

########################################################################################################################
# Re-run with GAMS
########################################################################################################################

if execute_gams:
    gams_console_log_path = "LEGO-GAMS/gams_console.log"
    gams_path = "C:/GAMS/49/gams.exe"
    lego_path = "LEGO.gms"
    max_runtime_in_seconds = 60

    import subprocess
    import psutil

    with open(gams_console_log_path, "w") as GAMSConsoleLogFile:

        # Create subprocess to execute LEGO model
        # Executing with argument string instead of list since GAMS has problems with double-quotes
        printer.information(f"Starting LEGO-GAMS with scenario folder \"{scenario_folder}\"")
        start_time = time.time()
        lego_process = subprocess.Popen(f"cd LEGO-GAMS && {gams_path} {lego_path} --scenarioFolder=\"../{scenario_folder}\"",
                                        stdout=GAMSConsoleLogFile, stderr=subprocess.STDOUT, shell=True)
        try:
            return_value = lego_process.wait(max_runtime_in_seconds)
            stop_time = time.time()

        except subprocess.TimeoutExpired:  # If it exceeds max_runtime_in_seconds, kill it incl. all child processes
            child_processes = psutil.Process(lego_process.pid).children(recursive=True)
            for child in child_processes:
                child.kill()
            printer.error(f"Runtime exceeded {max_runtime_in_seconds} seconds, killing (all) LEGO process(es)")
            gone, still_alive = psutil.wait_procs(child_processes, timeout=5)
            printer.information(f"Status child processes:\n{gone}\n{still_alive}")
            lego_process.kill()

            exit(-1)

    if return_value != 0:
        printer.error(f"Return value of process is {return_value} - please check log files")
        exit(-1)
    else:
        timing = stop_time - start_time
        printer.information(f"Executing GAMS took {timing:.2f} seconds")

        with open("LEGO-Gams/gams_console.log", "r") as file:
            for line in file:
                if "Objective:" in line:
                    objective_value_gams = float(line.split()[-1])
                    printer.information(f"Objective value: {objective_value_gams}")
                    break

########################################################################################################################
# Data input from case study
########################################################################################################################

if execute_pyomo:
    cs = CaseStudy(scenario_folder, do_not_merge_single_node_buses=True)
    cs.dPower_Network['pTecRepr'] = 'DC-OPF'

    lego = LEGO(cs)

    #####################################################################################################################
    # Evaluation
    #####################################################################################################################

    model, timing = lego.build_model()
    printer.information(f"Building LEGO model took {timing:.2f} seconds")
    model.write("model.mps", io_options={'labeler': NameLabeler()})

    if solve_pyomo:  # Solve LEGO model?
        results, timing = lego.solve_model()
        match results.solver.termination_condition:
            case pyo.TerminationCondition.optimal:
                printer.information(f"Optimal solution found after {timing:.2f} seconds")
                if "objective_value_gams" in locals():  # If GAMS has been executed and solved, compare objective values
                    digits = max(len(f"{pyo.value(model.objective):.4f}"), len(f"{objective_value_gams:.4f}"))
                    printer.information(f"Objective value Pyomo: {pyo.value(model.objective):>{digits}.4f}")
                    printer.information(f"Objective value GAMS : {objective_value_gams:>{digits}.4f}")
                    printer.information(f"Objective difference : {pyo.value(model.objective) - objective_value_gams:>{digits}.4f} | {100 * (pyo.value(model.objective) - objective_value_gams) / objective_value_gams:.2f}%")
                else:
                    printer.information(f"Objective value Pyomo: {pyo.value(model.objective):.4f}")

            case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                print(f"ERROR: Model is {results.solver.termination_condition}, logging infeasible constraints:")
                log_infeasible_constraints(model)
            case _:
                print("Solver terminated with condition:", results.solver.termination_condition)

if comparison_mps:
    compare_mps("model.mps", True, "LEGO-GAMS/LEGO-GAMS.mps", False, check_vars=check_vars, check_constraints=check_constraints, print_additional_information=print_additional_information,
                constraints_to_skip_from1=constraints_to_skip_from1, constraints_to_keep_from1=constraints_to_keep_from1, coefficients_to_skip_from1=coefficients_to_skip_from1,
                constraints_to_skip_from2=constraints_to_skip_from2, constraints_to_keep_from2=constraints_to_keep_from2, coefficients_to_skip_from2=coefficients_to_skip_from2, constraints_to_enforce_from2=constraints_to_enforce_from2)

print("Done")
