import logging

from pyomo.core import NameLabeler

from LEGO.CaseStudy import CaseStudy
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

########################################################################################################################
# Re-run with GAMS
########################################################################################################################

if True:
    gams_console_log_path = "LEGO-GAMS/gams_console.log"
    gams_path = "C:/GAMS/48/gams.exe"
    lego_path = "LEGO.gms"
    max_runtime_in_seconds = 60

    import subprocess
    import psutil

    with open(gams_console_log_path, "w") as GAMSConsoleLogFile:

        # Create subprocess to execute LEGO model
        # Executing with argument string instead of list since GAMS has problems with double-quotes
        lego_process = subprocess.Popen(f"cd LEGO-GAMS && {gams_path} {lego_path} --scenarioFolder=\"../{scenario_folder}\"",
                                        stdout=GAMSConsoleLogFile, stderr=subprocess.STDOUT, shell=True)
        try:
            return_value = lego_process.wait(max_runtime_in_seconds)

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

########################################################################################################################
# Data input from case study
########################################################################################################################

if True:
    cs = CaseStudy(scenario_folder, do_not_merge_single_node_buses=True)
    cs.dPower_Network['Technical Representation'] = 'DC-OPF'

    lego = LEGO(cs)

    #####################################################################################################################
    # Evaluation
    #####################################################################################################################

    model, timing = lego.build_model()
    printer.information(f"Building model took {timing:.2f} seconds")
    model.write("model.mps", io_options={'labeler': NameLabeler()})

constraints_to_skip_from1 = ["eStIntraRes", "eExclusiveChargeDischarge"]
constraints_to_skip_from2 = []
coefficients_to_skip_from1 = ["name"]
coefficients_to_skip_from2 = ["name",
                              "v2ndResDW", "vGenP1"]

compare_mps("model.mps", "LEGO-GAMS/LEGO-GAMS.mps", check_vars=False, print_additional_information=False,
            constraints_to_skip_from1=constraints_to_skip_from1, constraints_to_skip_from2=constraints_to_skip_from2,
            coefficients_to_skip_from1=coefficients_to_skip_from1, coefficients_to_skip_from2=coefficients_to_skip_from2)

print("Done")
