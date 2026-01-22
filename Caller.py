import argparse
import datetime
import os
import subprocess
import time

from InOutModule.printer import Printer

printer = Printer.getInstance()

parser = argparse.ArgumentParser(description='Calls the exact lines from the given file, can be called multiple times.')

parser.add_argument('jobs', type=str, help='Path to the text-file containing the commands to be called.')
parser.add_argument("--spawn", type=int, help='Number of jobs to spawn (if this is specified, it will call itself multiple times)', nargs='?', default=0)
args = parser.parse_args()
printer.information(f"Using jobs from '{args.jobs}'")

if args.spawn >= 1:
    printer.information(f"Spawning {args.spawn} parallel jobs")
    for i in range(args.spawn):
        subprocess.Popen([
            "cmd", "/c", "start", f"Caller {i}: {args.jobs}", "cmd", "/k",
            f"set POST_ACTIVATE_COMMAND=python Caller.py {args.jobs} && call Conda-Activation-Scripts/activate_environment_windows.bat"
        ])
    printer.information(f"Spawned {args.spawn} parallel jobs, exiting... ")
    exit(0)

while True:
    with open(args.jobs, 'r') as f:
        lines = f.readlines()

    found_one = False
    for i, line in enumerate(lines):
        started_job_flag = f"{args.jobs}.started{i}"
        finished_job_flag = f"{args.jobs}.finished{i}"
        error_job_flag = f"{args.jobs}.error{i}"
        if not os.path.exists(started_job_flag) and not os.path.exists(finished_job_flag) and not os.path.exists(error_job_flag):
            start_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(started_job_flag, 'w') as f:
                f.write(f"Command: {line.strip()}\n")
                f.write(f"Started at: {start_datetime}")
            found_one = True
            try:
                printer.information(f"Executing job {i} from '{args.jobs}': {line.strip()}")
                os.system(f"title Job {i} from '{args.jobs}': {line.strip()}")

                start_time = time.time()
                os.system(line.strip())
                end_time = time.time()
                with open(finished_job_flag, 'w') as f:
                    f.write(f"Command: {line.strip()}\n")
                    f.write(f"Started at:  {start_datetime}\n")
                    f.write(f"Finished at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n")
                    f.write(f"Execution time: {end_time - start_time:.2f} seconds (= {(end_time - start_time) / 60 / 60:.2f} hours)\n")

                printer.information(f"Finished job {i} from '{args.jobs}' after {end_time - start_time:.2f} seconds (= {(end_time - start_time) / 60 / 60:.2f} hours).")
            except Exception as e:
                printer.error(f"Error while executing job {i}: {e}")
                with open(error_job_flag, 'w') as f:
                    f.write(f"Command: {line.strip()}\n")
                    f.write(f"Error while executing job {i} from '{args.jobs}': {e}\n")
                    f.write(f"Started at:  {start_datetime}\n")
                    f.write(f"Occurred at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
            break

    if not found_one:
        printer.information(f"No more jobs to execute in '{args.jobs}', exiting.")
        break
