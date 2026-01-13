import argparse
import os

from InOutModule.printer import Printer

printer = Printer.getInstance()

parser = argparse.ArgumentParser(description='Calls the exact lines from the given file, can be called multiple times.')

parser.add_argument('jobs', type=str, help='Path to the text-file containing the commands to be called.')
args = parser.parse_args()

while True:
    printer.information(f"Reading jobs from '{args.jobs}'")
    with open(args.jobs, 'r') as f:
        lines = f.readlines()

    found_one = False
    for i, line in enumerate(lines):
        started_job_flag = f"{args.jobs}.started{i}"
        finished_job_flag = f"{args.jobs}.finished{i}"
        error_job_flag = f"{args.jobs}.error{i}"
        if not os.path.exists(started_job_flag) and not os.path.exists(finished_job_flag) and not os.path.exists(error_job_flag):
            with open(started_job_flag, 'w') as f:
                f.write(line.strip())
            found_one = True
            try:
                printer.information(f"Executing job {i}: {line.strip()}")
                os.system(line.strip())
                with open(finished_job_flag, 'r') as f:
                    f.write(line.strip())
                printer.information(f"Finished job {i}")
            except Exception as e:
                printer.error(f"Error while executing job {i}: {e}")
                with open(error_job_flag, 'w') as f:
                    f.write(f"Executed command: {line.strip()}\n")
                    f.write(f"Error while executing job {i}: {e}")
            break

    if not found_one:
        printer.information("No more jobs to execute, exiting.")
        break
