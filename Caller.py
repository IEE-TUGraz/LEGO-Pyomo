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
        finished_job_flag = f"{args.jobs}.finished{i}"
        if not os.path.exists(f"{finished_job_flag}"):
            open(finished_job_flag, 'w').close()
            found_one = True
            try:
                printer.information(f"Executing job {i}: {line.strip()}")
                os.system(line.strip())
                printer.information(f"Finished job {i}")
            except Exception as e:
                printer.error(f"Error while executing job {i}: {e}")
            break
    if not found_one:
        printer.information("No more jobs to execute, exiting.")
        break
