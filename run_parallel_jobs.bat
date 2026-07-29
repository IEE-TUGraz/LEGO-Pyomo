@echo off
REM ============================================================
REM  Reads the scenario Excel, writes a params.json per scenario,
REM  and runs all scenarios in parallel in the conda env
REM  LEGO-Pyomo_env (capped at WORKERS, never more than #scenarios).
REM
REM  Edit the paths below once, then just double-click.
REM  Run from the folder containing run_scenarios.py and LEGO.py.
REM ============================================================

REM --- Python interpreter INSIDE the conda env LEGO-Pyomo_env ---
REM Find it once with:  conda activate LEGO-Pyomo_env  &&  where python
set "ENV_PYTHON=C:\Users\Simon Malacek\anaconda3\envs\LEGO-Pyomo_env\python.exe"


set "CASE_STUDY_DIR=C:\Users\Simon Malacek\Code\LEGO-Pyomo\data\fullyear-resilience"
set "SCENARIO_FILE=C:\Users\Simon Malacek\Nextcloud\A_PhD-IEE\2026-04_ResearchStay_SelfSufficiency\data\benders_test\Scenario_Input.xlsx"
set "MODEL_TYPE=BENDERS"
set "WORKERS=12"

REM --- MPI ranks per scenario (decomposition methods only, e.g. BENDERS) ---
REM Each scenario is launched as "mpiexec -n %MPI_RANKS% python LEGO.py ..." so that
REM mpi-sppy spreads the (expensive) sub-problem build+solve across MPI_RANKS processes.
REM
REM   * MPI_RANKS=1  -> no MPI; behaves exactly as before (plain "python LEGO.py").
REM   * MPI_RANKS=N  -> N ranks per scenario. Useful N is capped by the number of
REM                     STOCHASTIC scenarios in the case study (rows in dGlobal_Scenarios,
REM                     currently 2) -- extra ranks just sit idle, so don't set it higher.
REM   * The launcher divides WORKERS by MPI_RANKS so the total process count stays
REM     ~= WORKERS and the CPUs are not oversubscribed (e.g. WORKERS=12, MPI_RANKS=2
REM     -> 6 scenarios run concurrently, each using 2 ranks).
REM   * Requires Microsoft MPI (mpiexec on PATH). Each rank opens its own Gurobi
REM     session -- fine for an academic named-user license on one machine.
REM Validate once that the objective matches the MPI_RANKS=1 run before trusting it.
set "MPI_RANKS=2"

REM Use the env python both for the launcher AND (via --python) for every scenario.
"%ENV_PYTHON%" run_scenarios.py "%CASE_STUDY_DIR%" "%SCENARIO_FILE%" %MODEL_TYPE% --workers %WORKERS% --mpi-ranks %MPI_RANKS% --windows --python "%ENV_PYTHON%"

pause