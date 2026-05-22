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
set "ENV_PYTHON=C:\Users\Server\anaconda3\envs\LEGO-Pyomo_env\python.exe"

 
set "CASE_STUDY_DIR=R:\01_Modelle\Simon\Resilience\data\10year"
set "SCENARIO_FILE=R:\01_Modelle\Simon\Resilience\scenarios\10y_test\Scenario_Input.xlsx"
set "MODEL_TYPE=EXTENSIVE_FORM"
set "WORKERS=12"
 
REM Use the env python both for the launcher AND (via --python) for every scenario.
"%ENV_PYTHON%" run_scenarios.py "%CASE_STUDY_DIR%" "%SCENARIO_FILE%" %MODEL_TYPE% --workers %WORKERS% --windows --python "%ENV_PYTHON%"
 
pause