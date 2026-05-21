import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import glob
import logging
import math
import os
import shutil
import sqlite3
import time
import typing

import pandas as pd

import gurobipy
import pyomo.environ as pyo
from pyomo.opt import SolutionStatus, SolverResults, SolverStatus
from pyomo.util.infeasible import log_infeasible_constraints
from rich_argparse import RichHelpFormatter

from InOutModule import SQLiteWriter, Utilities
from InOutModule.CaseStudy import CaseStudy
from InOutModule.ExcelWriter import ExcelWriter
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO
from LEGO.LEGOUtilities import add_UnitCommitmentSlack_And_FixVariables, markov_summand, markov_sum

########################################################################################################################
# Setup
########################################################################################################################

printer = Printer.getInstance()
printer.set_width(300)

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)


def write_results(lego, file_prefix: str, no_sqlite: bool, **run_parameters):
    if not no_sqlite:
        sqlite_timer = time.time()
        sqlite_file = f"{file_prefix}.sqlite"
        printer.information(f"Writing model to SQLite database: {sqlite_file}")
        SQLiteWriter.model_to_sqlite(lego.model, sqlite_file)
        SQLiteWriter.add_solver_statistics_to_sqlite(sqlite_file, lego)
        if run_parameters:
            SQLiteWriter.add_run_parameters_to_sqlite(sqlite_file, **run_parameters)
        printer.information(f"Writing model to SQLite database took {time.time() - sqlite_timer:.2f} seconds")


def _add_push_markov_constraints(lego: LEGO, thermalGeneratorRelaxed: dict):
    """Add ePushMarkov variables and constraints to a LEGO model (Markov-Strict only).

    These constraints force vStartup and vShutdown to be either 0 or the maximum they
    can be due to MinUp/DownTime, ensuring correct push behavior across representative
    period boundaries.
    """
    model = lego.model
    transition_matrix = lego.cs.rpTransitionMatrixRelativeFrom

    # Variables
    model.vU0 = pyo.Var(model.rp, model.k, model.thermalGenerators, domain=pyo.Binary, doc="Binary variable to indicate that vStartup is 0")
    model.vUX = pyo.Var(model.rp, model.k, model.thermalGenerators, domain=pyo.Binary, doc="Binary variable to indicate that vStartup is X (the maximum it can be due to MinDownTime)")
    model.vD0 = pyo.Var(model.rp, model.k, model.thermalGenerators, domain=pyo.Binary, doc="Binary variable to indicate that vShutdown is 0")
    model.vDY = pyo.Var(model.rp, model.k, model.thermalGenerators, domain=pyo.Binary, doc="Binary variable to indicate that vShutdown is Y (the maximum it can be due to MinUpTime)")

    model.pushMarkovCounter = pyo.Set(initialize=range(1, 11 + 1))
    model.ePushMarkov = pyo.Constraint(model.rp, model.k, model.thermalGenerators, model.pushMarkovCounter,
                                       doc="Constraints to force vStartup and vShutdown to be either 0 or the maximum it can be due to MinUp/DownTime")

    for t in model.thermalGenerators:
        if model.pMinDownTime[t] == 1 and model.pMinUpTime[t] == 1:
            continue
        is_relaxed = thermalGeneratorRelaxed.get(t, False)
        for k in model.k:
            if model.k.ord(k) > max(model.pMinDownTime[t], model.pMinUpTime[t]):
                break
            for rp in model.rp:
                if model.k.ord(k) == 1:
                    prev_commit = markov_summand(model.rp, rp, False, model.k.prevw(k), model.vCommit, transition_matrix, t)
                else:
                    prev_commit = model.vCommit[rp, model.k.prev(k), t]

                X = 1 - prev_commit - markov_sum(model.rp, rp, model.k, model.k.ord(k) - model.pMinDownTime[t] + 1, model.k.ord(k), model.vShutdown, transition_matrix, t)
                model.ePushMarkov[rp, k, t, 1] = (model.vStartup[rp, k, t] <= 1 - model.vU0[rp, k, t])
                model.ePushMarkov[rp, k, t, 2] = (model.vStartup[rp, k, t] <= X + (1 - model.vUX[rp, k, t]))
                model.ePushMarkov[rp, k, t, 3] = (model.vStartup[rp, k, t] >= X - (1 - model.vUX[rp, k, t]))
                model.ePushMarkov[rp, k, t, 4] = (model.vU0[rp, k, t] <= model.vDY[rp, k, t] + (1 - prev_commit))
                model.ePushMarkov[rp, k, t, 5] = (model.vUX[rp, k, t] <= model.vD0[rp, k, t])

                Y = prev_commit - markov_sum(model.rp, rp, model.k, model.k.ord(k) - model.pMinUpTime[t] + 1, model.k.ord(k), model.vStartup, transition_matrix, t)
                model.ePushMarkov[rp, k, t, 6] = (model.vShutdown[rp, k, t] <= 1 - model.vD0[rp, k, t])
                model.ePushMarkov[rp, k, t, 7] = (model.vShutdown[rp, k, t] <= Y + (1 - model.vDY[rp, k, t]))
                model.ePushMarkov[rp, k, t, 8] = (model.vShutdown[rp, k, t] >= Y - (1 - model.vDY[rp, k, t]))
                model.ePushMarkov[rp, k, t, 9] = (model.vD0[rp, k, t] <= model.vUX[rp, k, t] + prev_commit)
                model.ePushMarkov[rp, k, t, 10] = (model.vDY[rp, k, t] <= model.vU0[rp, k, t])
                model.ePushMarkov[rp, k, t, 11] = (1 <= model.vU0[rp, k, t] / 2 + model.vUX[rp, k, t] + model.vD0[rp, k, t] / 2 + model.vDY[rp, k, t])

                # Deactivate constraints for relaxed generators
                if is_relaxed:
                    for i in model.pushMarkovCounter:
                        model.ePushMarkov[rp, k, t, i].deactivate()


def _read_sqlite_run_info(sqlite_file: str) -> dict | None:
    """Read run_parameters and solver_statistics from a SQLite file."""
    try:
        cnx = sqlite3.connect(sqlite_file)
        cursor = cnx.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        result = {}
        if 'run_parameters' in tables:
            df = pd.read_sql("SELECT * FROM run_parameters", cnx)
            if not df.empty:
                result['params'] = df.iloc[0].to_dict()
        if 'solver_statistics' in tables:
            df = pd.read_sql("SELECT * FROM solver_statistics", cnx)
            if not df.empty:
                result['stats'] = df.iloc[0].to_dict()
        cnx.close()
        return result if result else None
    except Exception:
        return None


_SIBLING_COMPARE_KEYS = [
    'case_study_directory', 'limit_k', 'clusters', 'shift', 'stretch_demand',
    'scale_vres', 'thermal_invest_only', 'merge_generators',
    'relax_count', 'no_investment', 'rmip', 'no_crossover', 'force_barrier',
    'mip_gap', 'network', 'commit_consumption', 'startup_consumption', 'edge_handling',
    'shift_tm',
]


def _params_equal(a, b) -> bool:
    """Compare two run parameter values, tolerating int/float storage differences (e.g. 7 vs 7.0)."""
    sa, sb = str(a), str(b)
    if sa == sb:
        return True
    try:
        return float(sa) == float(sb)
    except (ValueError, TypeError):
        return False


def _find_sibling_runs(file_prefix: str, current_edge_params: dict) -> list[dict]:
    """Find existing SQLite files matching all run parameters except work_limit."""
    dir_path = os.path.dirname(os.path.abspath(file_prefix)) or '.'
    exact_file = os.path.abspath(f"{file_prefix}.sqlite")
    candidates = [
        f for f in glob.glob(os.path.join(dir_path, 'MK-*.sqlite'))
        if not f.endswith('-regret.sqlite')
           and not f.endswith('-invest-regret.sqlite')
           and os.path.abspath(f) != exact_file
    ]
    siblings = []
    for candidate in candidates:
        info = _read_sqlite_run_info(candidate)
        if info is None or 'params' not in info:
            continue
        params = info['params']
        match = all(
            _params_equal(current_edge_params.get(key), params.get(key, 'None'))
            for key in _SIBLING_COMPARE_KEYS
        )
        if not match:
            continue
        stats = info.get('stats', {})
        raw_wl = params.get('work_limit')
        prev_work_limit = None if str(raw_wl) in ('None', 'nan', '') else float(raw_wl)
        raw_wu = stats.get('work_units')
        prev_work_units = None if raw_wu is None or str(raw_wu) in ('None', 'nan', '') else float(raw_wu)
        siblings.append({
            'file': candidate,
            'work_limit': prev_work_limit,
            'termination_condition': stats.get('termination_condition'),
            'work_units': prev_work_units,
        })
    return siblings


def _should_skip_smart(current_work_limit: float | None, siblings: list[dict]) -> tuple[bool, str]:
    """
    Decide whether to skip a solve based on sibling run results (--no-overwrite smart check).

    Skip only when:
      - any sibling solved to optimality, OR
      - current has a finite work-limit AND a sibling with a strictly higher finite work-limit
        did not solve to optimality (higher budget already failed; lower won't do better).

    Run in all other cases, including:
      - no-limit sibling not optimal (likely aborted externally)
      - lower-limit sibling not optimal regardless of whether it reached its limit
    """
    if not siblings:
        return False, "no sibling runs found"

    for s in siblings:
        if s['termination_condition'] == 'optimal':
            return True, f"previous run (work_limit={s['work_limit']}) solved to optimality"

    if current_work_limit is not None:
        higher_finite = [
            s for s in siblings
            if s['work_limit'] is not None and s['work_limit'] > current_work_limit
        ]
        if higher_finite:
            best = max(higher_finite, key=lambda s: s['work_limit'])
            return True, (
                f"previous run with higher work_limit={best['work_limit']} "
                f"(used {best['work_units']:.1f} WU) did not solve to optimality"
                if best['work_units'] is not None
                else f"previous run with higher work_limit={best['work_limit']} did not solve to optimality"
            )

    limit_strs = ', '.join(
        f"work_limit={'none' if s['work_limit'] is None else s['work_limit']}" for s in siblings
    )
    return False, f"previous run(s) ({limit_strs}) not optimal — re-running"


def execute_case_studies(case_study_path: str, no_sqlite: bool = False,
                         calculate_regret: bool = False, relax_percentage: float = 0, skip_truth: bool = False,
                         enable_strict_markov: bool = False, invest_regret: bool = False,
                         no_investment: bool = False, rmip: bool = False, no_crossover: bool = False,
                         force_barrier: bool = False, mip_gap: float | None = None,
                         work_limit: float | None = None,
                         filter_zone: str | None = None, limitK: str | None = None, clusters: int = 1,
                         shift: int = 0, stretch_demand: float = 1.0, scale_vres: float = 1.0,
                         thermal_invest_only: bool = False, merge_generators: bool = False,
                         no_overwrite: bool = False, network: str | None = None,
                         commit_consumption: float = 1.0, startup_consumption: float = 1.0,
                         shift_tm: int | None = None,
                         cs: CaseStudy | None = None,
                         tee: bool = True) -> typing.Tuple[typing.List[str], typing.List[str], typing.Dict[str, LEGO]]:
    ########################################################################################################################
    # Data input from case study
    ########################################################################################################################

    if cs is None:
        # Load case study from Excels
        start_time = time.time()
        cs = CaseStudy(case_study_path, clip_method="none", clip_value=0)
        printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")
    else:
        printer.information(f"Using provided CaseStudy object (skipping Excel load)")

    # Build identifier parts for sqlite filenames (similar to TR/ID naming convention)
    identifier_parts = [f"data{case_study_path.rstrip('/').replace('/', '_').replace(' ', '')}"]
    if rmip:
        printer.information("Setting up case study as rMIP (relaxing all integer variables)")
        cs.dGlobal_Parameters["pEnableRMIP"] = True
        identifier_parts.append("rMIP")

    if no_crossover:
        printer.information("Disabling crossover for all solves")
        cs.dGlobal_Parameters["pDisableCrossover"] = True
        identifier_parts.append("noCrossover")

    if force_barrier:
        printer.information("Forcing barrier method for all solves")
        cs.dGlobal_Parameters["pForceBarrier"] = True
        identifier_parts.append("forceBarrier")

    if mip_gap is not None:
        printer.information(f"Setting MIP gap to {mip_gap}")
        cs.dGlobal_Parameters["pMIPGap"] = mip_gap
        identifier_parts.append(f"mipGap{mip_gap:g}")

    if work_limit is not None:
        printer.information(f"Setting work limit to {work_limit}")
        cs.dGlobal_Parameters["pWorkLimit"] = work_limit
        identifier_parts.append(f"workLimit{work_limit:g}")

    if network is not None:
        printer.information(f"Setting all lines to network representation '{network}'")
        cs.dPower_Network["pTecRepr"] = network
        identifier_parts.append(f"network{network}")

    if commit_consumption != 1.0:
        printer.information(f"Scaling CommitConsumption by {commit_consumption}")
        cs.dPower_ThermalGen['pInterVarCostEUR'] *= commit_consumption
        identifier_parts.append(f"commitConsumption{commit_consumption:g}")

    if startup_consumption != 1.0:
        printer.information(f"Scaling StartupConsumption by {startup_consumption}")
        cs.dPower_ThermalGen['pStartupCostEUR'] *= startup_consumption
        identifier_parts.append(f"startupConsumption{startup_consumption:g}")

    if scale_vres != 1.0:
        printer.information(f"Scaling VRES MaxProd by {scale_vres}")
        cs.dPower_VRES['MaxProd'] *= scale_vres
        identifier_parts.append(f"scaleVRES{scale_vres:g}")

    if thermal_invest_only:
        printer.information("Setting ExisUnits=1 for all non-thermal generators (thermalInvestOnly)")
        cs.dPower_VRES['ExisUnits'] = 1
        cs.dPower_VRES['EnableInvest'] = 0
        cs.dPower_Storage['ExisUnits'] = 1
        cs.dPower_Storage['EnableInvest'] = 0
        identifier_parts.append("thermalInvestOnly")

    if relax_percentage > 0:
        identifier_parts.append(f"relaxed{math.ceil(len(cs.dPower_ThermalGen.index) * relax_percentage)}")

    if shift_tm is not None:
        printer.information(f"Shifting transition matrix by {shift_tm} positions")
        cs.shift_transition_matrix(shift_tm, inplace=True)
        identifier_parts.append(f"shiftTM{shift_tm}")

    identifier = "-".join(identifier_parts)

    Utilities.plot_transition_matrix(cs.rpTransitionMatrixAbsolute, title=f"Not shifted from original" if shift_tm is None else f"Shifted by shift_tm={shift_tm}", output=f"MK-{identifier}.png")

    if any(cs.dPower_ThermalGen["MinUpTime"] > len(cs.dPower_WeightsK.index)) or any(cs.dPower_ThermalGen["MinDownTime"] > len(cs.dPower_WeightsK.index)):
        printer.warning(f"Some thermal generators have MinUpTime or MinDownTime greater than the number of K-values ({len(cs.dPower_WeightsK.index)}) - capping it to that number")
        cs.dPower_ThermalGen["MinUpTime"] = cs.dPower_ThermalGen["MinUpTime"].clip(upper=len(cs.dPower_WeightsK.index))
        cs.dPower_ThermalGen["MinDownTime"] = cs.dPower_ThermalGen["MinDownTime"].clip(upper=len(cs.dPower_WeightsK.index))

    # Create varied case studies
    start_time = time.time()
    printer.information(f"Creating varied case studies")
    cs_notEnforced = cs.copy()
    cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "notEnforced"
    cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "notEnforced"
    cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingIntraDayStorage"] = "notEnforced"

    cs_cyclic = cs_notEnforced.copy()
    cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "cyclic"
    cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "cyclic"
    cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingIntraDayStorage"] = "cyclic"

    cs_markov = cs_notEnforced.copy()
    cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "markov"
    cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "markov"
    cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingIntraDayStorage"] = "markov"

    if enable_strict_markov:
        cs_markov_strict = cs_notEnforced.copy()
        cs_markov_strict.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "markov"
        cs_markov_strict.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "markov"
        cs_markov_strict.dPower_Parameters["pReprPeriodEdgeHandlingIntraDayStorage"] = "markov"
    printer.information(f"Creating varied case studies took {time.time() - start_time:.2f} seconds")

    # Create "truth" case study for comparison
    if skip_truth:
        printer.information(f"Skipping truth case study as requested")
    else:
        start_time = time.time()
        printer.information(f"Creating truth case study (full-hourly)")
        cs_truth = cs.to_full_hourly_model(inplace=False)  # Create a full hourly model (which copies from notEnforced)
        printer.information(f"Creating truth case study (full-hourly) took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    printer.information(f"Building the LEGO models for adjustments")  # Note this is actually faster (1.5-2x) than copying already built models to re-use them
    lego_models = {}
    if not skip_truth:
        lego_models["Truth "] = LEGO(cs_truth)
    lego_models["NoEnf."] = LEGO(cs_notEnforced)
    lego_models["Cyclic"] = LEGO(cs_cyclic)
    lego_models["Markov"] = LEGO(cs_markov)
    if enable_strict_markov:
        lego_models["Markov-Strict"] = LEGO(cs_markov_strict)
    for name, lego in lego_models.items():
        _, build_time = lego.build_model()
        printer.information(f"Building model for case study '{name}' took {build_time:.2f} seconds")
    printer.information(f"Building the LEGO models took {time.time() - start_time:.2f} seconds overall")

    thermalGeneratorRelaxed = {}
    if relax_percentage == 0:
        printer.information(f"Not relaxing any unit commitment variables, all thermal generators stay binary")
        count_relaxed = 0
    else:
        thermalGenerators = cs.dPower_ThermalGen.copy()
        start_time = time.time()
        printer.information(f"Relaxing {relax_percentage * 100:.1f}% of unit commitment variables for thermal generators")
        count_relaxed = math.ceil(len(thermalGenerators.index) * relax_percentage)

        printer.information(f"Relaxing {count_relaxed} thermal generator(s), keeping {len(thermalGenerators.index) - count_relaxed} binary")
        thermalGenerators["MinUpDownTime-Sum"] = thermalGenerators["MinUpTime"] + thermalGenerators["MinDownTime"]
        thermalGenerators.sort_values(by=["MinUpDownTime-Sum"], inplace=True)

        thermalGeneratorRelaxed = {}
        for i, t in enumerate(thermalGenerators.index):
            thermalGeneratorRelaxed[t] = True if i < count_relaxed else False

        printer.information(f"Relaxing {count_relaxed} thermal generators: {[g for g in thermalGenerators.index if thermalGeneratorRelaxed[g]]}")
        for case_name, lego in lego_models.items():
            # Relax unit commitment variables for selected generators
            for g in lego.model.thermalGenerators:
                if thermalGeneratorRelaxed[g]:
                    for rp in lego.model.rp:
                        for k in lego.model.k:
                            lego.model.vCommit[rp, k, g].domain = pyo.PercentFraction
                            lego.model.vStartup[rp, k, g].domain = pyo.PercentFraction
                            lego.model.vShutdown[rp, k, g].domain = pyo.PercentFraction
        printer.information(f"Relaxing {count_relaxed} thermal generators took {time.time() - start_time:.2f} seconds")

    if enable_strict_markov:
        _add_push_markov_constraints(lego_models["Markov-Strict"], thermalGeneratorRelaxed)

    if no_investment:
        for name, lego in lego_models.items():
            for g in lego.model.g:
                lego.model.vGenInvest[g].value = 1
                lego.model.vGenInvest[g].fixed = True
        printer.information(f"Fixed vGenInvest to 1 for all generators in all models (--no-investment)")

    run_params = dict(
        case_study_directory=case_study_path,
        filter_zone=filter_zone,
        limit_k=limitK,
        clusters=clusters if clusters > 1 else None,
        shift=shift if shift != 0 else None,
        stretch_demand=stretch_demand if stretch_demand != 1.0 else None,
        scale_vres=scale_vres if scale_vres != 1.0 else None,
        thermal_invest_only=thermal_invest_only if thermal_invest_only else None,
        merge_generators=merge_generators if merge_generators else None,
        relax_count=count_relaxed if count_relaxed > 0 else None,
        no_investment=no_investment if no_investment else None,
        rmip=rmip if rmip else None,
        no_crossover=no_crossover if no_crossover else None,
        force_barrier=force_barrier if force_barrier else None,
        mip_gap=mip_gap,
        work_limit=work_limit,
        network=network,
        commit_consumption=commit_consumption if commit_consumption != 1.0 else None,
        startup_consumption=startup_consumption if startup_consumption != 1.0 else None,
        shift_tm=shift_tm,
    )
    sqlite_files, sqlite_labels, lego_models = execute_case_study(lego_models, identifier, no_sqlite, calculate_regret, skip_truth, invest_regret, run_params, no_overwrite, tee=tee)

    return sqlite_files, sqlite_labels, lego_models


def execute_case_study(lego_models: typing.Dict[str, LEGO], case_name: str, no_sqlite: bool, calculate_regret: bool, skip_truth: bool, invest_regret: bool = False, run_params: dict = None, no_overwrite: bool = False, tee: bool = True) -> typing.Tuple[typing.List[str], typing.List[str], typing.Dict[str, LEGO]]:
    ########################################################################################################################
    # Evaluation
    ########################################################################################################################
    sqlite_files = []
    sqlite_labels = []

    if not skip_truth:
        truth_lego = lego_models["Truth "]

    for edgeHandlingType, lego in lego_models.items():
        printer.information(f"\n\n{'=' * 60}\n{edgeHandlingType}\n{'=' * 60}")
        model = lego.model

        file_prefix = f"MK-{case_name}-{edgeHandlingType.strip().replace('.', '').replace(' ', '')}"

        # Part 1: Solve the main model (or skip if no-overwrite and result exists)
        case_skipped = False
        if no_overwrite:
            if os.path.exists(f"{file_prefix}.sqlite"):
                printer.information(f"  File '{file_prefix}.sqlite' already exists, skipping (--no-overwrite)")
                case_skipped = True
            elif run_params is not None:
                edge_handling_normalized = edgeHandlingType.strip().replace('.', '').replace(' ', '')
                current_edge_params = {**run_params, "edge_handling": edge_handling_normalized}
                siblings = _find_sibling_runs(file_prefix, current_edge_params)
                if siblings:
                    skip, reason = _should_skip_smart(run_params.get('work_limit'), siblings)
                    if skip:
                        printer.information(f"  Skipping (--no-overwrite smart check): {reason}")
                        case_skipped = True
                    else:
                        printer.information(f"  Running despite existing sibling run(s): {reason}")
        if not case_skipped:
            # Solve model
            optimizer = pyo.SolverFactory('gurobi_persistent')
            optimizer.set_instance(model)
            if getattr(model, 'pDisableCrossover', False):
                printer.information("Deactivating crossover")
                optimizer.options['Crossover'] = 0
            if getattr(model, 'pForceBarrier', False):
                printer.information("Forcing barrier method")
                optimizer.options['Method'] = 2
                optimizer.options['NodeMethod'] = 2
            mip_gap_value = getattr(model, 'pMIPGap', None)
            if mip_gap_value is not None:
                printer.information(f"Setting MIP gap to {mip_gap_value}")
                optimizer.options['MIPGap'] = mip_gap_value
            work_limit_value = getattr(model, 'pWorkLimit', None)
            if work_limit_value is not None:
                printer.information(f"Setting work limit to {work_limit_value}")
                optimizer.options['WorkLimit'] = work_limit_value
            start_time = time.time()
            solve_exception = None
            try:
                result = optimizer.solve(tee=tee, load_solutions=False)
            except Exception as e:
                solve_exception = e
                result = None
                printer.warning(f"Solver raised exception: {e}")
            has_solution = optimizer._solver_model.SolCount > 0
            if has_solution:
                if result is not None:
                    if result.solver.status == SolverStatus.error:
                        result.solver.status = SolverStatus.warning
                        if optimizer._solver_model.Status == gurobipy.GRB.WORK_LIMIT:
                            result.solver.termination_condition = "WorkLimit reached"
                        if len(result.solution) > 0:
                            result.solution[0].status = SolutionStatus.feasible
                    model.solutions.load_from(result)
                else:
                    # solve() threw (e.g. OOM) but solutions were found — load directly from Gurobi
                    try:
                        optimizer.load_vars()
                        printer.information("Loaded partial solution after solver exception")
                    except Exception as load_e:
                        printer.warning(f"Could not load solution after solver exception: {load_e}")
                        has_solution = False
            timing_solving = time.time() - start_time
            if result is not None:
                lego.results = result
            elif solve_exception is not None:
                exc_result = SolverResults()
                exc_result.solver.status = SolverStatus.error
                if isinstance(solve_exception, gurobipy.GurobiError) and solve_exception.errno == gurobipy.GRB.Error.OUT_OF_MEMORY:
                    exc_result.solver.termination_condition = "Out of Memory"
                lego.results = exc_result
            lego.work_units = optimizer._solver_model.Work
            printer.information(f"Solving model took {timing_solving:.2f} seconds ({lego.work_units:.2f} work units)")
            try:
                lego.mip_gap = optimizer._solver_model.MIPGap if optimizer._solver_model.IsMIP else None
                if not optimizer._solver_model.IsMIP:
                    printer.information("Model is an LP — no MIP gap stored in .sqlite")
            except Exception as e:
                printer.warning(f"Could not extract MIP gap from Gurobi: {e}")
                lego.mip_gap = None

            edge_params = {**(run_params or {}), "edge_handling": edgeHandlingType.strip().replace('.', '').replace(' ', '')}
            if has_solution:
                write_results(lego, file_prefix, no_sqlite, **edge_params)

            if result is not None:
                match result.solver.termination_condition:
                    case pyo.TerminationCondition.optimal:
                        printer.success("Optimal solution found")
                    case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                        printer.error(f"Model is {result.solver.termination_condition}, logging infeasible constraints:")
                        log_infeasible_constraints(model)
                    case _:
                        printer.warning("Solver terminated with condition:", result.solver.termination_condition)
            elif solve_exception is not None:
                printer.error(f"Solver terminated with exception: {solve_exception}")

        if not no_sqlite:
            sqlite_files.append(f"{file_prefix}.sqlite")
            sqlite_labels.append(edgeHandlingType)

        if calculate_regret and edgeHandlingType != "Truth " and not skip_truth:
            if no_overwrite and os.path.exists(f"{file_prefix}-regret.sqlite"):
                printer.information(f"  File '{file_prefix}-regret.sqlite' already exists, skipping regret (--no-overwrite)")
            elif case_skipped and not os.path.exists(f"{file_prefix}.sqlite"):
                printer.information(f"  Skipping regret: '{file_prefix}.sqlite' does not exist (smart-skipped)")
            else:
                try:
                    regret_lego = truth_lego.copy()

                    # Load vCommit values from sqlite if case was skipped
                    if case_skipped:
                        printer.information(f"Loading vCommit from '{file_prefix}.sqlite'")
                        cnx = sqlite3.connect(f"{file_prefix}.sqlite")
                        df_commit = pd.read_sql("SELECT * FROM vCommit", cnx)
                        cnx.close()
                        for _, row in df_commit.iterrows():
                            model.vCommit[row.iloc[0], row.iloc[1], row.iloc[2]].value = row['values']
                            model.vCommit[row.iloc[0], row.iloc[1], row.iloc[2]].stale = False

                    add_UnitCommitmentSlack_And_FixVariables(regret_lego, model, lego.cs.dPower_Hindex, lego.cs.dPower_ThermalGen, lego.cs.dPower_Parameters["pENSCost"])

                    # Re-solve the model
                    printer.information("Re-solving model with fixed variables for regret calculation")
                    regret_result, regret_timing_solving, regret_objective_value = regret_lego.solve_model(already_solved_ok=True)
                    printer.information(f"Solving regret model took {regret_timing_solving:.2f} seconds")

                    regret_params = {**(run_params or {}), "edge_handling": edgeHandlingType.strip().replace('.', '').replace(' ', ''), "run_type": "regret"}
                    write_results(regret_lego, f"{file_prefix}-regret", no_sqlite, **regret_params)

                    match regret_result.solver.termination_condition:
                        case pyo.TerminationCondition.optimal:
                            printer.success("Optimal solution found")
                        case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                            printer.error(f"Model is {regret_result.solver.termination_condition}, logging infeasible constraints:")
                            log_infeasible_constraints(regret_lego.model)
                        case _:
                            printer.warning("Solver terminated with condition:", regret_result.solver.termination_condition)
                except Exception as e:
                    printer.error(f"Regret calculation failed for '{edgeHandlingType}': {e}")

        # Part 2: Invest-regret (independent of whether main case was solved or skipped)
        if invest_regret and edgeHandlingType != "Truth ":
            if no_overwrite and os.path.exists(f"{file_prefix}-invest-regret.sqlite"):
                printer.information(f"  File '{file_prefix}-invest-regret.sqlite' already exists, skipping invest-regret (--no-overwrite)")
            elif case_skipped and not os.path.exists(f"{file_prefix}.sqlite"):
                printer.information(f"  Skipping invest-regret: '{file_prefix}.sqlite' does not exist (smart-skipped)")
            else:
                try:
                    printer.information(f"Calculating invest-regret for '{edgeHandlingType}': fixing vGenInvest in truth model")
                    invest_regret_lego = truth_lego.copy()

                    # Load vGenInvest values: from solved model if available, otherwise from existing sqlite
                    if case_skipped:
                        printer.information(f"Loading vGenInvest from '{file_prefix}.sqlite'")
                        cnx = sqlite3.connect(f"{file_prefix}.sqlite")
                        df_inv = pd.read_sql("SELECT * FROM vGenInvest", cnx)
                        cnx.close()
                        gen_invest_values = dict(zip(df_inv.iloc[:, 0], df_inv['values']))
                        for g in invest_regret_lego.model.g:
                            invest_regret_lego.model.vGenInvest[g].value = gen_invest_values.get(g, 1)
                            invest_regret_lego.model.vGenInvest[g].fixed = True
                    else:
                        for g in invest_regret_lego.model.g:
                            invest_regret_lego.model.vGenInvest[g].value = model.vGenInvest[g].value
                            invest_regret_lego.model.vGenInvest[g].fixed = True

                    # Re-solve the truth model with fixed investments
                    printer.information("Re-solving truth model with fixed vGenInvest for invest-regret calculation")
                    invest_regret_result, invest_regret_timing, invest_regret_objective = invest_regret_lego.solve_model(already_solved_ok=True)
                    printer.information(f"Solving invest-regret model took {invest_regret_timing:.2f} seconds")

                    invest_regret_params = {**(run_params or {}), "edge_handling": edgeHandlingType.strip().replace('.', '').replace(' ', ''), "run_type": "invest-regret"}
                    write_results(invest_regret_lego, f"{file_prefix}-invest-regret", no_sqlite, **invest_regret_params)

                    match invest_regret_result.solver.termination_condition:
                        case pyo.TerminationCondition.optimal:
                            printer.success(f"Optimal invest-regret solution: {invest_regret_objective:.4f}")
                        case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                            printer.error(f"Invest-regret model is {invest_regret_result.solver.termination_condition}, logging infeasible constraints:")
                            log_infeasible_constraints(invest_regret_lego.model)
                        case _:
                            printer.warning("Invest-regret solver terminated with condition:", invest_regret_result.solver.termination_condition)
                except Exception as e:
                    printer.error(f"Invest-regret calculation failed for '{edgeHandlingType}': {e}")

    return sqlite_files, sqlite_labels, lego_models


def copy_files_non_recursive(src_folder: str, dst_folder: str):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for item in os.listdir(src_folder):
        s = os.path.join(src_folder, item)
        d = os.path.join(dst_folder, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)


def main(caseStudyFolder: str, debug: bool = False, no_sqlite: bool = False, calculate_regret: bool = False,
         relax_percentage: float = 0.0, skip_truth: bool = False,
         clusters: int = 1, cluster_stepsize: int = 1, cluster_steps: int = 0,
         filter_zone: str | None = None, limitK: str | None = None,
         shift: int = 0, stretch_demand: float = 1, scale_vres: float = 1.0,
         thermal_invest_only: bool = False, merge_generators: bool = False,
         reuse_inputfiles: bool = False, enable_strict_markov: bool = False, invest_regret: bool = False,
         no_investment: bool = False, no_overwrite: bool = False, rmip: bool = False, no_crossover: bool = False,
         force_barrier: bool = False, mip_gap: float | None = None, work_limit: float | None = None,
         network: str | None = None, commit_consumption: float = 1.0, startup_consumption: float = 1.0,
         shift_tm: int | None = None):
    ew = ExcelWriter()

    if no_crossover != force_barrier:
        raise ValueError("Either both or none of no_crossover and force_barrier must be true")

    for folder in caseStudyFolder.split(","):
        try:
            if not folder.endswith("/"):
                folder += "/"

            if filter_zone is not None:
                printer.information(f"Filtering case study to zone '{filter_zone}'")
                new_folder = folder + f"filterZone{filter_zone}/"
                if reuse_inputfiles and os.path.exists(new_folder):
                    printer.information(f"Reusing already zone-filtered case study in '{new_folder}'")
                    folder = new_folder
                else:
                    copy_files_non_recursive(folder, new_folder)
                    folder = new_folder
                    printer.information(f"Copied original case study to '{folder}'")

                    cs = CaseStudy(folder, do_not_scale_units=True)
                    printer.information(f"Case study loaded, now filtering to zone '{filter_zone}'")
                    cs = cs.filter_zone(filter_zone)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    ew.write_caseStudy(cs, folder)
                    printer.information(f"Saved zone-filtered case study to '{folder}'")

            if limitK is not None:
                printer.information(f"Limiting K values to '{limitK}'")
                start_k, end_k = limitK.split("-")
                new_folder = folder + f"limitK{limitK}/"
                if reuse_inputfiles and os.path.exists(new_folder):
                    printer.information(f"Reusing already limited case study in '{new_folder}'")
                    folder = new_folder
                else:
                    copy_files_non_recursive(folder, new_folder)  # Copy original data to new folder
                    folder = new_folder
                    printer.information(f"Copied original case study to '{folder}'")

                    cs = CaseStudy(folder, do_not_scale_units=True)
                    printer.information(f"Case study loaded, now limiting timesteps")
                    cs = cs.filter_timesteps(start_k, end_k)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    printer.information(f"Limited, now writing to '{folder}'")
                    ew.write_caseStudy(cs, folder)
                    printer.information(f"Saved limited case study to '{folder}'")

            if shift != 0:
                printer.information(f"Shifting case study by {shift} hours")
                new_folder = folder + f"shift{shift}/"
                if reuse_inputfiles and os.path.exists(new_folder):
                    printer.information(f"Reusing already shifted case study in '{new_folder}'")
                    folder = new_folder
                else:
                    copy_files_non_recursive(folder, new_folder)  # Copy original data
                    folder = new_folder
                    printer.information(f"Copied original case study to '{folder}'")

                    cs = CaseStudy(folder, do_not_scale_units=True)
                    printer.information(f"Case study loaded, now shifting")
                    cs = cs.shift_ks(shift)
                    printer.information(f"Shifted by {shift}")
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    ew.write_caseStudy(cs, folder)
                    printer.information(f"Wrote shifted case study to '{folder}'")

            if stretch_demand != 1.0:
                printer.information(f"Stretching demand by factor {stretch_demand}")
                new_folder = folder + f"stretchDemand{stretch_demand:g}/"
                if reuse_inputfiles and os.path.exists(new_folder):
                    printer.information(f"Reusing already demand-stretched case study in '{new_folder}'")
                    folder = new_folder
                else:
                    copy_files_non_recursive(folder, new_folder)  # Copy original data
                    folder = new_folder
                    printer.information(f"Copied original case study to '{folder}'")

                    cs = CaseStudy(folder, do_not_scale_units=True)
                    printer.information(f"Case study loaded, now stretching demand for each bus around center")
                    center = cs.dPower_Demand.groupby("i")["value"].mean()
                    scaler = 1 + (stretch_demand - 1) / 2
                    for rp, k, i in cs.dPower_Demand.index:
                        cs.dPower_Demand.at[(rp, k, i), "value"] = center[i] + (cs.dPower_Demand.at[(rp, k, i), "value"] - center[i]) * scaler

                    # Fail if any of the values is negative
                    if (cs.dPower_Demand["value"] < 0).any():
                        to_clip = cs.dPower_Demand[cs.dPower_Demand['value'] < 0]
                        printer.warning(f"Stretching demand by factor {stretch_demand} leads to negative demand values, clipping {to_clip.shape[0]} values for {len(to_clip.index.get_level_values('i').unique())} nodes to 0")
                        printer.warning(f"Clipping for nodes: {", ".join([f"{i} for {to_clip[to_clip.index.get_level_values("i") == i].shape[0]} values" for i in to_clip.index.get_level_values('i').unique().tolist()])}")
                        cs.dPower_Demand["value"] = cs.dPower_Demand["value"].clip(lower=0)

                    printer.information(f"Stretched demand by factor {stretch_demand}")
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    ew.write_caseStudy(cs, folder)
                    printer.information(f"Wrote demand-stretched case study to '{folder}'")

            if merge_generators:
                printer.information(f"Merging generators of same technology at same bus")
                new_folder = folder + f"mergeGenerators/"
                if reuse_inputfiles and os.path.exists(new_folder):
                    printer.information(f"Reusing already generator-merged case study in '{new_folder}'")
                    folder = new_folder
                else:
                    copy_files_non_recursive(folder, new_folder)
                    folder = new_folder
                    printer.information(f"Copied original case study to '{folder}'")

                    cs = CaseStudy(folder, do_not_scale_units=True)
                    printer.information(f"Case study loaded, now merging generators")
                    cs = cs.merge_generators()
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    ew.write_caseStudy(cs, folder)
                    printer.information(f"Wrote generator-merged case study to '{folder}'")

            clusters = list(range(clusters, clusters + cluster_steps * cluster_stepsize + 1, cluster_stepsize))
            for cluster in clusters:
                cluster_folder = folder
                if cluster > 1:
                    cluster_folder = cluster_folder + f"{cluster} clusters/"
                    if reuse_inputfiles and os.path.exists(cluster_folder):
                        printer.information(f"Reusing already clustered case study in '{cluster_folder}'")
                    else:
                        copy_files_non_recursive(folder, cluster_folder)  # Copy original data to new folder

                        cs = CaseStudy(cluster_folder, do_not_scale_units=True)
                        cs_clustered = Utilities.apply_kmedoids_aggregation(cs, cluster, verbose=True)
                        ew.write_caseStudy(cs_clustered, cluster_folder)

                printer.information(f"Loading case study from '{cluster_folder}'")

                sqlite_files, case_labels, _ = execute_case_studies(cluster_folder, no_sqlite, calculate_regret, relax_percentage, skip_truth, enable_strict_markov, invest_regret,
                                                                    no_investment, rmip, no_crossover, force_barrier, mip_gap, work_limit, filter_zone=filter_zone, limitK=limitK,
                                                                    clusters=cluster, shift=shift, stretch_demand=stretch_demand, scale_vres=scale_vres, thermal_invest_only=thermal_invest_only,
                                                                    merge_generators=merge_generators, no_overwrite=no_overwrite, network=network,
                                                                    commit_consumption=commit_consumption, startup_consumption=startup_consumption,
                                                                    shift_tm=shift_tm)
        except Exception as e:
            printer.error(f"Exception while executing case study '{locals().get('cluster_folder', folder)}': {e}")  # locals-hack to always get correct folder-name
            if debug:
                raise e
            else:
                printer.console.print_exception()
                printer.error(f"Continuing with next case study")

    printer.success("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare edge-handling for given case-study", formatter_class=RichHelpFormatter)
    parser.add_argument("caseStudyFolder", type=str, help="Path to folder containing data for LEGO model. Can be a comma-separated list of multiple folders (executed after each other)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode where exceptions are passed on")
    parser.add_argument("--no-sqlite", action="store_true", help="Do not save results to SQLite database")
    parser.add_argument("--calculate-regret", action="store_true", help="Calculate regret by re-solving the truth model with fixed unit commitment from the other models (can take a while)")
    parser.add_argument("--relax-percentage", type=float, default=0, help="Percentage of thermal-generators to be relaxed (default: 0 = no relaxation, all binary)")
    parser.add_argument("--skip-truth", action="store_true", help="Skip solving the truth model")
    parser.add_argument("--clusters", type=int, default=1, help="Number of clusters (default: 1, i.e., no clustering)")
    parser.add_argument("--cluster-stepsize", type=int, default=1, help="If in-/decreasing number of clusters should be used (default: 1, leave cluster-steps default to not use in-/decreasing number of clusters)")
    parser.add_argument("--cluster-steps", type=int, default=0, help="Number of steps for in-/decreasing number of clusters (default: 0, i.e., leave clusters as given)")
    parser.add_argument("--filter-zone", type=str, default=None, help="Filter the case study to only include buses in the given zone (exact match of the 'z' column in Power_BusInfo), e.g. 'R1'")
    parser.add_argument("--limitK", type=str, help="Limit the ks, format: 'k0025-k0048'", nargs="?", default=None)
    parser.add_argument("--shift", type=int, default=0, help="Shift the time series by N hours (for testing purposes), e.g., 15 to shift by 15 hours")
    parser.add_argument("--stretch-demand", type=float, default=1.0, help="Stretch the demand by a factor (for testing purposes), e.g., 1.1 to increase max of demand by 5% and decrease min by 5%")
    parser.add_argument("--scale-vres", type=float, default=1.0, help="Scale the MaxProd of all VRES generators (PV, Wind, RoR) by this factor (default: 1.0, no change)")
    parser.add_argument("--thermal-invest-only", action="store_true", help="Set ExisUnits=1 for all non-thermal generators (VRES, Storage) so only thermal generators are investable")
    parser.add_argument("--merge-generators", action="store_true", help="Merge generators of the same technology at the same bus into one representative generator before clustering and solving")
    parser.add_argument("--reuse-inputfiles", action="store_true", help="Reuse input files (e.g., after shortening) instead of copying them to a new folder")
    parser.add_argument("--enable-strict-markov", action="store_true", help="Also execute the strict Markov variant (with push constraints active)")
    parser.add_argument("--invest-regret", action="store_true", help="Calculate invest-regret: fix vGenInvest from each edge-handling model into the truth model and compare objectives")
    parser.add_argument("--no-investment", action="store_true", help="Fix vGenInvest to 1 for all generators (skip investment decisions)")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip cases where the output .sqlite file already exists")
    parser.add_argument("--rmip", action="store_true", help="Relax all integer variables (rMIP) before solving")
    parser.add_argument("--no-crossover", action="store_true", help="Disable Gurobi crossover for all solves (faster LP solving, but solution may not be a vertex)")
    parser.add_argument("--force-barrier", action="store_true", help="Force Gurobi to use barrier method")
    parser.add_argument("--mip-gap", type=float, default=None, help="Set the MIP gap tolerance for the solver (e.g., 0.01 for 1%%; default: solver default)")
    parser.add_argument("--work-limit", type=float, default=None, help="Set the Gurobi WorkLimit (in work units) to stop after a given amount of work regardless of solution quality (default: no limit)")
    parser.add_argument("--network", type=str, default=None, choices=["DC-OPF", "TP", "SN"], help="Override network representation for all lines uniformly: DC-OPF, TP, or SN (default: no change, use values from data)")
    parser.add_argument("--commit-consumption", type=float, default=1.0, help="Multiplier for the CommitConsumption column of Power_ThermalGen (default: 1.0, no change)")
    parser.add_argument("--startup-consumption", type=float, default=1.0, help="Multiplier for the StartupConsumption column of Power_ThermalGen (default: 1.0, no change)")
    parser.add_argument("--shift-tm", type=int, default=None, help="Shift the transition matrix by <N> positions to the right")
    args = parser.parse_args()

    kwargs = vars(args)

    main(**kwargs)
