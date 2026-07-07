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

import pyomo.environ as pyo
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
    'scale_vres', 'scale_invest_cost', 'thermal_invest_only', 'merge_generators',
    'relax_count', 'no_investment', 'rmip', 'no_crossover', 'force_barrier',
    'mip_gap', 'network', 'commit_consumption', 'startup_consumption', 'edge_handling',
    'shift_tm', 'perturb_tm', 'run_type',
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
    # '-regret.sqlite' also covers '-invest-regret' and '-operational-regret'; '-operational'
    # files are kept as candidates and separated via the run_type compare key.
    candidates = [
        f for f in glob.glob(os.path.join(dir_path, 'MK-*.sqlite'))
        if not f.endswith('-regret.sqlite')
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


# Termination conditions that mean the solver genuinely consumed its budget
# (as opposed to being killed, running out of memory, or erroring out).
_CLEAN_TERMINATION_CONDITIONS = frozenset({'WorkLimit reached', 'optimal'})


def _should_skip_smart(current_work_limit: float | None, siblings: list[dict]) -> tuple[bool, str, str | None]:
    """
    Decide whether to skip a solve based on sibling run results (--no-overwrite smart check).

    Returns (skip, reason, sibling_file):
      - sibling_file is the path to the most relevant sibling SQLite when skip=True:
          Rule 1 (optimal sibling) → the optimal sibling's file.
          Rule 2 (WL comparison)   → the best clean sibling's file (may not be optimal).
        sibling_file is None when skip=False (re-run paths).

    Skip when:
      - any sibling solved to optimality, OR
      - the best "clean" sibling (highest work_limit among those that ran to budget exhaustion)
        has a work_limit >= current work_limit (no improvement possible with the same or lower budget).

    Re-run when:
      - current work_limit is strictly higher than all clean siblings' work_limits (None = unlimited = ∞), OR
      - all non-optimal siblings had non-ok status (killed / OOM / error — budget was not genuinely consumed).

    "Clean" = termination_condition in _CLEAN_TERMINATION_CONDITIONS (solver ran its budget normally).
    "Not ok" = any other termination_condition → re-run regardless of work_limit comparison.
    """
    if not siblings:
        return False, "no sibling runs found", None

    # Rule 1: any sibling solved optimally → always skip; return its file for downstream use
    for s in siblings:
        if s['termination_condition'] == 'optimal':
            return True, f"previous run (work_limit={s['work_limit']}) solved to optimality", s['file']

    # Separate clean siblings (ran budget to completion) from not-ok siblings (killed / error)
    clean_siblings = [s for s in siblings if s['termination_condition'] in _CLEAN_TERMINATION_CONDITIONS]

    if not clean_siblings:
        # All siblings had non-ok status — budget was not genuinely consumed, re-run in any case
        not_ok_tcs = ', '.join(str(s['termination_condition']) for s in siblings)
        return False, f"all previous run(s) had non-ok status ({not_ok_tcs}) — re-running", None

    # Rule 2: compare current work_limit against the best (highest) clean sibling.
    # None (unlimited) is treated as ∞.
    def _wl_sort_key(s):
        return float('inf') if s['work_limit'] is None else s['work_limit']

    best = max(clean_siblings, key=_wl_sort_key)
    best_wl = best['work_limit']

    # Is current strictly higher than best?  unlimited > finite; finite not > unlimited; equal → not higher
    if current_work_limit is None:
        current_higher = best_wl is not None  # unlimited > any finite
    else:
        current_higher = best_wl is not None and current_work_limit > best_wl

    if not current_higher:
        wu_str = f" (used {best['work_units']:.1f} WU)" if best['work_units'] is not None else ""
        wl_str = 'none (unlimited)' if best_wl is None else best_wl
        cur_str = 'none (unlimited)' if current_work_limit is None else current_work_limit
        return True, (
            f"previous run with work_limit={wl_str}{wu_str} did not solve to optimality; "
            f"current work_limit={cur_str} is not strictly higher"
        ), best['file']  # WL-comparison skip: pass the best sibling's file for downstream use (e.g. invest-regret)

    limit_strs = ', '.join(
        f"work_limit={'none' if s['work_limit'] is None else s['work_limit']}" for s in clean_siblings
    )
    return False, f"current work_limit={current_work_limit} is strictly higher than all clean run(s) ({limit_strs}) — re-running", None


def execute_case_studies(case_study_path: str, no_sqlite: bool = False,
                         calculate_regret: bool = False, relax_percentage: float = 0, skip_truth: bool = False,
                         enable_strict_markov: bool = False, invest_regret: bool = False,
                         no_investment: bool = False, rmip: bool = False, no_crossover: bool = False,
                         force_barrier: bool = False, mip_gap: float | None = None,
                         work_limit: float | None = None,
                         node_file_start: float | None = None, node_file_dir: str | None = None,
                         threads: int | None = None,
                         filter_zone: str | None = None, limitK: str | None = None, clusters: int = 1,
                         shift: int = 0, stretch_demand: float = 1.0, scale_vres: float = 1.0,
                         scale_invest_cost: float = 1.0,
                         thermal_invest_only: bool = False, merge_generators: bool = False,
                         no_overwrite: bool = False, operational: bool = False, operational_regret: bool = False, network: str | None = None,
                         commit_consumption: float = 1.0, startup_consumption: float = 1.0,
                         shift_tm: int | None = None,
                         perturb_tm: float | None = None,
                         cs: CaseStudy | None = None,
                         tee: bool = True) -> typing.Tuple[typing.List[str], typing.List[str], typing.Dict[str, LEGO]]:
    ########################################################################################################################
    # Data input from case study
    ########################################################################################################################

    if node_file_dir is not None and node_file_start is None:
        raise ValueError("node_file_dir requires node_file_start to be set (Gurobi NodefileStart must be enabled before NodefileDir is meaningful)")

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

    # Gurobi memory-management knobs: not in identifier or sibling-compare keys
    # (resource management, does not affect the solution).
    if node_file_start is not None:
        printer.information(f"Setting Gurobi NodefileStart to {node_file_start} GB")
        cs.dGlobal_Parameters["pNodeFileStart"] = node_file_start
        node_file_dir_base = node_file_dir if node_file_dir is not None else os.path.join(os.getcwd(), "gurobi-nodes")
        # Always append a per-PID subfolder so parallel spawns never collide on a shared dir,
        # even if the user passed a path without thinking about parallelism.
        node_file_dir = os.path.join(node_file_dir_base, str(os.getpid()))
        printer.information(f"NodeFileDir: '{node_file_dir}' (base '{node_file_dir_base}' + PID subfolder)")
        cs.dGlobal_Parameters["pNodeFileDir"] = node_file_dir

    if threads is not None:
        printer.information(f"Setting Gurobi Threads to {threads}")
        cs.dGlobal_Parameters["pThreads"] = threads

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

    if scale_invest_cost != 1.0:
        printer.information(f"Scaling InvestCostEUR by {scale_invest_cost}")
        cs.dPower_ThermalGen['InvestCostEUR'] *= scale_invest_cost
        cs.dPower_VRES['InvestCostEUR'] *= scale_invest_cost
        cs.dPower_Storage['InvestCostEUR'] *= scale_invest_cost
        identifier_parts.append(f"scaleInvestCost{scale_invest_cost:g}")

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

    if perturb_tm is not None:
        printer.information(f"Perturbing transition matrix with randomness={perturb_tm}")
        cs.perturb_transition_matrix(perturb_tm, inplace=True)
        identifier_parts.append(f"perturbTM{perturb_tm}")

    identifier = "-".join(identifier_parts)

    tm_title_parts = []
    if shift_tm is not None:
        tm_title_parts.append(f"Diagonal shifted to the right by {shift_tm} steps")
    if perturb_tm is not None:
        tm_title_parts.append(f"Perturbed by {perturb_tm}")
    Utilities.plot_transition_matrix(cs.rpTransitionMatrixAbsolute, title=", ".join(tm_title_parts), output=f"MK-{identifier}.png")

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
        scale_invest_cost=scale_invest_cost if scale_invest_cost != 1.0 else None,
        thermal_invest_only=thermal_invest_only if thermal_invest_only else None,
        merge_generators=merge_generators if merge_generators else None,
        relax_count=count_relaxed if count_relaxed > 0 else None,
        no_investment=no_investment if no_investment else None,
        rmip=rmip if rmip else None,
        no_crossover=no_crossover if no_crossover else None,
        force_barrier=force_barrier if force_barrier else None,
        mip_gap=mip_gap,
        work_limit=work_limit,
        node_file_start=node_file_start,
        node_file_dir=node_file_dir,
        threads=threads,
        network=network,
        commit_consumption=commit_consumption if commit_consumption != 1.0 else None,
        startup_consumption=startup_consumption if startup_consumption != 1.0 else None,
        shift_tm=shift_tm,
        perturb_tm=perturb_tm,
    )
    sqlite_files, sqlite_labels, lego_models = execute_case_study(lego_models, identifier, no_sqlite, calculate_regret, skip_truth, invest_regret, run_params, no_overwrite, operational=operational, operational_regret=operational_regret, cs=cs, thermal_generator_relaxed=thermalGeneratorRelaxed, tee=tee)

    return sqlite_files, sqlite_labels, lego_models


def _apply_solver_options(lego: LEGO, run_params: dict | None) -> None:
    """Explicitly re-stamp solver options from run_params onto a copied lego model.

    After truth_lego.copy() (deepcopy of a solved model) the plain model attributes
    (pWorkLimit, pMIPGap, pDisableCrossover, pForceBarrier) should survive, but
    re-setting them here makes the enforcement explicit, visible in logs, and robust
    against any future changes to how LEGO.copy() works.  Mirrors the logging done
    in the main solve loop inside execute_case_study().
    """
    if run_params is None:
        return
    model = lego.model
    wl = run_params.get('work_limit')
    if wl is not None:
        model.pWorkLimit = wl
        printer.information(f"  Work limit: {wl}")
    mg = run_params.get('mip_gap')
    if mg is not None:
        model.pMIPGap = mg
        printer.information(f"  MIP gap: {mg}")
    if run_params.get('force_barrier'):
        model.pForceBarrier = True
    if run_params.get('no_crossover'):
        model.pDisableCrossover = True
    nfs = run_params.get('node_file_start')
    if nfs is not None:
        model.pNodeFileStart = nfs
        printer.information(f"  NodefileStart: {nfs} GB")
    nfd = run_params.get('node_file_dir')
    if nfd is not None:
        model.pNodeFileDir = nfd
        printer.information(f"  NodefileDir: {nfd}")
    th = run_params.get('threads')
    if th is not None:
        model.pThreads = th
        printer.information(f"  Threads: {th}")


# Threshold above which a (possibly fractional, e.g. rMIP) vGenInvest counts as "invested".
_OPERATIONAL_INVEST_THRESHOLD = 0.5


def _find_optimal_truth_file(case_name: str, run_params: dict | None) -> str | None:
    """Return the path to a Truth .sqlite that solved to optimality (status ok).

    Prefers the exact Truth file for this case, then siblings that differ only in
    work_limit. Only files with termination_condition == 'optimal' qualify (so
    WorkLimit-reached / OOM Truth results are never used as the investment source).
    Returns None if no optimal Truth result exists on disk.
    """
    truth_prefix = f"MK-{case_name}-Truth"
    candidates = []
    exact = f"{truth_prefix}.sqlite"
    if os.path.exists(exact):
        candidates.append(exact)
    truth_edge_params = {**(run_params or {}), "edge_handling": "Truth", "run_type": None}
    candidates.extend(s['file'] for s in _find_sibling_runs(truth_prefix, truth_edge_params))
    for f in candidates:
        info = _read_sqlite_run_info(f)
        tc = info.get('stats', {}).get('termination_condition') if info else None
        if tc == 'optimal':
            return f
    return None


def _get_truth_geninvest(lego_models: typing.Dict[str, LEGO], case_name: str, run_params: dict | None) -> typing.Tuple[dict | None, str | None]:
    """Resolve the Truth investment decision for the operational runs.

    The investment must come from an *optimal* Truth result (a WorkLimit / OOM Truth
    solution is never used). Source priority:
      1. In-memory Truth model, only if it was solved this session to optimality.
      2. The best optimal Truth .sqlite on disk (exact file, then siblings).
    Returns (vGenInvest dict {g: value}, human-readable source) or (None, None) when
    no optimal Truth result is available anywhere.
    """
    truth_lego = lego_models.get("Truth ")
    if truth_lego is not None and getattr(truth_lego, 'has_solution', False):
        tc = truth_lego.results.solver.termination_condition if getattr(truth_lego, 'results', None) is not None else None
        if tc == pyo.TerminationCondition.optimal:
            try:
                invest = {g: pyo.value(truth_lego.model.vGenInvest[g]) for g in truth_lego.model.vGenInvest}
                return invest, "in-memory Truth model"
            except Exception as e:
                printer.warning(f"Could not read vGenInvest from in-memory Truth model: {e}")
        else:
            printer.warning(f"In-memory Truth solve did not reach optimality (termination_condition={tc}) — "
                            f"not using it as the operational investment source; looking for an optimal Truth .sqlite instead")
    truth_file = _find_optimal_truth_file(case_name, run_params)
    if truth_file is not None:
        try:
            cnx = sqlite3.connect(truth_file)
            df_inv = pd.read_sql("SELECT * FROM vGenInvest", cnx)
            cnx.close()
            invest = dict(zip(df_inv.iloc[:, 0], df_inv['values']))
            return invest, f"Truth sqlite '{truth_file}'"
        except Exception as e:
            printer.warning(f"Could not read vGenInvest from '{truth_file}': {e}")
    return None, None


def _build_full_hourly_truth_lego(cs: CaseStudy, thermal_generator_relaxed: dict | None) -> LEGO:
    """Build (but do not solve) the full-hourly Truth model on demand for Truth-operational.

    Used under --skip-truth, where the Truth model was never built but its operational
    re-solve is still wanted (e.g. to compare Truth's runtime against the other strategies).
    Mirrors the regular Truth build (cs.to_full_hourly_model) and applies the same unit
    commitment relaxation as the other models when --relax-percentage was set.
    """
    truth_cs = cs.to_full_hourly_model(inplace=False)
    truth_lego = LEGO(truth_cs)
    truth_lego.build_model()
    if thermal_generator_relaxed:
        for g in truth_lego.model.thermalGenerators:
            if thermal_generator_relaxed.get(g):
                for rp in truth_lego.model.rp:
                    for k in truth_lego.model.k:
                        truth_lego.model.vCommit[rp, k, g].domain = pyo.PercentFraction
                        truth_lego.model.vStartup[rp, k, g].domain = pyo.PercentFraction
                        truth_lego.model.vShutdown[rp, k, g].domain = pyo.PercentFraction
    return truth_lego


def execute_operational_runs(lego_models: typing.Dict[str, LEGO], case_name: str, no_sqlite: bool,
                             run_params: dict | None, no_overwrite: bool, tee: bool,
                             sqlite_files: typing.List[str], sqlite_labels: typing.List[str],
                             cs: CaseStudy | None = None, thermal_generator_relaxed: dict | None = None,
                             operational_regret: bool = False) -> None:
    """Solve an operational variant of each built edge-handling model (--operational).

    Each operational run fixes vGenInvest to the *Truth* investment decision (1 where
    Truth invested, 0 otherwise) and re-solves, isolating the operational problem of the
    edge handling under a common (Truth) investment. Results are written to
    'MK-{case_name}-{edge}-operational.sqlite'. If no Truth investment is
    available at all (no in-memory Truth and no optimal Truth .sqlite), all operational
    runs are skipped with an error.

    Under --skip-truth the Truth model is absent from lego_models; when `cs` is provided,
    the full-hourly Truth model is (re)built on demand so Truth-operational is still solved
    (its runtime is then comparable to the other operational runs).
    """
    printer.information(f"\n\n{'#' * 60}\nOperational runs (--operational): vGenInvest fixed to Truth's investment\n{'#' * 60}")

    truth_invest, source = _get_truth_geninvest(lego_models, case_name, run_params)
    if truth_invest is None:
        printer.error("--operational: no Truth investment available (neither solved in memory nor an "
                      "optimal Truth .sqlite found) — skipping all operational runs")
        return
    n_invested = sum(1 for v in truth_invest.values() if v > _OPERATIONAL_INVEST_THRESHOLD)
    printer.information(f"Using Truth investment from {source}: {n_invested} of {len(truth_invest)} generators invested")

    # Edge-handlings to operationalize. A None source model means "build the Truth model on
    # demand" — the --skip-truth case, where the Truth model was never built but its
    # operational re-solve is still wanted.
    edge_items = list(lego_models.items())
    if "Truth " not in lego_models:
        if cs is not None:
            edge_items = [("Truth ", None)] + edge_items
        else:
            printer.warning("--operational with no in-memory Truth model and no CaseStudy to rebuild it — "
                            "skipping Truth-operational (other operational runs still proceed)")

    # Full-hourly Truth base for --operational-regret re-solves, resolved on first actual need and
    # reused across edge handlings (mirrors the regret base in execute_case_study).
    truth_base = None

    for edgeHandlingType, lego in edge_items:
        normalized = edgeHandlingType.strip().replace('.', '').replace(' ', '')
        op_prefix = f"MK-{case_name}-{normalized}-operational"
        op_label = f"{normalized}-op"
        op_lego = None  # the in-memory operational solve, if it ran this iteration (else loaded from sqlite below)
        op_sibling_skip_file = None  # set to the skipping sibling's .sqlite when the operational run is smart-skipped

        # --no-overwrite: skip if the exact operational file is already optimal, or a smart
        # sibling check (matching run_params + edge_handling + run_type, differing only in
        # work_limit) says no improvement is possible.
        case_skipped = False
        if no_overwrite:
            if os.path.exists(f"{op_prefix}.sqlite"):
                info = _read_sqlite_run_info(f"{op_prefix}.sqlite")
                tc = info.get('stats', {}).get('termination_condition') if info else None
                if tc == 'optimal':
                    printer.information(f"  File '{op_prefix}.sqlite' already has optimal solution, skipping (--no-overwrite)")
                    case_skipped = True
                else:
                    printer.information(f"  File '{op_prefix}.sqlite' exists but status is '{tc or 'unknown'}' — re-running")
            elif run_params is not None:
                current_edge_params = {**run_params, "edge_handling": normalized, "run_type": "operational"}
                siblings = _find_sibling_runs(op_prefix, current_edge_params)
                if siblings:
                    skip, reason, sibling_file = _should_skip_smart(run_params.get('work_limit'), siblings)
                    if skip:
                        printer.information(f"  Skipping (--no-overwrite smart check): {reason}")
                        case_skipped = True
                        op_sibling_skip_file = sibling_file  # reuse its vCommit for operational-regret below
                    else:
                        printer.information(f"  Running despite existing sibling run(s): {reason}")

        if not case_skipped:
            printer.information(f"\n{'=' * 60}\n{normalized} (operational)\n{'=' * 60}")
            if lego is None:
                # Build the full-hourly Truth model on demand (lazily — only when actually
                # solving, so a --no-overwrite skip above avoids the expensive rebuild).
                printer.information("Building full-hourly Truth model on demand for Truth-operational (--skip-truth)")
                op_lego = _build_full_hourly_truth_lego(cs, thermal_generator_relaxed)
            else:
                op_lego = lego.copy()
            _apply_solver_options(op_lego, run_params)
            for g in op_lego.model.vGenInvest:
                op_lego.model.vGenInvest[g].value = 1 if truth_invest.get(g, 0) > _OPERATIONAL_INVEST_THRESHOLD else 0
                op_lego.model.vGenInvest[g].fixed = True

            op_result, op_timing, _ = op_lego.solve_model(tee=tee, already_solved_ok=True, raise_on_no_solution=False)
            work_units_str = f"{op_lego.work_units:.2f} work units" if op_lego.work_units is not None else "work units unavailable"
            printer.information(f"Solving operational model took {op_timing:.2f} seconds ({work_units_str})")

            op_params = {**(run_params or {}), "edge_handling": normalized, "run_type": "operational"}
            if op_lego.has_solution:
                write_results(op_lego, op_prefix, no_sqlite, **op_params)

            match op_result.solver.termination_condition:
                case pyo.TerminationCondition.optimal:
                    printer.success("Optimal solution found")
                case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                    printer.error(f"Model is {op_result.solver.termination_condition}, logging infeasible constraints:")
                    log_infeasible_constraints(op_lego.model)
                case _:
                    printer.warning("Solver terminated with condition:", op_result.solver.termination_condition)

        if not no_sqlite:
            sqlite_files.append(f"{op_prefix}.sqlite")
            sqlite_labels.append(op_label)

        # Operational-regret (--operational-regret): re-solve the full-hourly truth model with vGenInvest
        # fixed to the Truth investment (as above) and vCommit fixed to THIS edge handling's operational
        # commitment. Files get a '-operational-regret' suffix and are NOT appended to
        # sqlite_files/sqlite_labels (like the other regret variants).
        if operational_regret and edgeHandlingType != "Truth ":
            oregret_prefix = f"MK-{case_name}-{normalized}-operational-regret"
            run_oregret = True
            if no_overwrite and os.path.exists(f"{oregret_prefix}.sqlite"):
                or_info = _read_sqlite_run_info(f"{oregret_prefix}.sqlite")
                or_tc = or_info.get('stats', {}).get('termination_condition') if or_info else None
                if or_tc == 'optimal':
                    printer.information(f"  File '{oregret_prefix}.sqlite' already has optimal solution, skipping operational-regret (--no-overwrite)")
                    run_oregret = False
                else:
                    printer.information(f"  File '{oregret_prefix}.sqlite' exists but status is '{or_tc or 'unknown'}' — re-running operational-regret")
            if run_oregret:
                try:
                    # Source the operational commitment to pin: from the in-memory operational solve, or
                    # from the exact '{op_prefix}.sqlite' if the operational run was no-overwrite-skipped, or
                    # from the smart-skipping sibling (differs only in work_limit — its vCommit is a feasible
                    # commitment for the same operational problem; soft-fixed below so optimality is not required).
                    op_commit_model = None
                    op_commit_source = None
                    if op_lego is not None:
                        op_commit_model = op_lego.model
                    elif os.path.exists(f"{op_prefix}.sqlite"):
                        op_commit_source = f"{op_prefix}.sqlite"
                    elif op_sibling_skip_file is not None:
                        op_commit_source = op_sibling_skip_file
                    if op_commit_model is None and op_commit_source is not None:
                        printer.information(f"Loading operational vCommit from '{op_commit_source}'")
                        op_commit_lego = lego.copy()
                        cnx = sqlite3.connect(op_commit_source)
                        df_commit = pd.read_sql("SELECT * FROM vCommit", cnx)
                        cnx.close()
                        for _, row in df_commit.iterrows():
                            op_commit_lego.model.vCommit[row.iloc[0], row.iloc[1], row.iloc[2]].value = row['values']
                            op_commit_lego.model.vCommit[row.iloc[0], row.iloc[1], row.iloc[2]].stale = False
                        op_commit_model = op_commit_lego.model

                    # Resolve the full-hourly Truth base on first need, then reuse it across edge
                    # handlings (None = not yet resolved or unavailable).
                    if truth_base is None:
                        truth_base = lego_models.get("Truth ")
                        if truth_base is None and cs is not None:
                            printer.information("Building full-hourly Truth model on demand for operational-regret (--skip-truth)")
                            truth_base = _build_full_hourly_truth_lego(cs, thermal_generator_relaxed)
                    if truth_base is None:
                        printer.warning(f"  Skipping operational-regret for '{normalized}': no Truth model available and no CaseStudy to build one")
                    elif op_commit_model is None:
                        printer.information(f"  Skipping operational-regret for '{normalized}': no operational vCommit available (no in-memory solve, exact file, or skipping sibling)")
                    else:
                        printer.information(f"\n{'=' * 60}\n{normalized} (operational-regret)\n{'=' * 60}")
                        oregret_lego = truth_base.copy()
                        _apply_solver_options(oregret_lego, run_params)
                        # vGenInvest fixed to the Truth investment (same decision the operational runs use) ...
                        for g in oregret_lego.model.vGenInvest:
                            oregret_lego.model.vGenInvest[g].value = 1 if truth_invest.get(g, 0) > _OPERATIONAL_INVEST_THRESHOLD else 0
                            oregret_lego.model.vGenInvest[g].fixed = True
                        # ... and vCommit soft-fixed to this edge handling's operational commitment (hindex maps
                        # the reduced (rp, k) onto the truth model's full-hourly k).
                        add_UnitCommitmentSlack_And_FixVariables(oregret_lego, op_commit_model, lego.cs.dPower_Hindex, lego.cs.dPower_ThermalGen, lego.cs.dPower_Parameters["pENSCost"])

                        printer.information("Re-solving truth model with fixed vGenInvest (Truth) and vCommit (operational) for operational-regret")
                        oregret_result, oregret_timing, _ = oregret_lego.solve_model(already_solved_ok=True, raise_on_no_solution=False)
                        printer.information(f"Solving operational-regret model took {oregret_timing:.2f} seconds")

                        oregret_params = {**(run_params or {}), "edge_handling": normalized, "run_type": "operational-regret"}
                        if oregret_lego.has_solution:
                            write_results(oregret_lego, oregret_prefix, no_sqlite, **oregret_params)

                        match oregret_result.solver.termination_condition:
                            case pyo.TerminationCondition.optimal:
                                printer.success("Optimal solution found")
                            case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                                printer.error(f"Model is {oregret_result.solver.termination_condition}, logging infeasible constraints:")
                                log_infeasible_constraints(oregret_lego.model)
                            case _:
                                printer.warning("Solver terminated with condition:", oregret_result.solver.termination_condition)
                except Exception as e:
                    printer.error(f"Operational-regret calculation failed for '{edgeHandlingType}': {e}")


def execute_case_study(lego_models: typing.Dict[str, LEGO], case_name: str, no_sqlite: bool, calculate_regret: bool, skip_truth: bool, invest_regret: bool = False, run_params: dict = None, no_overwrite: bool = False, operational: bool = False, operational_regret: bool = False, cs: CaseStudy | None = None, thermal_generator_relaxed: dict | None = None, tee: bool = True) -> typing.Tuple[typing.List[str], typing.List[str], typing.Dict[str, LEGO]]:
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
        sibling_skip_file = None  # set to the best sibling's .sqlite path on a sibling-skip (optimal or WL-comparison)
        if no_overwrite:
            if os.path.exists(f"{file_prefix}.sqlite"):
                info = _read_sqlite_run_info(f"{file_prefix}.sqlite")
                tc = info.get('stats', {}).get('termination_condition') if info else None
                if tc == 'optimal':
                    printer.information(f"  File '{file_prefix}.sqlite' already has optimal solution, skipping (--no-overwrite)")
                    case_skipped = True
                else:
                    printer.information(f"  File '{file_prefix}.sqlite' exists but status is '{tc or 'unknown'}' — re-running")
            elif run_params is not None:
                edge_handling_normalized = edgeHandlingType.strip().replace('.', '').replace(' ', '')
                current_edge_params = {**run_params, "edge_handling": edge_handling_normalized}
                siblings = _find_sibling_runs(file_prefix, current_edge_params)
                if siblings:
                    skip, reason, sibling_file = _should_skip_smart(run_params.get('work_limit'), siblings)
                    if skip:
                        printer.information(f"  Skipping (--no-overwrite smart check): {reason}")
                        case_skipped = True
                        sibling_skip_file = sibling_file  # set on both skip rules (optimal and WL-comparison)
                    else:
                        printer.information(f"  Running despite existing sibling run(s): {reason}")
        if not case_skipped:
            # Solve model. All WorkLimit/OOM/exception recovery, partial-solution loading, and
            # work_units/mip_gap extraction live in LEGO.solve_model() — it never raises by default
            # and exposes lego.has_solution for the write gating below.
            if getattr(model, 'pDisableCrossover', False):
                printer.information("Deactivating crossover")
            if getattr(model, 'pForceBarrier', False):
                printer.information("Forcing barrier method")
            mip_gap_value = getattr(model, 'pMIPGap', None)
            if mip_gap_value is not None:
                printer.information(f"Setting MIP gap to {mip_gap_value}")
            work_limit_value = getattr(model, 'pWorkLimit', None)
            if work_limit_value is not None:
                printer.information(f"Setting work limit to {work_limit_value}")

            # raise_on_no_solution=False: the batch must survive an OOM/no-solution run (there is no
            # try/except here) and still write work_units etc. — recovery lives in solve_model.
            result, timing_solving, _ = lego.solve_model(tee=tee, raise_on_no_solution=False)
            has_solution = lego.has_solution
            work_units_str = f"{lego.work_units:.2f} work units" if lego.work_units is not None else "work units unavailable"
            printer.information(f"Solving model took {timing_solving:.2f} seconds ({work_units_str})")

            edge_params = {**(run_params or {}), "edge_handling": edgeHandlingType.strip().replace('.', '').replace(' ', '')}
            if has_solution:
                write_results(lego, file_prefix, no_sqlite, **edge_params)

            match result.solver.termination_condition:
                case pyo.TerminationCondition.optimal:
                    printer.success("Optimal solution found")
                case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                    printer.error(f"Model is {result.solver.termination_condition}, logging infeasible constraints:")
                    log_infeasible_constraints(model)
                case _:
                    printer.warning("Solver terminated with condition:", result.solver.termination_condition)

        if not no_sqlite:
            sqlite_files.append(f"{file_prefix}.sqlite")
            sqlite_labels.append(edgeHandlingType)

        if calculate_regret and edgeHandlingType != "Truth " and not skip_truth:
            # --no-overwrite: skip only if the existing -regret file is optimal (status-based, like the
            # operational / invest-regret checks) — an existence-only check would silently keep a
            # work-limited result, or an old-semantics file from before -regret also fixed vGenInvest.
            regret_file = f"{file_prefix}-regret.sqlite"
            run_regret = True
            if no_overwrite and os.path.exists(regret_file):
                r_info = _read_sqlite_run_info(regret_file)
                r_tc = r_info.get('stats', {}).get('termination_condition') if r_info else None
                if r_tc == 'optimal':
                    printer.information(f"  File '{regret_file}' already has optimal solution, skipping regret (--no-overwrite)")
                    run_regret = False
                else:
                    printer.information(f"  File '{regret_file}' exists but status is '{r_tc or 'unknown'}' — re-running regret")
            if run_regret and case_skipped and not os.path.exists(f"{file_prefix}.sqlite"):
                printer.information(f"  Skipping regret: '{file_prefix}.sqlite' does not exist (smart-skipped)")
                run_regret = False
            if run_regret:
                try:
                    regret_lego = truth_lego.copy()
                    _apply_solver_options(regret_lego, run_params)

                    # Source vCommit + vGenInvest from the edge-handling main run: in-memory when the main
                    # solve ran this session, or from '{file_prefix}.sqlite' when it was no-overwrite-skipped.
                    if case_skipped:
                        printer.information(f"Loading vCommit and vGenInvest from '{file_prefix}.sqlite'")
                        cnx = sqlite3.connect(f"{file_prefix}.sqlite")
                        df_commit = pd.read_sql("SELECT * FROM vCommit", cnx)
                        df_inv = pd.read_sql("SELECT * FROM vGenInvest", cnx)
                        cnx.close()
                        for _, row in df_commit.iterrows():
                            model.vCommit[row.iloc[0], row.iloc[1], row.iloc[2]].value = row['values']
                            model.vCommit[row.iloc[0], row.iloc[1], row.iloc[2]].stale = False
                        gen_invest_values = dict(zip(df_inv.iloc[:, 0], df_inv['values']))
                    else:
                        gen_invest_values = {g: model.vGenInvest[g].value for g in model.g}

                    # Hard-fix vGenInvest from the edge-handling main run into the truth copy, on top of the
                    # vCommit soft-fix below: -regret reflects the cost of the edge handling's full plan
                    # (investment + commitment) evaluated in the truth world.
                    for g in regret_lego.model.g:
                        regret_lego.model.vGenInvest[g].value = gen_invest_values.get(g, 1)
                        regret_lego.model.vGenInvest[g].fixed = True

                    add_UnitCommitmentSlack_And_FixVariables(regret_lego, model, lego.cs.dPower_Hindex, lego.cs.dPower_ThermalGen, lego.cs.dPower_Parameters["pENSCost"])

                    # Re-solve the model
                    printer.information("Re-solving model with fixed variables for regret calculation")
                    regret_result, regret_timing_solving, regret_objective_value = regret_lego.solve_model(already_solved_ok=True)
                    printer.information(f"Solving regret model took {regret_timing_solving:.2f} seconds")

                    regret_params = {**(run_params or {}), "edge_handling": edgeHandlingType.strip().replace('.', '').replace(' ', ''), "run_type": "regret"}
                    if regret_lego.has_solution:
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
            ir_file = f"{file_prefix}-invest-regret.sqlite"
            run_invest_regret = True
            if no_overwrite and os.path.exists(ir_file):
                ir_info = _read_sqlite_run_info(ir_file)
                ir_tc = ir_info.get('stats', {}).get('termination_condition') if ir_info else None
                if ir_tc == 'optimal':
                    printer.information(f"  File '{ir_file}' already has optimal solution, skipping invest-regret (--no-overwrite)")
                    run_invest_regret = False
                else:
                    printer.information(f"  File '{ir_file}' exists but status is '{ir_tc or 'unknown'}' — re-running invest-regret")
            elif case_skipped and not os.path.exists(f"{file_prefix}.sqlite"):
                if sibling_skip_file is not None:
                    printer.information(f"  Main solve was sibling-skipped; will use vGenInvest from sibling '{sibling_skip_file}'")
                else:
                    printer.information(f"  Skipping invest-regret: no main file and no usable sibling")
                    run_invest_regret = False
            if run_invest_regret:
                try:
                    printer.information(f"Calculating invest-regret for '{edgeHandlingType}': fixing vGenInvest in truth model")
                    invest_regret_lego = truth_lego.copy()
                    _apply_solver_options(invest_regret_lego, run_params)

                    # Load vGenInvest values: from in-memory model, or from an existing sqlite file.
                    # When the main solve was skipped, use the exact file if it exists, or the best
                    # sibling's file if the exact file was never created (sibling-skip path).
                    if case_skipped:
                        source_file = f"{file_prefix}.sqlite" if os.path.exists(f"{file_prefix}.sqlite") else sibling_skip_file
                        printer.information(f"Loading vGenInvest from '{source_file}'")
                        cnx = sqlite3.connect(source_file)
                        df_inv = pd.read_sql("SELECT * FROM vGenInvest", cnx)
                        cnx.close()
                        gen_invest_values = dict(zip(df_inv.iloc[:, 0], df_inv['values']))
                        for g in invest_regret_lego.model.g:
                            invest_regret_lego.model.vGenInvest[g].value = gen_invest_values.get(g, 1)
                            invest_regret_lego.model.vGenInvest[g].fixed = True
                    else:
                        # Main solve ran: read vGenInvest straight from the in-memory model. Even on a
                        # non-optimal solve (e.g. work limit), solve_model loads the partial solution
                        # into the model, so these values match what was written to the sqlite.
                        for g in invest_regret_lego.model.g:
                            invest_regret_lego.model.vGenInvest[g].value = model.vGenInvest[g].value
                            invest_regret_lego.model.vGenInvest[g].fixed = True

                    # Re-solve the truth model with fixed investments
                    printer.information("Re-solving truth model with fixed vGenInvest for invest-regret calculation")
                    invest_regret_result, invest_regret_timing, invest_regret_objective = invest_regret_lego.solve_model(already_solved_ok=True)
                    printer.information(f"Solving invest-regret model took {invest_regret_timing:.2f} seconds")

                    invest_regret_params = {**(run_params or {}), "edge_handling": edgeHandlingType.strip().replace('.', '').replace(' ', ''), "run_type": "invest-regret"}
                    if invest_regret_lego.has_solution:
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

    if operational:
        execute_operational_runs(lego_models, case_name, no_sqlite, run_params, no_overwrite, tee, sqlite_files, sqlite_labels,
                                 cs=cs, thermal_generator_relaxed=thermal_generator_relaxed, operational_regret=operational_regret)

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
         scale_invest_cost: float = 1.0,
         thermal_invest_only: bool = False, merge_generators: bool = False,
         reuse_inputfiles: bool = False, enable_strict_markov: bool = False, invest_regret: bool = False,
         no_investment: bool = False, operational: bool = False, operational_regret: bool = False, no_overwrite: bool = False, rmip: bool = False, no_crossover: bool = False,
         force_barrier: bool = False, mip_gap: float | None = None, work_limit: float | None = None,
         node_file_start: float | None = None, node_file_dir: str | None = None,
         threads: int | None = None,
         network: str | None = None, commit_consumption: float = 1.0, startup_consumption: float = 1.0,
         shift_tm: int | None = None, perturb_tm: float | None = None):
    ew = ExcelWriter()

    if no_crossover != force_barrier:
        raise ValueError("Either both or none of no_crossover and force_barrier must be true")

    if node_file_dir is not None and node_file_start is None:
        raise ValueError("--node-file-dir requires --node-file-start to be set")

    if operational_regret and not operational:
        raise ValueError("--operational-regret requires --operational (it re-solves with vCommit taken from each edge handling's operational run)")

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
                                                                    no_investment, rmip, no_crossover, force_barrier, mip_gap, work_limit,
                                                                    node_file_start=node_file_start, node_file_dir=node_file_dir, threads=threads,
                                                                    filter_zone=filter_zone, limitK=limitK,
                                                                    clusters=cluster, shift=shift, stretch_demand=stretch_demand, scale_vres=scale_vres, scale_invest_cost=scale_invest_cost, thermal_invest_only=thermal_invest_only,
                                                                    merge_generators=merge_generators, no_overwrite=no_overwrite, operational=operational,
                                                                    operational_regret=operational_regret, network=network,
                                                                    commit_consumption=commit_consumption, startup_consumption=startup_consumption,
                                                                    shift_tm=shift_tm, perturb_tm=perturb_tm)
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
    parser.add_argument("--calculate-regret", action="store_true", help="Calculate regret by re-solving the truth model with vGenInvest and vCommit fixed from each model's main run (can take a while)")
    parser.add_argument("--relax-percentage", type=float, default=0, help="Fraction (0-1) of thermal generators to be relaxed (default: 0 = no relaxation, all binary)")
    parser.add_argument("--skip-truth", action="store_true", help="Skip solving the truth model")
    parser.add_argument("--clusters", type=int, default=1, help="Number of clusters (default: 1, i.e., no clustering)")
    parser.add_argument("--cluster-stepsize", type=int, default=1, help="If in-/decreasing number of clusters should be used (default: 1, leave cluster-steps default to not use in-/decreasing number of clusters)")
    parser.add_argument("--cluster-steps", type=int, default=0, help="Number of steps for in-/decreasing number of clusters (default: 0, i.e., leave clusters as given)")
    parser.add_argument("--filter-zone", type=str, default=None, help="Filter the case study to only include buses in the given zone (exact match of the 'z' column in Power_BusInfo), e.g. 'R1'")
    parser.add_argument("--limitK", type=str, help="Limit the ks, format: 'k0025-k0048'", nargs="?", default=None)
    parser.add_argument("--shift", type=int, default=0, help="Shift the time series by N hours (for testing purposes), e.g., 15 to shift by 15 hours")
    parser.add_argument("--stretch-demand", type=float, default=1.0, help="Stretch the demand by a factor (for testing purposes), e.g., 1.1 to increase max of demand by 5%% and decrease min by 5%%")
    parser.add_argument("--scale-vres", type=float, default=1.0, help="Scale the MaxProd of all VRES generators (PV, Wind, RoR) by this factor (default: 1.0, no change)")
    parser.add_argument("--scale-invest-cost", type=float, default=1.0, help="Scale the investment cost (pInvestCost) of all generators (ThermalGen, VRES, Storage) by this factor (default: 1.0, no change)")
    parser.add_argument("--thermal-invest-only", action="store_true", help="Set ExisUnits=1 for all non-thermal generators (VRES, Storage) so only thermal generators are investable")
    parser.add_argument("--merge-generators", action="store_true", help="Merge generators of the same technology at the same bus into one representative generator before clustering and solving")
    parser.add_argument("--reuse-inputfiles", action="store_true", help="Reuse input files (e.g., after shortening) instead of copying them to a new folder")
    parser.add_argument("--enable-strict-markov", action="store_true", help="Also execute the strict Markov variant (with push constraints active)")
    parser.add_argument("--invest-regret", action="store_true", help="Calculate invest-regret: fix vGenInvest from each edge-handling model into the truth model and compare objectives")
    parser.add_argument("--no-investment", action="store_true", help="Fix vGenInvest to 1 for all generators (skip investment decisions)")
    parser.add_argument("--operational", action="store_true", help="Add an operational run for each edge-handling model (Truth, NoEnf, Cyclic, Markov, and Markov-Strict if enabled) that fixes vGenInvest to the Truth investment decision (1 where Truth invested, 0 otherwise) and re-solves. Truth investment is taken from the in-memory Truth solve, or an optimal Truth .sqlite if Truth was skipped; if neither is available, operational runs are skipped. Output files get a '-operational' suffix")
    parser.add_argument("--operational-regret", action="store_true", help="Calculate operational-regret: for each non-Truth edge handling, re-solve the full-hourly truth model with vGenInvest fixed to the Truth investment (as in --operational) and vCommit fixed to that edge handling's OPERATIONAL run. Isolates operational regret under the correct (Truth) fleet. Requires --operational. Output files get a '-operational-regret' suffix")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip cases that already solved to optimality (existing output .sqlite, or a sibling run differing only in work_limit); non-optimal results are re-run")
    parser.add_argument("--rmip", action="store_true", help="Relax all integer variables (rMIP) before solving")
    parser.add_argument("--no-crossover", action="store_true", help="Disable Gurobi crossover for all solves (faster LP solving, but solution may not be a vertex)")
    parser.add_argument("--force-barrier", action="store_true", help="Force Gurobi to use barrier method")
    parser.add_argument("--mip-gap", type=float, default=None, help="Set the MIP gap tolerance for the solver (e.g., 0.01 for 1%%; default: solver default)")
    parser.add_argument("--work-limit", type=float, default=None, help="Set the Gurobi WorkLimit (in work units) to stop after a given amount of work regardless of solution quality (default: no limit)")
    parser.add_argument("--node-file-start", type=float, default=None, help="Gurobi NodefileStart (in GB): when in-memory B&B node storage exceeds this, nodes spill to disk (default: no spilling)")
    parser.add_argument("--node-file-dir", type=str, default=None, help="Base directory for Gurobi NodefileDir (where spilled nodes are written). Requires --node-file-start. A '<pid>' subfolder is ALWAYS appended (e.g. 'E:/tmp/nodes' becomes 'E:/tmp/nodes/12345/') to guarantee parallel spawns never share a dir. Defaults to ./gurobi-nodes/ in the CWD when --node-file-start is set without this flag. Auto-created if missing")
    parser.add_argument("--threads", type=int, default=None, help="Gurobi Threads (default: 0 = use all cores). Lower this when running multiple Markov.py processes in parallel to avoid CPU oversubscription and per-thread memory overhead")
    parser.add_argument("--network", type=str, default=None, choices=["DC-OPF", "TP", "SN"], help="Override network representation for all lines uniformly: DC-OPF, TP, or SN (default: no change, use values from data)")
    parser.add_argument("--commit-consumption", type=float, default=1.0, help="Multiplier for the CommitConsumption column of Power_ThermalGen (default: 1.0, no change)")
    parser.add_argument("--startup-consumption", type=float, default=1.0, help="Multiplier for the StartupConsumption column of Power_ThermalGen (default: 1.0, no change)")
    parser.add_argument("--shift-tm", type=int, default=None, help="Shift the transition matrix by <N> positions to the right")
    parser.add_argument("--perturb-tm", type=float, default=None, help="Perturb the transition matrix with randomness in [0.0, 1.0]: new_prob = (1-r)*orig + r*random")
    args = parser.parse_args()

    kwargs = vars(args)

    main(**kwargs)
