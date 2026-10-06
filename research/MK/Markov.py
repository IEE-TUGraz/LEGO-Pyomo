import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import gc
import glob
import logging
import math
import os
import shutil
import sqlite3
import time
import typing
from collections import Counter
from contextlib import closing

import numpy as np
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


########################################################################################################################
# Per-run analysis, written into every result .sqlite next to the model results (see analyze_and_write):
#   mk_metrics                 one row: model size, transition-matrix nonzeros, memory, timings, cost split,
#                              curtailment, load shedding, unit-commitment corrections
#   mk_nonbinarity             one row: how non-binary vCommit/vStartup/vShutdown are (values and RP transitions)
#   mk_nonbinary_values        one row per fractional UC value (raw data for distribution plots)
#   mk_feasibility             one row: static check whether the UC schedule, laid out along the chronology
#                              (Hindex), is feasible in the full chronological model
#   mk_feasibility_violations  one row per (day, unit, check) with a violation
# All analysis functions only read values from an already solved model, they never modify it.
########################################################################################################################

UC_VARS = ['vCommit', 'vStartup', 'vShutdown']

# Distance to the nearest integer above which a value counts as fractional. Must be above Gurobi's
# IntFeasTol (default 1e-5), otherwise integrality noise of binary variables would count as fractional.
DEFAULT_NONBINARY_TOL = 1e-4
# Absolute tolerance for constraint residuals in the chronological feasibility check (model units)
DEFAULT_FEASIBILITY_TOL = 1e-4

FEASIBILITY_CHECKS = ['fractional', 'logic', 'ramp_up', 'ramp_down', 'max_out_shutdown', 'min_up', 'min_down']
ROUNDED_CHECKS = ['min_up_rounded', 'min_down_rounded']


def _series(component) -> pd.Series:
    """Values of an indexed Pyomo Var/Param as float Series (None -> NaN), keeping the tuple index."""
    return pd.Series(component.extract_values(), dtype=float)


def _weights(model) -> pd.Series:
    """pWeight_rp[rp] * pWeight_k[k] as Series indexed by (rp, k)."""
    w_rp = pd.Series(model.pWeight_rp.extract_values(), dtype=float)
    w_k = pd.Series(model.pWeight_k.extract_values(), dtype=float)
    idx = pd.MultiIndex.from_product([w_rp.index, w_k.index], names=['rp', 'k'])
    return pd.Series(np.outer(w_rp.values, w_k.values).ravel(), index=idx)


def _weighted_sum(var_series: pd.Series, weights: pd.Series, keep: set | None = None) -> float:
    """Sum over (rp, k, x) of value * weight[rp, k], optionally only for x in `keep`."""
    if var_series is None or len(var_series) == 0:
        return 0.0
    df = var_series.rename('v').reset_index()
    df.columns = ['rp', 'k', 'x', 'v']
    if keep is not None:
        df = df[df['x'].isin(keep)]
    w = weights.rename('w').reset_index()
    df = df.merge(w, on=['rp', 'k'], how='left')
    return float((df['v'].fillna(0) * df['w'].fillna(0)).sum())


def _chronology(lego) -> pd.DataFrame | None:
    """Chronological sequence of (rp, k) from the case study's Hindex, with the day index of each hour.

    A new 'day' (= representative-period instance) starts whenever k is the first k of the model.
    Returns None if the Hindex does not map onto the model's (rp, k) indices.
    """
    hindex = lego.cs.dPower_Hindex.reset_index()
    if 'scenario' in hindex.columns and hindex['scenario'].nunique() > 1:
        hindex = hindex[hindex['scenario'] == hindex['scenario'].iloc[0]]
    chron = hindex[['rp', 'k']].reset_index(drop=True)
    first_k = lego.model.k.first()
    chron['day'] = (chron['k'] == first_k).cumsum() - 1
    if chron['day'].iloc[0] < 0:  # Hindex does not start with the first k
        chron['day'] += 1
    if not (set(chron['rp']) <= set(lego.model.rp) and set(chron['k']) <= set(lego.model.k)):
        return None
    return chron


def _active_thermal_units(model) -> list:
    """Thermal generators with at least one (existing or invested) unit - others are forced to 0 anyway."""
    active = []
    for g in model.thermalGenerators:
        exis = pyo.value(model.pExisUnits[g])
        invest = model.vGenInvest[g].value if g in model.vGenInvest else 0
        if exis + (invest or 0) > 0.5:
            active.append(g)
    return active


def _uc_window(model, g) -> int:
    """Number of timesteps at the start of each RP in which Markov relaxes the UC variables (see vUC_domain)."""
    return int(max(pyo.value(model.pMinUpTime[g]), pyo.value(model.pMinDownTime[g])))


def _delta(values: np.ndarray) -> np.ndarray:
    """Distance to the nearest integer (for binaries in [0, 1]: min(v, 1 - v))."""
    return np.abs(values - np.round(values))


def process_memory_gb() -> tuple[float | None, float | None]:
    """(current RSS, peak RSS) of this process in GB. Peak is the lifetime high-water mark of the process."""
    rss = peak = None
    try:
        import psutil
        mem = psutil.Process().memory_info()
        rss = mem.rss / 1024 ** 3
        if getattr(mem, 'peak_wset', None) is not None:  # Windows
            peak = mem.peak_wset / 1024 ** 3
    except Exception:
        pass
    if peak is None:
        try:
            import resource  # Linux: ru_maxrss in kB
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        except Exception:
            pass
    return rss, peak


def compute_run_metrics(lego, tm_cs=None) -> dict:
    """Model size, transition matrix size, memory, timings, cost split, curtailment and load shedding.

    :param lego: Solved LEGO instance
    :param tm_cs: CaseStudy whose transition matrix is reported (default: lego.cs). Regret runs pass the
        edge handling's (reduced) case study, since their own model is full-hourly.
    """
    model = lego.model
    cs = lego.cs
    tm_cs = tm_cs if tm_cs is not None else cs
    metrics = {}

    # Model size (Gurobi's view if available, Pyomo's otherwise)
    solver_attributes = getattr(lego, 'solver_attributes', {}) or {}
    for key, attr in [('num_vars', 'NumVars'), ('num_bin_vars', 'NumBinVars'), ('num_int_vars', 'NumIntVars'),
                      ('num_constrs', 'NumConstrs'), ('num_nonzeros', 'NumNZs')]:
        metrics[key] = solver_attributes.get(attr)
    try:
        metrics['pyomo_num_vars'] = model.nvariables()
        metrics['pyomo_num_constrs'] = model.nconstraints()
    except Exception:
        pass

    # Representative periods and transition matrix
    tm = tm_cs.rpTransitionMatrixAbsolute
    metrics['num_rps'] = len(tm_cs.dPower_WeightsRP.index)
    metrics['num_k_per_rp'] = len(tm_cs.dPower_WeightsK.index)
    metrics['tm_entries'] = int(tm.size)
    metrics['tm_nonzero'] = int((tm > 0).sum().sum())

    # Memory and timings
    metrics['solver_max_mem_gb'] = solver_attributes.get('MaxMemUsed')
    rss, peak = process_memory_gb()
    metrics['process_rss_gb'] = rss
    metrics['process_peak_rss_gb'] = peak
    metrics['solver_runtime_s'] = solver_attributes.get('Runtime')
    metrics['solve_wall_time_s'] = lego.timings.get('model_solving')
    metrics['build_time_s'] = lego.timings.get('model_building')
    metrics['work_units'] = lego.work_units

    # Costs (objective is in the model's cost unit, i.e. after pCostScalingFactor)
    objective = pyo.value(model.objective) if getattr(lego, 'has_solution', False) else None
    metrics['objective'] = objective
    invest_cost = sum(pyo.value(model.pInvestCost[g]) * (model.vGenInvest[g].value or 0) for g in model.g)
    if hasattr(model, 'vLineInvest') and hasattr(model, 'pFixedCost'):
        invest_cost += sum(pyo.value(model.pFixedCost[i, j, c]) * (model.vLineInvest[i, j, c].value or 0) for i, j, c in model.lc)
    metrics['investment_cost'] = invest_cost

    weights = _weights(model)
    penalty = 0.0
    if hasattr(model, 'vCommitCorrectHigher'):
        corr = _series(model.vCommitCorrectHigher).fillna(0) + _series(model.vCommitCorrectLower).fillna(0)
        max_prod = cs.dPower_ThermalGen['MaxProd']
        corr_df = corr.rename('v').reset_index()
        corr_df.columns = ['rp', 'k', 'g', 'v']
        corr_df = corr_df.merge(weights.rename('w').reset_index(), on=['rp', 'k'], how='left')
        penalty = float((corr_df['v'] * corr_df['w'] * corr_df['g'].map(max_prod)).sum() * cs.dPower_Parameters['pENSCost'])
        metrics['uc_corrections'] = int((corr > DEFAULT_NONBINARY_TOL).sum())
        metrics['uc_corrections_weighted'] = float((corr_df['v'] * corr_df['w']).sum())
    metrics['uc_correction_penalty'] = penalty
    metrics['operating_cost'] = objective - invest_cost - penalty if objective is not None else None

    # Energy balances in MWh (model power = MW * power_scaling_factor)
    to_mwh = 1 / cs.power_scaling_factor
    demand = _weighted_sum(_series(model.pDemandP), weights) * to_mwh
    shed = _weighted_sum(_series(model.vPNS), weights) * to_mwh
    metrics['demand_mwh'] = demand
    metrics['load_shedding_mwh'] = shed
    metrics['load_shedding_pct'] = shed / demand * 100 if demand > 0 else None
    metrics['excess_power_mwh'] = _weighted_sum(_series(model.vEPS), weights) * to_mwh
    if hasattr(model, 'vCurtailment') and hasattr(model, 'vresGenerators'):
        vres = set(model.vresGenerators)
        curtailed = _weighted_sum(_series(model.vCurtailment), weights) * to_mwh
        produced = _weighted_sum(_series(model.vGenP), weights, keep=vres) * to_mwh
        metrics['vres_production_mwh'] = produced
        metrics['curtailment_mwh'] = curtailed
        metrics['curtailment_pct'] = curtailed / (curtailed + produced) * 100 if curtailed + produced > 0 else None

    return metrics


def compute_nonbinarity(lego, tol: float = DEFAULT_NONBINARY_TOL) -> tuple[dict, pd.DataFrame]:
    """Quantify how non-binary vCommit/vStartup/vShutdown are.

    Two views:
    - values: every (rp, k, g) value; `window` restricts to the first max(MinUp, MinDown) timesteps of each RP,
      i.e. exactly where the Markov edge handling relaxes the variables to [0, 1].
    - transitions: a transition instance is one chronological RP boundary (day d-1 -> d, taken from Hindex) times
      one active thermal unit. It counts as fractional if any UC variable of the unit in the window of RP(d) is
      fractional. Each (rp, g) therefore counts as often as rp is entered in the chronology.
      Only defined for models with more than one RP instance in the chronology (not for full-hourly models).

    Only active thermal units (existing or invested) are considered.
    Returns (summary dict, DataFrame of all fractional values).
    """
    model = lego.model
    empty = pd.DataFrame(columns=['rp', 'k', 'g', 'var', 'value', 'delta', 'in_window', 'transitions_into_rp'])
    if not hasattr(model, 'vCommit') or len(model.thermalGenerators) == 0:
        return {}, empty

    units = _active_thermal_units(model)
    k_ord = {k: i + 1 for i, k in enumerate(model.k)}
    window = {g: _uc_window(model, g) for g in units}

    frames = []
    for var_name in UC_VARS:
        df = _series(getattr(model, var_name)).rename('value').reset_index()
        df.columns = ['rp', 'k', 'g', 'value']
        df['var'] = var_name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df['g'].isin(units)].dropna(subset=['value'])
    df['delta'] = _delta(df['value'].to_numpy())
    df['in_window'] = df['k'].map(k_ord) <= df['g'].map(window)
    df['fractional'] = df['delta'] > tol

    chron = _chronology(lego)
    trans_into = {}
    if chron is not None:
        day_rps = chron.groupby('day')['rp'].first()
        if len(day_rps) > 1:
            trans_into = Counter(day_rps.iloc[1:])
    df['transitions_into_rp'] = df['rp'].map(trans_into).fillna(0).astype(int) if trans_into else np.nan

    summary = {'nb_tol': tol, 'nb_active_units': len(units)}
    for var_name in UC_VARS + ['all']:
        part = df if var_name == 'all' else df[df['var'] == var_name]
        win = part[part['in_window']]
        frac = part[part['fractional']]
        frac_win = win[win['fractional']]
        prefix = f"nb_{var_name}"
        summary[f"{prefix}_values"] = len(part)
        summary[f"{prefix}_fractional"] = len(frac)
        summary[f"{prefix}_window_values"] = len(win)
        summary[f"{prefix}_window_fractional"] = len(frac_win)
        summary[f"{prefix}_window_fractional_pct"] = len(frac_win) / len(win) * 100 if len(win) > 0 else None
        summary[f"{prefix}_max_delta"] = float(part['delta'].max()) if len(part) > 0 else None
        summary[f"{prefix}_mean_delta_fractional"] = float(frac['delta'].mean()) if len(frac) > 0 else None
        summary[f"{prefix}_mean_delta_window"] = float(win['delta'].mean()) if len(win) > 0 else None

    # Transition view
    if trans_into:
        win = df[df['in_window']]
        per_instance = win.groupby(['rp', 'g'])['delta'].max()  # max fractionality per (rp, unit) window
        n_instances = sum(trans_into.values()) * len(units)
        weights = per_instance.index.get_level_values('rp').map(lambda rp: trans_into.get(rp, 0)).to_numpy()
        frac_mask = per_instance.to_numpy() > tol
        n_frac = int(weights[frac_mask].sum())
        summary['nb_transition_instances'] = n_instances
        summary['nb_transitions_fractional'] = n_frac
        summary['nb_transitions_fractional_pct'] = n_frac / n_instances * 100 if n_instances > 0 else None
        summary['nb_transitions_max_delta'] = float(per_instance.max()) if len(per_instance) > 0 else None
        # Averages over instances weighted by how often the RP is entered
        total_w = weights.sum()
        summary['nb_transitions_mean_delta'] = float((per_instance.to_numpy() * weights).sum() / total_w) if total_w > 0 else None
        frac_w = weights[frac_mask].sum()
        summary['nb_transitions_mean_delta_fractional'] = float((per_instance.to_numpy()[frac_mask] * weights[frac_mask]).sum() / frac_w) if frac_w > 0 else None
    else:
        summary['nb_transition_instances'] = 0

    values = df[df['fractional']].drop(columns=['fractional']).reset_index(drop=True)
    return summary, values


def check_chronological_feasibility(lego, nonbinary_tol: float = DEFAULT_NONBINARY_TOL, tol: float = DEFAULT_FEASIBILITY_TOL) -> tuple[dict, pd.DataFrame]:
    """Lay out the solved UC schedule (vCommit, vStartup, vShutdown, vGenP1) along the chronology given by Hindex
    and check the constraints of the full chronological model (notEnforced/chronological formulation of
    thermalGen.py) hour by hour - without solving anything.

    Checks (per unit and hour t, violations are assigned to the day that contains the latest hour involved):
      fractional        any UC variable at t is non-binary (always counts as a violation)
      logic             u[t] - u[t-1] == su[t] - sd[t]
      ramp_up           p1[t] - p1[t-1] <= u[t] * RampUp
      ramp_down         p1[t] - p1[t-1] >= -u[t-1] * RampDw
      max_out_shutdown  p1[t] <= (Pmax - Pmin) * (u[t] - sd[t+1])
      min_up            sum(su[t-UT+1..t]) <= u[t]
      min_down          sum(sd[t-DT+1..t]) <= 1 - u[t]
    Rounded variant: u rounded at 0.5, start-ups/shut-downs derived from the rounded u, then min_up/min_down
    checked again (dispatch-related checks are not repeated, since the dispatch could always be adjusted to a
    feasible commitment).

    A transition instance is one chronological boundary (day d-1 -> d) times one active thermal unit; it is
    infeasible if any check fails in day d. Uses MinUp/MinDown/ramp parameters as used in the model.
    Returns (summary dict, DataFrame of violations aggregated per (day, unit, check)).
    """
    model = lego.model
    empty = pd.DataFrame(columns=['day', 'g', 'check', 'hours', 'max_residual'])
    if not hasattr(model, 'vCommit') or len(model.thermalGenerators) == 0:
        return {}, empty
    chron = _chronology(lego)
    if chron is None:
        printer.warning("Feasibility check: Hindex does not match the model's (rp, k) indices - skipping")
        return {}, empty

    units = _active_thermal_units(model)
    if len(units) == 0:
        return {'feas_units': 0}, empty
    idx = pd.MultiIndex.from_arrays([chron['rp'], chron['k']])

    def _matrix(var_name):
        wide = _series(getattr(model, var_name)).unstack(level=2)  # (rp, k) x g
        return wide.reindex(index=idx, columns=units).fillna(0).to_numpy()

    u, su, sd, p1 = (_matrix(v) for v in ['vCommit', 'vStartup', 'vShutdown', 'vGenP1'])
    T, G = u.shape
    day = chron['day'].to_numpy()
    ut = np.array([int(pyo.value(model.pMinUpTime[g])) for g in units])
    dt = np.array([int(pyo.value(model.pMinDownTime[g])) for g in units])
    ramp_up = np.array([pyo.value(model.pRampUp[g]) for g in units])
    ramp_dw = np.array([pyo.value(model.pRampDw[g]) for g in units])
    span = np.array([pyo.value(model.pMaxProd[g]) - pyo.value(model.pMinProd[g]) for g in units])

    residuals = {name: np.zeros((T, G)) for name in FEASIBILITY_CHECKS + ROUNDED_CHECKS}
    # Day to which the residual at row t is assigned (latest hour involved)
    assign_next = {'max_out_shutdown'}

    residuals['fractional'] = np.maximum.reduce([_delta(u), _delta(su), _delta(sd)])
    residuals['fractional'][residuals['fractional'] <= nonbinary_tol] = 0
    residuals['logic'][1:] = np.abs(u[1:] - u[:-1] - (su[1:] - sd[1:]))
    residuals['ramp_up'][1:] = p1[1:] - p1[:-1] - u[1:] * ramp_up
    residuals['ramp_down'][1:] = -(p1[1:] - p1[:-1] + u[:-1] * ramp_dw)
    residuals['max_out_shutdown'][:-1] = p1[:-1] - span * (u[:-1] - sd[1:])

    def _rolling_sum(x, window):
        c = np.cumsum(x, axis=0)
        out = c.copy()
        out[window:] = c[window:] - c[:-window]
        return out

    ur = (u >= 0.5).astype(float)
    sur = np.zeros_like(ur)
    sdr = np.zeros_like(ur)
    sur[1:] = np.maximum(ur[1:] - ur[:-1], 0)
    sdr[1:] = np.maximum(ur[:-1] - ur[1:], 0)
    for j in range(G):
        if ut[j] > 1:
            res = _rolling_sum(su[:, j], ut[j]) - u[:, j]
            res[:ut[j] - 1] = 0  # Window not yet complete (as in the chronological model)
            residuals['min_up'][:, j] = res
            res = _rolling_sum(sur[:, j], ut[j]) - ur[:, j]
            res[:ut[j] - 1] = 0
            residuals['min_up_rounded'][:, j] = res
        if dt[j] > 1:
            res = _rolling_sum(sd[:, j], dt[j]) - (1 - u[:, j])
            res[:dt[j] - 1] = 0
            residuals['min_down'][:, j] = res
            res = _rolling_sum(sdr[:, j], dt[j]) - (1 - ur[:, j])
            res[:dt[j] - 1] = 0
            residuals['min_down_rounded'][:, j] = res

    rows = []
    for name, res in residuals.items():
        t_idx, g_idx = np.nonzero(res > (0 if name == 'fractional' else tol))
        if len(t_idx) == 0:
            continue
        d = day[np.minimum(t_idx + 1, T - 1)] if name in assign_next else day[t_idx]
        viol = pd.DataFrame({'day': d, 'g': np.array(units)[g_idx], 'check': name, 'residual': res[t_idx, g_idx]})
        rows.append(viol.groupby(['day', 'g', 'check']).agg(hours=('residual', 'size'), max_residual=('residual', 'max')).reset_index())
    violations = pd.concat(rows, ignore_index=True) if rows else empty

    n_days = int(day.max()) + 1
    n_boundaries = n_days - 1
    n_instances = n_boundaries * G
    summary = {'feas_tol': tol, 'feas_nonbinary_tol': nonbinary_tol, 'feas_units': G, 'feas_boundaries': n_boundaries,
               'feas_instances': n_instances}
    at_boundary = violations[violations['day'] >= 1] if len(violations) > 0 else violations
    for name in FEASIBILITY_CHECKS + ROUNDED_CHECKS:
        part = at_boundary[at_boundary['check'] == name] if len(at_boundary) > 0 else at_boundary
        summary[f"feas_{name}_instances"] = int(len(part[['day', 'g']].drop_duplicates())) if len(part) > 0 else 0
        summary[f"feas_{name}_max_residual"] = float(part['max_residual'].max()) if len(part) > 0 else 0.0
    if n_instances > 0:
        strict = at_boundary[at_boundary['check'].isin(FEASIBILITY_CHECKS)] if len(at_boundary) > 0 else at_boundary
        rounded = at_boundary[at_boundary['check'].isin(ROUNDED_CHECKS)] if len(at_boundary) > 0 else at_boundary
        n_strict = len(strict[['day', 'g']].drop_duplicates()) if len(strict) > 0 else 0
        n_rounded = len(rounded[['day', 'g']].drop_duplicates()) if len(rounded) > 0 else 0
        summary['feas_infeasible_instances'] = n_strict
        summary['feas_infeasible_pct'] = n_strict / n_instances * 100
        summary['feas_infeasible_rounded_instances'] = n_rounded
        summary['feas_infeasible_rounded_pct'] = n_rounded / n_instances * 100
    # Constraint violations within the first day (no incoming boundary) - expected to be 0, non-zero hints at a mismatch
    # between this check and the model (fractional values are excluded, Markov legitimately has them in every RP)
    first_day = violations[(violations['day'] == 0) & (violations['check'] != 'fractional')] if len(violations) > 0 else violations
    summary['feas_violations_first_day'] = int(len(first_day))
    return summary, violations


def analyze_and_write(lego, sqlite_file: str | None, tm_cs=None, nonbinary_tol: float = DEFAULT_NONBINARY_TOL) -> dict:
    """Run all analyses on a solved model, print a one-line summary and (if sqlite_file is given) write the tables.
    Never raises - analysis problems must not stop a batch run."""
    summary = {}
    tables = {}
    for name, detail_name, fn in [('mk_metrics', None, lambda: (compute_run_metrics(lego, tm_cs), None)),
                                  ('mk_nonbinarity', 'mk_nonbinary_values', lambda: compute_nonbinarity(lego, nonbinary_tol)),
                                  ('mk_feasibility', 'mk_feasibility_violations', lambda: check_chronological_feasibility(lego, nonbinary_tol))]:
        try:
            result, detail = fn()
            tables[name] = pd.DataFrame([result])
            summary.update(result)
            if detail_name is not None:
                tables[detail_name] = detail
        except Exception as e:
            printer.warning(f"Analysis '{name}' failed: {e}")

    def _fmt(key, fmt="{:.2f}"):
        v = summary.get(key)
        return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)

    printer.information(f"Analysis: fractional UC transitions {_fmt('nb_transitions_fractional_pct')}% "
                        f"(max delta {_fmt('nb_all_max_delta', '{:.3f}')}), infeasible transitions {_fmt('feas_infeasible_pct')}% "
                        f"(rounded {_fmt('feas_infeasible_rounded_pct')}%), curtailment {_fmt('curtailment_pct')}%, "
                        f"load shedding {_fmt('load_shedding_pct', '{:.4f}')}%, solver memory {_fmt('solver_max_mem_gb')} GB")

    if sqlite_file is not None:
        try:
            with closing(sqlite3.connect(sqlite_file)) as cnx:
                for table, df in tables.items():
                    df.to_sql(table, cnx, if_exists='replace', index=False)
                cnx.commit()
        except Exception as e:
            printer.warning(f"Could not write analysis tables to '{sqlite_file}': {e}")
    return summary


########################################################################################################################
# Writing, solving and fixing results
########################################################################################################################

def write_results(lego, file_prefix: str, no_sqlite: bool, tm_cs: CaseStudy | None = None, **run_parameters):
    """Write model results, solver statistics and run parameters to '{file_prefix}.sqlite', then add the
    per-run analysis tables (see analyze_and_write).

    :param tm_cs: Case study whose transition matrix is reported in the metrics (the edge handling's reduced case
        study for runs that re-solve a full-hourly model; default: lego.cs)
    """
    sqlite_file = None
    if not no_sqlite:
        sqlite_timer = time.time()
        sqlite_file = f"{file_prefix}.sqlite"
        printer.information(f"Writing model to SQLite database: {sqlite_file}")
        SQLiteWriter.model_to_sqlite(lego.model, sqlite_file)
        SQLiteWriter.add_solver_statistics_to_sqlite(sqlite_file, lego)
        if run_parameters:
            SQLiteWriter.add_run_parameters_to_sqlite(sqlite_file, **run_parameters)
        printer.information(f"Writing model to SQLite database took {time.time() - sqlite_timer:.2f} seconds")
    analyze_and_write(lego, sqlite_file, tm_cs=tm_cs)


def _solve_and_write(lego: LEGO, file_prefix: str, no_sqlite: bool, params: dict, tee: bool, label: str, tm_cs: CaseStudy | None = None) -> None:
    """Solve an already set-up model (never raises on a failed solve), write it if a solution exists and log the
    outcome. The has_solution gate matters: an empty sqlite would pollute the --no-overwrite status checks."""
    result, timing, objective = lego.solve_model(tee=tee, already_solved_ok=True, raise_on_no_solution=False)
    work_units_str = f"{lego.work_units:.2f} work units" if lego.work_units is not None else "work units unavailable"
    printer.information(f"Solving {label} model took {timing:.2f} seconds ({work_units_str})")
    if lego.has_solution:
        write_results(lego, file_prefix, no_sqlite, tm_cs=tm_cs, **params)
    match result.solver.termination_condition:
        case pyo.TerminationCondition.optimal:
            printer.success(f"Optimal {label} solution: {objective:.4f}")
        case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
            printer.error(f"{label} model is {result.solver.termination_condition}, logging infeasible constraints:")
            log_infeasible_constraints(lego.model)
        case _:
            printer.warning(f"{label} solver terminated with condition:", result.solver.termination_condition)


def _read_gen_invest(sqlite_file: str) -> dict:
    """vGenInvest of a result file as {g: value}."""
    with closing(sqlite3.connect(sqlite_file)) as cnx:
        df = pd.read_sql("SELECT * FROM vGenInvest", cnx)
    return dict(zip(df.iloc[:, 0], df['values']))


def _load_commit(model: pyo.Model, sqlite_file: str) -> None:
    """Load the vCommit values of a result file into a (structurally identical, unsolved) model."""
    with closing(sqlite3.connect(sqlite_file)) as cnx:
        df = pd.read_sql("SELECT * FROM vCommit", cnx)
    for rp, k, g, value in zip(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df['values']):
        model.vCommit[rp, k, g].value = value


def _fix_gen_invest(lego: LEGO, values: dict, default: float) -> None:
    """Hard-fix vGenInvest to `values` (generators missing in `values` get `default`)."""
    for g in lego.model.vGenInvest:
        lego.model.vGenInvest[g].value = values.get(g, default)
        lego.model.vGenInvest[g].fixed = True


def _soft_fix_commit(lego: LEGO, commit_model: pyo.Model, edge_cs: CaseStudy) -> None:
    """Soft-fix vCommit of a full-chronological model to an edge handling's vCommit, mapped along the edge's Hindex.

    Pyomo's stale flags are global: loading any other solution marks commit_model's values stale, and
    add_UnitCommitmentSlack_And_FixVariables maps stale values to 0 - so the values are re-marked as fresh first.
    """
    for var in commit_model.vCommit.values():
        if var.value is not None:
            var.stale = False
    add_UnitCommitmentSlack_And_FixVariables(lego, commit_model, edge_cs.dPower_Hindex, edge_cs.dPower_ThermalGen, edge_cs.dPower_Parameters["pENSCost"])


def _solve_fixed_decisions(base_lego: LEGO, out_prefix: str, gen_invest: dict, gen_invest_default: float, commit_model: pyo.Model | None,
                           edge_cs: CaseStudy, run_params: dict | None, params: dict, no_sqlite: bool, tee: bool, label: str,
                           tm_cs: CaseStudy | None = None, copy_base: bool = True) -> None:
    """Re-solve a copy of a full-chronological model (Truth or the original) with vGenInvest hard-fixed and, if
    commit_model is given, vCommit soft-fixed to an edge handling's decisions - the common core of all regret runs.
    The transition-matrix metrics of the written file describe tm_cs (default: the edge handling's model, edge_cs).
    copy_base=False modifies and solves base_lego itself (single-solve --task runs: avoids holding two full-year models)."""
    fixed_lego = base_lego.copy() if copy_base else base_lego
    _apply_solver_options(fixed_lego, run_params)
    _fix_gen_invest(fixed_lego, gen_invest, gen_invest_default)
    if commit_model is not None:
        _soft_fix_commit(fixed_lego, commit_model, edge_cs)
    _solve_and_write(fixed_lego, out_prefix, no_sqlite, params, tee, label, tm_cs=tm_cs if tm_cs is not None else edge_cs)
    del fixed_lego
    gc.collect()


########################################################################################################################
# --no-overwrite checks
########################################################################################################################

def _normalize_edge(edge_handling_type: str) -> str:
    """Model dict key -> edge_handling value used in filenames and run_parameters (e.g. 'NoEnf.' -> 'NoEnf')."""
    return edge_handling_type.strip().replace('.', '').replace(' ', '')


def _read_sqlite_run_info(sqlite_file: str) -> dict | None:
    """Read run_parameters and solver_statistics from a SQLite file."""
    try:
        with closing(sqlite3.connect(sqlite_file)) as cnx:
            tables = {row[0] for row in cnx.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            result = {}
            for key, table in [('params', 'run_parameters'), ('stats', 'solver_statistics')]:
                if table in tables:
                    df = pd.read_sql(f"SELECT * FROM {table}", cnx)
                    if not df.empty:
                        result[key] = df.iloc[0].to_dict()
        return result if result else None
    except Exception:
        return None


def _termination_condition(sqlite_file: str) -> str | None:
    info = _read_sqlite_run_info(sqlite_file) if os.path.exists(sqlite_file) else None
    return info.get('stats', {}).get('termination_condition') if info else None


def _existing_optimal(sqlite_file: str, label: str = "") -> bool:
    """--no-overwrite check of one output file: True (skip) if it exists and solved to optimality. An existing
    non-optimal file (work limit, OOM, old semantics, ...) is logged and re-run."""
    if not os.path.exists(sqlite_file):
        return False
    tc = _termination_condition(sqlite_file)
    label = f" {label}" if label else ""
    if tc == 'optimal':
        printer.information(f"  File '{sqlite_file}' already has optimal solution, skipping{label} (--no-overwrite)")
        return True
    printer.information(f"  File '{sqlite_file}' exists but status is '{tc or 'unknown'}' — re-running{label}")
    return False


_SIBLING_COMPARE_KEYS = [
    'case_study_directory', 'limit_k', 'clusters', 'shift', 'stretch_demand',
    'scale_vres', 'scale_invest_cost', 'thermal_invest_only', 'merge_generators',
    'relax_count', 'no_investment', 'rmip', 'no_crossover', 'force_barrier',
    'mip_gap', 'network', 'commit_consumption', 'startup_consumption', 'edge_handling',
    'shift_tm', 'perturb_tm', 'run_type', 'reference',
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


def _no_overwrite_check(file_prefix: str, run_params: dict | None, edge_params: dict) -> typing.Tuple[bool, str | None]:
    """--no-overwrite check of a main or operational run: skip if '{file_prefix}.sqlite' is optimal; if it does not
    exist, let the sibling runs (same run_parameters + edge_params, any work_limit) decide (_should_skip_smart).
    Returns (skip, sibling_file) - sibling_file is the sibling whose result stands in for a sibling-skipped run."""
    if os.path.exists(f"{file_prefix}.sqlite"):
        return _existing_optimal(f"{file_prefix}.sqlite"), None
    if run_params is None:
        return False, None
    siblings = _find_sibling_runs(file_prefix, {**run_params, **edge_params})
    if not siblings:
        return False, None
    skip, reason, sibling_file = _should_skip_smart(run_params.get('work_limit'), siblings)
    printer.information(f"  Skipping (--no-overwrite smart check): {reason}" if skip else f"  Running despite existing sibling run(s): {reason}")
    return skip, sibling_file


def _edge_main_decisions(lego: LEGO, file_prefix: str, run_params: dict | None, normalized: str, need_commit: bool) -> typing.Tuple[dict | None, pyo.Model | None, str | None]:
    """vGenInvest (and optionally vCommit) of an edge handling's main run, for the regret runs.

    Source: the in-memory model if its main solve ran in this session (even a work-limited one - solve_model loads
    the partial solution, so the values match the written sqlite), else '{file_prefix}.sqlite', else the sibling
    that --no-overwrite skipped for (differs only in work_limit). File values of vCommit are loaded into the
    (unsolved) in-memory edge model, so it can be passed to _soft_fix_commit.
    Returns (vGenInvest dict, model holding vCommit or None, description) or (None, None, None).
    """
    model = lego.model
    if lego.results is not None:  # Main solve ran in this session
        if not lego.has_solution:
            return None, None, None
        return {g: model.vGenInvest[g].value for g in model.vGenInvest}, (model if need_commit else None), "in-memory model"
    return _main_decisions_from_file(file_prefix, run_params, normalized, model if need_commit else None)


def _main_decisions_from_file(file_prefix: str, run_params: dict | None, normalized: str,
                              commit_model: pyo.Model | None) -> typing.Tuple[dict | None, pyo.Model | None, str | None]:
    """File part of _edge_main_decisions: '{file_prefix}.sqlite', else the sibling --no-overwrite skipped for. vCommit
    is loaded into commit_model if given (None = vGenInvest only, no model needed)."""
    source = f"{file_prefix}.sqlite" if os.path.exists(f"{file_prefix}.sqlite") else None
    if source is None and run_params is not None:
        siblings = _find_sibling_runs(file_prefix, {**run_params, "edge_handling": normalized})
        _, _, source = _should_skip_smart(run_params.get('work_limit'), siblings)
    if source is None:
        return None, None, None
    if commit_model is not None:
        _load_commit(commit_model, source)
    return _read_gen_invest(source), commit_model, f"'{source}'"


########################################################################################################################
# Case study preparation
########################################################################################################################

def _data_identifier(case_study_path: str) -> str:
    """First part of the sqlite identifier, derived from the case study folder."""
    return f"data{case_study_path.rstrip('/').replace('/', '_').replace(' ', '')}"


def _node_file_dir_with_pid(node_file_start: float | None, node_file_dir: str | None) -> str | None:
    """Gurobi NodefileDir: always a per-PID subfolder of the given base (default ./gurobi-nodes), so parallel spawns
    never collide on a shared dir, even if the user passed a path without thinking about parallelism."""
    if node_file_start is None:
        return node_file_dir
    node_file_dir_base = node_file_dir if node_file_dir is not None else os.path.join(os.getcwd(), "gurobi-nodes")
    node_file_dir = os.path.join(node_file_dir_base, str(os.getpid()))
    printer.information(f"NodeFileDir: '{node_file_dir}' (base '{node_file_dir_base}' + PID subfolder)")
    return node_file_dir


def _apply_case_modifications(case_studies: typing.List[CaseStudy], rmip: bool = False, no_crossover: bool = False,
                              force_barrier: bool = False, mip_gap: float | None = None, work_limit: float | None = None,
                              node_file_start: float | None = None, node_file_dir: str | None = None,
                              threads: int | None = None, network: str | None = None,
                              commit_consumption: float = 1.0, startup_consumption: float = 1.0,
                              scale_vres: float = 1.0, scale_invest_cost: float = 1.0,
                              thermal_invest_only: bool = False) -> typing.List[str]:
    """Apply solver options and data modifications in place to all given case studies (the RP case study and, with
    --original-reference, the original one - so both see identical settings). node_file_dir must already contain
    the per-PID subfolder. Returns the identifier parts for the sqlite filenames (Gurobi resource options excluded)."""
    identifier_parts = []
    if rmip:
        printer.information("Setting up case study as rMIP (relaxing all integer variables)")
        identifier_parts.append("rMIP")
    if no_crossover:
        printer.information("Disabling crossover for all solves")
        identifier_parts.append("noCrossover")
    if force_barrier:
        printer.information("Forcing barrier method for all solves")
        identifier_parts.append("forceBarrier")
    if mip_gap is not None:
        printer.information(f"Setting MIP gap to {mip_gap}")
        identifier_parts.append(f"mipGap{mip_gap:g}")
    if work_limit is not None:
        printer.information(f"Setting work limit to {work_limit}")
        identifier_parts.append(f"workLimit{work_limit:g}")
    # Gurobi memory-management knobs: not in identifier or sibling-compare keys (resource management, does not affect the solution)
    if node_file_start is not None:
        printer.information(f"Setting Gurobi NodefileStart to {node_file_start} GB")
    if threads is not None:
        printer.information(f"Setting Gurobi Threads to {threads}")
    if network is not None:
        printer.information(f"Setting all lines to network representation '{network}'")
        identifier_parts.append(f"network{network}")
    if commit_consumption != 1.0:
        printer.information(f"Scaling CommitConsumption by {commit_consumption}")
        identifier_parts.append(f"commitConsumption{commit_consumption:g}")
    if startup_consumption != 1.0:
        printer.information(f"Scaling StartupConsumption by {startup_consumption}")
        identifier_parts.append(f"startupConsumption{startup_consumption:g}")
    if scale_vres != 1.0:
        printer.information(f"Scaling VRES MaxProd by {scale_vres}")
        identifier_parts.append(f"scaleVRES{scale_vres:g}")
    if scale_invest_cost != 1.0:
        printer.information(f"Scaling InvestCostEUR by {scale_invest_cost}")
        identifier_parts.append(f"scaleInvestCost{scale_invest_cost:g}")
    if thermal_invest_only:
        printer.information("Setting ExisUnits=1 for all non-thermal generators (thermalInvestOnly)")
        identifier_parts.append("thermalInvestOnly")

    for cs in case_studies:
        if rmip:
            cs.dGlobal_Parameters["pEnableRMIP"] = True
        if no_crossover:
            cs.dGlobal_Parameters["pDisableCrossover"] = True
        if force_barrier:
            cs.dGlobal_Parameters["pForceBarrier"] = True
        if mip_gap is not None:
            cs.dGlobal_Parameters["pMIPGap"] = mip_gap
        if work_limit is not None:
            cs.dGlobal_Parameters["pWorkLimit"] = work_limit
        if node_file_start is not None:
            cs.dGlobal_Parameters["pNodeFileStart"] = node_file_start
            cs.dGlobal_Parameters["pNodeFileDir"] = node_file_dir
        if threads is not None:
            cs.dGlobal_Parameters["pThreads"] = threads
        if network is not None:
            cs.dPower_Network["pTecRepr"] = network
        if commit_consumption != 1.0:
            cs.dPower_ThermalGen['pInterVarCostEUR'] *= commit_consumption
        if startup_consumption != 1.0:
            cs.dPower_ThermalGen['pStartupCostEUR'] *= startup_consumption
        if scale_vres != 1.0:
            cs.dPower_VRES['MaxProd'] *= scale_vres
        if scale_invest_cost != 1.0:
            cs.dPower_ThermalGen['InvestCostEUR'] *= scale_invest_cost
            cs.dPower_VRES['InvestCostEUR'] *= scale_invest_cost
            cs.dPower_Storage['InvestCostEUR'] *= scale_invest_cost
        if thermal_invest_only:
            cs.dPower_VRES['ExisUnits'] = 1
            cs.dPower_VRES['EnableInvest'] = 0
            cs.dPower_Storage['ExisUnits'] = 1
            cs.dPower_Storage['EnableInvest'] = 0
    return identifier_parts


def _build_run_params(case_study_path: str, clusters: int, relax_count: int, filter_zone: str | None, limit_k: str | None,
                      shift: int, stretch_demand: float, merge_generators: bool, no_investment: bool,
                      shift_tm: int | None = None, perturb_tm: float | None = None,
                      rmip: bool = False, no_crossover: bool = False, force_barrier: bool = False, mip_gap: float | None = None,
                      work_limit: float | None = None, node_file_start: float | None = None, node_file_dir: str | None = None,
                      threads: int | None = None, network: str | None = None, commit_consumption: float = 1.0,
                      startup_consumption: float = 1.0, scale_vres: float = 1.0, scale_invest_cost: float = 1.0,
                      thermal_invest_only: bool = False, **extra) -> dict:
    """run_parameters stored in every sqlite (and compared by the --no-overwrite sibling check). Options at their
    default value are stored as None. Takes the same modification keywords as _apply_case_modifications."""

    def non_default(value, default):
        return None if value == default else value

    return dict(
        case_study_directory=case_study_path,
        filter_zone=filter_zone,
        limit_k=limit_k,
        clusters=clusters if clusters > 1 else None,
        shift=non_default(shift, 0),
        stretch_demand=non_default(stretch_demand, 1.0),
        scale_vres=non_default(scale_vres, 1.0),
        scale_invest_cost=non_default(scale_invest_cost, 1.0),
        thermal_invest_only=non_default(thermal_invest_only, False),
        merge_generators=non_default(merge_generators, False),
        relax_count=non_default(relax_count, 0),
        no_investment=non_default(no_investment, False),
        rmip=non_default(rmip, False),
        no_crossover=non_default(no_crossover, False),
        force_barrier=non_default(force_barrier, False),
        mip_gap=mip_gap,
        work_limit=work_limit,
        node_file_start=node_file_start,
        node_file_dir=node_file_dir,
        threads=threads,
        network=network,
        commit_consumption=non_default(commit_consumption, 1.0),
        startup_consumption=non_default(startup_consumption, 1.0),
        shift_tm=shift_tm,
        perturb_tm=perturb_tm,
        **extra,
    )


def _cap_min_up_down_times(cs: CaseStudy, cap: int) -> None:
    """Cap MinUpTime/MinDownTime to `cap` timesteps (the RP length), as the RP models cannot represent longer times."""
    if any(cs.dPower_ThermalGen["MinUpTime"] > cap) or any(cs.dPower_ThermalGen["MinDownTime"] > cap):
        printer.warning(f"Some thermal generators have MinUpTime or MinDownTime greater than {cap} - capping it to that number")
        cs.dPower_ThermalGen["MinUpTime"] = cs.dPower_ThermalGen["MinUpTime"].clip(upper=cap)
        cs.dPower_ThermalGen["MinDownTime"] = cs.dPower_ThermalGen["MinDownTime"].clip(upper=cap)


def _select_relaxed_generators(cs: CaseStudy, relax_percentage: float) -> typing.Tuple[dict, int]:
    """Thermal generators whose UC variables are relaxed (--relax-percentage): the ones with the smallest
    MinUpTime + MinDownTime. Returns ({generator: relaxed?}, number relaxed)."""
    if relax_percentage == 0:
        return {}, 0
    thermalGenerators = cs.dPower_ThermalGen.copy()
    count_relaxed = math.ceil(len(thermalGenerators.index) * relax_percentage)
    thermalGenerators["MinUpDownTime-Sum"] = thermalGenerators["MinUpTime"] + thermalGenerators["MinDownTime"]
    thermalGenerators.sort_values(by=["MinUpDownTime-Sum"], inplace=True)
    return {t: i < count_relaxed for i, t in enumerate(thermalGenerators.index)}, count_relaxed


def _relax_unit_commitment(lego: LEGO, thermal_generator_relaxed: dict | None) -> None:
    """Relax vCommit/vStartup/vShutdown to [0, 1] for the selected generators."""
    if not thermal_generator_relaxed:
        return
    for g in lego.model.thermalGenerators:
        if thermal_generator_relaxed.get(g):
            for rp in lego.model.rp:
                for k in lego.model.k:
                    lego.model.vCommit[rp, k, g].domain = pyo.PercentFraction
                    lego.model.vStartup[rp, k, g].domain = pyo.PercentFraction
                    lego.model.vShutdown[rp, k, g].domain = pyo.PercentFraction


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


# Model dict key -> edge handling parameter value of the RP models ("Truth " is the full-hourly model)
_EDGE_HANDLINGS = {"NoEnf.": "notEnforced", "Cyclic": "cyclic", "Markov": "markov", "Markov-Strict": "markov"}


def _build_edge_lego(cs: CaseStudy, name: str, thermal_generator_relaxed: dict, no_investment: bool) -> LEGO:
    """Build (not solve) the model of one edge handling from the RP case study: "Truth " = full-hourly model built from
    the RP copies, otherwise a copy of cs with the edge handling set. Applies the UC relaxation, the Markov-Strict push
    constraints and --no-investment - every model of a grid point is built through here, also in single --task runs."""
    start_time = time.time()
    if name == "Truth ":
        edge_cs = cs.to_full_hourly_model(inplace=False)
    else:
        edge_cs = cs.copy()
        for parameter in ["pReprPeriodEdgeHandlingUnitCommitment", "pReprPeriodEdgeHandlingRamping", "pReprPeriodEdgeHandlingIntraDayStorage"]:
            edge_cs.dPower_Parameters[parameter] = _EDGE_HANDLINGS[name]
    lego = LEGO(edge_cs)
    lego.build_model()
    _relax_unit_commitment(lego, thermal_generator_relaxed)
    if name == "Markov-Strict":
        _add_push_markov_constraints(lego, thermal_generator_relaxed)
    if no_investment:
        _fix_gen_invest(lego, {}, default=1)
    printer.information(f"Building model for '{name}' took {time.time() - start_time:.2f} seconds")
    return lego


########################################################################################################################
# Experiment execution
########################################################################################################################

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
                         original_folder: str | None = None,
                         task: str | None = None, edge: str | None = None,
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

    # Original full-chronological reference (--original-reference): only meaningful for the unperturbed
    # transition matrix, since --shift-tm/--perturb-tm resample a synthetic chronology
    cs_original = None
    if original_folder is not None and (invest_regret or calculate_regret or (task or "").startswith("original-")):
        if shift_tm is not None or perturb_tm is not None:
            printer.warning("--original-reference is skipped for runs with --shift-tm/--perturb-tm (their chronology is resampled and does not match the original one)")
        else:
            start_time = time.time()
            cs_original = CaseStudy(original_folder, clip_method="none", clip_value=0)
            printer.information(f"Loading original (unclustered) case study from '{original_folder}' took {time.time() - start_time:.2f} seconds")

    # Solver options and data modifications, applied identically to the RP and the original case study
    modifications = dict(rmip=rmip, no_crossover=no_crossover, force_barrier=force_barrier, mip_gap=mip_gap, work_limit=work_limit,
                         node_file_start=node_file_start, node_file_dir=_node_file_dir_with_pid(node_file_start, node_file_dir),
                         threads=threads, network=network, commit_consumption=commit_consumption, startup_consumption=startup_consumption,
                         scale_vres=scale_vres, scale_invest_cost=scale_invest_cost, thermal_invest_only=thermal_invest_only)
    # Identifier parts for sqlite filenames (similar to TR/ID naming convention)
    identifier_parts = [_data_identifier(case_study_path)] + _apply_case_modifications([cs] + ([cs_original] if cs_original is not None else []), **modifications)

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

    if task is None or (task == "main" and edge == "Markov"):  # --task: plot once per grid point, not from every parallel job
        tm_title_parts = []
        if shift_tm is not None:
            tm_title_parts.append(f"Diagonal shifted to the right by {shift_tm} steps")
        if perturb_tm is not None:
            tm_title_parts.append(f"Perturbed by {perturb_tm}")
        Utilities.plot_transition_matrix(cs.rpTransitionMatrixAbsolute, title=", ".join(tm_title_parts), output=f"MK-{identifier}.png")

    _cap_min_up_down_times(cs, len(cs.dPower_WeightsK.index))
    if cs_original is not None:  # Same cap as the RP models, so only the time series differ between the models
        _cap_min_up_down_times(cs_original, len(cs.dPower_WeightsK.index))
    thermalGeneratorRelaxed, count_relaxed = _select_relaxed_generators(cs, relax_percentage)  # After the cap, which affects the sort order
    if count_relaxed == 0:
        printer.information(f"Not relaxing any unit commitment variables, all thermal generators stay binary")
    else:
        printer.information(f"Relaxing {count_relaxed} thermal generator(s), keeping {len(thermalGeneratorRelaxed) - count_relaxed} binary: {[g for g, relaxed in thermalGeneratorRelaxed.items() if relaxed]}")

    run_params = _build_run_params(case_study_path, clusters=clusters, relax_count=count_relaxed, filter_zone=filter_zone, limit_k=limitK,
                                   shift=shift, stretch_demand=stretch_demand, merge_generators=merge_generators, no_investment=no_investment,
                                   shift_tm=shift_tm, perturb_tm=perturb_tm, **modifications)

    if task is not None:
        out_prefix = execute_task(task, edge, cs, identifier, run_params, thermalGeneratorRelaxed, no_investment, no_sqlite, no_overwrite, tee,
                                  cs_original=cs_original)
        return [f"{out_prefix}.sqlite"], [f"{edge} {task}"], {}

    start_time = time.time()
    printer.information(f"Building the LEGO models")  # Note building is actually faster (1.5-2x) than copying already built models to re-use them
    edge_names = ([] if skip_truth else ["Truth "]) + ["NoEnf.", "Cyclic", "Markov"] + (["Markov-Strict"] if enable_strict_markov else [])
    lego_models = {name: _build_edge_lego(cs, name, thermalGeneratorRelaxed, no_investment) for name in edge_names}
    printer.information(f"Building the LEGO models took {time.time() - start_time:.2f} seconds overall")

    sqlite_files, sqlite_labels, lego_models = execute_case_study(lego_models, identifier, no_sqlite, calculate_regret, skip_truth, invest_regret, run_params, no_overwrite, operational=operational, operational_regret=operational_regret, cs=cs, thermal_generator_relaxed=thermalGeneratorRelaxed, tee=tee)

    if cs_original is not None:
        execute_original_reference_runs(lego_models, identifier, cs_original, run_params, no_sqlite, no_overwrite, invest_regret, calculate_regret,
                                        thermalGeneratorRelaxed, tee, tm_cs=cs)

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
    candidates = [f"{truth_prefix}.sqlite"] if os.path.exists(f"{truth_prefix}.sqlite") else []
    truth_edge_params = {**(run_params or {}), "edge_handling": "Truth", "run_type": None}
    candidates.extend(s['file'] for s in _find_sibling_runs(truth_prefix, truth_edge_params))
    return next((f for f in candidates if _termination_condition(f) == 'optimal'), None)


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
            return _read_gen_invest(truth_file), f"Truth sqlite '{truth_file}'"
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
    truth_lego = LEGO(cs.to_full_hourly_model(inplace=False))
    truth_lego.build_model()
    _relax_unit_commitment(truth_lego, thermal_generator_relaxed)
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

    With operational_regret, each non-Truth edge handling additionally gets an operational-regret run: the
    full-hourly Truth model re-solved with the same vGenInvest and vCommit soft-fixed to the edge's operational run.
    """
    printer.information(f"\n\n{'#' * 60}\nOperational runs (--operational): vGenInvest fixed to Truth's investment\n{'#' * 60}")

    truth_invest, source = _get_truth_geninvest(lego_models, case_name, run_params)
    if truth_invest is None:
        printer.error("--operational: no Truth investment available (neither solved in memory nor an "
                      "optimal Truth .sqlite found) — skipping all operational runs")
        return
    operational_invest = {g: 1 if v > _OPERATIONAL_INVEST_THRESHOLD else 0 for g, v in truth_invest.items()}
    printer.information(f"Using Truth investment from {source}: {sum(operational_invest.values())} of {len(operational_invest)} generators invested")

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
    # reused across edge handlings.
    truth_base = None

    for edgeHandlingType, lego in edge_items:
        normalized = _normalize_edge(edgeHandlingType)
        op_prefix = f"MK-{case_name}-{normalized}-operational"
        op_params = {**(run_params or {}), "edge_handling": normalized, "run_type": "operational"}
        op_lego = None  # the in-memory operational solve, if it ran this iteration (else loaded from sqlite below)

        # --no-overwrite: exact file optimal, or a sibling differing only in work_limit says no improvement is possible
        op_skipped, op_sibling_file = _no_overwrite_check(op_prefix, run_params, {"edge_handling": normalized, "run_type": "operational"}) if no_overwrite else (False, None)
        if not op_skipped:
            printer.information(f"\n{'=' * 60}\n{normalized} (operational)\n{'=' * 60}")
            if lego is None:
                # Build the full-hourly Truth model on demand (lazily — only when actually
                # solving, so a --no-overwrite skip above avoids the expensive rebuild).
                printer.information("Building full-hourly Truth model on demand for Truth-operational (--skip-truth)")
                op_lego = _build_full_hourly_truth_lego(cs, thermal_generator_relaxed)
            else:
                op_lego = lego.copy()
            _apply_solver_options(op_lego, run_params)
            _fix_gen_invest(op_lego, operational_invest, default=0)
            _solve_and_write(op_lego, op_prefix, no_sqlite, op_params, tee, f"{normalized} operational")

        if not no_sqlite:
            sqlite_files.append(f"{op_prefix}.sqlite")
            sqlite_labels.append(f"{normalized}-op")

        # Operational-regret: files get a '-operational-regret' suffix and are NOT appended to
        # sqlite_files/sqlite_labels (like the other regret variants).
        if not operational_regret or edgeHandlingType == "Truth ":
            continue
        oregret_prefix = f"MK-{case_name}-{normalized}-operational-regret"
        if no_overwrite and _existing_optimal(f"{oregret_prefix}.sqlite", "operational-regret"):
            continue
        try:
            # Operational commitment to pin: the in-memory operational solve, else the exact operational file
            # (no-overwrite-skipped), else the smart-skipping sibling (differs only in work_limit - a feasible
            # commitment for the same problem; soft-fixed, so optimality is not required). File values are loaded
            # into a copy, so the edge model keeps its main-run values (used by the original-reference runs).
            op_commit_model = None
            if op_lego is not None:
                op_commit_model = op_lego.model if op_lego.has_solution else None
            else:
                op_commit_source = f"{op_prefix}.sqlite" if os.path.exists(f"{op_prefix}.sqlite") else op_sibling_file
                if op_commit_source is not None:
                    printer.information(f"Loading operational vCommit from '{op_commit_source}'")
                    op_commit_model = lego.copy().model
                    _load_commit(op_commit_model, op_commit_source)

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
                _solve_fixed_decisions(truth_base, oregret_prefix, operational_invest, 0, op_commit_model, lego.cs, run_params,
                                       {**op_params, "run_type": "operational-regret"}, no_sqlite, tee, f"{normalized} operational-regret")
        except Exception as e:
            printer.error(f"Operational-regret calculation failed for '{normalized}': {e}")


def execute_case_study(lego_models: typing.Dict[str, LEGO], case_name: str, no_sqlite: bool, calculate_regret: bool, skip_truth: bool, invest_regret: bool = False, run_params: dict = None, no_overwrite: bool = False, operational: bool = False, operational_regret: bool = False, cs: CaseStudy | None = None, thermal_generator_relaxed: dict | None = None, tee: bool = True) -> typing.Tuple[typing.List[str], typing.List[str], typing.Dict[str, LEGO]]:
    """Solve the main run of every edge handling, followed by its regret runs: the full-hourly Truth model re-solved
    with vGenInvest hard-fixed ('-invest-regret') and additionally vCommit soft-fixed ('-regret') to the edge
    handling's main-run decisions (see _edge_main_decisions for their source)."""
    sqlite_files = []
    sqlite_labels = []

    truth_lego = lego_models.get("Truth ")
    regret_variants = ([("regret", True)] if calculate_regret else []) + ([("invest-regret", False)] if invest_regret else [])
    if regret_variants and truth_lego is None:
        printer.warning("--calculate-regret/--invest-regret re-solve the Truth model, which is skipped (--skip-truth) - no regret runs")
        regret_variants = []

    for edgeHandlingType, lego in lego_models.items():
        printer.information(f"\n\n{'=' * 60}\n{edgeHandlingType}\n{'=' * 60}")
        model = lego.model
        normalized = _normalize_edge(edgeHandlingType)
        file_prefix = f"MK-{case_name}-{normalized}"
        edge_params = {**(run_params or {}), "edge_handling": normalized}

        # Main solve (or skip if --no-overwrite and an optimal result / sufficient sibling exists)
        case_skipped, _ = _no_overwrite_check(file_prefix, run_params, {"edge_handling": normalized}) if no_overwrite else (False, None)
        if not case_skipped:
            if getattr(model, 'pDisableCrossover', False):
                printer.information("Deactivating crossover")
            if getattr(model, 'pForceBarrier', False):
                printer.information("Forcing barrier method")
            if getattr(model, 'pMIPGap', None) is not None:
                printer.information(f"Setting MIP gap to {model.pMIPGap}")
            if getattr(model, 'pWorkLimit', None) is not None:
                printer.information(f"Setting work limit to {model.pWorkLimit}")
            _solve_and_write(lego, file_prefix, no_sqlite, edge_params, tee, normalized)

        if not no_sqlite:
            sqlite_files.append(f"{file_prefix}.sqlite")
            sqlite_labels.append(edgeHandlingType)

        if edgeHandlingType == "Truth ":
            continue  # Regret of Truth itself is degenerate
        for run_type, fix_commit in regret_variants:
            out_prefix = f"{file_prefix}-{run_type}"
            if no_overwrite and _existing_optimal(f"{out_prefix}.sqlite", run_type):
                continue
            try:
                gen_invest, commit_model, source = _edge_main_decisions(lego, file_prefix, run_params, normalized, need_commit=fix_commit)
                if gen_invest is None:
                    printer.information(f"  Skipping {run_type} for '{normalized}': no main-run result available")
                    continue
                printer.information(f"\n{'=' * 60}\n{normalized} ({run_type}, decisions from {source})\n{'=' * 60}")
                _solve_fixed_decisions(truth_lego, out_prefix, gen_invest, 1, commit_model, lego.cs, run_params,
                                       {**edge_params, "run_type": run_type}, no_sqlite, tee, f"{normalized} {run_type}")
            except Exception as e:
                printer.error(f"{run_type} calculation failed for '{normalized}': {e}")

    if operational:
        execute_operational_runs(lego_models, case_name, no_sqlite, run_params, no_overwrite, tee, sqlite_files, sqlite_labels,
                                 cs=cs, thermal_generator_relaxed=thermal_generator_relaxed, operational_regret=operational_regret)

    return sqlite_files, sqlite_labels, lego_models


def _build_original_reference_lego(cs_original: CaseStudy, thermal_generator_relaxed: dict | None) -> LEGO:
    """Build (not solve) the original full-chronological model (--original-reference)."""
    start_time = time.time()
    cs_original.to_full_hourly_model(inplace=True)  # No-op for hourly data; rebuilds the chronology if the original data uses RPs
    original_lego = LEGO(cs_original)
    original_lego.build_model()
    _relax_unit_commitment(original_lego, thermal_generator_relaxed)
    printer.information(f"Building original full-chronological model took {time.time() - start_time:.2f} seconds")
    return original_lego


def execute_original_reference_runs(lego_models: typing.Dict[str, LEGO], case_name: str, cs_original: CaseStudy,
                                    run_params: dict | None, no_sqlite: bool, no_overwrite: bool, invest_regret: bool,
                                    calculate_regret: bool, thermal_generator_relaxed: dict | None, tee: bool, tm_cs: CaseStudy) -> None:
    """Evaluate each edge handling's decisions in the ORIGINAL full-chronological model (--original-reference).

    Mirrors --invest-regret (vGenInvest hard-fixed) and --calculate-regret (vGenInvest hard-fixed + vCommit soft-fixed
    via the edge handling's Hindex) against the original, unclustered time series instead of the chronology rebuilt
    from RP copies. Truth (RP copies) is included in invest-regret: its regret against the original isolates the
    error of the clustering itself. Files: 'MK-{case}-{edge}-original-{invest-regret|regret}.sqlite' with
    run_parameters reference='original'. The original model is built lazily (only if a run is not skipped), and the
    in-memory Truth (RP copies) model is released first, since both are full-year models.
    tm_cs is the RP case study whose transition matrix is reported in the metrics (also for Truth, whose own model
    has a single RP).
    """
    printer.information(f"\n\n{'#' * 60}\nOriginal-reference runs (--original-reference)\n{'#' * 60}")
    variants = ([("invest-regret", False)] if invest_regret else []) + ([("regret", True)] if calculate_regret else [])

    # Resolve all decisions first, so the large Truth (RP copies) model can be released before building the original model
    jobs = []
    for edgeHandlingType, lego in list(lego_models.items()):
        normalized = _normalize_edge(edgeHandlingType)
        file_prefix = f"MK-{case_name}-{normalized}"
        for run_type, fix_commit in variants:
            if fix_commit and edgeHandlingType == "Truth ":
                continue  # Only invest-regret for Truth (its vCommit would require keeping the full-year model)
            out_prefix = f"{file_prefix}-original-{run_type}"
            if no_overwrite and _existing_optimal(f"{out_prefix}.sqlite", f"original {run_type}"):
                continue
            gen_invest, commit_model, source = _edge_main_decisions(lego, file_prefix, run_params, normalized, need_commit=fix_commit)
            if gen_invest is None:
                printer.information(f"  Skipping original {run_type} for '{normalized}': no main-run result available")
                continue
            jobs.append((normalized, lego.cs, run_type, out_prefix, gen_invest, commit_model, source))  # No reference to the lego itself, so Truth can be released
    if "Truth " in lego_models:
        del lego_models["Truth "]
        gc.collect()

    original_base = None
    for normalized, edge_cs, run_type, out_prefix, gen_invest, commit_model, source in jobs:
        try:
            printer.information(f"\n{'=' * 60}\n{normalized} (original {run_type}, decisions from {source})\n{'=' * 60}")
            if original_base is None:
                original_base = _build_original_reference_lego(cs_original, thermal_generator_relaxed)
            params = {**(run_params or {}), "edge_handling": normalized, "run_type": run_type, "reference": "original"}
            _solve_fixed_decisions(original_base, out_prefix, gen_invest, 1, commit_model, edge_cs, run_params, params, no_sqlite, tee,
                                   f"{normalized} original {run_type}", tm_cs=tm_cs)
        except Exception as e:
            printer.error(f"Original {run_type} failed for '{normalized}': {e}")


def execute_original_truth(original_folder: str, no_sqlite: bool = False, relax_percentage: float = 0, no_investment: bool = False,
                           no_overwrite: bool = False, min_time_cap: int = 24, tee: bool = True, filter_zone: str | None = None,
                           limit_k: str | None = None, shift: int = 0, stretch_demand: float = 1.0, merge_generators: bool = False,
                           **modifications) -> str:
    """Solve the original full-chronological model once (--original-reference-only). It depends only on the data
    folder (incl. preprocessing such as --stretch-demand) and the model options, not on the number of RPs or the edge
    handling, so one solve serves all RP runs. MinUp/MinDown times are capped to `min_time_cap` (the RP length),
    as in the RP models. Writes 'MK-{identifier}-TruthOriginal.sqlite' with reference='original' and returns its prefix.

    :param modifications: The data/solver options of _apply_case_modifications
    """
    modifications['node_file_dir'] = _node_file_dir_with_pid(modifications.get('node_file_start'), modifications.get('node_file_dir'))
    cs = CaseStudy(original_folder, clip_method="none", clip_value=0)
    identifier_parts = [_data_identifier(original_folder)] + _apply_case_modifications([cs], **modifications)
    _cap_min_up_down_times(cs, min_time_cap)
    thermal_generator_relaxed, count_relaxed = _select_relaxed_generators(cs, relax_percentage)  # After the cap, as in the RP runs
    if count_relaxed > 0:
        identifier_parts.append(f"relaxed{count_relaxed}")
    file_prefix = f"MK-{'-'.join(identifier_parts)}-TruthOriginal"

    if no_overwrite and _existing_optimal(f"{file_prefix}.sqlite"):
        return file_prefix

    lego = _build_original_reference_lego(cs, thermal_generator_relaxed)
    if no_investment:
        _fix_gen_invest(lego, {}, default=1)
    params = _build_run_params(original_folder, clusters=1, relax_count=count_relaxed, filter_zone=filter_zone, limit_k=limit_k, shift=shift,
                               stretch_demand=stretch_demand, merge_generators=merge_generators, no_investment=no_investment, **modifications,
                               edge_handling="TruthOriginal", reference="original", min_time_cap=min_time_cap)
    printer.information(f"\n{'=' * 60}\nTruthOriginal ({original_folder})\n{'=' * 60}")
    _solve_and_write(lego, file_prefix, no_sqlite, params, tee, "TruthOriginal")


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
         clusters: int | str = 1, cluster_stepsize: int = 1, cluster_steps: int = 0,
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
         shift_tm: int | None = None, perturb_tm: float | None = None,
         prepare_only: bool = False, original_reference: bool = False, original_reference_only: bool = False,
         task: str | None = None, edge: str | None = None):
    ew = ExcelWriter()

    # --clusters accepts a single number or a comma-separated list (e.g. '3,5,7,10,14,18')
    cluster_list = [int(c) for c in str(clusters).split(",")]
    if len(cluster_list) > 1 and cluster_steps != 0:
        raise ValueError("--cluster-steps cannot be combined with a list of --clusters")
    if len(cluster_list) == 1:
        cluster_list = list(range(cluster_list[0], cluster_list[0] + cluster_steps * cluster_stepsize + 1, cluster_stepsize))

    if original_reference and not (invest_regret or calculate_regret):
        raise ValueError("--original-reference requires --invest-regret and/or --calculate-regret (it evaluates their decisions in the original model)")

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

            if original_reference_only:
                # Solve only the original full-chronological model of the (preprocessed) folder - independent of the
                # number of RPs, so run this once per folder before the RP runs that use --original-reference
                out_prefix = execute_original_truth(folder, no_sqlite=no_sqlite, relax_percentage=relax_percentage, no_investment=no_investment,
                                       no_overwrite=no_overwrite, filter_zone=filter_zone, limit_k=limitK, shift=shift,
                                       stretch_demand=stretch_demand, merge_generators=merge_generators,
                                       rmip=rmip, no_crossover=no_crossover, force_barrier=force_barrier, mip_gap=mip_gap, work_limit=work_limit,
                                       node_file_start=node_file_start, node_file_dir=node_file_dir, threads=threads, network=network,
                                       commit_consumption=commit_consumption, startup_consumption=startup_consumption, scale_vres=scale_vres,
                                       scale_invest_cost=scale_invest_cost, thermal_invest_only=thermal_invest_only)
                continue

            for cluster in cluster_list:
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

                if prepare_only:
                    printer.information(f"Prepared input files in '{cluster_folder}' (--prepare-only, not solving)")
                    continue

                printer.information(f"Loading case study from '{cluster_folder}'")

                sqlite_files, _, _ = execute_case_studies(cluster_folder, no_sqlite, calculate_regret, relax_percentage, skip_truth, enable_strict_markov, invest_regret,
                                     no_investment, rmip, no_crossover, force_barrier, mip_gap, work_limit,
                                     node_file_start=node_file_start, node_file_dir=node_file_dir, threads=threads,
                                     filter_zone=filter_zone, limitK=limitK,
                                     clusters=cluster, shift=shift, stretch_demand=stretch_demand, scale_vres=scale_vres, scale_invest_cost=scale_invest_cost, thermal_invest_only=thermal_invest_only,
                                     merge_generators=merge_generators, no_overwrite=no_overwrite, operational=operational,
                                     operational_regret=operational_regret, network=network,
                                     commit_consumption=commit_consumption, startup_consumption=startup_consumption,
                                     shift_tm=shift_tm, perturb_tm=perturb_tm,
                                     original_folder=folder if (original_reference or (task or "").startswith("original-")) and cluster > 1 else None,
                                     task=task, edge=edge)
                if task is not None:
                    _require_optimal_file(sqlite_files[0], f"--task {task} --edge {edge}")
        except Exception as e:
            printer.error(f"Exception while executing case study '{locals().get('cluster_folder', folder)}': {e}")  # locals-hack to always get correct folder-name
            if debug or task is not None:
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
    parser.add_argument("--clusters", type=str, default="1", help="Number of clusters (default: 1, i.e., no clustering). Can be a comma-separated list, e.g. '3,5,7,10,14,18' (executed after each other)")
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
    parser.add_argument("--prepare-only", action="store_true", help="Only create the preprocessed input folders (e.g. clusters, stretched demand) and exit without solving. Run this before starting parallel jobs that use --reuse-inputfiles, so they do not write the same folders concurrently")
    parser.add_argument("--original-reference", action="store_true", help="Additionally evaluate each model's decisions in the ORIGINAL (unclustered) full-chronological model: '-original-invest-regret' (with --invest-regret, incl. Truth) and '-original-regret' (with --calculate-regret). Only for runs without --shift-tm/--perturb-tm")
    parser.add_argument("--original-reference-only", action="store_true", help="Only solve the original full-chronological model of the (preprocessed) folder ('MK-...-TruthOriginal.sqlite') and exit. Independent of --clusters, so one run serves all RP counts")
    args = parser.parse_args()

    main(**vars(args))
