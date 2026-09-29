"""
Per-run analyses for the Markov (MK) experiments, computed by Markov.py right after each solve and written
into the result .sqlite next to the model results (see write_analysis_tables):

- mk_metrics                 one row: model size, transition-matrix nonzeros, memory, timings, cost split,
                             curtailment, load shedding, unit-commitment corrections
- mk_nonbinarity             one row: how non-binary vCommit/vStartup/vShutdown are (values and RP transitions)
- mk_nonbinary_values        one row per fractional UC value (raw data for distribution plots)
- mk_feasibility             one row: static check whether the UC schedule, laid out along the chronology
                             (Hindex), is feasible in the full chronological model
- mk_feasibility_violations  one row per (day, unit, check) with a violation

All functions only read values from an already solved model, they never modify it.
"""
import math
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from InOutModule.printer import Printer

printer = Printer.getInstance()

UC_VARS = ['vCommit', 'vStartup', 'vShutdown']

# Distance to the nearest integer above which a value counts as fractional. Must be above Gurobi's
# IntFeasTol (default 1e-5), otherwise integrality noise of binary variables would count as fractional.
DEFAULT_NONBINARY_TOL = 1e-4
# Absolute tolerance for constraint residuals in the chronological feasibility check (model units)
DEFAULT_FEASIBILITY_TOL = 1e-4

FEASIBILITY_CHECKS = ['fractional', 'logic', 'ramp_up', 'ramp_down', 'max_out_shutdown', 'min_up', 'min_down']
ROUNDED_CHECKS = ['min_up_rounded', 'min_down_rounded']


########################################################################################################################
# Helpers
########################################################################################################################

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
    model_rps = set(lego.model.rp)
    model_ks = set(lego.model.k)
    if not (set(chron['rp']) <= model_rps and set(chron['k']) <= model_ks):
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


########################################################################################################################
# Run metrics
########################################################################################################################

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


########################################################################################################################
# Non-binarity of the unit commitment variables
########################################################################################################################

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
        s = _series(getattr(model, var_name))
        df = s.rename('value').reset_index()
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


########################################################################################################################
# Static feasibility check of the UC schedule along the chronology
########################################################################################################################

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
        s = _series(getattr(model, var_name))
        wide = s.unstack(level=2)  # (rp, k) x g
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


########################################################################################################################
# Output
########################################################################################################################

def analyze_and_write(lego, sqlite_file: str | None, tm_cs=None, nonbinary_tol: float = DEFAULT_NONBINARY_TOL) -> dict:
    """Run all analyses on a solved model, print a one-line summary and (if sqlite_file is given) write the tables.
    Never raises - analysis problems must not stop a batch run."""
    summary = {}
    tables = {}
    for name, fn in [('mk_metrics', lambda: (compute_run_metrics(lego, tm_cs), None)),
                     ('mk_nonbinarity', lambda: compute_nonbinarity(lego, nonbinary_tol)),
                     ('mk_feasibility', lambda: check_chronological_feasibility(lego, nonbinary_tol))]:
        try:
            result, detail = fn()
            tables[name] = pd.DataFrame([result])
            summary.update(result)
            if detail is not None:
                tables[{'mk_nonbinarity': 'mk_nonbinary_values', 'mk_feasibility': 'mk_feasibility_violations'}[name]] = detail
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
            with sqlite3.connect(sqlite_file) as cnx:
                for table, df in tables.items():
                    df.to_sql(table, cnx, if_exists='replace', index=False)
        except Exception as e:
            printer.warning(f"Could not write analysis tables to '{sqlite_file}': {e}")
    return summary
