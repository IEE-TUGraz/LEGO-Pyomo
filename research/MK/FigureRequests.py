"""
FigureRequests.py - Figures and tables, computed directly from the MK-*.sqlite result files.

Run it where the result files are (e.g. on MUSICA, from the repo root), so that no result files need to be downloaded:

    python research/MK/FigureRequests.py [folder ...] --output-dir results/figure_requests [--only R1,R7] [--reuse]

- folder(s): folders with MK-*.sqlite files (default: current directory); several folders are combined (e.g. the
  one-third folders of the three case studies).
- Loading (run metrics, regret pairing incl. the original reference, chronological comparisons) is done with
  EvaluateMarkov.py, so the numbers match its markov_runs.csv. A second parallel pass extracts what the figures need
  from the RP result files (violations, storage boundary jumps, last-hour commitment). Both are cached in the output
  directory (markov_runs.csv, _cache/); --reuse skips the loading and reuses the cache.
- R4 (repair of the fractional flags) needs new runs and is skipped. R10 is drawn only if TruthOriginal and
  -original-invest-regret runs exist.
- Every figure: <name>.png (600 dpi) + <name>.pdf + the CSV behind it; 16 cm wide, 8 pt text. A failing figure is
  reported and skipped. See README.md ("FigureRequests.py") for the definitions of all quantities.
"""
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import os
import pickle
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing

import numpy as np
import pandas as pd

import EvaluateMarkov as em

########################################################################################################################
# Constants and style
########################################################################################################################

CASES_ORDER = ["RTS-GMLC", "NREL-118", "TX-123BT"]
CLUSTERS = [3, 5, 7, 14, 21, 28]
METHODS = ["NoEnf", "Cyclic", "Markov"]
LABEL = {"NoEnf": "No Enf.", "Cyclic": "Cyclic", "Markov": "Markov", "Truth": "Reference"}
COLOR = {"NoEnf": "#9467bd", "Cyclic": "#ff7f0e", "Markov": "#1f77b4", "Truth": "#52514e"}
CASE_MARKER = {"RTS-GMLC": "o", "NREL-118": "s", "TX-123BT": "^"}
CASE_COLOR = {"RTS-GMLC": "#1b9e77", "NREL-118": "#d95f02", "TX-123BT": "#7570b3"}
TM_COLOR = {"original": "#333333", "shiftTM1": "#2ca02c", "shiftTM2": "#d62728"}
TM_LABEL = {"original": "original TM", "shiftTM1": "TM shifted by 1", "shiftTM2": "TM shifted by 2"}
KEYS = ["case", "tm", "clusters", "sd"]
WIDTH_IN = 16 / 2.54
TOL = 1e-4  # Markov.DEFAULT_FEASIBILITY_TOL / DEFAULT_NONBINARY_TOL
ST_TOL = em.STORAGE_BOUNDARY_TOL

OUT = Path(".")
CACHE = Path(".")
MIP_GAP = 0.005


def style():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                                "ytick.labelsize": 8, "legend.fontsize": 8, "savefig.dpi": 600, "pdf.fonttype": 42,
                                "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name, data=None):
    import matplotlib.pyplot as plt
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    if data is not None:
        data.to_csv(OUT / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"  wrote {name}")


def method_handles(methods=METHODS, truth=False):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(facecolor=COLOR[m], alpha=0.6, edgecolor=COLOR[m], label=LABEL[m]) for m in methods]
    if truth:
        h.append(Line2D([], [], color=COLOR["Truth"], ls="--", label=LABEL["Truth"]))
    return h


def boxes(ax, groups, positions, colors, width=0.25, strip_below=30):
    """Box per group, or a strip of points + median line if fewer than `strip_below` values. Nothing is dropped."""
    for vals, pos, col in zip(groups, positions, colors):
        vals = np.asarray([v for v in vals if np.isfinite(v)])
        if len(vals) == 0:
            continue
        if len(vals) < strip_below:
            jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * width * 0.8
            ax.scatter(pos + jitter, vals, s=6, color=col, alpha=0.8, linewidths=0, zorder=3)
            ax.hlines(np.median(vals), pos - width / 2, pos + width / 2, color=col, lw=1.2, zorder=4)
        else:
            bp = ax.boxplot(vals, positions=[pos], widths=width, patch_artist=True, manage_ticks=False,
                            flierprops=dict(marker="o", markersize=2, markerfacecolor=col, markeredgecolor=col, alpha=0.6),
                            medianprops=dict(color="black", lw=1), whiskerprops=dict(color=col),
                            capprops=dict(color=col), boxprops=dict(edgecolor=col))
            bp["boxes"][0].set_facecolor(col)
            bp["boxes"][0].set_alpha(0.6)


def cat_xaxis(ax, every=1):
    ax.set_xticks(range(len(CLUSTERS)))
    ax.set_xticklabels([c if i % every == 0 else "" for i, c in enumerate(CLUSTERS)])


XPOS = {c: i for i, c in enumerate(CLUSTERS)}


def present(cases):
    return [c for c in CASES_ORDER if c in set(cases)] + sorted(set(cases) - set(CASES_ORDER))


########################################################################################################################
# Loading: run table (EvaluateMarkov) + per-file extraction
########################################################################################################################

def load_runs(folders, reuse, no_chronology):
    path = OUT / "markov_runs.csv"
    if reuse and path.exists():
        runs = pd.read_csv(path, low_memory=False)
    else:
        entries = []
        for folder in folders:
            e, _ = em.load_entries(folder, analysis=True)
            entries += e
        if not entries:
            sys.exit(f"No MK-*.sqlite files in {folders}")
        em.attach_regret(entries)
        if not no_chronology:
            em.attach_chronology(entries)
        runs = em.runs_dataframe(entries)
        runs.to_csv(path, index=False)
    runs["case"] = runs["dataset"]
    runs["tm"] = [("original" if pd.isna(s) or s in ("None", "") else f"shiftTM{int(float(s))}")
                  + ("" if pd.isna(p) else f"-perturbTM{float(p):g}") for s, p in zip(runs["shift_tm"], runs["perturb_tm"])]
    runs["sd"] = pd.to_numeric(runs["stretch_demand"], errors="coerce").fillna(1.0)
    runs["clusters"] = pd.to_numeric(runs["clusters"], errors="coerce")
    runs["original_ref"] = runs["reference"].astype(str) == "original"
    runs["optimal"] = runs["termination_condition"] == "optimal"
    runs["method"] = runs["edge_handling"]
    return runs


def select(runs, include_perturbed):
    """Optimal runs of the standard TM variants (perturbed TMs only with --include-perturbed)."""
    r = runs[runs["optimal"]]
    if not include_perturbed:
        r = r[~r["tm"].str.contains("perturb")]
    return r


def truth_subcases(runs):
    t = runs[(runs["kind"] == "main") & (runs["method"] == "Truth") & ~runs["original_ref"] & runs["clusters"].notna()]
    return set(zip(t["case"], t["tm"], t["clusters"].astype(int), t["sd"]))


def vs_truth(runs, kind, column, signed=True):
    """Per non-Truth run of `kind`: 100 (x - x_Truth) / x_Truth vs the Truth run of the same kind and sub-case."""
    sub = runs[(runs["kind"] == kind) & ~runs["original_ref"] & runs["clusters"].notna()]
    truth = sub[sub["method"] == "Truth"].set_index(KEYS)[column]
    truth = truth[~truth.index.duplicated()]
    out = sub[sub["method"].isin(METHODS)][KEYS + ["method", column]].copy()
    out["ref"] = [truth.get(tuple(k), np.nan) for k in out[KEYS].itertuples(index=False)]
    out["value"] = 100 * (out[column] - out["ref"]) / out["ref"]
    if not signed:
        out["value"] = out["value"].abs()
    return out.dropna(subset=["value"])


def _table(conn, name, cols=None):
    try:
        df = pd.read_sql(f'SELECT * FROM "{name}"', conn)
    except Exception:
        return None
    df = df.drop(columns=["index"], errors="ignore")
    if cols is not None:
        idx = [c for c in df.columns if c != "values"]
        df = df.rename(columns=dict(zip(idx, cols)))
    return df


def _per_unit(conn, name):
    df = _table(conn, name, ["u"])
    return df.set_index("u")["values"].astype(float) if df is not None and len(df) else pd.Series(dtype=float)


def _active(conn):
    exis, inv = _per_unit(conn, "pExisUnits"), _per_unit(conn, "vGenInvest")
    therm = _table(conn, "thermalGenerators")
    units = therm.iloc[:, -1].tolist() if therm is not None else []
    return [g for g in units if exis.get(g, 0) + inv.get(g, 0) > 0.5]


def _hindex(conn):
    h = _table(conn, "hindex", ["p", "rp", "k"]).sort_values("p").reset_index(drop=True)
    nk = h["k"].nunique()
    h["day"] = np.arange(len(h)) // nk
    return h, nk


def _spells(on, day, need, kind):
    """Rounded-schedule min-up ('up') / min-down ('down') shortfalls in hours, assigned to the day of the hour that ends
    the spell. Spells cut by the horizon start / end are not counted."""
    T, target, rows, t = len(on), 1 if kind == "up" else 0, [], 0
    while t < T:
        if on[t] != target:
            t += 1
            continue
        s = t
        while t < T and on[t] == target:
            t += 1
        if s > 0 and t < T and t - s < need:
            rows.append((int(day[t]), need - (t - s)))
    return rows


def _storage(conn, h, day_rps):
    level = _table(conn, "vStIntraRes", ["rp", "k", "u"])
    if level is None or level.empty:
        return None
    ks = sorted(h["k"].unique())
    k1, kl = ks[0], ks[-1]
    exis, inv = _per_unit(conn, "pExisUnits"), _per_unit(conn, "vGenInvest")
    units = sorted(level["u"].unique())
    n = pd.Series({u: exis.get(u, 0) + inv.get(u, 0) for u in units})
    cap = _per_unit(conn, "pMaxReserve").reindex(units) * n
    lo = _per_unit(conn, "pMinReserve").reindex(units).fillna(0) * n
    units = [u for u in units if cap.get(u, 0) > 1e-9]
    if not units:
        return None
    L = level.pivot_table(index=["rp", "k"], columns="u", values="values")[units]
    rps = L.xs(k1, level="k").index

    def at_k1(name):
        df = _table(conn, name, ["rp", "k", "u"])
        if df is None or df.empty:
            return pd.DataFrame(0.0, index=rps, columns=units)
        df = df[(df["k"] == k1) & df["u"].isin(units)].pivot(index="rp", columns="u", values="values")
        return df.reindex(index=rps, columns=units).fillna(0)

    wk = _per_unit(conn, "pWeight_k").get(k1, 1.0)
    ed = _per_unit(conn, "pDisEffic").reindex(units).fillna(1)
    ec = _per_unit(conn, "pChEffic").reindex(units).fillna(1)
    implied = L.xs(k1, level="k") + at_k1("vGenP") * wk / ed - at_k1("vConsump") * wk * ec - at_k1("pStorageInflows") + at_k1("vStorageSpillage")
    end = L.xs(kl, level="k")
    curves = {rp: L.xs(rp, level="rp").reindex(ks) for rp in rps}
    rows = []
    for u in units:
        c, actual_end = cap[u], None
        for d, rp in enumerate(day_rps):
            traj = curves[rp][u].to_numpy()
            offset = 0.0 if d == 0 else actual_end - implied.at[rp, u]
            if d > 0:
                jump = implied.at[rp, u] - end.at[day_rps[d - 1], u]
                breach = bool(((traj + offset) < lo[u] - ST_TOL * c).any() or ((traj + offset) > c + ST_TOL * c).any())
                rows.append((d, u, rp, day_rps[d - 1], 100 * jump / c, breach))
            actual_end = traj[-1] + offset
    return {"jumps": pd.DataFrame(rows, columns=["day", "u", "rp", "prev_rp", "jump_pct", "carried_breach"]), "capacity": cap[units]}


def _extract_rp(path):
    out = {}
    with closing(em._connect_ro(path)) as conn:
        h, nk = _hindex(conn)
        day_rps = h["rp"].tolist()[::nk]
        out["day_rps"] = day_rps
        out["nonbinary"] = _table(conn, "mk_nonbinary_values")
        out["viol"] = _table(conn, "mk_feasibility_violations")
        units = _active(conn)
        idx = pd.MultiIndex.from_arrays([h["rp"], h["k"]])

        def wide(v):
            d = _table(conn, v, ["rp", "k", "g"])
            return d.pivot_table(index=["rp", "k"], columns="g", values="values").reindex(index=idx, columns=units).fillna(0).to_numpy()

        u, su, sd, p1 = (wide(v) for v in ["vCommit", "vStartup", "vShutdown", "vGenP1"])
        day = h["day"].to_numpy()
        T = len(day)
        pmax = _per_unit(conn, "pMaxProd").reindex(units).to_numpy()
        pmin = _per_unit(conn, "pMinProd").reindex(units).to_numpy()
        ru = _per_unit(conn, "pRampUp").reindex(units).to_numpy()
        rd = _per_unit(conn, "pRampDw").reindex(units).to_numpy()
        ut = _per_unit(conn, "pMinUpTime").reindex(units).fillna(1).astype(int).to_numpy()
        dt = _per_unit(conn, "pMinDownTime").reindex(units).fillna(1).astype(int).to_numpy()
        span = pmax - pmin
        rows = []
        for name, nxt in (("ramp_up", False), ("ramp_down", False), ("max_out_shutdown", True)):
            r = np.zeros_like(p1)
            if name == "ramp_up":
                r[1:] = p1[1:] - p1[:-1] - u[1:] * ru
            elif name == "ramp_down":
                r[1:] = -(p1[1:] - p1[:-1] + u[:-1] * rd)
            else:
                r[:-1] = p1[:-1] - span * (u[:-1] - sd[1:])
            t, g = np.nonzero(r > TOL)
            d = day[np.minimum(t + 1, T - 1)] if nxt else day[t]
            rows.append(pd.DataFrame({"day": d, "g": np.array(units, dtype=object)[g], "check": name, "residual_phys": 100 * r[t, g] / pmax[g]}))
        ur = (u >= 0.5).astype(int)
        for j, g in enumerate(units):
            for kind, need in (("up", ut[j]), ("down", dt[j])):
                if need > 1:
                    for d, short in _spells(ur[:, j], day, need, kind):
                        rows.append(pd.DataFrame({"day": [d], "g": [g], "check": [f"min_{kind}_rounded"], "residual_phys": [float(short)]}))
        phys = pd.concat(rows, ignore_index=True)
        out["phys"] = phys.groupby(["day", "g", "check"]).agg(max_residual_phys=("residual_phys", "max")).reset_index()
        b = np.nonzero((np.arange(T) % nk == 0) & (np.arange(T) > 0))[0]
        both_on = (u[b] > 0.5) & (u[b - 1] > 0.5)
        dp = p1[b] - p1[b - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            up_ratio = np.where(both_on & (ru < span), dp / ru, np.nan)
            dn_ratio = np.where(both_on & (rd < span), -dp / rd, np.nan)
        out["ramp"] = {"units": len(units), "units_ru_below_span": int((ru < span).sum()), "units_rd_below_span": int((rd < span).sum()),
                       "pairs": int(b.size * len(units)), "pairs_both_on": int(both_on.sum()),
                       "pairs_bindable_up": int((both_on & (ru < span)).sum()), "pairs_bindable_down": int((both_on & (rd < span)).sum()),
                       "max_up_ratio": float(np.nanmax(up_ratio)) if np.isfinite(up_ratio).any() else np.nan,
                       "max_down_ratio": float(np.nanmax(dn_ratio)) if np.isfinite(dn_ratio).any() else np.nan,
                       "max_abs_dp_pct_pmax": float(np.nanmax(np.abs(dp) / pmax * 100)) if dp.size else np.nan}
        last_k = sorted(h["k"].unique())[-1]
        cw = _table(conn, "vCommit", ["rp", "k", "g"])
        cw = cw[(cw["k"] == last_k) & cw["g"].isin(units)].pivot(index="rp", columns="g", values="values")
        out["last_u"] = (cw >= 0.5).astype(int)
        out["storage"] = _storage(conn, h, day_rps)
    return out


def _extract_truth(path):
    with closing(em._connect_ro(path)) as conn:
        units = _active(conn)
        cw = _table(conn, "vCommit", ["rp", "k", "g"])
        cw = cw[cw["g"].isin(units)].pivot(index="k", columns="g", values="values").sort_index()
        return {"last_u_days": (cw.iloc[23::24] >= 0.5).astype(int).reset_index(drop=True)}  # all RPs are 24 h


def _extract_job(args):
    path, method, target = args
    try:
        data = _extract_truth(path) if method == "Truth" else _extract_rp(path)
    except Exception as e:
        return path, repr(e)
    with open(target, "wb") as fh:
        pickle.dump(data, fh)
    return path, None


def extract(runs, workers):
    """Per-file pickles for the main / operational runs (RP runs and their Truth runs)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = runs[runs["kind"].isin(["main", "operational"]) & ~runs["original_ref"] & runs["clusters"].notna()
                & runs["method"].isin(METHODS + ["Truth"])]
    jobs = [(f, m, CACHE / (os.path.basename(f) + ".pkl")) for f, m in zip(todo["file"], todo["method"])]
    jobs = [j for j in jobs if not j[2].exists()]
    print(f"Extracting {len(jobs)} result file(s) with {workers} processes ...")
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (path, err) in enumerate(ex.map(_extract_job, jobs, chunksize=2), 1):
            if err:
                print(f"  extraction failed for {path}: {err}")
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)


def cached(runs, methods, kind, keep=None):
    """Extracted data of the selected runs (with meta), optionally restricted to the sub-cases in `keep`."""
    sub = runs[runs["method"].isin([methods] if isinstance(methods, str) else methods) & (runs["kind"] == kind)
               & ~runs["original_ref"] & runs["clusters"].notna()]
    out = []
    for _, r in sub.iterrows():
        key = (r["case"], r["tm"], int(r["clusters"]), r["sd"])
        p = CACHE / (os.path.basename(r["file"]) + ".pkl")
        if (keep is None or key in keep) and p.exists():
            with open(p, "rb") as fh:
                d = pickle.load(fh)
            d["meta"] = dict(zip(KEYS, key)) | {"method": r["method"], "kind": kind, "file": r["file"]}
            out.append(d)
    return out


########################################################################################################################
# R1 - fractional UC flags
########################################################################################################################

def r1(ctx):
    import matplotlib.pyplot as plt
    runs, keep = ctx["runs"], ctx["keep"]
    VARS = ["vCommit", "vStartup", "vShutdown"]
    tables = []
    for kind in ("main", "operational"):
        rows = []
        for d in cached(runs, "Markov", kind, keep):
            v = d["nonbinary"]
            if v is not None and len(v):
                rows.append(v.assign(**{k: d["meta"][k] for k in KEYS}))
        if not rows:
            print(f"  R1 {kind}: no fractional values")
            continue
        df = pd.concat(rows, ignore_index=True)
        df["in_window"] = df["in_window"].astype(bool)
        mk = runs[(runs["method"] == "Markov") & (runs["kind"] == kind) & ~runs["original_ref"]]
        mk = mk[[k in keep for k in zip(mk["case"], mk["tm"], mk["clusters"].astype(int), mk["sd"])]]
        cases = present(df["case"])
        sfx = "" if kind == "main" else "_operational"
        bins = np.linspace(0, 1, 21)
        fig, axes = plt.subplots(len(cases), 3, figsize=(WIDTH_IN, 1.8 * len(cases) + 0.4), sharex=True, sharey=True, squeeze=False)
        hist = []
        for i, case in enumerate(cases):
            for j, var in enumerate(VARS):
                ax = axes[i, j]
                v = df[(df["case"] == case) & (df["var"] == var)]["value"]
                n_all = mk.loc[mk["case"] == case, f"nb_{var}_values"].sum()
                if len(v):
                    h, _ = np.histogram(v, bins=bins)
                    ax.bar(bins[:-1], 100 * h / len(v), width=0.05, align="edge", color=COLOR["Markov"], alpha=0.8, edgecolor="white", lw=0.3)
                    hist += [{"case": case, "var": var, "bin_lo": lo, "count": c, "share_pct": 100 * c / len(v)} for lo, c in zip(bins[:-1], h)]
                ax.text(0.5, 0.95, f"n = {len(v):,}\n{100 * len(v) / n_all if n_all else 0:.3f} % of all {var}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=6.5)
                ax.set_ylim(0, 28)
                ax.grid(axis="y", color="#e5e4e0", lw=0.5)
                if i == 0:
                    ax.set_title(var)
                if j == 0:
                    ax.set_ylabel(f"{case}\n% of fractional values")
                if i == len(cases) - 1:
                    ax.set_xlabel("value")
        fig.tight_layout()
        save(fig, f"R1a_fractional_value_histogram{sfx}", pd.DataFrame(hist))
        cmap = plt.get_cmap("viridis")
        fig, axes = plt.subplots(1, len(cases), figsize=(WIDTH_IN, 2.6), sharey=True, squeeze=False)
        ecdf = []
        for ax, case in zip(axes[0], cases):
            for k, n in enumerate(CLUSTERS):
                dl = np.sort(df[(df["case"] == case) & (df["clusters"] == n)]["delta"].to_numpy())
                if len(dl):
                    y = np.arange(1, len(dl) + 1) / len(dl)
                    ax.step(dl, y, where="post", color=cmap(k / (len(CLUSTERS) - 1)), lw=1, label=f"{n} RPs (n={len(dl)})")
                    ecdf.append(pd.DataFrame({"case": case, "clusters": n, "delta": dl, "ecdf": y}))
            ax.set_xlim(0, 0.5)
            ax.set_title(case)
            ax.set_xlabel("distance to nearest integer")
            ax.grid(color="#e5e4e0", lw=0.5)
            ax.legend(frameon=False, fontsize=6, loc="lower right")
        axes[0][0].set_ylabel("cumulative share of\nfractional values")
        fig.tight_layout()
        save(fig, f"R1b_fractional_distance_ecdf{sfx}", pd.concat(ecdf) if ecdf else None)
        for case in cases:
            for n in CLUSTERS:
                rn = mk[(mk["case"] == case) & (mk["clusters"] == n)]
                if rn.empty:
                    continue
                d = df[(df["case"] == case) & (df["clusters"] == n)]
                dl = d["delta"]
                tables.append({"case": case, "kind": kind, "clusters": n, "n_runs": len(rn), "n_flag_values": int(rn["nb_all_values"].sum()),
                               "n_fractional": len(d), "pct_fractional": 100 * len(d) / rn["nb_all_values"].sum(),
                               "delta_mean": dl.mean(), "delta_median": dl.median(), "delta_p90": dl.quantile(0.9) if len(dl) else np.nan,
                               "delta_max": dl.max(), "share_delta_below_0.05_pct": 100 * (dl < 0.05).mean() if len(dl) else np.nan,
                               "share_delta_above_0.25_pct": 100 * (dl > 0.25).mean() if len(dl) else np.nan,
                               "share_in_window_pct": 100 * d["in_window"].mean() if len(dl) else np.nan,
                               "n_outside_window": int((~d["in_window"]).sum())})
    pd.DataFrame(tables).to_csv(OUT / "R1_fractional_summary.csv", index=False)


########################################################################################################################
# R2 - violation magnitudes
########################################################################################################################

R2_CHECKS = [("logic", "Logic\n[flag]"), ("ramp_up", "Ramp up\n[% Pmax]"), ("ramp_down", "Ramp down\n[% Pmax]"),
             ("max_out_shutdown", "Output bef.\nshutdown\n[% Pmax]"), ("min_up_rounded", "Min. up,\nrounded [h]"),
             ("min_down_rounded", "Min. down,\nrounded [h]")]


def r2(ctx):
    import matplotlib.pyplot as plt
    runs, keep = ctx["runs"], ctx["keep"]
    summary, ramps = [], []
    for kind in ("main", "operational"):
        rows = []
        for d in cached(runs, METHODS, kind, keep):
            meta = {k: d["meta"][k] for k in KEYS + ["method"]}
            st = d["viol"]
            st = st[st["day"] >= 1] if st is not None and len(st) else pd.DataFrame(columns=["day", "g", "check", "max_residual"])
            for check in ("logic", "fractional", "min_up", "min_down"):
                p = st[st["check"] == check]
                rows.append(pd.DataFrame({"check": check, "residual": p["max_residual"].astype(float), "unit": "flag"}).assign(**meta))
            ph = d["phys"]
            ph = ph[ph["day"] >= 1]
            for check in ("ramp_up", "ramp_down", "max_out_shutdown", "min_up_rounded", "min_down_rounded"):
                p = ph[ph["check"] == check]
                rows.append(pd.DataFrame({"check": check, "residual": p["max_residual_phys"], "unit": "h" if "min_" in check else "% Pmax"}).assign(**meta))
            ramps.append(meta | {"kind": kind} | d["ramp"])
        if not rows:
            continue
        inst = pd.concat(rows, ignore_index=True)
        sfx = "" if kind == "main" else "_operational"
        inst.to_csv(OUT / f"R2a_violation_magnitude{sfx}.csv", index=False)
        cases = present(inst["case"])
        variants = [[(c, "original") for c in cases]] + ([[("RTS-GMLC", "shiftTM2")]] if ((inst["case"] == "RTS-GMLC") & (inst["tm"] == "shiftTM2")).any() else [])
        for case_tms in variants:
            fig, axes = plt.subplots(len(case_tms), len(R2_CHECKS), figsize=(WIDTH_IN, 1.9 * len(case_tms) + 0.9), squeeze=False)
            for i, (case, tm) in enumerate(case_tms):
                for j, (check, title) in enumerate(R2_CHECKS):
                    ax = axes[i, j]
                    cell = inst[(inst["case"] == case) & (inst["tm"] == tm) & (inst["check"] == check)]
                    if cell.empty:
                        ax.text(0.5, 0.5, "none", ha="center", va="center", transform=ax.transAxes, fontsize=7, color="gray")
                    boxes(ax, [cell.loc[cell["method"] == m, "residual"] for m in METHODS], [0, 1, 2], [COLOR[m] for m in METHODS], width=0.6)
                    for k, m in enumerate(METHODS):
                        n = int((cell["method"] == m).sum())
                        ax.text(k, 1.0, f"{n / 1000:.0f}k" if n >= 10000 else (f"{n / 1000:.1f}k" if n >= 1000 else f"{n}"),
                                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5, color=COLOR[m])
                    ax.set_xlim(-0.6, 2.6)
                    ax.set_xticks([])
                    ax.set_ylim(bottom=0)
                    ax.grid(axis="y", color="#e5e4e0", lw=0.5)
                    if i == 0:
                        ax.set_title(title, pad=10)
                    if j == 0:
                        ax.set_ylabel(case if tm == "original" else f"{case}\n{tm}")
            fig.legend(handles=method_handles(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            save(fig, f"R2a_violation_magnitude{sfx}" + ("" if case_tms[0][1] == "original" else "_RTS-GMLC_shiftTM2"))
        # R2b: share of infeasible transitions by RPs, as solved / after rounding (original TM)
        sub = runs[(runs["kind"] == kind) & (runs["tm"] == "original") & runs["method"].isin(METHODS) & ~runs["original_ref"]]
        sub = sub[[k in keep for k in zip(sub["case"], sub["tm"], sub["clusters"].astype(int), sub["sd"])]]
        fig, axes = plt.subplots(2, len(cases), figsize=(WIDTH_IN, 3.8), sharex=True, squeeze=False)
        data = []
        for j, case in enumerate(cases):
            for i, (col, lab) in enumerate((("feas_infeasible_pct", "as solved"), ("feas_infeasible_rounded_pct", "after rounding"))):
                ax = axes[i, j]
                for m in METHODS:
                    g = sub[(sub["case"] == case) & (sub["method"] == m)].groupby("clusters")[col].agg(["mean", "min", "max", "size"])
                    x = [XPOS[int(c)] for c in g.index]
                    ax.plot(x, g["mean"], color=COLOR[m], marker="o", ms=2.5, lw=1.1)
                    ax.fill_between(x, g["min"], g["max"], color=COLOR[m], alpha=0.15, lw=0)
                    data.append(g.reset_index().assign(case=case, method=m, metric=col))
                cat_xaxis(ax)
                ax.set_ylim(bottom=0)
                ax.grid(color="#e5e4e0", lw=0.5)
                if i == 0:
                    ax.set_title(case)
                if j == 0:
                    ax.set_ylabel(f"Infeasible transitions [%]\n{lab}")
                if i == 1:
                    ax.set_xlabel("Number of RPs")
        fig.legend(handles=method_handles(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        save(fig, f"R2b_violation_share_by_rps{sfx}", pd.concat(data))
        k_runs = runs[(runs["kind"] == kind) & runs["method"].isin(METHODS) & ~runs["original_ref"]]
        k_runs = k_runs[[k in keep for k in zip(k_runs["case"], k_runs["tm"], k_runs["clusters"].astype(int), k_runs["sd"])]]
        for (case, tm, m), g in k_runs.groupby(["case", "tm", "method"]):
            checked = g["feas_instances"].sum()
            for check in [c for c, _ in R2_CHECKS] + ["fractional", "min_up", "min_down"]:
                n = g[f"feas_{check}_instances"].sum() if f"feas_{check}_instances" in g else np.nan
                r = inst[(inst["case"] == case) & (inst["tm"] == tm) & (inst["method"] == m) & (inst["check"] == check)]["residual"]
                summary.append({"case": case, "tm": tm, "kind": kind, "method": m, "check": check, "rounded": check.endswith("_rounded"),
                                "n_runs": len(g), "n_checked_instances": int(checked), "n_violating": n,
                                "pct_violating": 100 * n / checked if checked else np.nan,
                                "residual_unit": "flag" if check in ("logic", "fractional", "min_up", "min_down") else ("h" if "min_" in check else "% Pmax"),
                                "mean_residual": r.mean(), "max_residual": r.max()})
    pd.DataFrame(summary).to_csv(OUT / "R2_violation_summary.csv", index=False)
    pd.DataFrame(ramps).to_csv(OUT / "R2_ramp_bindability.csv", index=False)


########################################################################################################################
# R3 - storage along the chronology
########################################################################################################################

def r3(ctx):
    import matplotlib.pyplot as plt
    runs, keep = ctx["runs"], ctx["keep"]
    jmp_by_kind = {}
    for kind in ("operational", "main"):
        rows = [d["storage"]["jumps"].assign(**{k: d["meta"][k] for k in KEYS + ["method"]})
                for d in cached(runs, METHODS, kind, keep) if d["storage"] is not None]
        if not rows:
            continue
        jmp = pd.concat(rows, ignore_index=True)
        jmp["abs_jump"] = jmp["jump_pct"].abs()
        jmp["rp_changes"] = jmp["rp"] != jmp["prev_rp"]
        jmp_by_kind[kind] = jmp
        jmp.to_csv(OUT / f"R3b_storage_boundary_jumps_{kind}.csv", index=False)
        panels = [(c, t) for c, t in [("RTS-GMLC", "original"), ("RTS-GMLC", "shiftTM2")] + [(c, "original") for c in present(jmp["case"]) if c != "RTS-GMLC"]
                  if ((jmp["case"] == c) & (jmp["tm"] == t)).any()]
        for split in (False, True):
            fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH_IN, 2.6), sharey=True, squeeze=False)
            for ax, (case, tm) in zip(axes[0], panels):
                cell = jmp[(jmp["case"] == case) & (jmp["tm"] == tm)]
                for m in METHODS:
                    groups = [(cell["method"] == m) & ~cell["rp_changes"], (cell["method"] == m) & cell["rp_changes"]] if split else [cell["method"] == m]
                    for g, ls in zip(groups, ["--", "-"] if split else ["-"]):
                        v = np.sort(cell.loc[g, "abs_jump"].to_numpy())
                        if len(v):
                            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COLOR[m], ls=ls, lw=1)
                ax.set_xscale("symlog", linthresh=1)
                ax.set_xlim(0, cell["abs_jump"].max() * 1.2)
                ax.axvline(100, color="gray", lw=0.5, ls=":")
                ax.set_title(case if tm == "original" else f"{case}, {TM_LABEL.get(tm, tm)}")
                ax.set_xlabel("|Boundary jump| [% of capacity]")
                ax.grid(color="#e5e4e0", lw=0.5)
            axes[0][0].set_ylabel("Cumulative share of\n(boundary, unit) pairs")
            from matplotlib.lines import Line2D
            h = method_handles() + ([Line2D([], [], color="k", ls="--", label="RP follows itself"), Line2D([], [], color="k", ls="-", label="RP changes")] if split else [])
            fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1))
            fig.tight_layout(rect=(0, 0.08, 1, 1))
            save(fig, "R3b_storage_boundary_jump_distribution" + ("_split" if split else "") + ("" if kind == "operational" else "_main"))
    if "operational" in jmp_by_kind:
        _r3a(runs, jmp_by_kind["operational"])
    rows = []
    for kind, jmp in jmp_by_kind.items():
        for (case, tm, m, n), g in jmp.groupby(["case", "tm", "method", "clusters"]):
            rows.append({"case": case, "tm": tm, "kind": kind, "method": m, "clusters": n, "n_runs": g["sd"].nunique(), "n_pairs": len(g),
                         "pct_jump_above_1pct": 100 * (g["abs_jump"] > 1).mean(), "mean_abs_jump_pct": g["abs_jump"].mean(),
                         "max_abs_jump_pct": g["abs_jump"].max(), "pct_jump_above_100pct": 100 * (g["abs_jump"] > 100).mean(),
                         "n_pairs_carried_breach": int(g["carried_breach"].sum()),
                         "n_boundaries_carried_breach": int(g[g["carried_breach"]].groupby(["sd", "day"]).ngroups),
                         "same_rp_mean_abs_jump_pct": g.loc[~g["rp_changes"], "abs_jump"].mean(),
                         "rp_change_mean_abs_jump_pct": g.loc[g["rp_changes"], "abs_jump"].mean()})
    pd.DataFrame(rows).to_csv(OUT / "R3_storage_summary.csv", index=False)


def _chron_level(path, unit, truth):
    with closing(em._connect_ro(path)) as conn:
        h, nk = _hindex(conn)
        lv = _table(conn, "vStIntraRes", ["rp", "k", "u"])
        lv = lv[lv["u"] == unit]
        n = _per_unit(conn, "pExisUnits").get(unit, 0) + _per_unit(conn, "vGenInvest").get(unit, 0)
        cap = _per_unit(conn, "pMaxReserve")[unit] * n
        if truth:
            s = lv.set_index("k")["values"].reindex([p.replace("h", "k") for p in h["p"]]).to_numpy()
        else:
            s = h.merge(lv, on=["rp", "k"], how="left")["values"].to_numpy()
        return 100 * s / cap, h["rp"].tolist()[::nk]


def _r3a(runs, jmp):
    """One operational sub-case per case study with storage: 7 RPs (else the closest), demand 1.0 (else the highest)."""
    import matplotlib.pyplot as plt
    picks = []
    for case in present(jmp["case"]):
        cand = jmp[(jmp["case"] == case) & (jmp["tm"] == "original")][["clusters", "sd"]].drop_duplicates()
        if cand.empty:
            continue
        cand = cand.assign(d=(cand["clusters"] - 7).abs()).sort_values(["d", "sd"], ascending=[True, False])
        picks.append((case, int(cand.iloc[0]["clusters"]), float(cand.iloc[0]["sd"])))
    if not picks:
        return
    fig, axes = plt.subplots(2 * len(picks), 1, figsize=(WIDTH_IN, 3.3 * len(picks)), gridspec_kw={"height_ratios": [3, 1.3] * len(picks)})
    data, notes = [], []
    for i, (case, n, sd) in enumerate(picks):
        files = runs[(runs["case"] == case) & (runs["tm"] == "original") & (runs["clusters"] == n) & (runs["sd"] == sd)
                     & (runs["kind"] == "operational") & ~runs["original_ref"]].set_index("method")["file"].to_dict()
        if "Truth" not in files:
            continue
        with closing(em._connect_ro(files["Truth"])) as conn:
            intra = _table(conn, "intraStorageUnits").iloc[:, -1].tolist()
            nunits = _per_unit(conn, "pExisUnits").add(_per_unit(conn, "vGenInvest"), fill_value=0)
            caps = (_per_unit(conn, "pMaxReserve") * nunits.reindex(_per_unit(conn, "pMaxReserve").index).fillna(0)).reindex(intra)
        caps = caps[caps > 1e-9].sort_values(ascending=False)
        unit = caps.index[0]
        _, day_rps = _chron_level(files["Markov"], unit, False)
        changes = np.array([day_rps[d] != day_rps[d - 1] for d in range(1, len(day_rps))])
        start = next((s for s in range(1, len(day_rps) - 14) if changes[s - 1:s + 13].sum() >= 4), 1)
        hours = np.arange(start * 24, (start + 14) * 24)
        ax, axj = axes[2 * i], axes[2 * i + 1]
        axj.sharex(ax)
        for m in ["Truth", "Cyclic", "Markov", "NoEnf"]:
            if m not in files:
                continue
            s, _ = _chron_level(files[m], unit, m == "Truth")
            ax.plot(hours / 24 + 1, s[hours], color=COLOR[m], lw=1.1 if m != "NoEnf" else 0.9, ls="--" if m == "Truth" else (":" if m == "NoEnf" else "-"))
            data.append(pd.DataFrame({"case": case, "unit": unit, "method": m, "hour": hours + 1, "level_pct": s[hours]}))
        for d in range(start, start + 14):
            ax.axvline(d + 1, color="gray", lw=0.4, alpha=0.6)
            ax.text(d + 1.5, 1.01, day_rps[d].replace("rp0", "").replace("rp", ""), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6)
        ax.set_ylabel("State of charge\n[% of capacity]")
        ax.set_xlim(start + 1, start + 15)
        ax.tick_params(labelbottom=False)
        ax.set_title(f"{case}, {n} RPs, demand {sd:g} - unit {unit}", pad=12)
        ax.grid(axis="y", color="#e5e4e0", lw=0.5)
        j = jmp[(jmp["case"] == case) & (jmp["tm"] == "original") & (jmp["clusters"] == n) & (jmp["sd"] == sd) & (jmp["u"] == unit)]
        for k, m in enumerate(METHODS):
            jj = j[(j["method"] == m) & j["day"].between(start, start + 13)]
            axj.bar(jj["day"] + 1 + (k - 1) * 0.25, jj["jump_pct"], width=0.25, color=COLOR[m], alpha=0.8)
        axj.axhline(0, color="gray", lw=0.5)
        axj.set_ylabel("Boundary jump\n[% of cap.]")
        axj.set_xlabel("Day of the horizon (bars at the boundary into that day)")
        axj.grid(axis="y", color="#e5e4e0", lw=0.5)
        notes.append({"case": case, "clusters": n, "sd": sd, "unit": unit, "capacity": caps.iloc[0], "window_days_1based": f"{start + 1}-{start + 14}",
                      "rp_changes": int(changes[start - 1:start + 13].sum()), "other_units": ", ".join(f"{u} ({c:.3g})" for u, c in caps.iloc[1:].items())})
    fig.legend(handles=method_handles(truth=True), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "R3a_storage_trajectory", pd.concat(data))
    pd.DataFrame(notes).to_csv(OUT / "R3a_selected_units.csv", index=False)


########################################################################################################################
# R5 - TM structure vs benefit of Markov
########################################################################################################################

def r5(ctx):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import spearmanr
    runs, keep = ctx["runs"], ctx["keep"]
    dev = vs_truth(runs, "operational", "vShutdown").pivot_table(index=KEYS, columns="method", values="value")
    inv = runs[(runs["kind"] == "invest_regret") & ~runs["original_ref"]].pivot_table(index=KEYS, columns="method", values="regret_pct")
    tm = runs[(runs["kind"] == "main") & (runs["method"] == "Markov") & ~runs["original_ref"]].drop_duplicates(KEYS).set_index(KEYS)[["tm_off_diag_mass", "tm_entropy_norm", "tm_nonzero"]]
    df = pd.DataFrame({f"oper_dev_{m.lower()}_pct": dev.get(m) for m in METHODS}).join(
        pd.DataFrame({f"invest_regret_{m.lower()}_pct": inv.get(m) for m in METHODS}), how="outer").join(tm, how="left")
    df["improvement_oper_pp"] = df["oper_dev_cyclic_pct"].abs() - df["oper_dev_markov_pct"].abs()
    df["improvement_regret_pp"] = df["invest_regret_cyclic_pct"] - df["invest_regret_markov_pct"]
    df = df.reset_index()
    ref = {tuple(d["meta"][k] for k in KEYS): d for d in cached(runs, "Truth", "operational", keep)}
    rows = []
    for d in cached(runs, "Cyclic", "operational", keep):
        key = tuple(d["meta"][k] for k in KEYS)
        day_rps, lu = d["day_rps"], d["last_u"]
        diffs = [(lu.loc[p] != lu.loc[r]).mean() for p, r in zip(day_rps[:-1], day_rps[1:])]
        chg = [x for x, p, r in zip(diffs, day_rps[:-1], day_rps[1:]) if p != r]
        prev = day_rps[-1:] + day_rps[:-1]
        self_share = sum(1 for p, c in zip(prev, day_rps) if p == c) / len(day_rps)
        refm = np.nan
        if key in ref:
            U = ref[key]["last_u_days"].to_numpy(dtype=float)[:len(day_rps)]
            arr = np.array(day_rps)
            typical = {rp: U[arr == rp].mean(axis=0) for rp in set(day_rps)}
            refm = float(np.mean([np.abs(U[t - 1] - typical[day_rps[t]]).mean() for t in range(1, len(day_rps))]))
        rows.append(dict(zip(KEYS, key)) | {"self_transition_share": self_share, "cyclic_endstate_mismatch": float(np.mean(diffs)),
                                            "cyclic_endstate_mismatch_rp_change": float(np.mean(chg)) if chg else np.nan, "ref_endstate_mismatch": refm})
    df = df.merge(pd.DataFrame(rows), on=KEYS, how="left")
    df = df[[k in keep for k in zip(df["case"], df["tm"], df["clusters"].astype(int), df["sd"])]]
    df.to_csv(OUT / "R5_tm_metrics.csv", index=False)
    XS = [("tm_off_diag_mass", "Off-diagonal\nmass"), ("tm_entropy_norm", "Entropy rate\n(normalized)"),
          ("self_transition_share", "Weighted self-\ntransition share"), ("cyclic_endstate_mismatch", "End-state mismatch\n(Cyclic run)")]
    cases = present(df["case"])

    def rho(x, y):
        lines = []
        for label, part in [("all", df)] + [(c.split("-")[0], df[df["case"] == c]) for c in cases]:
            p = part[[x, y]].dropna()
            if len(p) > 3:
                r, pv = spearmanr(p[x], p[y])
                lines.append(f"{label}: ρ={r:+.2f}{'*' if pv < 0.05 else ''}")
        return "\n".join(lines)

    sizes = {3: 6, 5: 10, 7: 14, 14: 22, 21: 30, 28: 40}
    rho_rows = []
    for y, ylabel, name, band in (("improvement_oper_pp", "|dev. Cyclic| - |dev. Markov|\n[pp of shutdowns]", "R5a_improvement_vs_tm_metric", False),
                                  ("improvement_regret_pp", "Invest-regret Cyclic - Markov\n[% of ref. objective]", "R5b_regret_vs_tm_metric", True)):
        fig, axes = plt.subplots(1, len(XS), figsize=(WIDTH_IN, 3.6), sharey=True)
        for ax, (x, xl) in zip(axes, XS):
            for case in cases:
                for t, col in TM_COLOR.items():
                    p = df[(df["case"] == case) & (df["tm"] == t)]
                    if len(p):
                        ax.scatter(p[x], p[y], marker=CASE_MARKER.get(case, "D"), s=p["clusters"].map(sizes), facecolors="none", edgecolors=col, lw=0.6, alpha=0.8)
            ax.axhline(0, color="gray", lw=0.6)
            if band:
                ax.axhspan(-100 * MIP_GAP, 100 * MIP_GAP, color="#bbbbbb", alpha=0.35, lw=0)
            ax.set_xlabel(xl)
            ax.grid(color="#e5e4e0", lw=0.5)
            txt = rho(x, y)
            rho_rows.append({"y": y, "x": x, "spearman": txt.replace("\n", " | ")})
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=5.5, bbox=dict(facecolor="white", alpha=0.7, lw=0, pad=1))
        axes[0].set_ylabel(ylabel)
        h = [Line2D([], [], marker=CASE_MARKER.get(c, "D"), ls="", mfc="none", mec="k", label=c) for c in cases] + \
            [Line2D([], [], marker="o", ls="", mfc="none", mec=col, label=TM_LABEL[t]) for t, col in TM_COLOR.items() if (df["tm"] == t).any()] + \
            [Line2D([], [], marker="o", ls="", mfc="none", mec="gray", ms=np.sqrt(sizes[n]), label=f"{n} RPs") for n in (3, 7, 28)]
        fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1))
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        save(fig, name)
    for x in ("ref_endstate_mismatch", "cyclic_endstate_mismatch_rp_change"):
        for y in ("improvement_oper_pp", "improvement_regret_pp"):
            rho_rows.append({"y": y, "x": x, "spearman": rho(x, y).replace("\n", " | ")})
    pd.DataFrame(rho_rows).to_csv(OUT / "R5_spearman.csv", index=False)


########################################################################################################################
# R6 - scalability
########################################################################################################################

R6_COLS = [("num_constrs", "Constraints\n[million]", False), ("num_nonzeros", "Nonzeros\n", True), ("tm_nonzero", "Nonzero TM\nentries", False),
           ("solver_max_mem_gb", "Peak solver\nmemory [GB]", True), ("solver_runtime_s", "Solver\ntime [s]", True), ("work_units", "Work\nunits", True)]


def r6(ctx):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    runs = ctx["runs"][~ctx["runs"]["original_ref"] & ctx["runs"]["clusters"].notna()]
    cases = present(runs["case"])

    def fig_a(sub, kind, name):
        fig, axes = plt.subplots(len(cases), len(R6_COLS), figsize=(WIDTH_IN, 2.3 * len(cases) + 0.4), squeeze=False)
        for i, case in enumerate(cases):
            for j, (col, title, log) in enumerate(R6_COLS):
                ax = axes[i, j]
                for m in METHODS + ["Truth"]:
                    if m == "Truth" and col == "tm_nonzero":
                        continue
                    g = sub[(sub["case"] == case) & (sub["method"] == m) & (sub["kind"] == kind)].groupby("clusters")[col].agg(["median", "min", "max"]).dropna()
                    if g.empty:
                        continue
                    if col == "num_constrs":
                        g = g / 1e6
                    x = [XPOS[int(c)] for c in g.index]
                    ax.plot(x, g["median"], color=COLOR[m], ls="--" if m == "Truth" else "-", lw=1.1, marker=None if m == "Truth" else "o", ms=2.5)
                    ax.fill_between(x, g["min"], g["max"], color=COLOR[m], alpha=0.15, lw=0)
                if col == "tm_nonzero":
                    ax.plot(range(len(CLUSTERS)), np.array(CLUSTERS) ** 2, color="lightgray", lw=0.8, zorder=0)
                if log:
                    ax.set_yscale("log")
                cat_xaxis(ax, 2)
                ax.grid(color="#e5e4e0", lw=0.5)
                if i == 0:
                    ax.set_title(title)
                if j == 0:
                    ax.set_ylabel(case)
                if i == len(cases) - 1:
                    ax.set_xlabel("Number of RPs")
        fig.legend(handles=method_handles(truth=True) + [Line2D([], [], color="lightgray", label="RPs² (full TM)")], loc="lower center", ncol=5,
                   frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        save(fig, name)

    for kind in ("main", "operational"):
        fig_a(runs[runs["tm"] == "original"], kind, f"R6a_scalability_{kind}")
        shifted = runs[(runs["case"] == "RTS-GMLC") & runs["tm"].isin(["shiftTM1", "shiftTM2"])]
        if not shifted.empty:
            cases_bak = cases
            cases = ["RTS-GMLC"]
            fig_a(shifted, kind, f"R6a_scalability_{kind}_RTS-GMLC_shiftedTM")
            cases = cases_bak
    cols = ["num_nonzeros", "solver_max_mem_gb", "solver_runtime_s", "work_units", "tm_nonzero"]
    sub = runs[runs["kind"].isin(["main", "operational"])]
    k5 = KEYS + ["kind"]
    mk = sub[sub["method"] == "Markov"].drop_duplicates(k5).set_index(k5)[cols]
    cy = sub[sub["method"] == "Cyclic"].drop_duplicates(k5).set_index(k5)[cols]
    tr = sub[sub["method"] == "Truth"].drop_duplicates(k5).set_index(k5)[["work_units"]]
    r = (mk / cy).drop(columns="tm_nonzero").add_suffix("_ratio")
    r["tm_nonzero"] = mk["tm_nonzero"]
    r["markov_wu_pct_of_ref"] = 100 * mk["work_units"] / tr["work_units"].reindex(mk.index)
    r = r.reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH_IN, 2.7), sharey=True)
    for ax, (col, title) in zip(axes, [("num_nonzeros_ratio", "Nonzeros"), ("solver_max_mem_gb_ratio", "Peak solver memory"),
                                       ("solver_runtime_s_ratio", "Solver time"), ("work_units_ratio", "Work units")]):
        for case in cases:
            for kind, ls in (("main", "-"), ("operational", "--")):
                g = r[(r["tm"] == "original") & (r["case"] == case) & (r["kind"] == kind)].groupby("clusters")[col].agg(["median", "min", "max"]).dropna()
                if g.empty:
                    continue
                x = [XPOS[int(c)] for c in g.index]
                ax.plot(x, g["median"], color=CASE_COLOR.get(case, "k"), ls=ls, marker="o", ms=2.5, lw=1)
                ax.fill_between(x, g["min"], g["max"], color=CASE_COLOR.get(case, "k"), alpha=0.12, lw=0)
        ax.axhline(1, color="gray", lw=0.6)
        ax.set_yscale("log")
        cat_xaxis(ax, 2)
        ax.set_title(title)
        ax.set_xlabel("Number of RPs")
        ax.grid(color="#e5e4e0", lw=0.5)
    axes[0].set_ylabel("Markov / Cyclic")
    h = [Line2D([], [], color=CASE_COLOR.get(c, "k"), label=c) for c in cases] + [Line2D([], [], color="k", ls="-", label="investment (main)"),
                                                                                   Line2D([], [], color="k", ls="--", label="operational")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "R6b_markov_overhead", r[r["tm"] == "original"])
    fig, ax = plt.subplots(figsize=(WIDTH_IN * 0.75, 3.0))
    for case in cases:
        for kind, filled in (("main", True), ("operational", False)):
            p = r[(r["case"] == case) & (r["kind"] == kind)]
            ax.scatter(p["tm_nonzero"], p["work_units_ratio"], marker=CASE_MARKER.get(case, "D"), s=12, lw=0.7, alpha=0.8,
                       facecolors=CASE_COLOR.get(case, "k") if filled else "none", edgecolors=CASE_COLOR.get(case, "k"),
                       label=f"{case}, {'investment' if filled else 'operational'}")
    ax.axhline(1, color="gray", lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nonzero transition probabilities")
    ax.set_ylabel("Work units Markov / Cyclic")
    ax.grid(color="#e5e4e0", lw=0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    save(fig, "R6c_workunits_vs_tm_nonzeros", r)
    tcols = [c for c, _, _ in R6_COLS] + ["num_vars", "num_bin_vars"]
    g = sub.groupby(["case", "tm", "kind", "clusters", "method"])
    t = g[tcols].agg(["median", "max"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t["n_runs"] = g.size()
    t.reset_index().to_csv(OUT / "R6_scalability.csv", index=False)
    r[r["markov_wu_pct_of_ref"] > 100].sort_values(["case", "kind", "clusters", "sd"]).to_csv(OUT / "R6_markov_above_reference.csv", index=False)


########################################################################################################################
# R7 - operational fidelity metrics
########################################################################################################################

R7_ROWS = [("opcost", "Operating cost\ndev. [%]"), ("dispatch", "Hourly dispatch\ndev. [% of ref.]"), ("curtail", "Curtailment\n- ref. [pp]"),
           ("shed", "Load shedding\n- ref. [pp]"), ("storage", "Storage level\ndev. [% of cap.]"), ("shutdown", "Shutdowns\ndev. [%]"),
           ("infeas", "Transitions with\nUC violation,\nrounded [%]")]
R7_SIGNED = {"opcost", "curtail", "shed", "shutdown"}


def r7(ctx):
    import matplotlib.pyplot as plt
    runs = ctx["runs"]
    op = runs[(runs["kind"] == "operational") & ~runs["original_ref"] & runs["clusters"].notna()]
    truth = op[op["method"] == "Truth"].drop_duplicates(KEYS).set_index(KEYS)
    rows = []
    for _, r in op[op["method"].isin(METHODS)].iterrows():
        key = (r["case"], r["tm"], r["clusters"], r["sd"])
        if key not in truth.index:
            continue
        t = truth.loc[key]
        vals = {"opcost": 100 * (r["operating_cost"] - t["operating_cost"]) / t["operating_cost"], "dispatch": r.get("dispatch_dev_pct", np.nan),
                "curtail": r["curtailment_pct"] - t["curtailment_pct"], "shed": r["load_shedding_pct"] - t["load_shedding_pct"],
                "storage": r.get("storage_dev_pct", np.nan), "shutdown": 100 * (r["vShutdown"] - t["vShutdown"]) / t["vShutdown"],
                "infeas": r["feas_infeasible_rounded_pct"]}
        for m, v in vals.items():
            rows.append(dict(zip(KEYS, key)) | {"method": r["method"], "file": r["basename"], "metric": m, "value": v,
                                                "load_shedding_pct": r["load_shedding_pct"], "ref_load_shedding_pct": t["load_shedding_pct"]})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  R7: no operational runs with a reference")
        return
    df.to_csv(OUT / "R7_operational_metrics_values.csv", index=False)
    columns = [(c, t, l) for c, t, l in [("RTS-GMLC", "original", "RTS-GMLC\noriginal TM"), ("RTS-GMLC", "shiftTM2", "RTS-GMLC\nTM shifted by 2")]
               + [(c, "original", c) for c in present(df["case"]) if c != "RTS-GMLC"] if ((df["case"] == c) & (df["tm"] == t)).any()]
    for methods, name in ((METHODS, "R7_operational_metrics"), (["Cyclic", "Markov"], "R7_operational_metrics_noNoEnf")):
        fig, axes = plt.subplots(len(R7_ROWS), len(columns), figsize=(WIDTH_IN, 9.6), sharey="row", squeeze=False)
        for i, (metric, ylabel) in enumerate(R7_ROWS):
            part = df[(df["metric"] == metric) & df["method"].isin(methods)]
            v = part["value"].abs().dropna()
            lin = max(np.percentile(v, 90), 0.01) if len(v) and v.max() > 8 * max(np.percentile(v, 90), 1e-9) else None
            for j, (case, tm, title) in enumerate(columns):
                ax = axes[i, j]
                cell = part[(part["case"] == case) & (part["tm"] == tm)]
                if cell["value"].notna().sum() == 0:
                    ax.text(0.5, 0.5, "no intra-day\nstorage" if metric == "storage" else "n/a", ha="center", va="center", transform=ax.transAxes, fontsize=7, color="gray")
                boxes(ax, [cell.loc[cell["method"] == m, "value"] for m in methods], range(len(methods)), [COLOR[m] for m in methods], width=0.6)
                if metric in R7_SIGNED:
                    ax.axhline(0, color="gray", lw=0.6)
                if lin is not None:
                    ax.set_yscale("symlog", linthresh=lin)
                ax.set_xticks([])
                ax.grid(axis="y", color="#e5e4e0", lw=0.5)
                if i == 0:
                    ax.set_title(title)
                if j == 0:
                    ax.set_ylabel(ylabel)
        fig.legend(handles=method_handles(methods), loc="lower center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        save(fig, name)
    agg = df.dropna(subset=["value"]).groupby(["case", "tm", "method", "metric"])["value"].agg(median="median", min="min", max="max", n="size").reset_index()
    agg["latex"] = [f"{m:.2f} [{lo:.2f}, {hi:.2f}]" for m, lo, hi in zip(agg["median"], agg["min"], agg["max"])]
    agg.to_csv(OUT / "R7_operational_metrics.csv", index=False)
    out = []
    for _, g in df.dropna(subset=["value"]).groupby(["case", "tm", "method", "metric"]):
        q1, q3 = np.percentile(g["value"], [25, 75])
        out.append(g[(g["value"] < q1 - 1.5 * (q3 - q1)) | (g["value"] > q3 + 1.5 * (q3 - q1))])
    pd.concat(out).to_csv(OUT / "R7_outliers.csv", index=False)


########################################################################################################################
# R8 - Markov order and stationarity (RP sequences only)
########################################################################################################################

def _entropy(counter):
    c = np.array(list(counter.values()), dtype=float)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def _entropies(x):
    t = range(2, len(x))
    return (_entropy(Counter(x[i] for i in t)),
            _entropy(Counter((x[i - 1], x[i]) for i in t)) - _entropy(Counter(x[i - 1] for i in t)),
            _entropy(Counter((x[i - 2], x[i - 1], x[i]) for i in t)) - _entropy(Counter((x[i - 2], x[i - 1]) for i in t)))


def _fit(x, K):
    c = np.zeros((K, K))
    for a, b in zip(x[:-1], x[1:]):
        c[a, b] += 1
    rows = c.sum(axis=1, keepdims=True)
    return np.divide(c, rows, out=np.full_like(c, 1 / K), where=rows > 0), c


def _simulate(P, x0, n, rng):
    x = [x0]
    for _ in range(n - 1):
        x.append(int(rng.choice(len(P), p=P[x[-1]])))
    return x


def _tv_halves(x, K):
    h = len(x) // 2
    P1, c1 = _fit(x[:h + 1], K)
    P2, c2 = _fit(x[h:], K)
    seen = (c1.sum(axis=1) > 0) & (c2.sum(axis=1) > 0)
    w = np.bincount(x, minlength=K).astype(float) * seen
    tv = 0.5 * np.abs(P1 - P2).sum(axis=1)
    return (float((w * tv).sum() / w.sum()) if w.sum() else np.nan), P1, P2, c1, c2


def r8(ctx):
    import matplotlib.pyplot as plt
    runs, rng = ctx["runs"], np.random.default_rng(42)
    base = runs[(runs["kind"] == "main") & runs["method"].isin(METHODS) & (runs["tm"] == "original") & (runs["sd"] == 1.0) & ~runs["original_ref"]]
    seqs = {}
    for (case, n, lk), g in base.groupby(["case", "clusters", base["limit_k"].fillna("full year")]):
        with closing(em._connect_ro(g["file"].iloc[0])) as conn:
            h, nk = _hindex(conn)
        x = [int(r.replace("rp", "")) - 1 for r in h["rp"].tolist()[::nk]]
        seqs[(f"{lk} ({len(x)} days)" if lk != "full year" else f"full year ({len(x)} days)", case, int(n))] = x
    rows = []
    for (hz, case, n), x in seqs.items():
        h0, h1, h2 = _entropies(x)
        P, c = _fit(x, n)
        sims = [_simulate(P, x[0], len(x), rng) for _ in range(200)]
        s2 = np.array([_entropies(s)[2] for s in sims])
        tv = _tv_halves(x, n)[0]
        tvs = np.array([_tv_halves(s, n)[0] for s in sims])
        rows.append({"horizon": hz, "case": case, "clusters": n, "days": len(x), "transitions": len(x) - 1, "tm_nonzero": int((c > 0).sum()),
                     "transitions_per_entry": (len(x) - 1) / n ** 2, "transitions_per_nonzero": (len(x) - 1) / max((c > 0).sum(), 1),
                     "H0_bits": h0, "H1_bits": h1, "H2_bits": h2, "H2_sim_median": np.median(s2), "H2_sim_p05": np.percentile(s2, 5),
                     "H2_sim_p95": np.percentile(s2, 95), "H2_below_band": h2 < np.percentile(s2, 5), "tv_halves": tv,
                     "tv_sim_median": np.nanmedian(tvs), "tv_sim_p05": np.nanpercentile(tvs, 5), "tv_sim_p95": np.nanpercentile(tvs, 95)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "R8_markov_order.csv", index=False)
    horizons, cases = list(dict.fromkeys(df["horizon"])), present(df["case"])
    fig, axes = plt.subplots(len(horizons), len(cases), figsize=(WIDTH_IN, 2.3 * len(horizons) + 0.4), sharey=True, squeeze=False)
    for i, hz in enumerate(horizons):
        for ax, case in zip(axes[i], cases):
            d = df[(df["horizon"] == hz) & (df["case"] == case)].sort_values("clusters")
            x = [XPOS[c] for c in d["clusters"]]
            ax.plot(x, d["H0_bits"], "k-", marker="o", ms=2.5, lw=1, label="H(X$_t$)")
            ax.plot(x, d["H1_bits"], color="#1f77b4", marker="o", ms=2.5, lw=1, label="H(X$_t$|X$_{t-1}$)")
            ax.plot(x, d["H2_bits"], color="#d62728", marker="o", ms=2.5, lw=1, label="H(X$_t$|X$_{t-1}$,X$_{t-2}$)")
            ax.fill_between(x, d["H2_sim_p05"], d["H2_sim_p95"], color="#d62728", alpha=0.15, lw=0, label="2nd order on 1st-order\nsimulations (5-95 %)")
            ax.plot(x, d["H2_sim_median"], color="#d62728", ls=":", lw=0.8)
            cat_xaxis(ax)
            ax.grid(color="#e5e4e0", lw=0.5)
            if i == 0:
                ax.set_title(case)
            if i == len(horizons) - 1:
                ax.set_xlabel("Number of RPs")
        axes[i][0].set_ylabel(f"{hz}\nentropy [bits]")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "R8a_markov_order")
    hm_hz = horizons[0]
    hm_cases = [c for c in cases if (hm_hz, c, 7) in seqs]
    fig = plt.figure(figsize=(WIDTH_IN, 6.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1])
    hm = []
    for j, case in enumerate(hm_cases[:3]):
        tv, P1, P2, c1, c2 = _tv_halves(seqs[(hm_hz, case, 7)], 7)
        for k, (P, c, ttl) in enumerate(((P1, c1, "1st half"), (P2, c2, "2nd half"))):
            ax = fig.add_subplot(gs[j // 2, (j % 2) * 2 + k])
            ax.imshow(np.where(c.sum(axis=1, keepdims=True) > 0, P, np.nan), vmin=0, vmax=1, cmap="Blues")
            for a in range(7):
                for b in range(7):
                    if c[a, b]:
                        ax.text(b, a, int(c[a, b]), ha="center", va="center", fontsize=4.5, color="white" if P[a, b] > 0.6 else "black")
            ax.set_xticks(range(7))
            ax.set_yticks(range(7))
            ax.set_xticklabels(range(1, 8), fontsize=5)
            ax.set_yticklabels(range(1, 8), fontsize=5)
            ax.set_title(f"{case}, {ttl}" + (f"\nTV = {tv:.2f}" if k == 1 else ""), fontsize=8)
            ax.set_xlabel("to RP", fontsize=7)
            ax.set_ylabel("from RP", fontsize=7)
            hm.append(pd.DataFrame(P, index=range(1, 8), columns=range(1, 8)).stack().rename("p").reset_index()
                      .rename(columns={"level_0": "from_rp", "level_1": "to_rp"}).assign(case=case, half=ttl, count=c.ravel()))
    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    ax.text(0, 0.5, f"Heat maps: {hm_hz}, 7 RPs, demand 1.0.\nColour = row-normalized probability,\nnumbers = observed transitions.\n"
                    "TV = RP-weighted total variation distance.", fontsize=7, va="center")
    for j, case in enumerate(cases[:3]):
        ax = fig.add_subplot(gs[2, j])
        for hz, ls, col in zip(horizons, ["-", "--", "-."], ["gray", "#1f77b4", "#2ca02c"]):
            d = df[(df["horizon"] == hz) & (df["case"] == case)].sort_values("clusters")
            x = [XPOS[c] for c in d["clusters"]]
            ax.plot(x, d["tv_halves"], color="k", ls=ls, marker="o", ms=2.5, lw=1, label=f"{hz}, observed")
            ax.fill_between(x, d["tv_sim_p05"], d["tv_sim_p95"], color=col, alpha=0.2, lw=0, label=f"{hz}, stationary 1st-order\nsimulations (5-95 %)")
        cat_xaxis(ax)
        ax.set_ylim(0, 1)
        ax.set_title(case)
        ax.set_xlabel("Number of RPs")
        ax.grid(color="#e5e4e0", lw=0.5)
        if j == 0:
            ax.set_ylabel("TV distance\n1st vs 2nd half")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "R8b_stationarity", pd.concat(hm) if hm else None)


########################################################################################################################
# R9 - all case studies in the paper's layout; TX-123BT unit commitment
########################################################################################################################

def r9ab(ctx):
    import matplotlib.pyplot as plt
    runs = ctx["runs"]
    panels = [(c, t, l) for c, t, l in [("RTS-GMLC", "original", "RTS-GMLC"), ("RTS-GMLC", "shiftTM1", "RTS-GMLC\nTM shifted by 1"),
                                        ("RTS-GMLC", "shiftTM2", "RTS-GMLC\nTM shifted by 2")]
              + [(c, "original", c) for c in present(runs["case"].dropna()) if c != "RTS-GMLC"] if ((runs["case"] == c) & (runs["tm"] == t)).any()]
    summary, upper = [], []
    for clusters, tag in (([3, 5, 7], "357"), (CLUSTERS, "all")):
        sub = runs[runs["clusters"].isin(clusters)]
        for signed in (False, True):
            d = vs_truth(sub, "operational", "vShutdown", signed=signed)
            fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH_IN, 2.6), sharey=True, squeeze=False)
            for ax, (case, tm, title) in zip(axes[0], panels):
                cell = d[(d["case"] == case) & (d["tm"] == tm)]
                boxes(ax, [cell.loc[cell["method"] == m, "value"] for m in ("Cyclic", "Markov")], [0, 1], [COLOR["Cyclic"], COLOR["Markov"]], width=0.6)
                for m in ("Cyclic", "Markov"):
                    v = cell.loc[cell["method"] == m, "value"]
                    summary.append({"case": case, "tm": tm, "method": m, "n": len(v), "median": v.median(), "mean": v.mean(), "min": v.min(),
                                    "max": v.max(), "rps": tag, "kind": "signed" if signed else "magnitude"})
                if signed:
                    ax.axhline(0, color="gray", lw=0.6)
                ax.set_title(title)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Cyclic", "Markov"])
                ax.grid(axis="y", color="#e5e4e0", lw=0.5)
            axes[0][0].set_ylabel(("Shutdown deviation" if signed else "|Shutdown deviation|") + " [%]")
            fig.tight_layout()
            save(fig, f"R9a_operational_deviation_all_case_studies_{tag}_{'signed' if signed else 'magnitude'}",
                 d[KEYS + ["method", "vShutdown", "ref", "value"]])
        inv = sub[(sub["kind"] == "invest_regret") & ~sub["original_ref"] & sub["method"].isin(METHODS)]
        inv = inv[KEYS + ["method", "regret", "reference_objective", "regret_pct"]].dropna(subset=["regret_pct"])
        if inv.empty:
            continue
        top = max(inv[inv["method"] != "NoEnf"]["regret_pct"].max(), 100 * MIP_GAP) * 1.15
        bottom = min(inv["regret_pct"].min(), -100 * MIP_GAP) * 1.15
        high = inv["regret_pct"] > top
        fig = plt.figure(figsize=(WIDTH_IN, 3.2))
        gs = fig.add_gridspec(2, len(panels), height_ratios=[1, 3], hspace=0.08) if high.any() else fig.add_gridspec(1, len(panels))
        for j, (case, tm, title) in enumerate(panels):
            cell = inv[(inv["case"] == case) & (inv["tm"] == tm)]
            axs = [fig.add_subplot(gs[1, j]), fig.add_subplot(gs[0, j])] if high.any() else [fig.add_subplot(gs[0, j])]
            for k, ax in enumerate(axs):
                ax.axhspan(-100 * MIP_GAP, 100 * MIP_GAP, color="#bbbbbb", alpha=0.35, lw=0)
                ax.axhline(0, color="gray", lw=0.5)
                boxes(ax, [cell.loc[cell["method"] == m, "regret_pct"] for m in METHODS], [0, 1, 2], [COLOR[m] for m in METHODS], width=0.6)
                ax.grid(axis="y", color="#e5e4e0", lw=0.5)
                ax.set_xlim(-0.6, 2.6)
                if k == 0:
                    ax.set_ylim(bottom, top)
                    ax.set_xticks([0, 1, 2])
                    ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=30, ha="right")
                    if j:
                        ax.set_yticklabels([])
                else:
                    ax.set_ylim(top, inv["regret_pct"].max() * 1.1)
                    ax.set_yscale("log")
                    ax.set_xticks([])
                    ax.spines["bottom"].set_visible(False)
                    if j:
                        ax.set_yticklabels([], minor=True)
                        ax.set_yticklabels([])
                    ax.plot([-0.03, 0.03], [-0.04, 0.04], transform=ax.transAxes, color="k", clip_on=False, lw=0.8)
                if k == len(axs) - 1:
                    ax.set_title(title)
            if j == 0:
                axs[0].set_ylabel("Invest-regret [% of reference objective]")
        from matplotlib.patches import Patch
        fig.legend(handles=method_handles() + [Patch(color="#bbbbbb", alpha=0.35, label=f"MIP gap (±{100 * MIP_GAP:g} %)")], loc="lower center", ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, -0.08))
        save(fig, f"R9b_invest_regret_all_case_studies_{tag}", inv)
        upper.append(inv[high].assign(rps=tag, lower_axis_top=top))
    pd.DataFrame(summary).to_csv(OUT / "R9ab_summary.csv", index=False)
    if upper:
        pd.concat(upper).to_csv(OUT / "R9b_upper_axis_values.csv", index=False)


def _chron_uc(path, truth):
    with closing(em._connect_ro(path)) as conn:
        h, _ = _hindex(conn)
        g = _table(conn, "gtec")
        tec = dict(zip(g.iloc[:, -2], g.iloc[:, -1]))
        units = _active(conn)
        out = {}
        for v in ("vCommit", "vStartup", "vShutdown"):
            d = _table(conn, v, ["rp", "k", "g"])
            d = d[d["g"].isin(units)]
            if truth:
                w = d.pivot(index="k", columns="g", values="values").reindex([p.replace("h", "k") for p in h["p"]])
            else:
                w = d.pivot_table(index=["rp", "k"], columns="g", values="values").reindex(pd.MultiIndex.from_arrays([h["rp"], h["k"]]))
            out[v] = w.reset_index(drop=True).reindex(columns=units).fillna(0)
    return out, tec, h["rp"].tolist()


def r9c(ctx, case="TX-123BT"):
    import matplotlib.pyplot as plt
    runs = ctx["runs"]
    dev = vs_truth(runs[(runs["case"] == case) & (runs["tm"] == "original")], "operational", "vShutdown")
    dev = dev[dev["method"] == "Markov"]
    if dev.empty:
        print(f"  R9c: no {case} Markov operational run with a reference")
        return
    surplus_rows = []
    for _, w in dev.iterrows():  # surplus by hour of the day for every sub-case
        f = runs[(runs["case"] == case) & (runs["tm"] == "original") & (runs["clusters"] == w["clusters"]) & (runs["sd"] == w["sd"])
                 & (runs["kind"] == "operational") & ~runs["original_ref"]].set_index("method")["file"]
        sd_t = _chron_uc(f["Truth"], True)[0]["vShutdown"].sum(axis=1).to_numpy()
        sd_m = _chron_uc(f["Markov"], False)[0]["vShutdown"].sum(axis=1).to_numpy()
        s = pd.Series(sd_m - sd_t).groupby(np.arange(len(sd_t)) % 24 + 1).sum()
        surplus_rows.append({"clusters": int(w["clusters"]), "sd": w["sd"], "markov_dev_pct": w["value"], "surplus_total": s.sum(),
                             "ref_total": sd_t.sum()} | {f"h{h}": v for h, v in s.items()})
    pd.DataFrame(surplus_rows).to_csv(OUT / "R9c_surplus_by_hour_all_subcases.csv", index=False)
    worst = dev.sort_values("value").iloc[-1]
    n, sd = int(worst["clusters"]), float(worst["sd"])
    files = runs[(runs["case"] == case) & (runs["tm"] == "original") & (runs["clusters"] == n) & (runs["sd"] == sd)
                 & (runs["kind"] == "operational") & ~runs["original_ref"]].set_index("method")["file"]
    data = {m: _chron_uc(files[m], m == "Truth") for m in ("Truth", "Cyclic", "Markov")}
    tec = data["Truth"][1]
    rps = data["Markov"][2]
    day_rps = rps[::24]
    T = len(data["Truth"][0]["vCommit"])
    hod = np.arange(T) % 24 + 1
    sdn = {m: data[m][0]["vShutdown"].sum(axis=1).to_numpy() for m in data}
    sun = {m: data[m][0]["vStartup"].sum(axis=1).to_numpy() for m in data}
    surplus = sdn["Markov"] - sdn["Truth"]
    daily = pd.Series(surplus).groupby(np.arange(T) // 24).sum()
    start = max(int(daily.rolling(6).sum().idxmax()) - 5, 0)
    win = np.arange(start * 24, min(start * 24 + 144, T))
    fig = plt.figure(figsize=(WIDTH_IN, 7.4))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.2, 1.2, 1.5], hspace=0.55, wspace=0.4)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
    for m in ("Truth", "Cyclic", "Markov"):
        ls = "--" if m == "Truth" else "-"
        ax1.step(win + 1, data[m][0]["vCommit"].sum(axis=1).to_numpy()[win], where="mid", color=COLOR[m], ls=ls, lw=1, label=LABEL[m])
        ax2.step(win + 1, sdn[m][win], where="mid", color=COLOR[m], ls=ls, lw=1)
        ax3.step(win + 1, sun[m][win], where="mid", color=COLOR[m], ls=ls, lw=1)
    for ax, yl in ((ax1, "Committed\nunits"), (ax2, "Shutdowns\nper hour"), (ax3, "Start-ups\nper hour")):
        for d in range(start, start + 7):
            ax.axvline(d * 24 + 0.5, color="gray", lw=0.4)
        ax.set_ylabel(yl)
        ax.grid(axis="y", color="#e5e4e0", lw=0.5)
    for d in range(start, min(start + 6, len(day_rps))):
        ax1.text(d * 24 + 12.5, 1.02, day_rps[d].replace("rp0", "RP ").replace("rp", "RP "), transform=ax1.get_xaxis_transform(), ha="center", fontsize=6)
    ax3.set_xlabel("Hour of the horizon")
    ax1.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.35))
    ax1.set_xlim(win[0] + 0.5, win[-1] + 1.5)
    axh = fig.add_subplot(gs[3, 0])
    by_hod = pd.DataFrame({m: pd.Series(sdn[m] - sdn["Truth"]).groupby(hod).sum() for m in ("Cyclic", "Markov")})
    for k, m in enumerate(("Cyclic", "Markov")):
        axh.bar(by_hod.index + (k - 0.5) * 0.4, by_hod[m], width=0.4, color=COLOR[m])
    axh.axhline(0, color="gray", lw=0.5)
    axh.set_xlabel("Hour of the day (1 = first hour of an RP)")
    axh.set_ylabel("Shutdowns - reference\n(whole horizon)")
    axh.set_xticks([1, 6, 12, 18, 24])
    axh.grid(axis="y", color="#e5e4e0", lw=0.5)
    axt = fig.add_subplot(gs[3, 1])
    techs = sorted({tec[u] for m in data for u in data[m][0]["vShutdown"].columns})
    by_tec = pd.DataFrame({m: pd.Series({t: data[m][0]["vShutdown"][[u for u in data[m][0]["vShutdown"].columns if tec[u] == t]].to_numpy().sum()
                                         for t in techs}) for m in data})
    diff = by_tec[["Cyclic", "Markov"]].sub(by_tec["Truth"], axis=0)
    for k, m in enumerate(("Cyclic", "Markov")):
        axt.bar(np.arange(len(techs)) + (k - 0.5) * 0.4, diff[m], width=0.4, color=COLOR[m])
    axt.axhline(0, color="gray", lw=0.5)
    axt.set_xticks(range(len(techs)))
    axt.set_xticklabels(techs, rotation=30, ha="right")
    axt.set_ylabel("Shutdowns - reference")
    axt.grid(axis="y", color="#e5e4e0", lw=0.5)
    fig.align_ylabels()
    hours = pd.DataFrame({"hour": np.arange(T) + 1, "day": np.arange(T) // 24 + 1, "hour_of_day": hod, "rp": rps}
                         | {f"{v}_{m}": data[m][0][v].sum(axis=1).to_numpy() for m in data for v in ("vCommit", "vStartup", "vShutdown")})
    save(fig, "R9c_unit_commitment", hours)
    by_hod.to_csv(OUT / "R9c_surplus_shutdowns_by_hour_of_day.csv")
    by_tec.assign(**{f"{m}_minus_ref": diff[m] for m in diff}).to_csv(OUT / "R9c_shutdowns_by_technology.csv")

    def chron_sd(m):
        u = (data[m][0]["vCommit"].to_numpy() >= 0.5).astype(int)
        s = np.zeros_like(u)
        s[1:] = np.maximum(u[:-1] - u[1:], 0)
        return s.sum()

    pd.DataFrame({"shutdowns_model": {m: v.sum() for m, v in sdn.items()},
                  "shutdowns_rounded_flags": {m: (data[m][0]["vShutdown"].to_numpy() >= 0.5).sum() for m in data},
                  "shutdowns_from_rounded_commitment_chronological": {m: chron_sd(m) for m in data},
                  "startups_model": {m: v.sum() for m, v in sun.items()},
                  "startups_rounded_flags": {m: (data[m][0]["vStartup"].to_numpy() >= 0.5).sum() for m in data}}).assign(
        case=case, clusters=n, sd=sd, markov_dev_pct=worst["value"], window_days_1based=f"{start + 1}-{start + 6}").to_csv(OUT / "R9c_totals.csv")


########################################################################################################################
# R10 - comparison with the original full chronology (TruthOriginal)
########################################################################################################################

def r10(ctx):
    import matplotlib.pyplot as plt
    runs = ctx["runs"]
    to = runs[(runs["method"] == "TruthOriginal")].drop_duplicates(["case", "sd"]).set_index(["case", "sd"])
    oir = runs[(runs["kind"] == "invest_regret") & runs["original_ref"] & (runs["tm"] == "original") & runs["clusters"].notna()]
    if to.empty or oir.empty:
        print("  R10: no TruthOriginal / -original-invest-regret runs - skipped")
        return
    metrics = [("Objective", "Objective / invest-regret [%]"), ("operating_cost", "Operating cost [%]"), ("vStartup", "Start-ups [%]")]
    rows = []
    for (case, sd, n), g in oir.groupby(["case", "sd", "clusters"]):
        if (case, sd) not in to.index:
            continue
        ref = to.loc[(case, sd)]
        by = g.drop_duplicates("method").set_index("method")
        for col, _ in metrics:
            if col not in by or pd.isna(ref.get(col)) or not ref[col]:
                continue
            clus = 100 * (by[col]["Truth"] - ref[col]) / abs(ref[col]) if "Truth" in by.index else np.nan
            for m in [m for m in METHODS if m in by.index]:
                total = 100 * (by[col][m] - ref[col]) / abs(ref[col])
                rows.append({"case": case, "sd": sd, "clusters": int(n), "method": m, "metric": col, "truth_original": ref[col],
                             "reference_on_original": by[col].get("Truth", np.nan), "method_on_original": by[col][m],
                             "clustering_error_pct": clus, "edge_error_pct": total - clus, "total_error_pct": total})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  R10: no sub-case has both TruthOriginal and -original-invest-regret runs - skipped")
        return
    df.to_csv(OUT / "R10_error_decomposition_values.csv", index=False)
    cases = present(df["case"])
    fig, axes = plt.subplots(len(cases), len(metrics), figsize=(WIDTH_IN, 2.4 * len(cases) + 0.5), squeeze=False)
    agg = df.groupby(["case", "clusters", "method", "metric"])[["clustering_error_pct", "edge_error_pct", "total_error_pct"]].median().reset_index()
    for i, case in enumerate(cases):
        for j, (col, title) in enumerate(metrics):
            ax = axes[i, j]
            a = agg[(agg["case"] == case) & (agg["metric"] == col)]
            for k, m in enumerate(METHODS):
                p = a[a["method"] == m]
                x = np.array([XPOS[int(c)] for c in p["clusters"]]) + (k - 1) * 0.27
                c = p["clustering_error_pct"].fillna(0).to_numpy()
                ax.bar(x, c, width=0.25, color="#bbbbbb", edgecolor="#777777", lw=0.3, hatch="///")
                ax.bar(x, p["edge_error_pct"], bottom=c, width=0.25, color=COLOR[m], alpha=0.85)
                ax.scatter(x, p["total_error_pct"], s=4, color="k", zorder=3)
            ax.axhline(0, color="gray", lw=0.5)
            cat_xaxis(ax)
            ax.grid(axis="y", color="#e5e4e0", lw=0.5)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"{case}\nvs. original model")
            if i == len(cases) - 1:
                ax.set_xlabel("Number of RPs")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(facecolor="#bbbbbb", edgecolor="#777777", hatch="///", label="clustering error (reference vs original)")] + method_handles() + \
        [Line2D([], [], marker="o", ls="", color="k", ms=2, label="median total")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "R10_error_decomposition", agg)
    # compare_original_* equivalent: regret vs TruthOriginal (incl. the reference's own = clustering error)
    for kind, name in (("invest_regret", "R10b_original_invest_regret"), ("regret", "R10c_original_regret")):
        sub = runs[(runs["kind"] == kind) & runs["original_ref"] & (runs["tm"] == "original")].dropna(subset=["regret_pct"])
        if sub.empty:
            continue
        ms = [m for m in ["Truth"] + METHODS if (sub["method"] == m).any()]
        fig, axes = plt.subplots(1, len(present(sub["case"])), figsize=(WIDTH_IN, 2.8), squeeze=False)
        for ax, case in zip(axes[0], present(sub["case"])):
            cell = sub[sub["case"] == case]
            ax.axhspan(-100 * MIP_GAP, 100 * MIP_GAP, color="#bbbbbb", alpha=0.35, lw=0)
            boxes(ax, [cell.loc[cell["method"] == m, "regret_pct"] for m in ms], range(len(ms)), [COLOR[m] for m in ms], width=0.6)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels([LABEL[m] for m in ms], rotation=30, ha="right")
            ax.set_title(case)
            ax.grid(axis="y", color="#e5e4e0", lw=0.5)
        axes[0][0].set_ylabel("Regret vs original model\n[% of its objective]")
        fig.tight_layout()
        save(fig, name, sub[KEYS + ["method", "Objective", "reference_objective", "regret", "regret_pct"]])


########################################################################################################################
# Main
########################################################################################################################

FIGURES = {"R1": r1, "R2": r2, "R3": r3, "R5": r5, "R6": r6, "R7": r7, "R8": r8, "R9": lambda c: (r9ab(c), r9c(c)), "R10": r10}


def main():
    global OUT, CACHE
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="*", default=["."], help="Folders with MK-*.sqlite files (default: current directory)")
    p.add_argument("--output-dir", default="results/figure_requests", help="Output directory (default: results/figure_requests)")
    p.add_argument("--only", default=None, help="Comma-separated subset of R1,R2,R3,R5,R6,R7,R8,R9,R10")
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 4),
                   help="Max. processes of every pool (EvaluateMarkov's loading and the extraction pass); e.g. 4-8 on a login node")
    p.add_argument("--reuse", action="store_true", help="Reuse markov_runs.csv and the extraction cache of a previous run")
    p.add_argument("--include-perturbed", action="store_true", help="Also use the perturbed-TM runs (excluded by default)")
    p.add_argument("--require-truth", action="store_true",
                   help="Restrict R1-R3 to sub-cases with an optimal Truth main run (comparisons with the reference always need it)")
    p.add_argument("--no-chronology", action="store_true", help="Skip EvaluateMarkov's chronological comparisons (dispatch / storage deviation, TM structure)")
    p.add_argument("--mip-gap", type=float, default=0.005, help="Only runs with this mip_gap run parameter (default 0.005); other runs in the folder are ignored")
    args = p.parse_args()
    # Cap every process pool (also EvaluateMarkov's, which sizes its pools with os.cpu_count) - e.g. on a login node
    os.cpu_count = lambda: args.workers
    OUT = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE = OUT / "_cache"
    style()
    runs_all = load_runs(args.folders, args.reuse, args.no_chronology)
    gap = pd.to_numeric(runs_all["mip_gap"], errors="coerce")
    dropped = int((~np.isclose(gap, args.mip_gap)).sum())
    if dropped:
        print(f"Ignoring {dropped} run(s) with mip_gap != {args.mip_gap}")
    runs_all = runs_all[np.isclose(gap, args.mip_gap)]
    dup = runs_all[runs_all["optimal"]].duplicated(["case", "tm", "clusters", "sd", "kind", "original_ref", "method"], keep=False) & runs_all["clusters"].notna()
    if dup.any():
        print(f"WARNING: {int(dup.sum())} runs share (case, TM, RPs, demand, kind, method) - other run parameters differ "
              f"(e.g. limitK / network); only the first of each is used. See markov_runs.csv.")
        runs_all = runs_all[~(runs_all["optimal"] & runs_all.duplicated(["case", "tm", "clusters", "sd", "kind", "original_ref", "method"]) & runs_all["clusters"].notna())]
    runs = select(runs_all, args.include_perturbed)
    runs[runs["clusters"].notna() | (runs["method"] == "TruthOriginal")].groupby(
        ["case", "tm", "kind", "original_ref", "method"]).size().rename("optimal_runs").to_csv(OUT / "run_counts.csv")
    rp = runs[runs["clusters"].notna()]
    all_subcases = set(zip(rp["case"], rp["tm"], rp["clusters"].astype(int), rp["sd"]))
    ctx = {"runs": runs, "keep": truth_subcases(runs) if args.require_truth else all_subcases}
    extract(runs, args.workers)
    wanted = [s.strip().upper() for s in args.only.split(",")] if args.only else list(FIGURES)
    print("R4: skipped (needs new runs: dispatch re-solved with rounded, fixed UC flags)")
    failed = []
    for name in wanted:
        print(f"{name} ...", flush=True)
        try:
            FIGURES[name](ctx)
        except Exception:
            traceback.print_exc()
            failed.append(name)
    print("done" + (f"; FAILED: {', '.join(failed)}" if failed else "") + f" - outputs in {OUT}")


if __name__ == "__main__":
    main()
