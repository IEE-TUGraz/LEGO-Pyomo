# Markov Chain Edge Handling (MK) Experiments

Experiments comparing strategies for handling the boundaries between representative periods in energy system
optimization. When using clustered time series, transitions between representative periods need special handling for
unit commitment, ramping, and intra-day storage constraints.

## Running Experiments

```bash
# Run basic experiment
python research/MK/Markov.py data/example

# Run with regret calculation
python research/MK/Markov.py data/example --calculate-regret

# Run with limited time horizon
python research/MK/Markov.py data/NREL-118 --limitK k0001-k0168

# Run with clustering and relaxation
python research/MK/Markov.py data/example --clusters 3 --relax-percentage 0.5

# Run with strict Markov variant
python research/MK/Markov.py data/example --enable-strict-markov

# Run multiple case studies
python research/MK/Markov.py data/folder1,data/folder2

# Resume an interrupted batch run
python research/MK/Markov.py data/NREL-118 --no-overwrite
```

## Key Concepts

**Edge Handling Strategies** — four approaches compared:

- `notEnforced`: No constraints between representative periods (baseline)
- `cyclic`: Wrap-around constraints (last timestep connects to first)
- `markov`: Markov chain-based transition constraints with push constraints deactivated
- `markov-strict` (opt-in via `--enable-strict-markov`): Full Markov variant with push constraints active

**Truth Model**: A full-hourly (non-aggregated) model used as ground truth for comparison. Skip with `--skip-truth`.

**Regret Calculation**: Each edge handling's decisions are fixed into the full-hourly truth model, which is then
re-solved to measure the cost of using simplified edge handling. Three variants isolate different decisions
(all re-solve the truth model; all skip Truth itself, which has zero regret):

| Flag                  | `vGenInvest` fixed from | `vCommit` fixed from        | Isolates                              |
|-----------------------|-------------------------|-----------------------------|---------------------------------------|
| `--calculate-regret`  | edge handling main run  | edge handling main run      | total regret of the full plan         |
| `--invest-regret`     | edge handling main run  | (free / re-optimized)       | investment-decision regret            |
| `--operational-regret`| Truth                   | edge handling operational run | operational regret under the correct fleet |

`vCommit` is soft-fixed (a slack with an EPS penalty allows deviations); `vGenInvest` is hard-fixed.

**Relaxation**: A percentage of thermal generators can have binary unit commitment variables relaxed to continuous,
ordered by sum of MinUpTime + MinDownTime.

## Script Reference

### `Markov.py` — Main experiment script

Produces `.sqlite` files with model results, run parameters, and solver statistics.

| Parameter                | Default        | Description                                                                                                 |
|--------------------------|----------------|-------------------------------------------------------------------------------------------------------------|
| `caseStudyFolder`        | —              | Path to data folder (comma-separated list for multiple)                                                     |
| `--debug`                | off            | Re-raise exceptions instead of continuing with the next case study                                          |
| `--calculate-regret`     | off            | Re-solve truth model with `vGenInvest` **and** `vCommit` fixed from each model's main run (total regret)     |
| `--skip-truth`           | off            | Skip solving the full-hourly truth model                                                                    |
| `--relax-percentage`     | 0              | Fraction of thermal generators to relax from binary to continuous                                           |
| `--clusters`             | 1              | Number of k-medoids clusters (1 = no clustering)                                                            |
| `--cluster-stepsize`     | 1              | Step size when sweeping cluster counts                                                                      |
| `--cluster-steps`        | 0              | Number of additional cluster count steps                                                                    |
| `--filter-zone`          | —              | Restrict to buses in a single zone (exact match of `z` column in Power_BusInfo, e.g. `R1`)                  |
| `--limitK`               | —              | Restrict timesteps, e.g. `k0001-k0168`                                                                      |
| `--shift`                | 0              | Shift time series by N hours                                                                                |
| `--stretch-demand`       | 1.0            | Stretch demand around its mean by a factor                                                                  |
| `--scale-vres`           | 1.0            | Multiply `MaxProd` of all VRES generators (PV, Wind, RoR) by this factor                                    |
| `--scale-invest-cost`    | 1.0            | Multiply `pInvestCost` of all generators (ThermalGen, VRES, Storage) by this factor                         |
| `--thermal-invest-only`  | off            | Set `ExisUnits=1` for all VRES and Storage generators — only thermal generators remain investable           |
| `--merge-generators`     | off            | Merge generators of same technology at same bus before clustering                                           |
| `--enable-strict-markov` | off            | Also run the Markov-Strict variant (push constraints active)                                                |
| `--invest-regret`        | off            | Fix vGenInvest from each model into truth and compare objectives                                            |
| `--no-investment`        | off            | Fix all vGenInvest to 1 (skip investment decisions)                                                         |
| `--operational`          | off            | Add an operational run per edge-handling model (Truth, NoEnf, Cyclic, Markov, +Markov-Strict if enabled): fixes `vGenInvest` to the Truth investment (1 where Truth invested, 0 otherwise) and re-solves. See "Operational runs" below |
| `--operational-regret`   | off            | Per non-Truth edge handling, re-solve the truth model with `vGenInvest` fixed to Truth's and `vCommit` fixed from that edge handling's **operational** run (operational regret under the correct fleet). **Requires `--operational`.** See "Operational-regret runs" below |
| `--no-overwrite`         | off            | Skip runs that already solved to optimality (existing file, or a sibling run differing only in `work_limit`); non-optimal results are re-run |
| `--rmip`                 | off            | Relax all integer variables before solving                                                                  |
| `--no-crossover`         | off            | Disable Gurobi crossover (must be paired with `--force-barrier`)                                            |
| `--force-barrier`        | off            | Force Gurobi barrier method (must be paired with `--no-crossover`)                                          |
| `--mip-gap`              | solver default | MIP gap tolerance, e.g. `0.01` for 1%                                                                       |
| `--work-limit`           | no limit       | Gurobi WorkLimit (in work units): stop after the given budget regardless of solution quality                |
| `--node-file-start`      | no spilling    | Gurobi NodefileStart (GB): B&B nodes spill to disk when in-memory storage exceeds this — useful when MIP runs OOM |
| `--node-file-dir`        | `./gurobi-nodes/`  | Base directory for spilled nodes (auto-created). Requires `--node-file-start`. A `<pid>` subfolder is ALWAYS appended (e.g. `E:/tmp/nodes` becomes `E:/tmp/nodes/12345/`) to guarantee parallel spawns never share a dir. Use a fast local SSD, NOT NFS  |
| `--threads`              | 0 (all cores)  | Gurobi Threads. Lower when running multiple processes in parallel (e.g. via `Caller.py --spawn N`) to avoid oversubscription |
| `--network`              | no change      | Override `pTecRepr` for all lines uniformly: `DC-OPF`, `TP`, or `SN` (omit to use values from data)         |
| `--commit-consumption`   | 1.0            | Multiplier for `CommitConsumption` in `Power_ThermalGen`                                                    |
| `--startup-consumption`  | 1.0            | Multiplier for `StartupConsumption` in `Power_ThermalGen`                                                   |
| `--shift-tm`             | —              | Cyclically shift each row of the transition matrix right by N positions, then normalize and resample Hindex |
| `--perturb-tm`           | —              | Perturb each row of the transition matrix: `new_prob = (1-r)*orig + r*random`, with `r` in [0.0, 1.0]      |
| `--no-sqlite`            | off            | Do not save results to SQLite                                                                               |
| `--reuse-inputfiles`     | off            | Reuse already-prepared input folders (e.g. after limitK)                                                    |

**Output naming**: `MK-{identifier}-{edgeHandling}.sqlite`. Regret files append `-regret`, `-invest-regret`, or
`-operational-regret`; operational runs (`--operational`) append `-operational`.
Non-default parameters are encoded in the identifier (e.g. `filterZoneR1`, `relaxed3`, `rMIP`, `mipGap0.01`,
`workLimit500`, `networkTP`, `commitConsumption0.5`, `startupConsumption2`, `shiftTM2`, `perturbTM0.5`,
`scaleVRES0.8`, `scaleInvestCost0.5`).

**Operational runs** (`--operational`): solves a variant of each edge-handling model with `vGenInvest` fixed to the
**Truth** investment decision (1 where Truth invested above 0.5, 0 otherwise), isolating the operational cost of each
edge handling under a common investment. The Truth investment must come from an *optimal* Truth result: the in-memory
Truth solve (only if it reached optimality), otherwise an optimal Truth `.sqlite` (the exact file, or a sibling differing
only in `work_limit`; WorkLimit/OOM Truth results are never used). If no optimal Truth result is available anywhere, all
operational runs are skipped with an error and the run continues. Under `--skip-truth` the Truth model isn't in memory, so
`Truth-operational` rebuilds the full-hourly Truth model on demand (applying `--relax-percentage` consistently) and
re-solves it with the fixed investment — so its solve time is comparable to the other operational runs. `--no-overwrite`
applies to operational files using the same exact-file / smart-sibling logic as the main runs (and the on-demand Truth
rebuild is lazy, happening only when the run isn't skipped).

**Operational-regret runs** (`--operational-regret`, requires `--operational`): for each non-Truth edge handling,
re-solves the full-hourly truth model with `vGenInvest` hard-fixed to the **Truth** investment (as in `--operational`)
and `vCommit` soft-fixed to that edge handling's **operational** run — isolating its operational regret while the fleet
is held at the correct (Truth) investment. The operational `vCommit` is taken from the in-memory operational solve, the
`-operational.sqlite` file if that run was no-overwrite-skipped, or the skipping sibling's `.sqlite` if it was
sibling-skipped; only when none of these sources exists is operational-regret skipped for that edge handling. Output files append
`-operational-regret`; like the other regret variants they are written but not returned for downstream plotting.
`--no-overwrite` skips an operational-regret file only when it already solved to optimality.

### `EvaluateMarkov.py` — Result evaluation

Reads all `MK-*.sqlite` files in a folder and prints comparison tables. Operational runs (`-operational.sqlite`) and
each regret variant (`-regret.sqlite`, `-invest-regret.sqlite`, `-operational-regret.sqlite`) are shown in their own
separate table per group, below the main comparison (regret tables have no Truth row, so their `%` columns read
against Markov).

```bash
python research/MK/EvaluateMarkov.py                             # current directory
python research/MK/EvaluateMarkov.py path/to/results             # specific folder
python research/MK/EvaluateMarkov.py --plot --case-study-folder data/example
```

| Parameter             | Description                                              |
|-----------------------|----------------------------------------------------------|
| `folder`              | Folder with `.sqlite` files (default: current directory) |
| `--plot`              | Show unit commitment plots                               |
| `--case-study-folder` | Case study folder for plots                              |
| `--number-of-hours`   | Number of hours to show in plots (default: 144)          |
| `--start-hour`        | Start hour for plots (default: 1)                        |
| `--no-show`           | Only save the plot, don't display it                     |

### `CompareMarkov.py` — Cross-run comparison boxplots

Reads all `MK-*.sqlite` files in a folder and produces **boxplot PNGs** comparing the
edge-handling strategies (NoEnf, Cyclic, Markov — plus Markov-Strict with `--markov-strict`)
against the Truth model. Each figure has one subplot per `(shift_tm, perturb_tm)` combination
(shared y-axis), and within each subplot one boxplot per strategy. Every box aggregates over the
**sub-cases** sharing that TM combination — i.e. the other run parameters that vary (`clusters`,
`stretch_demand`, …). Truth is the deviation/regret reference, never drawn as a box.

There are **14 logical plots**, and **each is emitted twice** — once with all strategies and once
with NoEnf excluded (a `_noNoEnf` suffix), since NoEnf's large deviations often compress the scale
— giving up to **28 PNGs**:

| Base filename                                  | Content                                                          |
|------------------------------------------------|------------------------------------------------------------------|
| `compare_workunits_operational_absolute.png`   | A — Work units, operational runs                                 |
| `compare_workunits_operational_relative.png`   | A — Work units as % of Truth, operational runs                   |
| `compare_vshutdown_operational_absolute.png`   | B — vShutdown deviation vs Truth-operational (absolute)          |
| `compare_vshutdown_operational_relative.png`   | B — vShutdown deviation vs Truth-operational [%]                 |
| `compare_workunits_investment_absolute.png`    | C — Work units, investment (main) runs                           |
| `compare_workunits_investment_relative.png`    | C — Work units as % of Truth, investment runs                    |
| `compare_vshutdown_investment_absolute.png`    | D — vShutdown deviation vs Truth-main (absolute)                 |
| `compare_vshutdown_investment_relative.png`    | D — vShutdown deviation vs Truth-main [%]                        |
| `compare_invest_regret_absolute.png`           | E — Invest-regret (absolute) over Truth-main objective           |
| `compare_invest_regret_relative.png`           | E — Invest-regret [%] over Truth-main objective                  |
| `compare_regret_absolute.png`                  | F — Regret (absolute) over Truth-main objective (invest+commit fixed) |
| `compare_regret_relative.png`                  | F — Regret [%] over Truth-main objective                         |
| `compare_operational_regret_absolute.png`      | G — Operational-regret (absolute) over Truth-operational objective |
| `compare_operational_regret_relative.png`      | G — Operational-regret [%] over Truth-operational objective      |

"Operational runs" are the `--operational` runs (vGenInvest fixed to Truth's investment);
"investment runs" are the regular main runs. The three regret plots reference the **same-kind** Truth
objective: invest-regret (E) and regret (F) against Truth-main, operational-regret (G) against
Truth-operational. The relative work-units plots show `work_units / truth_work_units * 100`
(100% == as expensive as Truth), so they need a solver that reports work units (Gurobi); under solvers
that don't (e.g. HiGHS) those plots are empty. A category with no files is skipped with a message (no
crash). Reads only what each plot needs (run parameters, solver `work_units`, a SQL-aggregated weighted
`vShutdown` sum, and the objective), loaded concurrently with a thread pool.

```bash
python research/MK/CompareMarkov.py                                    # current directory
python research/MK/CompareMarkov.py path/to/results                    # specific folder
python research/MK/CompareMarkov.py results/ --output-dir plots/ --no-show
python research/MK/CompareMarkov.py results/ --include-nonoptimal      # don't drop non-optimal runs
python research/MK/CompareMarkov.py results/ --markov-strict           # add a Markov-Strict box
python research/MK/CompareMarkov.py results/ --logscale                # log y-axis for work-units plots
python research/MK/CompareMarkov.py results/ --nrOfClusters 3,5,7      # only runs with 3, 5 or 7 clusters
python research/MK/CompareMarkov.py results/ --separateClusters        # one plot set per cluster count
python research/MK/CompareMarkov.py results/ --tm none:0.2 --tm 1:none # only those two TM subplots
python research/MK/CompareMarkov.py results/ --tm "base,1:*"           # base + everything with shiftTM=1
```

| Parameter             | Default        | Description                                                                                 |
|-----------------------|----------------|---------------------------------------------------------------------------------------------|
| `folder`              | `.`            | Folder with `MK-*.sqlite` files                                                             |
| `--output-dir`        | input folder   | Directory to save the PNGs in                                                               |
| `--no-show`           | off            | Suppress interactive display (for headless/batch runs)                                      |
| `--include-nonoptimal`| off            | Include runs with `termination_condition != 'optimal'` (default: optimal-only)              |
| `--markov-strict`     | off            | Also draw a Markov-Strict box (only meaningful for `--enable-strict-markov` runs)           |
| `--logscale`          | off            | Log-scale y-axis for the work-units plots (A/C); no effect on deviation/regret plots        |
| `--nrOfClusters`      | all            | Comma-separated list of cluster counts; only runs whose `clusters` run-parameter is in the list are included (e.g. `3,5,7`) |
| `--separateClusters`  | off            | Emit the full plot set once per cluster count found in the (filtered) data; filenames get a `_clusters{N}` suffix and titles a ` — N clusters` suffix |
| `--tm`                | all            | Select which `(shift_tm, perturb_tm)` subplots to show. Repeatable and/or comma-separated specs `SHIFT:PERTURB`, each side a number, `none` (parameter unset) or `*` (any); `base` = `none:none`. E.g. `--tm none:0.2 --tm 1:*` |

Subplots are ordered by shift first, then perturb: base, perturbTM, shiftTM, shiftTM+perturbTM, … .
Invest-regret (E) uses Truth's `Objective` as the reference, emitted both relative —
`(invest-regret obj − truth obj) / |truth obj| * 100` (same convention as `EvaluateMarkov.py`'s `%`
columns) — and absolute (`invest-regret obj − truth obj`, native objective units). Sub-cases whose
Truth objective is missing / 0 / -1 are skipped in both, so the two plots cover the same sub-cases.
