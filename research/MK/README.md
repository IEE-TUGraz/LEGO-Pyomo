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
| `--clusters`             | 1              | Number of k-medoids clusters (1 = no clustering); comma-separated list runs several, e.g. `3,5,7,10,14,18`  |
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
| `--prepare-only`         | off            | Only create the preprocessed input folders (limitK, stretch demand, clusters, …) and exit without solving   |
| `--original-reference`   | off            | Also evaluate each model's decisions in the **original** (unclustered) full-chronological model — see "Original reference" below. Requires `--invest-regret` and/or `--calculate-regret` |
| `--original-reference-only` | off         | Only solve the original full-chronological model of the (preprocessed) folder (`MK-…-TruthOriginal.sqlite`) and exit |


**Output naming**: `MK-{identifier}-{edgeHandling}.sqlite`. Regret files append `-regret`, `-invest-regret`, or
`-operational-regret`; operational runs (`--operational`) append `-operational`.
Non-default parameters are encoded in the identifier (e.g. `filterZoneR1`, `relaxed3`, `rMIP`, `mipGap0.01`,
`workLimit500`, `networkTP`, `commitConsumption0.5`, `startupConsumption2`, `shiftTM2`, `perturbTM0.5`,
`scaleVRES0.8`, `scaleInvestCost0.5`).

**Regret runs** (`--calculate-regret`, `--invest-regret`): the edge handling's main-run decisions come from the
in-memory model if its main run was solved in this session, otherwise from its `.sqlite` file, otherwise (when the main
run was skipped by `--no-overwrite` because of a sibling run differing only in `work_limit`) from that sibling's file.

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

**Per-run analysis tables**: every written `.sqlite` (main, operational and all regret runs) additionally gets
tables computed right after the solve (section "Per-run analysis" in `Markov.py`):

| Table                        | Content                                                                                                                  |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `mk_metrics`                 | Model size (Gurobi `NumVars`, `NumBinVars`, `NumConstrs`, `NumNZs`), number of RPs and nonzero transition probabilities, memory (Gurobi `MaxMemUsed`, process peak RSS), solver runtime, work units, objective split into investment/operating cost, curtailment and load shedding (MWh and %), vCommit corrections of soft-fixed runs |
| `mk_nonbinarity`             | How non-binary `vCommit`/`vStartup`/`vShutdown` are: share of RP transitions with a fractional value, maximum and mean distance to 0/1, per variable and for the relaxed window (first `max(MinUp, MinDown)` hours of each RP) |
| `mk_nonbinary_values`        | Every fractional UC value (raw data for the distribution plots)                                                          |
| `mk_feasibility`             | Static check (no solve) whether the UC schedule, laid out along the chronology, would be feasible in the full chronological model |
| `mk_feasibility_violations`  | Violations per day, unit and check                                                                                       |

The transition-matrix columns of `mk_metrics` (`num_rps`, `num_k_per_rp`, `tm_entries`, `tm_nonzero`) always describe
the RP model whose decisions are evaluated. In regret files the re-solved model is full-hourly (1 RP × 8760 h), so they
report the edge handling's RP model instead — e.g. for 7 RPs the main file and its `-invest-regret` file both show
`num_rps = 7`, `tm_entries = 49`, while `num_vars` etc. describe the model actually solved. (Truth's own main file
reports its single RP; its `-original-invest-regret` file reports the RP case study Truth was built from.)

A **transition** is one chronological RP boundary (day d-1 → d) times one active thermal unit (existing or invested).
Tolerance for "fractional" is `1e-4` (above Gurobi's `IntFeasTol`). The feasibility check lays the RP schedule out via
Hindex and checks start-up/shut-down logic, ramping, the shut-down output limit and minimum up/down times hour by hour.
Every fractional value counts as a violation (`feas_infeasible_pct`); additionally the commitment is rounded at 0.5, the
start-ups/shut-downs are derived from it and minimum up/down times are checked again (`feas_infeasible_rounded_pct` —
dispatch-related checks are not repeated there, since the dispatch could always be adjusted to a given commitment).
So the strict value also counts start-up/shut-down decisions that do not match the chronology (e.g. Cyclic starts a unit
at the beginning of an RP although it was already running in the chronologically preceding day), while the rounded value
only counts commitments that are infeasible as such (minimum up/down times violated across the boundary).

**Original reference** (`--original-reference`): Truth is the full-chronological model rebuilt from copies of the RPs.
With this flag each model's decisions are additionally evaluated in the model with the *original* time series (the
folder before clustering): `-original-invest-regret` (vGenInvest fixed, also for Truth — isolating the clustering error)
and `-original-regret` (vGenInvest + vCommit fixed, like `--calculate-regret`). These files carry `reference=original`
in `run_parameters`. The original model's own optimum is solved once per folder with `--original-reference-only`
(`MK-…-TruthOriginal.sqlite`), since it does not depend on the number of RPs. MinUp/MinDown times are capped to the RP
length in the original model as well, so only the time series differ. Skipped (with a warning) for `--shift-tm` /
`--perturb-tm`, whose chronology is resampled and has no original counterpart.

### Revision runs (`jobs-revision.txt`)

All runs for the revision: RTS-GMLC, NREL-118 and TX-123BT × 3/5/7/10/14/18 RPs × demand variability
(100/50/70/90%) × transition matrix (original, shifted by 1, shifted by 2). Stages must run in order:

0. `--prepare-only` jobs — create all input folders (parallel jobs with `--reuse-inputfiles` would otherwise write the
   same folders concurrently).
1. `--original-reference-only` jobs — one original full-chronological solve per dataset and demand variability.
2. RP jobs — independent of each other and of stage 1 (can run in parallel).
3. Evaluation (`EvaluateMarkov.py all`).

`--node-file-dir $TMPDIR/gurobi-nodes` assumes a Linux cluster with node-local `$TMPDIR`.

### `EvaluateMarkov.py` — Result evaluation

Reads all `MK-*.sqlite` files of a folder (with `--recursive` also of its subfolders) and evaluates them with one of
four subcommands:

| Subcommand | Output |
|------------|--------|
| `tables`   | Per-group comparison tables in the terminal (optionally unit-commitment plots with `--plot`) |
| `plots`    | Boxplot PNGs of the edge handlings vs Truth + the aggregated results table behind them (`compare_markov_results.txt/.csv`) |
| `summary`  | `markov_runs.csv`, `markov_summary.csv` and the non-binarity / feasibility plots of the per-run analysis tables |
| `all`      | All of the above from a single load of the files |

```bash
python research/MK/EvaluateMarkov.py tables results/                                  # comparison tables
python research/MK/EvaluateMarkov.py tables results/ --plot --case-study-folder data/example
python research/MK/EvaluateMarkov.py plots results/ --output-dir plots/ --no-show
python research/MK/EvaluateMarkov.py plots results/ --nrOfClusters 3,5,7 --separateClusters
python research/MK/EvaluateMarkov.py plots results/ --tm none:0.2 --tm 1:none         # only those two TM subplots
python research/MK/EvaluateMarkov.py plots results/ --tm "base,1:*"                   # base + everything with shiftTM=1
python research/MK/EvaluateMarkov.py summary results/ --output-dir results/summary
python research/MK/EvaluateMarkov.py all results/ --no-show --separateClusters
```

| Parameter               | Subcommands    | Default      | Description |
|-------------------------|----------------|--------------|-------------|
| `folder`                | all            | `.`          | Folder with `MK-*.sqlite` files |
| `--recursive`           | all            | off          | Also search subfolders |
| `--output-dir`          | all            | input folder | Directory for PNGs, CSVs and the results table |
| `--include-nonoptimal`  | plots, summary | off          | Also use runs with `termination_condition != 'optimal'` (`tables` always shows all runs, with the status highlighted) |
| `--nrOfClusters`        | all            | all          | Comma-separated cluster counts; only runs whose `clusters` run parameter is in the list (e.g. `3,5,7`). Unclustered runs (incl. `TruthOriginal`) are dropped when given |
| `--tm`                  | all            | all          | Select `(shift_tm, perturb_tm)` combinations. Repeatable and/or comma-separated specs `SHIFT:PERTURB`, each side a number, `none` (parameter unset) or `*` (any); `base` = `none:none` |
| `--no-show`             | all            | off          | Don't display figures (only save them) |
| `--plot`                | tables         | off          | Unit-commitment plot of each group's main runs |
| `--case-study-folder`   | tables         | —            | Case study folder for the plot (fallback if the `.sqlite` has no `hindex`) |
| `--number-of-hours`     | tables         | 144          | Hours shown in the plot |
| `--start-hour`          | tables         | 1            | First hour shown in the plot |
| `--logscale`            | plots          | off          | Log-scale y-axis for the work-units plots |
| `--markov-strict`       | plots          | off          | Also draw a Markov-Strict box (runs with `--enable-strict-markov`) |
| `--separateClusters`    | plots          | off          | Emit the full plot set (and results table) once per cluster count; filenames get a `_clusters{N}` suffix |
| `--no-results-table`    | plots          | off          | Skip the aggregated results table |
| `--no-plots`            | summary        | off          | Only write the CSV tables |

**Run kinds** are told apart by the file suffix: main, `-operational`, `-regret`, `-invest-regret`,
`-operational-regret` (and `-original-regret` / `-original-invest-regret`, which carry `reference=original`).
**Regret** is computed once for all subcommands as `objective − reference objective`: regret and invest-regret
against the optimal Truth main run of the same TM variant and sub-case, operational-regret against Truth-operational,
`-original-*` runs against `TruthOriginal` (same folder before clustering, same other run parameters). `plots` and
`summary` use the optimal main/operational runs that pass the filters, plus the optimal regret runs whose base run
(the run whose decisions they evaluate) was selected.

#### `tables`

One block per run-parameter group: the main comparison table (objective, first-stage share, work units, status,
MIP gap, weighted `vGenP`/`vCommit`/`vStartup`/`vShutdown`/`vPNS`/`vEPS`, with `%` columns relative to Truth), the
investment table (`vGenInvest`) and the invested-capacity table (`vGenInvest * pMaxProd`), each in total and per
technology. Operational runs and every regret variant are printed in their own table per group (regret tables have no
Truth row, so their `%` columns read against Markov). Original-reference runs form their own group.

#### `plots`

Boxplot PNGs comparing the edge handlings (NoEnf, Cyclic, Markov — plus Markov-Strict with `--markov-strict`) against
Truth. Each figure has one subplot per `(shift_tm, perturb_tm)` combination (shared y-axis, ordered base, perturbTM,
shiftTM, shiftTM+perturbTM, …), and within each subplot one box per edge handling. Every box aggregates over the
**sub-cases** sharing that TM combination — the other run parameters that vary (`clusters`, `stretch_demand`, …).
Truth is the reference, never a box. There are **18 logical plots**, each emitted twice — with all edge handlings and
with NoEnf excluded (`_noNoEnf` suffix), since NoEnf's large deviations often compress the scale:

| Base filename                                          | Content                                                          |
|--------------------------------------------------------|------------------------------------------------------------------|
| `compare_workunits_{operational,investment}_absolute`  | Work units                                                       |
| `compare_workunits_{operational,investment}_relative`  | Work units as % of Truth (Gurobi only; empty under HiGHS)        |
| `compare_vshutdown_{operational,investment}_absolute`  | vShutdown deviation vs Truth (signed, y-axis symmetric around 0) |
| `compare_vshutdown_{operational,investment}_relative`  | vShutdown deviation vs Truth [%]                                 |
| `compare_vshutdown_{operational,investment}_*_magnitude` | \|deviation\| (absolute and relative), y-axis from 0           |
| `compare_invest_regret_{absolute,relative}`            | Invest-regret over the Truth-main objective, with MIP-gap band   |
| `compare_regret_{absolute,relative}`                   | Regret (investment + commitment fixed) over the Truth-main objective |
| `compare_operational_regret_{absolute,relative}`       | Operational-regret over the Truth-operational objective          |

"Operational" plots use the `--operational` runs (vGenInvest fixed to Truth's investment) and their Truth-operational
reference; "investment" plots use the regular main runs. The regret plots' y-axis reaches only as far below 0 as
needed (regret is expected to be non-negative, small negative values are solver noise). Both invest-regret plots draw
a light-red **MIP-gap noise band**: a regret inside it could be explained by solver tolerance alone. It uses the
requested `mip_gap` run parameter — `mip_gap * |invest-regret objective|` on the absolute plot (a band, since the
objectives differ between sub-cases), `±mip_gap * 100 %` on the relative plot (a line for a uniform `mip_gap`).

**Aggregated results table**: the **mean** (plus median / min / max) *behind* the boxplots, one row per
**(TM_variant, method)** cell, printed and saved as `compare_markov_results.txt` (+ `.csv`; `_clusters{N}` suffix under
`--separateClusters`). `TM_variant` is `Original` (no TM shift), `ShiftN`, with `+perturbX` if perturbed.

| Column                 | From the plot                              | Notes |
|------------------------|--------------------------------------------|-------|
| `oper_dev_*_pct`       | `compare_vshutdown_operational_relative`   | relative vShutdown deviation vs Truth-operational; mean/median/min/max |
| `invest_regret_*_MEUR` | `compare_invest_regret_absolute`           | absolute invest-regret; the LEGO objective is already in **M EUR** |
| `wu_oper_mean_pct`     | `compare_workunits_operational_relative`   | operational work units as % of Truth-operational (Gurobi only) |
| `wu_invest_mean_pct`   | `compare_workunits_investment_relative`    | investment work units as % of Truth-main (Gurobi only) |

A diagnostics table adds the start-up deviation (differs from shut-downs for NoEnf, equal for Cyclic/Markov) and the
per-metric run counts, and a reference table the mean Truth objective per `TM_variant`. For the
`Original/Shift1/Shift2` × `{3,5,7}` grid, pass e.g. `--nrOfClusters 3,5,7 --tm base --tm 1:none --tm 2:none`.

#### `summary`

Collects the per-run analysis tables (see above) into `markov_runs.csv` (one row per file — all files, incl.
non-optimal ones — with run parameters, solver statistics, all metrics and the regret) and `markov_summary.csv` (mean
over the variants per dataset × number of RPs × edge handling × run kind; run kinds of original-reference runs get an
`@original` suffix), and plots per dataset the distribution of how non-binary the Markov RP transitions are
(`nonbinarity_{dataset}.png`) and the share of infeasible transitions (`feasibility_{dataset}.png`).
