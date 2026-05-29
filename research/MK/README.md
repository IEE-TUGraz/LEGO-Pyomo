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

**Regret Calculation**: After solving, each model's unit commitment decisions are fixed into the truth model, which is
then re-solved to measure the cost of using simplified edge handling.

**Relaxation**: A percentage of thermal generators can have binary unit commitment variables relaxed to continuous,
ordered by sum of MinUpTime + MinDownTime.

## Script Reference

### `Markov.py` — Main experiment script

Produces `.sqlite` files with model results, run parameters, and solver statistics.

| Parameter                | Default        | Description                                                                                                 |
|--------------------------|----------------|-------------------------------------------------------------------------------------------------------------|
| `caseStudyFolder`        | —              | Path to data folder (comma-separated list for multiple)                                                     |
| `--calculate-regret`     | off            | Re-solve truth model with fixed unit commitment from each model                                             |
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
| `--no-overwrite`         | off            | Skip runs where output `.sqlite` already exists                                                             |
| `--rmip`                 | off            | Relax all integer variables before solving                                                                  |
| `--no-crossover`         | off            | Disable Gurobi crossover (must be paired with `--force-barrier`)                                            |
| `--force-barrier`        | off            | Force Gurobi barrier method (must be paired with `--no-crossover`)                                          |
| `--mip-gap`              | solver default | MIP gap tolerance, e.g. `0.01` for 1%                                                                       |
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

**Output naming**: `MK-{identifier}-{edgeHandling}.sqlite`. Regret files append `-regret` or `-invest-regret`;
operational runs (`--operational`) append `-operational`.
Non-default parameters are encoded in the identifier (e.g. `filterZoneR1`, `relaxed3`, `rMIP`, `mipGap0.01`,
`networkTP`, `commitConsumption0.5`, `startupConsumption2`, `shiftTM2`, `perturbTM0.5`, `scaleVRES0.8`, `scaleInvestCost0.5`).

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

### `EvaluateMarkov.py` — Result evaluation

Reads all `MK-*.sqlite` files in a folder and prints comparison tables. Operational runs (`-operational.sqlite`,
from `--operational`) are shown in a separate "Operational runs" table per group, below the main comparison.

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
| `--number-of-hours`   | Number of hours to show in plots                         |
| `--start-hour`        | Start hour for plots                                     |
