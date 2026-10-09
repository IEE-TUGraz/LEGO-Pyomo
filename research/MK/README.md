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
| `--task`                 | —              | Run exactly **one** solve of a grid point (one `--clusters` value): `main`, `operational`, `regret`, `invest-regret`, `operational-regret`, `original-regret`, `original-invest-regret` or `truth-original` (= `--original-reference-only`). Inputs from earlier solves are read from their result files; exits non-zero unless the output is optimal. Used by `cluster.py` — see "Cluster runs" |
| `--edge`                 | —              | Edge handling of the `--task` solve: `Truth`, `NoEnf`, `Cyclic`, `Markov`, `Markov-Strict` |


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

### `cluster.py` — Cluster runs (Slurm)

`cluster.py` expands the grid of a TOML experiment file into one task per solve with dependencies: every model is solved
once, and each fixed-decision run starts as soon as its decision source has finished. The tasks run either as one Slurm
job each (`submit`, job arrays with dependencies) or in a **task pool** on whole nodes (`submit-workers`, see "Task
pool" below) - the mode used on MUSICA, which prefers full-node jobs.
`experiments/experiment.toml` holds all runs of the paper: RTS-GMLC, TX-123BT, NREL-118 × 3/5/7/14/21/28 RPs × demand
variability 100/50/70/90 % × transition matrix base / shifted by 1 / shifted by 2, plus a random TM (100 % demand only,
queued last).

```
prepare (dataset, demand)               --prepare-only: clustered / demand-stretched input folders, once
 ├─ truth-original (dataset, demand)    Original Truth (raw data)
 └─ per grid point (dataset, demand, clusters, TM):
     ├─ main/Truth                      Adapted Truth (full year from RP copies)
     │   ├─ operational/<edge>          RP model / Truth with the Truth investment
     │   │   └─ operational-regret/<edge>   (also needs main/Truth)
     │   └─ original-invest-regret/Truth    (base TM only)
     └─ main/<edge>                     NoEnf, Cyclic, Markov (RP models)
         ├─ regret/<edge>, invest-regret/<edge>
         └─ original-regret/<edge>, original-invest-regret/<edge>   (base TM only)
evaluate                                after all jobs have ended (successful or not)
```

```bash
# on the login node, from the repo root (Python >= 3.11, standard library only)
export MK_MAIL_USER=you@example.org   # optional Slurm mails (arrays: mail_type, default FAIL; evaluation: END,FAIL)
python research/MK/cluster.py submit research/MK/experiments/experiment.toml --dry-run   # writes runs/<name>-dryrun/
python research/MK/cluster.py submit research/MK/experiments/experiment.toml
python research/MK/cluster.py status experiment                        # counts per task kind, failed/blocked/unknown tasks
python research/MK/cluster.py status experiment --list running --filter 'TX-123BT/*'
python research/MK/cluster.py log experiment TX-123BT/sd0.5/c21/base/main/Truth
python research/MK/cluster.py restart experiment --failed --mem-factor 1.5          # all failed tasks
python research/MK/cluster.py restart experiment 'TX-123BT/sd0.5/c21/*' --time 3-00:00:00
python research/MK/cluster.py prioritize experiment --nice-step 300    # change the queue-order nice of queued jobs
python research/MK/cluster.py cancel experiment
```

**Task keys**: `{dataset}/sd{demand}/c{clusters}/{tm}/{task}/{edge}` (e.g. `TX-123BT/sd0.5/c21/shiftTM1/regret/Markov`),
`{dataset}/sd{demand}/prepare`, `{dataset}/sd{demand}/truth-original` and `evaluate`. `restart`, `log`, `local` and
`--filter` take glob patterns.

**Local runs** (no Slurm, e.g. on a laptop to test tasks before submitting): `local` builds the same commands from the
config (file or name in `experiments/`) and runs the matching tasks in dependency order, from the repo root with the
current Python. Results go to the usual places, so a later cluster run reuses them via `--no-overwrite` (and vice versa).

```bash
python research/MK/cluster.py local experiment '*/prepare' -j 12 --keep-going    # all prepare jobs in parallel
python research/MK/cluster.py local pilot TX-123BT/sd1/prepare                   # one task, output to the console
python research/MK/cluster.py local pilot 'TX-123BT/sd1/c28/base/regret/Markov' --with-deps --dry-run
```

`local` is the pool worker (see "Task pool" below) with a throw-away pool: `-j/--jobs N` runs up to N tasks at once,
each as soon as its dependencies have finished; output then always goes to log files `<key>.<attempt>.log` (`--log-dir`,
default `runs/<name>-local/logs/`). Memory packing is off (`-j` is the concurrency; the TOML memory is sized for Slurm),
but the memory guard still stops the newest task if the machine runs out of memory. All prepare jobs are independent and
write disjoint folders, so they can run fully in parallel (~3 GB RAM each). Mind the memory with solves: full-year
models need tens of GB each. `--with-deps` adds all upstream tasks (finished ones skip quickly via `--no-overwrite`).
Without `--keep-going`, a failure stops starting new tasks (running ones finish); with it, only the tasks depending on
the failure are skipped. Gurobi threads per task = the task's `cpus`, at most CPU count / jobs (`--threads`). The exit
code is 1 if a task failed. For long local solves, run at low priority (PowerShell: `(Get-Process -Id $PID).PriorityClass = 'Idle'` first; cmd:
prefix the command with `start /low /b /wait`, and quote patterns with `"` instead of `'`).

**Job names** start with an 8-character code, because `squeue` shows only 8 characters: dataset (2, first two letters
of the folder name; lower case for low-priority runs) + task (4) + edge (2), then `_` and the array name, e.g.
`TXmainMk_main-Markov-TX-123BT`. Task codes: `prep` prepare, `trOr` truth-original, `main`, `oper` operational, `rgrt`
regret, `iRgr` invest-regret, `oRgr` operational-regret, `OiRg` original-invest-regret, `ORgr` original-regret. Edge
codes: `Tr` Truth, `NE` NoEnf, `Cy` Cyclic, `Mk` Markov, `--` none. Full names: `squeue -o "%.18i %.40j %.8T %.10M"`.

**Queue order**: `grid.datasets` order (RTS-GMLC, TX-123BT, NREL-118), then the runs of `low_priority` TM variants
(own arrays ending in `-low`, e.g. `rtmainMk_main-Markov-RTS-GMLC-low`). Every job gets `sbatch --nice` = rank ×
`[slurm] nice_step` (default 100): 0 / 100 / 200 for the datasets, 300 / 400 / 500 for their low-priority runs. Nice
is fixed per job (also on `restart`); it acts like submitting the job nice / 30 hours later (MUSICA: age weight 10000
over 14 days), against your own and other users' jobs. So it orders your jobs only among those that became eligible
within that margin of each other; a larger step orders more strictly but delays the later ranks more against other
users. Check with `squeue -u $USER -t PD -S -p -o "%.12i %.32j %.10Q %.6y %.20r"` (`%y` = nice). `prioritize
--nice-step N` re-applies the nice of all queued jobs with a new step (stored for later restarts).

**Status categories**: `done` (completed = optimal result file), `running`, `pending`, `blocked` (waits for a failed
task, listed under it), `failed` (FAILED, OUT_OF_MEMORY, TIMEOUT, CANCELLED, …, with peak vs requested memory),
`unknown` (no Slurm record).

**Restart** resubmits the selected tasks (`--failed` and/or patterns; queued ones only with `--include-pending`) as
single jobs, plus all unfinished downstream tasks (their old jobs are cancelled), and re-queues the evaluation behind
them. `--mem`/`--mem-factor`/`--cpus`/`--time` apply to the selected tasks only. Finished tasks are never resubmitted;
delete a result file to recompute it.

**Config** (see `experiments/experiment.toml`):
- `[slurm]`: account, partition, QoS, optional `nice_step` (see Queue order), optional `mail_user`/`mail_type` (leave them out of versioned configs and use
  `$MK_MAIL_USER`), `repo` (repo path on the cluster), `setup` (environment lines).
- `[markov]`: `args` for every solve; `node_file_start`: Gurobi `NodefileStart` in GB, one number or a table per dataset
  (`{ default = 8, "TX-123BT" = 16 }`), identical for all solves of a dataset because it affects the runtime. It limits
  only the B&B tree in memory (not the model): once the tree is larger, Gurobi compresses nodes and writes them to the
  node-local `$TMPDIR`. Keep pilot and full run identical. `--threads` is always the allocated core count.
- `[slurm] disk_min_free_gb` (default 10): every job logs its node-file peak and the free space of the node-file disk
  (`Node files: peak … MB, min free disk … MB` at the end of the log) and warns below this.
- `[pool]` (task pool only): `seed_runs` (runs whose measurements seed the estimates, e.g. `["pilot"]`), `cores` and
  `mem` per worker node (required by `submit-workers`), `walltime` (72 h), `chain_after_hours` (24), `workers` (3),
  `mem_fraction` (share of `mem` that is packed, 0.9), `idle_hours` (3), `mem_safety` (1.2), `time_safety` (1.5),
  `disk_min_free_gb` (10), `rank_offset_hours` (soft dataset order, see Task pool; absent = strict), `mail_type` for the worker jobs (default: `[slurm] mail_type`; address from `[slurm] mail_user` /
  `$MK_MAIL_USER`, written into `worker.sbatch` at `submit-workers` - edit that file to change it for later successors).
- `[grid]`: `datasets`, `stretch_demand`, `clusters`, `edges`, `tasks`, `truth_original`, and `tm` entries `base` /
  `shift:N` / `perturb:R`, or an inline table `{ spec = "perturb:1.0", stretch_demand = [1.0], low_priority = true }`
  (only these demand levels / queued last).
- `[resources.<class>]`: `cpus`, `mem`, `time` in a `default` table, overridden per dataset folder name. Classes:
  `prepare`, `rp` (main/operational of the RP models), `full` (all full-year solves), `truth-original` (falls back to
  `full`), `evaluate`. All solves of a dataset need the same `cpus` (comparable work units / solver times): `submit`
  refuses other configs, `restart --cpus` warns.
- `[evaluate]`: `args`, `output_dir`.

Run state, command files and logs go to `research/MK/runs/<name>/` (not versioned).

**Task pool** (alternative to one Slurm job per task, sized for whole nodes as MUSICA prefers): `worker` runs the
tasks of a config on the machine it is started on, packing them by **estimated memory** and **cores**, and pulls them
from a shared pool in `runs/<name>/pool/`. Any number of workers (one per node) can work on the same pool at once; each
task runs exactly once. A task starts when its dependencies are done, it fits (sum of `max(estimate, current RSS)` of the
running tasks + its estimate ≤ `mem_fraction` × memory; enough cores), and - if its runtime is known from measurements -
it finishes before the worker's walltime. Priority: the longest chain of dependent tasks (critical path, from the
runtime estimates) minus `[pool] rank_offset_hours` (12) per position in `grid.datasets` - so RTS-GMLC goes first among
equal chains, but a TX-123BT task whose chain is more than 12 h longer goes before it. Without `rank_offset_hours` the
order is strict (all ready tasks of the first dataset first). Ties: largest memory first. If the top task waits too long
(30 min), smaller tasks stop jumping ahead of it. `low_priority` TM variants (the random TM) only use spare capacity:
they start once every ready normal task has been placed; once started they run to the end like any other task (they are
never stopped to make room), but the memory guard evicts low-priority tasks first. Tasks never use more Gurobi threads
in total than the worker's cores.

```bash
python research/MK/cluster.py resources pilot --save                # pilot measurements -> seed of the estimates
python research/MK/cluster.py submit-workers experiment --estimate-only   # remaining work, days for 1/2/3/5/10 chains
python research/MK/cluster.py submit-workers experiment --dry-run   # job script + sbatch calls
python research/MK/cluster.py submit-workers experiment             # [pool] workers chains (default 3); repeat to add more
python research/MK/cluster.py cancel experiment                     # all worker jobs incl. queued successors
python research/MK/cluster.py worker experiment                     # by hand; on a Slurm node: until the job ends
python research/MK/cluster.py worker experiment 'TX-123BT/*' --mem 700G --cores 192 --hours 72
python research/MK/cluster.py status experiment                     # pool mode: counts, workers, retried attempts
python research/MK/cluster.py resources experiment                  # from the pool's measurements
python research/MK/cluster.py restart experiment --failed           # failed tasks become runnable again
```

- **Estimates** come from measurements (peak RSS, runtime) of this run and of `[pool] seed_runs`, most specific first:
  the same task, the same task/edge/RP count, the same task/edge, the resource class (memory only); else the TOML
  `[resources.*]`. Value = largest measurement × `mem_safety` / `time_safety`. They are recomputed continuously, so
  later tasks start with better estimates. Seed a pool with a Slurm run's measurements via `resources <run> --save`
  (`runs/<run>/measurements.json`). Runs that had nothing to do (output existed, folders reused; `Markov.py` prints
  `MK-NOOP`) are not measurements.
- **Memory guard**: no per-task memory limits. If free memory gets short (or the tasks use more than the worker's
  `--mem`), the most recently started task is stopped and retried later with its peak as lower bound (never the only
  running task). Same for the walltime: shortly before the end, running tasks are stopped and retried by another worker.
- **Idle**: a worker without anything to run waits up to `idle_hours` for tasks to become ready (e.g. while another
  worker runs a Truth solve), and exits at once only when nothing is left.
- **Node files** go to `--node-dir` (default `$TMPDIR`); the worker records each task's node-file peak and pauses new
  tasks while the disk has less than `disk_min_free_gb` free. The first log line of a worker shows the node-file
  directory and its free space - it must be the node-local NVMe (`$TMPDIR` in a job), not a small `/tmp`.
- **Whole-node jobs and chaining** (`submit-workers`): each job takes a full node (`--exclusive --mem=0`, `[pool]
  walltime`, default 72 h) and runs one worker with `[pool] cores`/`mem` (MUSICA: 192 / `740G`). After
  `chain_after_hours` (24) it submits the same script again if tasks remain, so the successor waits in the queue while
  it still runs (with ~2 days queue time it starts about when its predecessor ends; if it starts earlier, both share the
  work). A chain ends by itself when no tasks are left at that point. This needs `sbatch` on compute nodes; a failed
  submission is logged as `ERROR` in `runs/<name>/pool/jobs/worker_<job>.out` and retried every 10 min. Tasks longer
  than the walltime can never finish (Gurobi cannot resume) - keep `walltime` at the QoS maximum.
- **How many chains**: `--estimate-only` sums the estimated memory × runtime (and cores × runtime) of the open tasks
  per node and prints the days for 1-10 chains, bounded below by the longest dependency chain. Before measurements
  exist, the TOML `time` (a limit, not a runtime) makes this a wild upper bound - seed with the pilot first. Start
  with a few chains and add more with another `submit-workers --workers N` once the estimate is based on
  measurements; `evaluate` runs automatically after all other tasks.
- Per-attempt logs: `runs/<name>/pool/logs/<key>.<attempt>.log`. A dead worker's tasks are retried after 10 min without
  heartbeat - at once if a new worker starts on the same machine and finds the old worker's process gone.
- **Stopping workers** started by hand (no Slurm job to cancel): `cluster.py stop <run> [--host NAME ...] [--now]`.
  Default: the workers start no new tasks and exit once their running tasks have finished; `--now` interrupts the
  running tasks (retried by other / later workers). Only workers that started before the request are affected.

**Windows servers** (a separate site from MUSICA - results of the two sites are never mixed):
the same task pool, one worker per server, started by hand over Remote Desktop. All workers of a pool must see the same
files - pool, input data and the `MK-*.sqlite` results, which `Markov.py` writes into the repo root - so the servers
share **one clone of the repo on the network drive**; each server needs its own conda env and Gurobi license.
Config: `experiments/experiment-win.toml` - same grid and model options as `experiment.toml`, with `cpus`, memory
and `node_file_start` sized for the servers. Gurobi node files belong on a fast local disk (`--node-dir`; the network
drive works but is slow). Use **one hardware type per pool**: solver times and work units of different CPUs are not
comparable, so other servers need their own config (own `name`) and datasets. The memory of TX-123BT's full-year
solves and Original Truths is capped just below the servers' RAM, so they are tried, each alone on its server. If they
run out of memory, stop scheduling the dataset, e.g. restart the workers with task patterns
(`worker experiment-win "RTS-GMLC/*" "NREL-118/*" ...`).

1. Clone the repo onto the share and create the conda env on every server (`environment.yml`).
2. On each server, in a terminal with the env active, from the repo on the share:

```bash
python research/MK/cluster.py worker experiment-win --detach --node-dir D:/gurobi-nodes
```

3. Control all workers from any one server (or any machine with the share):

```bash
python research/MK/cluster.py status experiment-win                 # counts, workers per server (active / stopping / exited)
python research/MK/cluster.py stop experiment-win                   # all workers: no new tasks, exit after the running ones
python research/MK/cluster.py stop experiment-win --host SERVER2 --now   # one server, interrupt its tasks (retried elsewhere)
python research/MK/cluster.py restart experiment-win --failed       # failed tasks become runnable again for all workers
```

`--detach` starts the worker in the background (normal CPU priority) and returns; its output goes to
`runs/<name>/pool/worker-logs/<server>-<time>-<pid>.log` (`--log` does the same for a worker in the foreground). It keeps
running when the terminal is closed or the Remote Desktop session is *disconnected* - **signing out ends it**, and so
does a reboot (Windows Updates): start it again afterwards; a new worker on the same server retries the tasks of the
previous one at once. If the worker dies, Windows also ends its tasks (job object), so no orphaned solve keeps writing
a result file that another worker retries. Other options as for any worker: `--cores` (default: all **physical** cores),
`--mem` (default: all RAM; `[pool] mem_fraction` of it is packed), `--node-dir` (Gurobi node files - a large local disk;
default `%TEMP%`), `--idle-hours`, `--max-tasks`, task patterns. A worker never starts a task whose `cpus` exceed its
cores (fewer threads would make work units and solver times incomparable), nor a not-yet-attempted task whose memory
estimate exceeds its RAM (it would only swap; estimates between `mem_fraction` x RAM and the RAM run alone): it logs
`skipping ...` and leaves the task to the other workers. The memory check uses the current estimate, so a measurement
(also from `seed_runs`) can lift or impose it; an evicted task's retry is never skipped.

**Pilot before the full run** (per-task mode; optional for the task pool, which learns its estimates while it runs
and can be seeded with a pilot via `[pool] seed_runs`): the resources in `experiment.toml` are guesses. `experiments/pilot.toml` runs one grid
point per dataset (100 % demand, 28 clusters = largest RP models, base TM, all edges and tasks, Original Truth; 78 jobs)
with generous memory. The full run reuses its results via `--no-overwrite` (same model options), so submit it only
after the pilot has finished.

```bash
python research/MK/cluster.py submit research/MK/experiments/pilot.toml
python research/MK/cluster.py resources pilot              # max peak / elapsed per dataset and class, suggested values
python research/MK/cluster.py status pilot --list done     # per task: 'mem' column = peak RSS / requested memory
seff <jobid>                                                # per job, incl. CPU efficiency
```

`resources` groups the finished tasks by dataset and `[resources.*]` class (`prepare`, `truth-original`, `rp`, `full`)
and suggests `mem` = 1.3× the largest peak and `time` = 2× the longest run (at least 30 min; `--mem-factor`,
`--time-factor`, `--min-time`), naming the tasks behind both maxima. `--save` writes the measurements to
`runs/<run>/measurements.json`, the seed for pool runs (`[pool] seed_runs`). Tasks that hit `OUT_OF_MEMORY`/`TIMEOUT` are listed
with the limit they exceeded (the class needs more than that). Copy the values per dataset into the full experiment's
TOML; `restart --mem-factor` covers outliers. Keep a larger margin on time: MIP solve times vary more between demand
levels and TMs than memory does, and the pilot covers only its grid points.

**Tests** of `cluster.py` / `pool.py` (fake tasks that sleep, allocate memory and exit with a given code - no Gurobi,
no Slurm, about 1 min):

```bash
pytest research/MK/tests
```

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
| `--no-chronology`       | plots, summary | off          | Skip the chronological comparisons (see below) - they read the large variable tables |
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
Truth is the reference, never a box. There are **28 logical plots**, each emitted twice — with all edge handlings and
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
| `compare_original_invest_regret_{absolute,relative}`   | Invest-regret against the **original full chronology** (`TruthOriginal`), incl. a Truth box: Truth = clustering error alone, edge handlings = total error |
| `compare_original_regret_{absolute,relative}`          | Regret (investment + commitment fixed) against `TruthOriginal`   |
| `compare_operating_cost_operational_relative`          | Operating cost of the model vs Truth-operational [%]             |
| `compare_curtailment_operational_absolute`             | Renewable-curtailment share vs Truth-operational [pp]            |
| `compare_load_shedding_operational_absolute`           | Load-shedding share vs Truth-operational [pp]                    |
| `compare_dispatch_deviation_operational`               | Hourly dispatch deviation from Truth-operational, Σ\|Δ vGenP\| / Σ vGenP_Truth [%] |
| `compare_storage_deviation_operational`                | Hourly storage-level deviation from Truth-operational [% of energy capacity] |
| `compare_storage_boundary_investment`                  | Storage-level jump at the chronological RP boundaries of the investment runs [% of energy capacity] |

**Chronological comparisons** (`plots` and `summary`, skipped with `--no-chronology`), laid out along the stored
`hindex` without solving anything: the **transition-matrix structure** of every grid point (`tm_off_diag_mass` = share
of period-to-period transitions that change the RP, i.e. where the cyclic assumption is wrong; `tm_entropy_norm` =
entropy rate / log(#RPs)); the **storage boundary check** of the RP runs (at every chronological boundary, the level
the first timestep of an RP starts from - recovered from its energy balance - against the level the predecessor
period actually ends with; `st_boundary_violation_pct` = share of (boundary, unit) instances above 0.1 % of the
capacity, `st_boundary_{mean,max}_dev_pct`); and the hour-by-hour **dispatch / storage-level deviation** of the
operational runs from Truth-operational (same fleet: `dispatch_dev_pct`, `storage_dev_pct`).

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
per-metric run counts, a reference table the mean Truth objective per `TM_variant`, and an **operational fidelity /
full chronology** table the means behind the new plots (operating cost, curtailment, load shedding, dispatch and
storage deviation, storage boundary jump, regret against `TruthOriginal` - with a Truth row for the clustering error). For the
`Original/Shift1/Shift2` × `{3,5,7}` grid, pass e.g. `--nrOfClusters 3,5,7 --tm base --tm 1:none --tm 2:none`.

#### `summary`

Collects the per-run analysis tables (see above) into `markov_runs.csv` (one row per file — all files, incl.
non-optimal ones — with run parameters, solver statistics, all metrics and the regret) and `markov_summary.csv` (mean
over the variants per dataset × number of RPs × edge handling × run kind; run kinds of original-reference runs get an
`@original` suffix), and plots per dataset the distribution of how non-binary the Markov RP transitions are
(`nonbinarity_{dataset}.png`) and the share of infeasible transitions (`feasibility_{dataset}.png`). Further plots per
dataset: the share of transitions violating each feasibility check (`feasibility_checks_{dataset}.png`; mean/max
violation per check as `feas_{check}_{pct,mean_residual,max_residual}` in the CSVs), model size, nonzero
transition-matrix entries, solver memory, solver time and work units over the number of RPs (`scalability_{dataset}.png`),
and the operational vShutdown deviation, invest-regret and operational-regret over the off-diagonal mass of the
transition matrix (`offdiagonality_{dataset}.png`, one point per sub-case and TM variant).
