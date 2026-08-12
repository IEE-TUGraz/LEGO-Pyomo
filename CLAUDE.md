# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LEGO (Low-carbon Expansion Generation Optimization) is a mixed integer quadratically constrained optimization
model for energy systems (unit commitment through generation/transmission expansion planning). This repo,
**LEGO-Pyomo**, is a Pyomo re-implementation of the original **LEGO-GAMS** model (kept in this repo under
`LEGO-GAMS/LEGO.gms` as the reference implementation to check behavior against — see "GAMS compatibility" below).

The model is built from **thematic modules** that can be toggled on/off per case study via parameters read from
Excel input files, following a "swiss army knife" philosophy of modularity and flexibility.

This is a git repo with submodules:
- `InOutModule/` — data I/O (Excel reading/writing, `CaseStudy` class, comparison/plot utilities). Installed as an
  editable pip package (`-e InOutModule` in `environment.yml`), so it's imported as `InOutModule.X`, not `LEGO.InOutModule.X`.
- `Conda-Activation-Scripts/` — environment setup scripts, not project logic.

## Environment setup

Requires an MPI implementation installed first (MPICH/OpenMPI/MS-MPI depending on OS), then:

```bash
conda env create -f environment.yml
conda activate LEGO-Pyomo_env
```

Or use the provided activation scripts (`activate_environment_windows.bat` / `source activate_environment_unix.sh`),
which create the env from `environment.yml` if it doesn't exist yet and then activate it.

Key dependencies: `pyomo` (modeling), `gurobi`/`cplex`/`highspy` (solvers), `mpi4py`/`mpi-sppy` (stochastic
programming — Extensive Form, Benders, Progressive Hedging), `tsam` (time series aggregation), `py7zr` (MPS archive
compression), `openpyxl`/`python-calamine` (Excel I/O).

## Running tests

```bash
python -m pytest tests/
```

CI (`.github/workflows/ci.yaml`) runs `python -m pytest tests/ --ignore=tests/test_gamsCompatibility.py` on
Linux for every branch, and additionally on Windows for `main` (GAMS compatibility tests need a GAMS install, so
they're excluded from the general CI run and are meant to be run locally when touching model equations).

Run a single test:

```bash
python -m pytest tests/test_stochasticity.py::test_deterministicVsExtensiveWithNoScenarios
```

Test files:
- `tests/test_examples.py` — validates every folder under `data/` round-trips through the Excel reader/writer
  unchanged, and compares generated MPS files against an archived reference (`tests/data/mps-archive/`, `.mps.7z`
  compressed). Also runs the SOCP variant.
- `tests/test_stochasticity.py` — checks that a deterministic model and an Extensive Form model with no/duplicate
  scenarios produce equivalent MPS output.
- `tests/test_gamsCompatibility.py` — compares Pyomo-generated MPS output against the original GAMS model
  (`LEGO-GAMS/LEGO.gms`); requires a local GAMS installation, excluded from most CI runs.
- `InOutModule/tests/test_ExcelReaderWriter.py` — belongs to the `InOutModule` submodule.

When adding/changing MPS archive comparisons, `tests/data/mps-archive/mps-file-descriptions.txt` must have a
matching description entry for every `.mps`/`.mps.7z` file in the archive (enforced by
`test_documentationMPSArchive`).

## Architecture

### Model assembly (`LEGO/LEGO.py`)

`_build_model(cs: CaseStudy)` is the core model builder: it creates a `pyo.ConcreteModel`, then calls each
module's `add_element_definitions_and_bounds(model, cs)` followed later by `add_constraints(model, cs)`, gated
behind the relevant `cs.dPower_Parameters["pEnableXxx"]` flag. `power.py` (buses, lines, generic sets/params) and
the OPF module (`dcOpf.py`, or `acOpfBfm.py`/`acOpfNim.py` when SOCP is enabled, per `pChooseAC-OPF-Model`) are
always active; everything else (thermal generators, VRES, storage, second reserve, import/export, soft line load
limits, DSM, DGA) is opt-in per case study.

The `LEGO` class wraps a `CaseStudy` + built `pyo.Model`, and supports multiple `ModelType`s for solving:
`DETERMINISTIC` (direct solve), `EXTENSIVE_FORM`, `BENDERS`, `PROGRESSIVE_HEDGING` (all three via `mpi-sppy`,
built through `_scenario_creator`, one Pyomo model per scenario). Benders and Progressive Hedging are marked as
not fully tested.

### Module pattern (`LEGO/modules/*.py`)

Every module follows the same two-function contract, enforced by decorators in `LEGO/LEGOUtilities.py`:

```python
@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model, cs) -> (list[pyo.Var], list[pyo.Var]):
    ...
    return first_stage_variables, second_stage_variables

@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model, cs) -> pyo.Expression:  # returns first_stage_objective contribution
    ...
```

- `add_element_definitions_and_bounds` must return every newly-added `pyo.Var` split into
  **first-stage** (common across all stochastic scenarios) and **second-stage** (scenario-specific) lists — the
  decorator raises if any new variable is missing from either list, or if a listed variable wasn't actually added.
- `add_constraints` must declare (via the second decorator) which `add_element_definitions_and_bounds` functions
  are required to have already run; the decorator also prevents a module's setup functions from running twice.
- Constraints/costs for an optional module must be written so that a disabled module leaves the model unaffected
  (see the `if cs.dPower_Parameters["pEnableDSM"]:` branch in `acOpfBfm.py`'s power balance — this pattern must be
  copied whenever a new optional module's variable is referenced from another module's constraint).
- Module data flows in exclusively through the `CaseStudy` object (`cs.dPower_*`, `cs.dGlobal_*` DataFrames/dicts
  parsed from Excel) — modules don't read Excel directly.

New modules should follow an existing similar module as a template (e.g. `DGA.py` for another per-node,
per-timestep dynamic-limit module) rather than being written from scratch — data reading is registered by adding
new dataframe names to the relevant list in `InOutModule/CaseStudy.py` (`rpk_dependent_dataframes` etc.), then
adding matching `get_*`/`write_*` functions in `InOutModule/ExcelReader.py`/`ExcelWriter.py`.

### Data (`InOutModule/CaseStudy.py`)

`CaseStudy` loads a full case study from a folder of Excel files (defaults under `data/example`, `data/NREL-118`,
`data/exampleStochastic`) into pandas DataFrames (`dPower_*`, `dGlobal_*`), in parallel where dependencies allow.
Dataframes are categorized by their time/scenario dependency (`rpk_dependent_dataframes`,
`rp_only_dependent_dataframes`, `k_only_dependent_dataframes`, `non_time_dependent_dataframes`,
`non_dependent_dataframes`) — this categorization drives scenario filtering (`filter_scenario`) and time-window
filtering (`filter_timestamps`) used for stochastic programming and moving-window solves. All physical values in
the Excel files are in MW.

### Comparisons and diagnostics (`LEGO/helpers/CompareModels.py`, `LEGO/LEGOUtilities.py`)

`compareModels` (in `LEGO/helpers/CompareModels.py`) builds two models (deterministic/extensive-form/GAMS/an
existing MPS file, via `ModelTypeForComparison`) and diffs their generated MPS files — this is the backbone of
most tests. `LEGOUtilities.py` also holds solver-agnostic diagnostics (`analyze_infeasible_constraints`), MPS
7z (de)compression helpers (`MPSFileManager`), Zone-of-Interest objective evaluation, and unit-commitment plotting.

### Running jobs (`Caller.py`)

Standalone script (independent of the `LEGO` package) for queueing/running many CLI-invoked runs sequentially or
in parallel from a text file of shell commands, with `---` lines acting as barriers and file-based flags
(`.started`/`.finished`/`.error`) for coordination between spawned worker processes. Used for batch experiment runs.

## Notes

- `LEGO/notes.md` has design-decision notes worth checking when touching line power bounds, slack/PNS cost,
  Unit-Commitment formulation (based on Damcı-Kurt et al., "Tight and Compact MILP Formulation for the Thermal
  Unit Commitment Problem", doi:10.1109/TPWRS.2013.2251373), ramping constraints, or forced line-investment order.
- GAMS compatibility (`test_gamsCompatibility.py`, `LEGO-GAMS/LEGO.gms`) is a correctness reference — when GAMS
  parity matters for a change, run that test locally (requires GAMS installed) since CI skips it.
