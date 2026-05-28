# CLAUDE.md — LEGO Package

## Overview

The `LEGO/` package contains the core optimization model builder. The main class `LEGO` orchestrates model construction, solving, and result extraction. It uses a modular architecture where constraint modules in `modules/` are conditionally loaded based on case study parameters.

## Key Files

- **`LEGO.py`**: Main `LEGO` class and supporting functions (model building, solving, objective evaluation)
- **`LEGOUtilities.py`**: Decorators, ZOI objective evaluation, MPS file handling, Markov chain helpers, unit commitment slack for regret runs
- **`modules/`**: Modular constraint definitions (see `modules/CLAUDE.md`)
- **`helpers/`**: Model comparison utilities (see `helpers/CLAUDE.md`)

## LEGO Class

### Constructor
```python
LEGO(cs: CaseStudy = None, model: pyo.Model = None, results=None)
```

### Public Methods

| Method                                  | Returns                            | Description                                       |
|-----------------------------------------|------------------------------------|---------------------------------------------------|
| `build_model(model_type=DETERMINISTIC)` | `(model, build_time)`              | Builds Pyomo model by calling all enabled modules |
| `solve_model(model_type=DETERMINISTIC)` | `(results, solve_time, objective)` | Solves model with configured solver               |
| `get_objective_value(zoi=False)`        | `float`                            | Total or ZOI-filtered objective                   |
| `get_number_of_variables()`             | `int`                              | Count of variable instances                       |
| `get_number_of_constraints()`           | `int`                              | Count of constraint instances                     |
| `copy()`                                | `LEGO`                             | Deep copy of instance                             |

#### `solve_model` recovery behavior (Gurobi path)

`solve_model(..., tee=True, raise_on_no_solution=False)` uses `gurobi_persistent` with `load_solutions=False` and owns all solver-failure recovery:
- **WorkLimit / error-with-solution**: if the solver returns `status=error` but `SolCount > 0` (e.g. work limit hit mid-solve), it promotes the status to `warning`, patches `termination_condition` to `"WorkLimit reached"` (only when Gurobi's `Status == GRB.WORK_LIMIT`), then loads the partial solution.
- **Exception-with-solution** (e.g. OOM): if `solve()` throws but a solution exists, it recovers via `optimizer.load_vars()`.
- **No solution**: builds a synthetic `SolverResults` (`status=error`, `termination_condition="Out of Memory"` for a Gurobi OOM `GurobiError`). By default (`raise_on_no_solution=True`) it raises a `RuntimeError`, otherwise it logs via `printer.error` and returns.
- **Side effects**: sets `self.results`, `self.work_units`, `self.mip_gap` (extracted even after a crash), and `self.has_solution` (gate writes/result handling on this). Options are applied silently (callers/`_apply_solver_options` do the logging).

Non-Gurobi solvers take the `else` branch: solutions load automatically (`load_solutions=True`), `self.has_solution` is derived from the termination condition (`optimal`/`feasible`), and `work_units`/`mip_gap` stay `None`. The solver is chosen from the case study's `pSolver` unless `solver_name` is passed. Solver-option handling differs: `pMIPGap` is applied as `mip_rel_gap` for HiGHS; `pWorkLimit`/`pDisableCrossover`/`pForceBarrier`/`pNodeFileStart`/`pNodeFileDir`/`pThreads` are Gurobi-only and `pMIPGap` on any other non-Gurobi solver — each is reported via `printer.error` and ignored rather than silently dropped.

**Gurobi memory-management options** (Gurobi path only): `pNodeFileStart` (GB → `NodefileStart`) spills B&B nodes to disk when in-memory storage exceeds the threshold — fix for MIP OOM. `pNodeFileDir` (path → `NodefileDir`) is the spill directory; `solve_model()` auto-creates it via `os.makedirs(..., exist_ok=True)` before passing to Gurobi. `pThreads` (int → `Threads`) caps the per-process thread count — useful when running many Pythons in parallel to avoid CPU oversubscription and per-thread memory overhead. All three are read from `cs.dGlobal_Parameters` and stored as plain model attrs by `_build_model()`.

### Model Types (ModelType enum)

- **`DETERMINISTIC`**: Single optimization problem (default, most common)
- **`EXTENSIVE_FORM`**: All stochastic scenarios in one model (uses mpi-sppy)
- **`BENDERS`**: Benders decomposition (experimental)
- **`PROGRESSIVE_HEDGING`**: Progressive hedging algorithm (experimental)

### Typical Usage

```python
from InOutModule.CaseStudy import CaseStudy
from LEGO.LEGO import LEGO

cs = CaseStudy("data/example")
lego = LEGO(cs)
model, build_time = lego.build_model()
results, solve_time, objective = lego.solve_model()
```

For regret runs (fixing investment decisions from a source model):
```python
from LEGO.LEGO import build_from_clone_with_fixed_results
regret_model = build_from_clone_with_fixed_results(dc_model, source_model, variables_to_fix=['vGenInvest'])
```

### Model Building Flow

1. `_build_model(cs)` creates a `pyo.ConcreteModel()`
2. Calls `add_element_definitions_and_bounds()` on each enabled module (sets, parameters, variables)
3. Calls `add_constraints()` on each enabled module (constraints, objective contributions)
4. Optionally relaxes integer variables (if `pEnableRMIP`)
5. Module execution order: power → thermalGen → vres → storage → secondReserve → importExport → softLineLoadLimits

### First-Stage vs Second-Stage Variables

For stochastic models, variables are split into:
- **First-stage (investment)**: Shared across scenarios — `vGenInvest`, `vLineInvest`
- **Second-stage (operational)**: Scenario-specific — all dispatch, commitment, flow variables

Each module's `add_element_definitions_and_bounds()` returns `(first_stage_vars, second_stage_vars)` lists, validated by the `@safetyCheck_AddElementDefinitionsAndBounds` decorator.

## LEGOUtilities.py

### Module Safety Decorators

- **`@safetyCheck_AddElementDefinitionsAndBounds`**: Prevents duplicate module execution, validates all variables are assigned to first/second stage lists
- **`@safetyCheck_addConstraints(required_functions)`**: Ensures dependency modules ran first, prevents duplicate execution

### ZOI Objective Functions

- `evaluate_zoi_objective(model, line_filter)`: Computes objective for ZOI components only. `line_filter="both"` assigns 50% weight to inter-zone lines
- `extract_zoi_objective_data(model)`: Extracts lightweight data dict for storage/pickling
- `evaluate_zoi_objective_from_data(zoi_data, new_zoi_i, line_filter)`: Recomputes ZOI objective with different ZOI bus definitions

### Regret Run Helpers

- `add_UnitCommitmentSlack_And_FixVariables(regret_lego, original_model, ...)`: Adds slack variables for unit commitment in regret runs, fixing `vCommit` to source values with penalty for deviations

### Other Utilities

- `addToSet(model, set_name, values)` / `addToParameter(model, param_name, values)`: Used by modules to extend shared model components
- `set_range_cyclic()` / `set_range_non_cyclic()`: Set element range helpers for time-indexed constraints
- `markov_summand()` / `markov_sum()`: Markov chain-based representative period edge handling
- `MPSFileManager`: Context manager for MPS file compression/decompression (7z)
- `evaluate_gen_investment_by_technology(model, filter_zoi)`: Aggregates vGenInvest by technology

## Standalone Functions in LEGO.py

- `get_objective_value(model, zoi)`: Calculates objective (overall or ZOI-filtered)
- `build_from_clone_with_fixed_results(model_to_clone, model_with_results, variables_to_fix)`: Clones model and fixes specified variables (used for regret runs)
- `addToSet()` / `addToParameter()`: Module helpers for extending sets/parameters
