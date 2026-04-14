# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.
See `README.md` for project overview, setup, usage, and data structure.

## Architecture Notes

### Non-obvious behaviors

- `CaseStudy.merge_single_node_buses()` preserves the `z` (zone) column as a sorted unique union string of all merged zones (e.g. `"R1_R2"`)
- `CaseStudy` categorizes all dataframes by time dependency: `rpk` (rp+k indexed), `rp`-only, `k`-only, or non-time-dependent
- Use `cs.copy()` to create independent case study variants — it is a full deep copy, safe to modify without affecting the original

### Model building flow

1. `CaseStudy(data_folder)` loads all Excel files
2. `LEGO(cs)` creates an instance
3. `lego.build_model()` creates a `pyo.ConcreteModel`, then calls `add_element_definitions_and_bounds()` and `add_constraints()` on each enabled module in fixed order: `power → thermalGen → vres → storage → secondReserve → importExport → softLineLoadLimits`
4. `lego.solve_model()` — for Gurobi, uses `gurobi_persistent` to access work units; solver options (`pDisableCrossover`, `pForceBarrier`, `pMIPGap`) are stored as plain attributes on the Pyomo model object in `_build_model()` and applied in `solve_model()`
5. `SQLiteWriter.model_to_sqlite()` exports results

### Research scripts

Each `research/` subfolder has a `README.md` (usage/parameters for humans) and a `CLAUDE.md` (non-obvious patterns). See `research/CLAUDE.md` for an index.

### InOutModule

Non-obvious I/O patterns (CaseStudy read order, ExcelReader version checks, SQLiteWriter behavior, Caller barriers) are documented in `InOutModule/CLAUDE.md`. Human-facing usage is in `InOutModule/README.md`.

## Development Notes

- Modules use `@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds` to prevent duplicate execution; all modules must return `(first_stage_vars, second_stage_vars)` — validated by the decorator
- Use `@LEGOUtilities.safetyCheck_addConstraints([dep_fn])` on `add_constraints()` to enforce module execution order
- Models can be exported to MPS: `model.write("model.mps", io_options={'labeler': NameLabeler()})`
- Tests compare against archived MPS files in `tests/data/mps-archive/` — not relevant when working on research scripts (research scripts do not alter the core LEGO model)

### SQLite Run Parameters

- `SQLiteWriter.add_run_parameters_to_sqlite()` stores all run configuration in the `run_parameters` table
- Evaluation scripts read from this table first (more reliable than filename parsing)
- Parameters stored vary by experiment; common ones: `case_study_directory`, `zoi`, `limit_k`, `dc_buffer`, `tp_buffer`, `scale_demand`, `scale_pmax`