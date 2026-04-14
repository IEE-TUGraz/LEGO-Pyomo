# CLAUDE.md

This file provides guidance to Claude Code when working on MK experiment scripts.
See [`README.md`](README.md) for usage, key concepts, and CLI parameter reference.

## Non-obvious patterns

**Solver options flow**: A CLI arg (e.g. `--mip-gap`) sets a key in `cs.dGlobal_Parameters` (e.g. `"pMIPGap"`) inside `execute_case_studies()`. `LEGO._build_model()` reads this and stores it as a plain attribute on the Pyomo model (e.g. `model.pMIPGap`). Both `LEGO.solve_model()` and the direct `execute_case_study()` solve path then read this attribute and apply it to `optimizer.options`. Both code paths must be kept in sync when adding new solver options.

**`--no-crossover` and `--force-barrier` are coupled**: `main()` raises if exactly one is set. This is intentional — disabling crossover without barrier produces an interior-point solution that is not a vertex; the barrier flag is the companion that makes this meaningful.

**`edge_handling` value in run_parameters**: The stored string is the model dict key after `.strip().replace('.', '').replace(' ', '')` — e.g. `"NoEnf"`, `"Cyclic"`, `"Markov"`, `"MarkovStrict"`. Use this normalized form when filtering or grouping SQLite results.

**Evaluation grouping**: `EvaluateMarkov.py` groups results by `(case_study_directory, limit_k, clusters, shift, stretch_demand, relax_count, no_investment, rmip, no_crossover, force_barrier, mip_gap)`. A new run parameter that should split comparison groups must be added to this grouping — and to `load_file_metadata` so it is read from SQLite. When implementing a new run parameter, proactively ask the user whether and how it should be added to `EvaluateMarkov.py` before closing the task.

**SQLite file discovery**: `EvaluateMarkov.py` discovers all `MK-*.sqlite` files excluding those ending in `-regret.sqlite` or `-invest-regret.sqlite`.