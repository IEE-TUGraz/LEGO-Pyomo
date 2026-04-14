# CLAUDE.md — LEGO Helpers

## Overview

Helper modules for comparing and validating LEGO models. Primarily used by the test suite.

## Files

### CompareModels.py

Utilities for comparing LEGO model results across different solving approaches and with GAMS.

**Key functions:**
- `execute_gams(data_folder, ...)`: Runs the GAMS version of the LEGO model and extracts the objective value for cross-validation
- Supports comparison between model types: `DETERMINISTIC`, `EXTENSIVE_FORM`, `BENDERS`, `PROGRESSIVE_HEDGING`, `GAMS`, `MPS_FILE`

### mpsCompare.py

Loads and compares MPS (Mathematical Programming System) files to verify model structure consistency.

**Key functions:**
- `load_mps(filepath)`: Loads an MPS file using CPLEX
- `get_model_data(model)`: Extracts model structure — variables, bounds, objective coefficients, constraints, RHS values, constraint senses (supports linear and quadratic terms)

Used by `tests/test_examples.py::test_comparisonAgainstMPS` to verify that the current model matches archived reference MPS files.
