# CLAUDE.md

This file provides guidance to Claude Code when working on TR experiment scripts.
See [`README.md`](README.md) for usage, key concepts, and CLI parameter reference.

## Non-obvious patterns

**Zone `None` vs string `"None"` (critical)**: The ZOI can be Python `None` (when loaded from SQLite `run_parameters`) or the string `"None"` (when parsed from a filename). Always check both: `(zone is None or zone == "None")` when identifying the zoiNone baseline. Checking only one will silently miss cases.

**Regret files for investment data**: `EvaluateGenInvestByTechnology.py` reads investment results from regret `.sqlite` files, not from source files. Regret runs use full DC-OPF topology which produces complete bus-zone mappings in `i_zone`. DC baselines use their own file directly (no regret run is needed for DC).

**`prevent_cross_zone_sn()` and the `--preventCrossZoneMerging` flag**: The function is called before every SN run regardless. The flag controls whether the cross-zone SN→TP upgrade is actually applied inside it. Check the flag at the call site, not inside the function.

**File naming and grouping**: Only non-default parameter values appear in the filename. Evaluation scripts group by `(input_dir, limit_k, demand, pmax)` — dcBuffer and tpBuffer are display columns, not grouping keys.