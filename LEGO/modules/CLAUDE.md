# CLAUDE.md — LEGO Modules

## Overview

Each module defines sets, parameters, variables, and constraints for a specific aspect of the power system model. Modules are conditionally loaded based on flags in `cs.dPower_Parameters`.

## Module Interface

Every module implements two functions:

```python
@safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model, cs):
    # Define sets, parameters, variables with bounds
    return first_stage_vars, second_stage_vars

@safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model, cs):
    # Add constraints, modify objective expression
    return first_stage_objective_contribution
```

The `@safetyCheck` decorators prevent duplicate execution and validate that all variables are properly categorized as first-stage or second-stage.

## Module Execution Order

In `_build_model()`, modules execute in this fixed order (both for definitions and constraints):

1. `power` (always)
2. `thermalGen` (if `pEnableThermalGen`)
3. `vres` (if `pEnableVRES`)
4. `storage` (if `pEnableStorage`)
5. `secondReserve` (if `p2ndResUp > 0` or `p2ndResDW > 0`)
6. `importExport` (if `pEnablePowerImportExport`)
7. `softLineLoadLimits` (if `pEnableSoftLineLoadLimits`)

## Modules

### power.py — Core Power System

Always loaded. Defines the fundamental network structure.

**Key sets**: `i` (buses), `la`/`le`/`lc` (lines: all/existing/candidate), `g` (generators, populated by other modules), `gi` (generator-bus), `tec`/`gtec` (technologies), `rp`/`k`/`p` (time), `zoi_i` (ZOI buses), `i_zone` (bus-zone mapping)

**First-stage variables**: `vLineInvest[la]` (binary line investment), `vGenInvest[g]` (integer generator investment)

**Second-stage variables**: `vTheta[rp,k,i]` (voltage angle), `vLineP[rp,k,la]` (power flow), `vGenP[rp,k,g]` (generation), `vPNS[rp,k,i]` / `vEPS[rp,k,i]` (slack: power not served / excess power served)

**Constraints**: Power balance (DC or SOCP), DC-OPF line flow equations (for lines with `pTecRepr='DC-OPF'`), slack node fixing. Optionally adds SOCP (AC-OPF) variables and constraints if `pEnableSOCP`.

### thermalGen.py — Thermal Generator Unit Commitment

**Key sets**: `thermalGenerators` (extends `g`, `gi`, `tec`, `gtec`)

**Second-stage variables**: `vCommit[rp,k,g]` (binary commitment), `vStartup[rp,k,g]` / `vShutdown[rp,k,g]` (binary), `vGenP1[rp,k,g]` (power above minimum)

**Constraints**: Unit commitment logic (startup/shutdown tracking), minimum up/down time, ramp up/down limits, max production during startup/shutdown. Supports three edge handling modes for representative period boundaries: `notEnforced`, `cyclic`, `markov`.

**Objective contribution**: Commitment cost (`vCommit * pInterVarCost`) + startup cost (`vStartup * pStartupCost`)

### vres.py — Variable Renewable Energy Sources

Handles wind, solar, and run-of-river hydro generators.

**Key sets**: `vresGenerators` (extends `g`, `gi`, `tec`, `gtec`)

**Second-stage variables**: `vCurtailment[rp,k,g]` (curtailment)

**Constraints**: `vGenP + vCurtailment = capacity * capacityFactor` — available production must be used or curtailed. Auto-clips inflows exceeding max production.

### storage.py — Energy Storage

Handles batteries, pumped hydro, and long-duration energy storage (LDES).

**Key sets**: `storageUnits`, `intraStorageUnits`, `interStorageUnits` (LDES for multi-period), `hydroStorageUnits`

**Second-stage variables**: `vConsump[rp,k,g]` (charging), `vStIntraRes[rp,k,g]` (energy level within period), `vStInterRes[p,g]` (energy level across periods for LDES), `vStorageSpillage[rp,k,g]` (hydro spillage)

**Constraints**: Intra-period energy balance (charge/discharge/inflows/spillage), inter-period balance for LDES, optional charge/discharge exclusion (`pEnableChDisPower`)

### secondReserve.py — Secondary Reserve Requirements

Enabled when `p2ndResUp > 0` or `p2ndResDW > 0`.

**Key sets**: `secondReserveGenerators` (union of thermal generators and storage units)

**Second-stage variables**: `v2ndResUP[rp,k,g]`, `v2ndResDW[rp,k,g]`

**Constraints**: System-wide upward/downward reserve requirements as fraction of demand. Limits reserve provision by generator capacity headroom. Modifies thermal generator max output constraint to account for upward reserve.

### importExport.py — Import/Export Hubs

**Key sets**: `hubs`, `hubConnections` (hub-bus pairs)

**Second-stage variables**: `vImpExp[rp,k,hub,i]` (positive=import, negative=export)

**Constraints**: Import/export limits per hub (supports fixed or max quantity types). Adds `vImpExp` to the power balance equation.

**Objective contribution**: `vImpExp * pImpExpPrice` (import cost / export revenue)

### softLineLoadLimits.py — Soft Line Capacity Limits

Allows lines to exceed a hard capacity fraction (`pMaxLineLoad`) up to full capacity, at a penalty cost.

**Second-stage variables**: `vLineOverload[rp,k,la]` (overload fraction, 0 to `1-pMaxLineLoad`)

**Constraints**: Line flow bounded by `pPmax * pMaxLineLoad + vLineOverload * pPmax` (separate constraints for existing/candidate lines, positive/negative direction)

**Objective contribution**: Overload penalty (`vLineOverload * pLOLCost`)

## How Modules Extend Shared Sets

Modules like thermalGen, vres, and storage add their generators to the shared `g`, `gi`, `tec`, `gtec` sets defined by `power.py` using `LEGOUtilities.addToSet()`. This allows `power.py` constraints (power balance, investment) to operate over all generator types without knowing the specifics.
