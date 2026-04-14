# Technical Representation (TR) Experiments

Experiments on representing electricity networks with different formulations (DC-OPF, Transport Problem, Single Node). The goal is to achieve faster solves by using a detailed formulation (e.g. DC-OPF) in a zone of interest (ZOI) while simplifying the rest of the network.

## Running Experiments

```bash
# Run basic experiment
python research/TR/TechnicalRepresentation.py data/example

# Run all zones (DC, TP, SN, and all zones from Power_BusInfo), including regret runs
python research/TR/TechnicalRepresentation.py data/NREL-118 --limitK k0001-k0024 --all

# Run for a specific ZOI with scaling and regret
python research/TR/TechnicalRepresentation.py data/NREL-118 --limitK k0001-k0720 \
    --scaleDemand 0.7 --scalePMax 0.5 --noOverwrite --zoi R1
```

## Key Concepts

**Technical Representation (`pTecRepr`)** — three network formulations:
- `DC-OPF`: Full DC optimal power flow with voltage angles
- `TP`: Transport model (simplified, no angles)
- `SN`: Single node (buses collapsed via `merge_single_node_buses()`)

**Zone of Interest (ZOI)**: Buses flagged as ZOI receive the most detailed representation. `TechnicalRepresentation.py` assigns formulations based on BFS distance from ZOI.

**Special ZOI values**: `--zoi DC` (uniform DC-OPF), `--zoi TP` (uniform TP), `--zoi SN` (uniform SN). Omit `--zoi` to use the formulations as specified in the Excel files.

**SN Cross-Zone Prevention**: `prevent_cross_zone_sn()` upgrades cross-zone SN connections to TP before `merge_single_node_buses()`, preserving zone boundaries. Use `--preventCrossZoneMerging` to enable this.

**Regret Runs**: Fix all `vGenInvest` from a source model into a full DC-OPF model with all buses as ZOI, then re-solve. Regret = (regret objective − DC baseline objective), showing the cost of using a simplified representation.

**Per-Zone Analysis**: The `i_zone` set (bus, zone tuples) in the SQLite output enables per-zone objective breakdowns. Evaluation scripts use regret files for investment data because they have complete DC-OPF bus-zone mappings.

## Script Reference

### `TechnicalRepresentation.py` — Main experiment script

| Parameter | Default | Description |
|---|---|---|
| `caseStudyFolder` | — | Path to data folder |
| `--zoi` | None (use Excel) | Zone of interest, or `DC`/`TP`/`SN` for uniform representation |
| `--all` | off | Run DC, TP, SN, all zones, plus regret runs |
| `--limitK` | — | Restrict timesteps, e.g. `k0001-k0024` |
| `--dcBuffer` | 0 | Extra buffer layers assigned DC-OPF around ZOI |
| `--tpBuffer` | 0 | Extra buffer layers assigned TP around DC-OPF region |
| `--scaleDemand` | 1.0 | Scale all demand values |
| `--scalePMax` | 1.0 | Scale all generator max production values |
| `--noOverwrite` | off | Skip runs where output `.sqlite` already exists |
| `--preventCrossZoneMerging` | off | Upgrade cross-zone SN connections to TP before merging |

**Output naming**: `TR-data{name}-limitK{l}-demand{d}-pmax{p}-dcBuffer{dc}-tpBuffer{tp}-zoi{zone}{-regret}.sqlite`. Only non-default parameters are included in the name.

**Run order with `--all`**: Normal runs first (DC, TP, SN, zone-specific), then regret runs (TP, SN, zone-specific). No regret for DC (would be identical to DC itself).

### `EvaluateGenInvestByTechnology.py` — Generator investment analysis

Analyzes generator investment capacity by technology from `.sqlite` files. Uses regret files for investment data (complete DC-OPF bus-zone mappings); DC baselines use their own file.

```bash
python research/TR/EvaluateGenInvestByTechnology.py              # current directory
python research/TR/EvaluateGenInvestByTechnology.py <folder>
```

Prints one table per technology per comparison group. Columns: Source, DC-Buf, TP-Buf, Total, Sum(zones), Check, Rel% vs DC, WU, WU%, per-zone values + per-zone Rel%.

### `EvaluateZOIObjective.py` — ZOI objective analysis

Analyzes ZOI objectives with per-zone breakdowns (objectives, PNS, ENS).

```bash
python research/TR/EvaluateZOIObjective.py                       # current directory
python research/TR/EvaluateZOIObjective.py <folder>
```

Prints 3 tables per comparison group: **Objectives (+WU)**, **PNS**, **ENS**. Groups results by (input_dir, limit_k, demand, pmax). DC baseline is used as reference for relative percentages.