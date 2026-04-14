# Inflow Details (ID) Experiments

Experiments on how temporal aggregation of hydro inflow data affects optimization results. Hourly inflows serve as the reference; yearly/monthly/weekly/daily averages represent progressively simplified representations. Regret runs measure the cost of using simplified inflow data by fixing investment decisions and re-solving with hourly data.

## Running Experiments

```bash
# Run basic experiment (no clustering)
python research/ID/InflowDetails.py data/example

# Run with clustering into representative periods
python research/ID/InflowDetails.py data/NREL-118 --numberOfRPs 10 --lengthOfRPs 24

# Run with scaling and timestep limiting
python research/ID/InflowDetails.py data/NREL-118 --limitK k0001-k0720 \
    --scaleDemand 0.7 --scaleInflows 1.5 --rMIP --noOverwrite

# Run as single node with inflow scaling
python research/ID/InflowDetails.py data/NREL-118 --singleNode --scaleInflows 2.0 --scaleRoRToInflowScaling
```

## Key Concepts

**Inflow Aggregation Levels** — five case study variants from the same base data:
- `hourly`: Original data (reference/baseline)
- `daily`: Inflows averaged over 24-hour windows
- `weekly`: Inflows averaged over 168-hour windows
- `monthly`: Inflows averaged over 720-hour windows
- `yearly`: Single average per generator across all timesteps

**Regret Runs**: Fix `vGenInvest` from a simplified model and re-solve with the original hourly inflows. Regret = (regret objective − hourly baseline objective). No regret run is performed for the hourly model itself.

**Clustering**: Optional k-medoids clustering into representative periods. `--clusterOnOriginalData` clusters all aggregation levels using clusters derived from hourly data; without it each aggregation level is clustered independently.

## Script Reference

### `InflowDetails.py` — Main experiment script

| Parameter | Default | Description |
|---|---|---|
| `caseStudyFolder` | — | Path to data folder |
| `--numberOfRPs` | — | Number of representative periods for clustering |
| `--lengthOfRPs` | — | Length of each representative period in hours |
| `--limitK` | — | Restrict timesteps, e.g. `k0001-k0720` |
| `--scaleDemand` | 1.0 | Scale all demand values |
| `--scaleInflows` | 1.0 | Scale all inflow values |
| `--scaleVRESMaxProd` | 1.0 | Scale VRES max production |
| `--scalePMax` | 1.0 | Scale generator max production |
| `--clusterOnOriginalData` | off | Use hourly-derived clusters for all aggregation levels |
| `--scaleRoRToInflowScaling` | off | Also apply inflow scaling to run-of-river generators |
| `--rMIP` | off | Relax all integer variables before solving |
| `--singleNode` | off | Merge all buses to a single node |
| `--noOverwrite` | off | Skip runs where output `.sqlite` already exists |

**Output naming**: `ID-data{name}{-non-default-params}-{aggregation}{-regret}.sqlite`.

**Run order**: Hourly first (establishes baseline objective), then yearly/monthly/weekly/daily each followed by its regret run.