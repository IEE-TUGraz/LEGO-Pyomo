# CLAUDE.md

This file provides guidance to Claude Code when working on ID experiment scripts.
See [`README.md`](README.md) for usage, key concepts, and CLI parameter reference.

## Non-obvious patterns

**Run order matters**: The hourly model always runs first to establish the baseline objective value. All regret calculations for daily/weekly/monthly/yearly variants subtract this baseline. If the hourly run is skipped or fails, regret values will be wrong.

**Clustering on original data vs. per-level**: With `--clusterOnOriginalData`, hourly data drives cluster assignment and the same cluster mapping is applied to all aggregation levels — this ensures clusters are comparable across levels. Without this flag, each level is clustered independently, so the representative periods differ between levels.

**File naming**: Only non-default parameters appear in the identifier portion of the filename. The aggregation level (`hourly`, `daily`, `weekly`, `monthly`, `yearly`) is always the final segment before `-regret`.