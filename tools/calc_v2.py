import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("../InOutModule")
from ExcelWriter import ExcelWriter
import ExcelReader

IN_DIR = "../data/hydroExample/"
OUT_DIR = "../output/inflow_spinup/"
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIO_MAIN = "Scenario_2015"
SCENARIO_SPIN = "Scenario_2014"
SPINUP = 168
HOURS = 8760

# Read input files
dPower_VRES = ExcelReader.get_Power_VRES(IN_DIR + "Power_VRES.xlsx", keep_excluded_entries=True, fail_on_wrong_version=False)
dPower_Storage = ExcelReader.get_Power_Storage(IN_DIR + "Power_Storage.xlsx", keep_excluded_entries=True, fail_on_wrong_version=False)
dPower_HydroNetwork = ExcelReader.get_Power_HydroNetwork(IN_DIR + "Power_HydroNetwork.xlsx", keep_excluded_entries=True, fail_on_wrong_version=False)
dPower_HydroAssets = ExcelReader.get_Power_HydroAssets(IN_DIR + "Power_HydroAssets.xlsx", keep_excluded_entries=True, fail_on_wrong_version=False)
dPower_Inflows_WaterAmount = ExcelReader.get_Power_Inflows_WaterAmount(IN_DIR + "Power_Inflows_WaterAmount.xlsx", fail_on_wrong_version=False)

# Identify hydro assets with outbound pump connections (TurbineOrPump == 1)
pump_assets = set(dPower_HydroNetwork[dPower_HydroNetwork["TurbineOrPump"] == 1].reset_index()["i"].unique())

# Build network dictionaries
pp_pre = {}
pp_follow = {}
for i, j in dPower_HydroNetwork.index:
    pp_follow.setdefault(i, []).append(j)
    pp_pre.setdefault(j, []).append(i)
    pp_pre.setdefault(i, [])

plants = list(pp_pre.keys())
edges = list(dPower_HydroNetwork.index)

inflow_main = dPower_Inflows_WaterAmount[dPower_Inflows_WaterAmount["scenario"] == SCENARIO_MAIN]
inflow_spin = dPower_Inflows_WaterAmount[dPower_Inflows_WaterAmount["scenario"] == SCENARIO_SPIN]

# Add missing plants with zero inflow
rp = inflow_main.index.get_level_values('rp').unique()
k = inflow_main.index.get_level_values('k').unique()
full_idx = pd.MultiIndex.from_product([rp, k, plants], names=['rp', 'k', 'g'])
inflow_main = inflow_main.reindex(full_idx, fill_value=0)
inflow_spin = inflow_spin.reindex(full_idx, fill_value=0)

# =========================
# SIMULATION
# =========================
prod = {p: np.zeros(HOURS) for p in plants}

# Water amount that flowed through p in the PREVIOUS HOUR
inflow_prev = {p: 0.0 for p in plants}

# Spin-up
for t in range(SPINUP):
    inflow_curr = {p: float(inflow_spin.loc[("rp01", f"k{t + 1 + HOURS - SPINUP:04}", p), "value"]) for p in plants}
    for u, v in edges:
        inflow_curr[v] += inflow_prev[u]
    inflow_prev = inflow_curr

# Main calculation
for t in range(HOURS):
    inflow_curr = {p: float(inflow_main.loc[("rp01", f"k{t + 1:04}", p), "value"]) for p in plants}
    for u, v in edges:
        inflow_curr[v] += inflow_prev[u]

    for p in plants:
        prod[p][t] = inflow_curr[p] * dPower_HydroAssets.loc[p, "PowerFactorTurbine"]

    inflow_prev = inflow_curr

# Save Excel file with production per plant
df_energy = pd.DataFrame(prod)
df_energy = df_energy.reset_index(names="k").melt(id_vars="k", var_name="g", value_name="value")
df_energy["scenario"] = SCENARIO_MAIN
df_energy["id"] = None
df_energy["rp"] = "rp01"
df_energy["k"] = (df_energy["k"] + 1).astype(str).str.zfill(4).radd("k")
df_energy["dataPackage"] = "TO BE FILLED"
df_energy["dataSource"] = "TO BE FILLED"

ew = ExcelWriter()
ew.write_Power_Inflows(df_energy, OUT_DIR)

# Calculate MaxProd and MinProd in MWh by multiplying water amounts with PowerFactors
dPower_HydroAssets['MaxProd'] = dPower_HydroAssets['MaxProdWater'] * dPower_HydroAssets['PowerFactorTurbine']
dPower_HydroAssets['MinProd'] = 0.0  # MinProdWater not available in HydroAssets
dPower_HydroAssets['MaxCons'] = dPower_HydroAssets['MaxPumpWater'] * dPower_HydroAssets['PowerFactorPump']

# Split hydro assets: those with pump connections go to Storage, others to VRES
df_hydro_for_storage = dPower_HydroAssets[dPower_HydroAssets.index.isin(pump_assets)]
df_hydro_for_vres = dPower_HydroAssets[~dPower_HydroAssets.index.isin(pump_assets)]

# Add hydro assets to Power_VRES
df_vres_combined = pd.concat([dPower_VRES, df_hydro_for_vres])
ew.write_Power_VRES(df_vres_combined, OUT_DIR)

# Add hydro assets to Power_Storage
df_storage_combined = pd.concat([dPower_Storage, df_hydro_for_storage])
ew.write_Power_Storage(df_storage_combined, OUT_DIR)

# Plotting
month_edges = np.array([0, 744, 1416, 2160, 2880, 3624, 4344,
                        5088, 5832, 6552, 7296, 8016, 8760])
month_centers = (month_edges[:-1] + month_edges[1:]) / 2
month_labels = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

x = np.arange(HOURS)

for p in plants:
    plt.figure()
    plt.title(f"{p} ({SCENARIO_MAIN})")
    plt.plot(x, prod[p])
    plt.xticks(month_centers, month_labels)
    plt.xlabel("Time (Months)")
    plt.ylabel("Energy (MWh)")

    plt.gca().set_xticks(month_edges, minor=True)
    plt.grid(True, which="minor", axis="x")
    plt.grid(True, axis="y")
    plt.xlim(0, HOURS - 1)

    plt.savefig(os.path.join(OUT_DIR, f"{p}_{SCENARIO_MAIN}.png"), dpi=150)
    plt.close()

# =========================
# Production of the whole network
# =========================
total_energy = np.zeros(HOURS)
for t in range(HOURS):
    s = 0.0
    for p in plants:
        s += prod[p][t]
    total_energy[t] = s

window = 168  # 168h moving mean
total_movmean = np.convolve(total_energy, np.ones(window) / window, mode="same")

plt.figure()
plt.plot(x, total_energy, alpha=0.3, label="Original")
plt.plot(x, total_movmean, linewidth=2, label="Moving Mean (168h)")

plt.xticks(month_centers, month_labels)
plt.xlabel("Time (Months)")
plt.ylabel("Total Production [MWh]")
plt.title(f"Total Production ({SCENARIO_MAIN}) with Smoothing")
plt.legend()

plt.gca().set_xticks(month_edges, minor=True)
plt.grid(True, which="minor", axis="x")
plt.grid(True, axis="y")
plt.xlim(0, HOURS - 1)

plt.savefig(os.path.join(OUT_DIR, f"Total_Production_{SCENARIO_MAIN}_with_Movmean.png"), dpi=150)
plt.close()
