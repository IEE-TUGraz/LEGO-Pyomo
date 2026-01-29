import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ExcelWriter import ExcelWriter

# Parameters
IN_DIR = "../data/hydroExample/"
OUT_DIR = "../output/inflow_spinup/"  # hier beliebiges Verzeichnis wählen
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIO_MAIN = "Scenario_2015"
SCENARIO_SPIN = "Scenario_2014"
SPINUP = 168
HOURS = 8760

# =========================
# Read in network
# =========================
df_network = pd.read_excel(
    IN_DIR + "Power_HydroNetwork.xlsx",
    sheet_name="Scenario_2014",
    usecols="C:E",  # Spalten für From und To
    skiprows=7,
    header=None
)
df_network.columns = ["From", "To", "TurbineOrPump"]
df_network = df_network[df_network["TurbineOrPump"] == 0]  # Filter out all pump-connections, otherwise water would flow backwards
df_network = df_network.dropna().drop_duplicates()

df_network["From"] = df_network["From"].astype(str).str.strip()
df_network["To"] = df_network["To"].astype(str).str.strip()

# Build network dictionaries
pp_pre = {}
pp_follow = {}
for _, r in df_network.iterrows():
    u = r["From"]
    v = r["To"]
    pp_follow.setdefault(u, []).append(v)
    pp_pre.setdefault(v, []).append(u)
    pp_pre.setdefault(u, [])

plants = list(pp_pre.keys())
edges = list(zip(df_network["From"].tolist(), df_network["To"].tolist()))

# =========================
# Read in assets
# =========================
df_assets = pd.read_excel(
    IN_DIR + "Power_HydroAssets.xlsx",
    sheet_name="Scenario_2014",
    usecols="C,I",  # C ist generator unit und I der power factor
    skiprows=7,
    header=None
)
df_assets.columns = ["generator_unit", "power_factor"]
df_assets = df_assets.dropna()

df_assets["generator_unit"] = df_assets["generator_unit"].astype(str).str.strip()
power_factor = dict(zip(df_assets["generator_unit"], df_assets["power_factor"]))


# =========================
# Read in inflows  (WICHTIG: Mehrere Zeilen pro generator_unit werden addiert)
# =========================
def read_inflows(sheet):
    df = pd.read_excel(
        IN_DIR + "Power_Inflows_WaterAmount.xlsx",
        sheet_name=sheet,
        skiprows=7,
        header=None
    )
    names = df.iloc[:, 3].astype(str).str.strip()  # Column D (generator_unit)
    vals = df.iloc[:, 7:].to_numpy(dtype=float)  # Column H..end (k0001..k8760)

    d = {}
    for i in range(len(names)):
        key = names[i]
        if key in d:
            d[key] += vals[i, :]
        else:
            d[key] = vals[i, :].copy()
    return d


inflow_main = read_inflows(SCENARIO_MAIN)
inflow_spin = read_inflows(SCENARIO_SPIN)

# Fill missing plants with zero inflow
for p in plants:
    if p not in inflow_main:
        inflow_main[p] = np.zeros(HOURS)
    if p not in inflow_spin:
        inflow_spin[p] = np.zeros(HOURS)

# Spinup-Arrays: 168h before main simulation
inflow_spin_tail = {p: inflow_spin[p][-SPINUP:] for p in plants}

# =========================
# SIMULATION
# =========================
prod = {p: np.zeros(HOURS) for p in plants}

# Water amount that flowed through p in the PREVIOUS HOUR
inflow_prev = {p: 0.0 for p in plants}

# Spin-up
for t in range(SPINUP):
    inflow_curr = {p: float(inflow_spin_tail[p][t]) for p in plants}
    for u, v in edges:
        inflow_curr[v] += inflow_prev[u]
    inflow_prev = inflow_curr

# Main calculation
for t in range(HOURS):
    inflow_curr = {p: float(inflow_main[p][t]) for p in plants}
    for u, v in edges:
        inflow_curr[v] += inflow_prev[u]

    for p in plants:
        prod[p][t] = inflow_curr[p] * power_factor.get(p, 0.0)

    inflow_prev = inflow_curr

# =========================
# Save Excel file with production per plant
# =========================
df_energy = pd.DataFrame(prod)

# Prepare df_energy for ExcelWriter
df_energy = df_energy.reset_index(names="k").melt(id_vars="k", var_name="g", value_name="value")
df_energy["scenario"] = SCENARIO_MAIN
df_energy["id"] = None
df_energy["rp"] = "rp01"
df_energy["k"] = (df_energy["k"] + 1).astype(str).str.zfill(4).radd("k")
df_energy["dataPackage"] = "TO BE FILLED"
df_energy["dataSource"] = "TO BE FILLED"

ew = ExcelWriter()
ew.write_Power_Inflows(df_energy, OUT_DIR)

# =========================
# Plotting per plant
# =========================
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
