import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
OUT_DIR = r"../output/inflow_no_spinup/"
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIO_MAIN = "Scenario_2015"
HOURS = 8760

# =========================
# Read in network
#   From = Col C, To = Col D, start at row 8
# =========================
df_network = pd.read_excel(
    "Power_HydroNetwork.xlsx",
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

# Edges of the network
edges = list(zip(df_network["From"].tolist(), df_network["To"].tolist()))

# Optional: Check for unintended split (1 -> multiple)
succ_set = {}
bad = []
for u, v in edges:
    succ_set.setdefault(u, set()).add(v)
for u in succ_set:
    if len(succ_set[u]) > 1:
        bad.append((u, list(succ_set[u])))
if len(bad) > 0:
    print("WARNING: Found From with multiple To (split). This will duplicate water:", bad)

# =========================
# Read in assets
#   generator_unit = Col C, power_factor = Col I, start at row 8
# =========================
df_assets = pd.read_excel(
    "Power_HydroAssets.xlsx",
    sheet_name="Scenario_2014",  
    usecols="C,I",
    skiprows=7,
    header=None
)
df_assets.columns = ["generator_unit", "power_factor"]
df_assets = df_assets.dropna()

df_assets["generator_unit"] = df_assets["generator_unit"].astype(str).str.strip()
power_factor = dict(zip(df_assets["generator_unit"], df_assets["power_factor"]))

# =========================
# Read in inflows (MAIN)
#   generator_unit = Col D, inflows = Col H..end, start at row 8
# =========================
def read_inflows(sheet):
    df = pd.read_excel(
        "Power_Inflows_WaterAmount.xlsx",
        sheet_name=sheet,
        skiprows=7,
        header=None
    )
    names = df.iloc[:, 3].astype(str).str.strip()       # Column D
    vals = df.iloc[:, 7:].to_numpy(dtype=float)         # Column H..end

    d = {}
    for i in range(len(names)):
        key = names[i]
        if key in d:
            d[key] += vals[i, :]        # addieren, falls schon vorhanden
        else:
            d[key] = vals[i, :].copy()  # neu anlegen
    return d, df

inflow_main, df_in_main_raw = read_inflows(SCENARIO_MAIN)

# Fill missing plants with zero inflow
for p in plants:
    if p not in inflow_main:
        inflow_main[p] = np.zeros(HOURS)

# =========================
# Export inflows per generator_unit (MAIN)
# =========================
out_inflows = df_in_main_raw.iloc[:, [3] + list(range(7, df_in_main_raw.shape[1]))].copy()
out_inflows.columns = ["generator_unit"] + [f"k{str(i).zfill(4)}" for i in range(1, out_inflows.shape[1])]
out_inflows = out_inflows.dropna(subset=["generator_unit"])
out_inflows.to_excel(os.path.join(OUT_DIR, f"Inflows_per_generator_unit_{SCENARIO_MAIN}.xlsx"), index=False)

# =========================
# SIMULATION (NO SPINUP)
# =========================
prod = {p: np.zeros(HOURS) for p in plants}

# Water amount that flowed through p in the PREVIOUS HOUR
inflow_prev = {p: 0.0 for p in plants}

for t in range(HOURS):
    inflow_curr = {p: float(inflow_main[p][t]) for p in plants}

    # add upstream contributions (1h travel time)
    for u, v in edges:
        inflow_curr[v] += inflow_prev[u]

    # production per plant at hour t
    for p in plants:
        prod[p][t] = inflow_curr[p] * power_factor.get(p, 0.0)

    # shift state
    inflow_prev = inflow_curr

# =========================
# Save Excel file with production per plant
# =========================
df_energy = pd.DataFrame(prod)
df_energy.index = range(1, HOURS + 1)
df_energy.to_excel(os.path.join(OUT_DIR, f"Energy_Production_{SCENARIO_MAIN}_8760h.xlsx"))

# =========================
# Plotting per plant (x-axis months, grid at month edges)
# =========================
month_edges = np.array([0, 744, 1416, 2160, 2880, 3624, 4344,
                        5088, 5832, 6552, 7296, 8016, 8760])
month_centers = (month_edges[:-1] + month_edges[1:]) / 2
month_labels = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

x = np.arange(HOURS)

for p in plants:
    plt.figure()
    plt.plot(x, prod[p])
    plt.title(f"{p} ({SCENARIO_MAIN})")
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
# Total production of the whole network + moving mean (same plot)
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