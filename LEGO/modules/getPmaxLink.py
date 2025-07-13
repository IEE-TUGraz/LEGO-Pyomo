import pandas as pd
import os

# Set the data path relative to current working directory
data_path = os.path.join("data", "example")

# 1. Load Power_Demand.xlsx
demand_raw = pd.read_excel(os.path.join(data_path, "Power_Demand.xlsx"), header=None)

# Extract time labels from row 3 (index 2), columns G to AD (6 to 29)
time_labels = demand_raw.iloc[2, 6:30].tolist()

# Extract data from row 8 (index 7), columns A to AD (0 to 29)
demand_data = demand_raw.iloc[7:, 0:30].copy()

# Assign column names: first 6 columns from row 8, then time labels
first_six_cols = demand_raw.iloc[7, 0:6].tolist()
demand_data.columns = first_six_cols + time_labels

# 2. Load Power_VRES.xlsx
vres_raw = pd.read_excel(os.path.join(data_path, "Power_VRES.xlsx"), header=None)

# Extract data from row 8 (index 7), columns A to G (0 to 6)
vres_data = vres_raw.iloc[7:, 0:7].copy()

# Assign column names
vres_data.columns = ['Ignore1', 'Ignore2', 'Ignore3', 'Technology', 'Node', 'ColF', 'ColG']

# Calculate installed capacity = ColF * ColG
vres_data['Installed_Capacity'] = vres_data['ColF'] * vres_data['ColG']

# Sum installed capacity by Technology and Node
installed_capacity = vres_data.groupby(['Technology', 'Node'], as_index=False)['Installed_Capacity'].sum()

# 3. Load Power_VRESProfiles.xlsx
profiles_raw = pd.read_excel(os.path.join(data_path, "Power_VRESProfiles.xlsx"), header=None)

# Extract time labels from row 3 (index 2), columns G to AD (6 to 29)
profiles_time_labels = profiles_raw.iloc[2, 6:30].tolist()

# Extract data from row 8 (index 7), columns A to AD (0 to 29)
profiles_data = profiles_raw.iloc[7:, 0:30].copy()

# Assign columns: first 6 columns from row 8, then time labels
first_six_cols_profiles = profiles_raw.iloc[7, 0:6].tolist()
profiles_data.columns = first_six_cols_profiles + profiles_time_labels

# Melt profiles to long format
profiles_melted = profiles_data.melt(
    id_vars=[profiles_data.columns[2], profiles_data.columns[3]],  # Representative Day, Technology
    value_vars=profiles_time_labels,
    var_name='Time',
    value_name='Profile_Value'
)
profiles_melted.columns = ['Representative_Day', 'Technology', 'Time', 'Profile_Value']

# Merge installed capacity
profiles_with_capacity = profiles_melted.merge(installed_capacity, on='Technology', how='left')

# Calculate generation = profile value * installed capacity
profiles_with_capacity['Generation'] = profiles_with_capacity['Profile_Value'] * profiles_with_capacity['Installed_Capacity']

# 4. Melt demand data to long format
demand_melted = demand_data.melt(
    id_vars=[demand_data.columns[2], demand_data.columns[3]],  # Representative Day, Node
    value_vars=time_labels,
    var_name='Time',
    value_name='Demand'
)
demand_melted.columns = ['Representative_Day', 'Node', 'Time', 'Demand']

# 5. Sum generation by Time, Representative Day, Node
generation_sum = profiles_with_capacity.groupby(['Time', 'Representative_Day', 'Node'], as_index=False)['Generation'].sum()

# 6. Merge demand and generation with left join to keep all demand nodes
comparison_df = demand_melted.merge(generation_sum, on=['Time', 'Representative_Day', 'Node'], how='left')

# Fill missing generation with 0
comparison_df['Generation'] = comparison_df['Generation'].fillna(0)

# 7. Calculate absolute difference
comparison_df['Abs_Difference'] = (comparison_df['Demand'] - comparison_df['Generation']).abs()

# 8. Get max absolute difference (PMAX) per node
pmax_df = comparison_df.groupby('Node', as_index=False)['Abs_Difference'].max()
pmax_df.columns = ['Node', 'PMAX']

# 9. Save results
output_file = os.path.join(data_path, "PMAX_Results.xlsx")
pmax_df.to_excel(output_file, index=False)

print(f"✅ PMAX results saved to: {output_file}")
