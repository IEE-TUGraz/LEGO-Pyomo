import pandas as pd
import yaml
import os
import numpy as np

def read_data_settings(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def read_tab_separated_file(file_path: str, file_name: str) -> pd.DataFrame:
    """
    Reads a tab-separated file and returns a DataFrame.

    :param file_path: Path to the tab-separated file.
    :return: DataFrame containing the data from the file.
    """
    file = os.path.join(file_path, file_name)
    if not os.path.exists(file):
        raise FileNotFoundError(f"The file {file} does not exist.")

    try:
        data = pd.read_csv(file, sep='\t', header=[0], skiprows=[1], encoding='latin1')
        # Drop the last column if it is completely empty or unnamed
        if data.columns[-1][0].startswith('Unnamed'):
            data = data.iloc[:, :-1]

    except Exception as e:
        raise ValueError(f"Error reading the file: {e}")

    return data




def aggregate_TS(settings, df_PV, type: str = "mean"):
    if settings["aggregation"]["enabled"]:
        # aggregate each steps
        df_PV["invervall_group"] = df_PV.index // settings["aggregation"]["intervall"]

        if type == "mean":
            df_PV_sum = df_PV.groupby("invervall_group", as_index=False)["value"].mean()
        elif type == "sum":
            df_PV_sum = df_PV.groupby("invervall_group", as_index=False)["value"].sum()
        else:
            raise ValueError(f"Unknown aggregation type: {type}")

        # delete last element
        df_PV_sum = df_PV_sum.iloc[:-1]
    else:
        df_PV_sum = df_PV.copy()
    return df_PV_sum



def get_dHeat_Demand(file_path: str) -> pd.DataFrame:
    settings = read_data_settings(os.path.join(file_path, "DataSettings.yaml"))
    df_raw = read_tab_separated_file(file_path, settings["heat_demand"]["filename"])

    df_demand = df_raw[[settings["heat_demand"]["column"]]].copy()
    df_demand.columns = ["value"]

    # calc total demand
    total_heat_demand = df_demand["value"].sum() / 60 / 1000  # in MWh
    print(f"Total heat heat demand in the raw data: {total_heat_demand} MWh")

    df_demand_sum = aggregate_TS(settings, df_demand, "mean")

    num_elements = df_demand_sum["value"].count()

    #df_heat_demand = build_dummy_df(settings, num_elements, "i", "Node_1")
    df_heat_demand = pd.DataFrame(columns=['rp','k','i','value'])
    df_heat_demand['rp'] = ['rp01'] * num_elements
    df_heat_demand['k'] = [f"k{str(i).zfill(4)}" for i in range(1, num_elements + 1)]
    df_heat_demand['i'] = ['Node_1'] * num_elements

    df_heat_demand.loc[:, "value"] = df_demand_sum["value"].values * 1e-3  # calculate the demand in MW (raw data in kW)

    return df_heat_demand



# Example usage
if __name__ == "__main__":
    data_folder = os.path.join("data", "rings")
    settings = read_data_settings(os.path.join(data_folder, "DataSettings.yaml"))
    # print(settings)
    df_heat_demand = get_dHeat_Demand("data/rings", "Building_5600m2.out")
    print(df_heat_demand.head(24))


    # import ExcelReader

    # path = os.path.join("data", "rings_base_example")
    # df_k = ExcelReader.get_dPower_WeightsK(os.path.join(path, "Power_WeightsK.xlsx"))
    # df_hindex = ExcelReader.get_dPower_Hindex(os.path.join(path, "Power_Hindex.xlsx"))

    # print(df_k.head(24))

    # df_test = create_kWeights(24)
    # print(df_test.head(24))
