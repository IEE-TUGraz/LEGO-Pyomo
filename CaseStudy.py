import pandas as pd


class CaseStudy:

    def __init__(self, example_folder: str):
        self.example_folder = example_folder

    def get_dPower_Parameters(self):
        dPower_Parameters = pd.read_excel(self.example_folder + "Power_Parameters.xlsx", skiprows=[0, 1])
        dPower_Parameters = dPower_Parameters.drop(dPower_Parameters.columns[0], axis=1)
        dPower_Parameters = dPower_Parameters.dropna(how="all")
        dPower_Parameters = dPower_Parameters.set_index('General')
        return dPower_Parameters

    def get_dPower_BusInfo(self):
        dPower_BusInfo = pd.read_excel(self.example_folder + "Power_BusInfo.xlsx", skiprows=[0, 1, 3, 4, 5])
        dPower_BusInfo = dPower_BusInfo.drop(dPower_BusInfo.columns[0], axis=1)
        dPower_BusInfo = dPower_BusInfo.rename(columns={dPower_BusInfo.columns[0]: "i", dPower_BusInfo.columns[1]: "System"})
        dPower_BusInfo = dPower_BusInfo.set_index('i')
        return dPower_BusInfo

    def get_dPower_Network(self):
        dPower_Network = pd.read_excel(self.example_folder + "Power_Network.xlsx", skiprows=[0, 1, 3, 4, 5])
        dPower_Network = dPower_Network.drop(dPower_Network.columns[0], axis=1)
        dPower_Network = dPower_Network.rename(columns={dPower_Network.columns[0]: "i", dPower_Network.columns[1]: "j", dPower_Network.columns[2]: "Circuit ID"})
        dPower_Network = dPower_Network.set_index(['i', 'j'])
        return dPower_Network

    def get_dPower_ThermalGen(self):
        return self.read_generator_data(self.example_folder + "Power_ThermalGen.xlsx")

    def get_dPower_RoR(self):
        return self.read_generator_data(self.example_folder + "Power_RoR.xlsx")

    def get_dPower_VRES(self):
        return self.read_generator_data(self.example_folder + "Power_VRES.xlsx")

    def get_dPower_Demand(self):
        dPower_Demand = pd.read_excel(self.example_folder + "Power_Demand.xlsx", skiprows=[0, 1, 3, 4, 5])
        dPower_Demand = dPower_Demand.drop(dPower_Demand.columns[0], axis=1)
        dPower_Demand = dPower_Demand.rename(columns={dPower_Demand.columns[0]: "rp", dPower_Demand.columns[1]: "i"})
        dPower_Demand = dPower_Demand.melt(id_vars=['rp', 'i'], var_name='k', value_name='Demand')
        dPower_Demand = dPower_Demand.set_index(['rp', 'i', 'k'])
        return dPower_Demand

    def get_dPower_Inflows(self):
        dPower_Inflows = pd.read_excel(self.example_folder + "Power_Inflows.xlsx", skiprows=[0, 1, 3, 4, 5])
        dPower_Inflows = dPower_Inflows.drop(dPower_Inflows.columns[0], axis=1)
        dPower_Inflows = dPower_Inflows.rename(columns={dPower_Inflows.columns[0]: "rp", dPower_Inflows.columns[1]: "g"})
        dPower_Inflows = dPower_Inflows.melt(id_vars=['rp', 'g'], var_name='k', value_name='Inflow')
        dPower_Inflows = dPower_Inflows.set_index(['rp', 'g', 'k'])
        return dPower_Inflows

    def get_dPower_VRESProfiles(self):
        dPower_VRESProfiles = pd.read_excel(self.example_folder + "Power_VRESProfiles.xlsx", skiprows=[0, 1, 3, 4, 5])
        dPower_VRESProfiles = dPower_VRESProfiles.drop(dPower_VRESProfiles.columns[0], axis=1)
        dPower_VRESProfiles = dPower_VRESProfiles.rename(columns={dPower_VRESProfiles.columns[0]: "rp", dPower_VRESProfiles.columns[1]: "i", dPower_VRESProfiles.columns[2]: "tec"})
        dPower_VRESProfiles = dPower_VRESProfiles.melt(id_vars=['rp', 'i', 'tec'], var_name='k', value_name='Capacity')
        dPower_VRESProfiles = dPower_VRESProfiles.set_index(['rp', 'i', 'k', 'tec'])
        return dPower_VRESProfiles

    # Function to read generator data
    @staticmethod
    def read_generator_data(file_path):
        d_generator = pd.read_excel(file_path, skiprows=[0, 1, 3, 4, 5])
        d_generator = d_generator.drop(d_generator.columns[0], axis=1)
        d_generator = d_generator.rename(columns={d_generator.columns[0]: "g", d_generator.columns[1]: "tec", d_generator.columns[2]: "i"})
        d_generator = d_generator.set_index('g')
        d_generator = d_generator[d_generator["ExisUnits"] > 0]
        return d_generator
