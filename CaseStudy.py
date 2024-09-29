import warnings

import numpy as np
import pandas as pd


class CaseStudy:

    def __init__(self, example_folder: str, do_not_merge_single_node_buses: bool = False,
                 power_parameters_file: str = "Power_Parameters.xlsx", dPower_Parameters: pd.DataFrame = None,
                 power_businfo_file: str = "Power_BusInfo.xlsx", dPower_BusInfo: pd.DataFrame = None,
                 power_network_file: str = "Power_Network.xlsx", dPower_Network: pd.DataFrame = None,
                 power_thermalgen_file: str = "Power_ThermalGen.xlsx", dPower_ThermalGen: pd.DataFrame = None,
                 power_ror_file: str = "Power_RoR.xlsx", dPower_RoR: pd.DataFrame = None,
                 power_vres_file: str = "Power_VRES.xlsx", dPower_VRES: pd.DataFrame = None,
                 power_demand_file: str = "Power_Demand.xlsx", dPower_Demand: pd.DataFrame = None,
                 power_inflows_file: str = "Power_Inflows.xlsx", dPower_Inflows: pd.DataFrame = None,
                 power_vresprofiles_file: str = "Power_VRESProfiles.xlsx", dPower_VRESProfiles: pd.DataFrame = None,
                 power_Storage: str = "Power_Storage.xlsx", dPower_Storage: pd.DataFrame = None):
        self.example_folder = example_folder
        self.do_not_merge_single_node_buses = do_not_merge_single_node_buses

        if dPower_Parameters is not None:
            self.dPower_Parameters = dPower_Parameters
        else:
            self.power_parameters_file = power_parameters_file
            self.dPower_Parameters = self.get_dPower_Parameters()

        if dPower_BusInfo is not None:
            self.dPower_BusInfo = dPower_BusInfo
        else:
            self.power_businfo_file = power_businfo_file
            self.dPower_BusInfo = self.get_dPower_BusInfo()

        if dPower_Network is not None:
            self.dPower_Network = dPower_Network
        else:
            self.power_network_file = power_network_file
            self.dPower_Network = self.get_dPower_Network()

        if dPower_ThermalGen is not None:
            self.dPower_ThermalGen = dPower_ThermalGen
        else:
            self.power_thermalgen_file = power_thermalgen_file
            self.dPower_ThermalGen = self.get_dPower_ThermalGen()

        if dPower_RoR is not None:
            self.dPower_RoR = dPower_RoR
        else:
            self.power_ror_file = power_ror_file
            self.dPower_RoR = self.get_dPower_RoR()

        if dPower_VRES is not None:
            self.dPower_VRES = dPower_VRES
        else:
            self.power_vres_file = power_vres_file
            self.dPower_VRES = self.get_dPower_VRES()

        if dPower_Demand is not None:
            self.dPower_Demand = dPower_Demand
        else:
            self.power_demand_file = power_demand_file
            self.dPower_Demand = self.get_dPower_Demand()

        if dPower_Inflows is not None:
            self.dPower_Inflows = dPower_Inflows
        else:
            self.power_inflows_file = power_inflows_file
            self.dPower_Inflows = self.get_dPower_Inflows()

        if dPower_VRESProfiles is not None:
            self.dPower_VRESProfiles = dPower_VRESProfiles
        else:
            self.power_vresprofiles_file = power_vresprofiles_file
            self.dPower_VRESProfiles = self.get_dPower_VRESProfiles()

        if dPower_Storage is not None:
            self.dPower_Storage = dPower_Storage
        else:
            self.power_storage_file = power_Storage
            self.dPower_Storage = self.get_dPower_Storage()

        self.pMaxAngleDCOPF = self.dPower_Parameters.loc["pMaxAngleDCOPF"].iloc[0] * np.pi / 180  # Read and convert to radians
        self.pSBase = self.dPower_Parameters.loc["pSBase"].iloc[0]

        # Dataframe that shows connections between g and i, only concatenating g and i from the dataframes
        self.hGenerators_to_Buses = self.update_hGenerators_to_Buses()

        if not do_not_merge_single_node_buses:
            self.merge_single_node_buses()

    def copy(self):
        return CaseStudy(example_folder=self.example_folder, do_not_merge_single_node_buses=True,
                         dPower_Parameters=self.dPower_Parameters.copy(), dPower_BusInfo=self.dPower_BusInfo.copy(),
                         dPower_Network=self.dPower_Network.copy(), dPower_ThermalGen=self.dPower_ThermalGen.copy(),
                         dPower_RoR=self.dPower_RoR.copy(), dPower_VRES=self.dPower_VRES.copy(), dPower_Demand=self.dPower_Demand.copy(),
                         dPower_Inflows=self.dPower_Inflows.copy(), dPower_VRESProfiles=self.dPower_VRESProfiles.copy())

    def get_dPower_Parameters(self):
        dPower_Parameters = pd.read_excel(self.example_folder + self.power_parameters_file, skiprows=[0, 1])
        dPower_Parameters = dPower_Parameters.drop(dPower_Parameters.columns[0], axis=1)
        dPower_Parameters = dPower_Parameters.dropna(how="all")
        dPower_Parameters = dPower_Parameters.set_index('General')
        return dPower_Parameters

    def get_dPower_BusInfo(self):
        dPower_BusInfo = pd.read_excel(self.example_folder + self.power_businfo_file, skiprows=[0, 1, 3, 4, 5])
        dPower_BusInfo = dPower_BusInfo.drop(dPower_BusInfo.columns[0], axis=1)
        dPower_BusInfo = dPower_BusInfo.rename(columns={dPower_BusInfo.columns[0]: "i", dPower_BusInfo.columns[1]: "System"})
        dPower_BusInfo = dPower_BusInfo.set_index('i')
        return dPower_BusInfo

    def get_dPower_Network(self):
        dPower_Network = pd.read_excel(self.example_folder + self.power_network_file, skiprows=[0, 1, 3, 4, 5])
        dPower_Network = dPower_Network.drop(dPower_Network.columns[0], axis=1)
        dPower_Network = dPower_Network.rename(columns={dPower_Network.columns[0]: "i", dPower_Network.columns[1]: "j", dPower_Network.columns[2]: "Circuit ID"})
        dPower_Network = dPower_Network.set_index(['i', 'j'])
        return dPower_Network

    def get_dPower_ThermalGen(self):
        dPower_ThermalGen = self.read_generator_data(self.example_folder + self.power_thermalgen_file)

        dPower_ThermalGen['pSlopeVarCostEUR'] = (dPower_ThermalGen['OMVarCost'] * 1e-3 +
                                                 dPower_ThermalGen['SlopeVarCost'] * 1e-3 * dPower_ThermalGen['FuelCost'])

        dPower_ThermalGen['pInterVarCostEUR'] = dPower_ThermalGen['InterVarCost'] * 1e-6 * dPower_ThermalGen['FuelCost']
        dPower_ThermalGen['pStartupCostEUR'] = dPower_ThermalGen['StartupCost'] * 1e-6 * dPower_ThermalGen['FuelCost']

        # Fill NaN values with 0 for MinUpTime and MinDownTime
        dPower_ThermalGen['MinUpTime'] = dPower_ThermalGen['MinUpTime'].fillna(0)
        dPower_ThermalGen['MinDownTime'] = dPower_ThermalGen['MinDownTime'].fillna(0)

        # Check that both MinUpTime and MinDownTime are integers and raise error if not
        if not dPower_ThermalGen['MinUpTime'].apply(float.is_integer).all():
            raise ValueError("MinUpTime must be an integer.")
        if not dPower_ThermalGen['MinDownTime'].apply(float.is_integer).all():
            raise ValueError("MinDownTime must be an integer.")
        dPower_ThermalGen['MinUpTime'] = dPower_ThermalGen['MinUpTime'].astype(int)
        dPower_ThermalGen['MinDownTime'] = dPower_ThermalGen['MinDownTime'].astype(int)

        return dPower_ThermalGen

    def get_dPower_RoR(self):
        return self.read_generator_data(self.example_folder + self.power_ror_file)

    def get_dPower_VRES(self):
        return self.read_generator_data(self.example_folder + self.power_vres_file)

    def get_dPower_Storage(self):
        dPower_Storage = self.read_generator_data(self.example_folder + self.power_storage_file)
        dPower_Storage['pOMVarCostEUR'] = dPower_Storage['OMVarCost'] * 1e-3
        return dPower_Storage

    def get_dPower_Demand(self):
        dPower_Demand = pd.read_excel(self.example_folder + self.power_demand_file, skiprows=[0, 1, 3, 4, 5])
        dPower_Demand = dPower_Demand.drop(dPower_Demand.columns[0], axis=1)
        dPower_Demand = dPower_Demand.rename(columns={dPower_Demand.columns[0]: "rp", dPower_Demand.columns[1]: "i"})
        dPower_Demand = dPower_Demand.melt(id_vars=['rp', 'i'], var_name='k', value_name='Demand')
        dPower_Demand = dPower_Demand.set_index(['rp', 'i', 'k'])
        return dPower_Demand

    def get_dPower_Inflows(self):
        dPower_Inflows = pd.read_excel(self.example_folder + self.power_inflows_file, skiprows=[0, 1, 3, 4, 5])
        dPower_Inflows = dPower_Inflows.drop(dPower_Inflows.columns[0], axis=1)
        dPower_Inflows = dPower_Inflows.rename(columns={dPower_Inflows.columns[0]: "rp", dPower_Inflows.columns[1]: "g"})
        dPower_Inflows = dPower_Inflows.melt(id_vars=['rp', 'g'], var_name='k', value_name='Inflow')
        dPower_Inflows = dPower_Inflows.set_index(['rp', 'g', 'k'])
        return dPower_Inflows

    def get_dPower_VRESProfiles(self):
        dPower_VRESProfiles = pd.read_excel(self.example_folder + self.power_vresprofiles_file, skiprows=[0, 1, 3, 4, 5])
        dPower_VRESProfiles = dPower_VRESProfiles.drop(dPower_VRESProfiles.columns[0], axis=1)
        dPower_VRESProfiles = dPower_VRESProfiles.rename(columns={dPower_VRESProfiles.columns[0]: "rp", dPower_VRESProfiles.columns[1]: "i", dPower_VRESProfiles.columns[2]: "tec"})
        dPower_VRESProfiles = dPower_VRESProfiles.melt(id_vars=['rp', 'i', 'tec'], var_name='k', value_name='Capacity')
        dPower_VRESProfiles = dPower_VRESProfiles.set_index(['rp', 'i', 'k', 'tec'])
        return dPower_VRESProfiles

    def update_hGenerators_to_Buses(self):
        return pd.concat([self.dPower_ThermalGen[['i']], self.dPower_RoR[['i']], self.dPower_VRES[['i']], self.dPower_Storage[['i']]])

    # Function to read generator data
    @staticmethod
    def read_generator_data(file_path):
        d_generator = pd.read_excel(file_path, skiprows=[0, 1, 3, 4, 5])
        d_generator = d_generator.drop(d_generator.columns[0], axis=1)
        d_generator = d_generator.rename(columns={d_generator.columns[0]: "g", d_generator.columns[1]: "tec", d_generator.columns[2]: "i"})
        d_generator = d_generator.set_index('g')
        d_generator = d_generator[d_generator["ExisUnits"] > 0]
        return d_generator

    @staticmethod
    def get_connected_buses(connection_matrix, bus: str):
        connected_buses = []
        stack = [bus]
        while stack:
            current_bus = stack.pop()
            connected_buses.append(current_bus)

            connected_to_current_bus = [multiindex[0] for multiindex in connection_matrix.loc[current_bus][connection_matrix.loc[current_bus] == True].index.tolist()]
            for node in connected_to_current_bus:
                if node not in connected_buses and node not in stack:
                    stack.append(node)

        connected_buses.sort()
        return connected_buses

    def merge_single_node_buses(self):
        # Create a connection matrix
        connectionMatrix = pd.DataFrame(index=self.dPower_BusInfo.index, columns=[self.dPower_BusInfo.index], data=False)

        for index, entry in self.dPower_Network.iterrows():
            if entry["Technical Representation"] == "SN":
                connectionMatrix.loc[index] = True
                connectionMatrix.loc[index[1], index[0]] = True

        merged_buses = set()  # Set of buses that have been merged already

        for index, entry in connectionMatrix.iterrows():
            if index in merged_buses or entry[entry == True].empty:  # Skip if bus has already been merged or has no connections
                continue

            connected_buses = self.get_connected_buses(connectionMatrix, str(index))

            for bus in connected_buses:
                merged_buses.add(bus)

            new_bus_name = "merged-" + "-".join(connected_buses)

            ### Adapt dPower_BusInfo
            dPower_BusInfo_entry = self.dPower_BusInfo.loc[connected_buses]  # Entry for the new bus
            zoneOfInterest = "yes" if any(dPower_BusInfo_entry['ZoneOfInterest'] == "yes") else "no"
            aggregation_methods_for_columns = {
                # 'System': 'max',
                # 'BaseVolt': 'mean',
                # 'maxVolt': 'max',
                # 'minVolt': 'min',
                # 'Bs': 'mean',
                # 'Gs': 'mean',
                # 'PowerFactor': 'mean',
                'YearCom': 'mean',
                'YearDecom': 'mean',
                'lat': 'mean',
                'long': 'mean'
            }
            dPower_BusInfo_entry = dPower_BusInfo_entry.agg(aggregation_methods_for_columns)
            dPower_BusInfo_entry['ZoneOfInterest'] = zoneOfInterest
            dPower_BusInfo_entry = dPower_BusInfo_entry.to_frame().T
            dPower_BusInfo_entry.index = [new_bus_name]

            self.dPower_BusInfo = self.dPower_BusInfo.drop(connected_buses)
            with warnings.catch_warnings():  # Suppressing FutureWarning because some entries might include NaN values
                warnings.simplefilter(action='ignore', category=FutureWarning)
                self.dPower_BusInfo = pd.concat([self.dPower_BusInfo, dPower_BusInfo_entry])

            ### Adapt dPower_Network
            self.dPower_Network = self.dPower_Network.reset_index()
            rows_to_drop = []
            for i, row in self.dPower_Network.iterrows():
                if row['i'] in connected_buses and row['j'] in connected_buses:
                    rows_to_drop.append(i)
                elif row['i'] in connected_buses:
                    row['i'] = new_bus_name
                    self.dPower_Network.iloc[i] = row
                elif row['j'] in connected_buses:
                    row['j'] = new_bus_name
                    self.dPower_Network.iloc[i] = row
            self.dPower_Network = self.dPower_Network.drop(rows_to_drop)

            # Always put new_bus_name to 'j' (handles case where e.g. 2->3 and 4->2 would lead to 2->34 and 34->2 (because 3 and 4 are merged))
            for i, row in self.dPower_Network.iterrows():
                if row['i'] == new_bus_name:
                    row['i'] = row['j']
                    row['j'] = new_bus_name
                    self.dPower_Network.loc[i] = row

            # Handle case where e.g. 2->3 and 2->4 would lead to 2->34 and 2->34 (because 3 and 4 are merged); also incl. handling 2->3 and 4->2
            self.dPower_Network['Technical Representation'] = self.dPower_Network.groupby(['i', 'j'])['Technical Representation'].transform(lambda series: 'DC-OPF' if 'DC-OPF' in series.values else series.iloc[0])
            aggregation_methods_for_columns = {
                # 'Circuit ID': 'first',
                # 'InService': 'max',
                # 'R': 'mean',
                'X': lambda x: x.map(lambda a: 1 / a).sum() ** -1,  # Formula: 1/X = sum((i,j), 1/Xij)) (e.g., 1/X = 1/Xij_1 +1/Xij_2 + 1/Xij_3...)
                # 'Bc': 'mean',
                # 'TapAngle': 'mean',
                # 'TapRatio': 'mean',
                'Pmax': lambda x: x.min() * x.count(),  # Number of lines times the minimum Pmax for new Pmax of the merged lines TODO: Calculate this based on more complex method (flow is relative to R, talk to Benjamin)
                # 'FixedCost': 'mean',
                # 'FxChargeRate': 'mean',
                'Technical Representation': 'first',
                'LineID': 'first',
                'YearCom': 'mean',
                'YearDecom': 'mean'
            }
            self.dPower_Network = self.dPower_Network.groupby(['i', 'j']).agg(aggregation_methods_for_columns)

            ### Adapt dPower_ThermalGen
            for i, row in self.dPower_ThermalGen.iterrows():
                if row['i'] in connected_buses:
                    row['i'] = new_bus_name
                    self.dPower_ThermalGen.loc[i] = row

            # Adapt dPower_RoR
            for i, row in self.dPower_RoR.iterrows():
                if row['i'] in connected_buses:
                    row['i'] = new_bus_name
                    self.dPower_RoR.loc[i] = row

            # Adapt dPower_VRES
            for i, row in self.dPower_VRES.iterrows():
                if row['i'] in connected_buses:
                    row['i'] = new_bus_name
                    self.dPower_VRES.loc[i] = row

            # Adapt dPower_Storage
            for i, row in self.dPower_Storage.iterrows():
                if row['i'] in connected_buses:
                    row['i'] = new_bus_name
                    self.dPower_Storage.loc[i] = row

            # Adapt dPower_Demand
            self.dPower_Demand = self.dPower_Demand.reset_index()
            for i, row in self.dPower_Demand.iterrows():
                if row['i'] in connected_buses:
                    row['i'] = new_bus_name
                    self.dPower_Demand.loc[i] = row
            self.dPower_Demand = self.dPower_Demand.groupby(['rp', 'i', 'k']).sum()

            # Adapt dPower_VRESProfiles
            self.dPower_VRESProfiles = self.dPower_VRESProfiles.reset_index()
            for i, row in self.dPower_VRESProfiles.iterrows():
                if row['i'] in connected_buses:
                    row['i'] = new_bus_name
                    self.dPower_VRESProfiles.loc[i] = row

            self.dPower_VRESProfiles = self.dPower_VRESProfiles.groupby(['rp', 'i', 'k', 'tec']).mean()  # TODO: Aggregate using more complex method (capacity * productionCapacity * ... * / Total Production Capacity)
            self.dPower_VRESProfiles.sort_index(inplace=True)

            # Update hGenerators_to_Buses
            self.hGenerators_to_Buses = self.update_hGenerators_to_Buses()
