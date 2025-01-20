import warnings

import numpy as np
import pandas as pd


class CaseStudy:

    def __init__(self, example_folder: str, do_not_merge_single_node_buses: bool = False,
                 global_parameters_file: str = "Global_Parameters.xlsx", dGlobal_Parameters: pd.DataFrame = None,
                 power_parameters_file: str = "Power_Parameters.xlsx", dPower_Parameters: pd.DataFrame = None,
                 power_businfo_file: str = "Power_BusInfo.xlsx", dPower_BusInfo: pd.DataFrame = None,
                 power_network_file: str = "Power_Network.xlsx", dPower_Network: pd.DataFrame = None,
                 power_thermalgen_file: str = "Power_ThermalGen.xlsx", dPower_ThermalGen: pd.DataFrame = None,
                 power_ror_file: str = "Power_RoR.xlsx", dPower_RoR: pd.DataFrame = None,
                 power_vres_file: str = "Power_VRES.xlsx", dPower_VRES: pd.DataFrame = None,
                 power_demand_file: str = "Power_Demand.xlsx", dPower_Demand: pd.DataFrame = None,
                 power_inflows_file: str = "Power_Inflows.xlsx", dPower_Inflows: pd.DataFrame = None,
                 power_vresprofiles_file: str = "Power_VRESProfiles.xlsx", dPower_VRESProfiles: pd.DataFrame = None,
                 power_storage_file: str = "Power_Storage.xlsx", dPower_Storage: pd.DataFrame = None,
                 power_weightsrp_file: str = "Power_WeightsRP.xlsx", dPower_WeightsRP: pd.DataFrame = None,
                 power_weightsk_file: str = "Power_WeightsK.xlsx", dPower_WeightsK: pd.DataFrame = None,
                 power_hindex_file: str = "Power_Hindex.xlsx", dPower_Hindex: pd.DataFrame = None,
                 power_impexphubs_file: str = "Power_ImpExpHubs.xlsx", dPower_ImpExpHubs: pd.DataFrame = None,
                 power_impexpprofiles_file: str = "Power_ImpExpProfiles.xlsx", dPower_ImpExpProfiles: pd.DataFrame = None):
        self.example_folder = example_folder
        self.do_not_merge_single_node_buses = do_not_merge_single_node_buses

        if dGlobal_Parameters is not None:
            self.dGlobal_Parameters = dGlobal_Parameters
        else:
            self.global_parameters_file = global_parameters_file
            self.dGlobal_Parameters = self.get_dGlobal_Parameters()

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
            self.power_storage_file = power_storage_file
            self.dPower_Storage = self.get_dPower_Storage()

        if dPower_WeightsRP is not None:
            self.dPower_WeightsRP = dPower_WeightsRP
        else:
            self.power_weightsrp_file = power_weightsrp_file
            self.dPower_WeightsRP = self.get_dPower_WeightsRP()

        if dPower_WeightsK is not None:
            self.dPower_WeightsK = dPower_WeightsK
        else:
            self.power_weightsk_file = power_weightsk_file
            self.dPower_WeightsK = self.get_dPower_WeightsK()

        if dPower_Hindex is not None:
            self.dPower_Hindex = dPower_Hindex
        else:
            self.power_hindex_file = power_hindex_file
            self.dPower_Hindex = self.get_dPower_Hindex()

        if self.dPower_Parameters["pEnablePowerImportExport"]:
            if dPower_ImpExpHubs is not None:
                self.dPower_ImpExpHubs = dPower_ImpExpHubs
            else:
                self.power_impexphubs_file = power_impexphubs_file
                self.dPower_ImpExpHubs = self.get_dPower_ImpExpHubs()

            if dPower_ImpExpProfiles is not None:
                self.dPower_ImpExpProfiles = dPower_ImpExpProfiles
            else:
                self.power_impexpprofiles_file = power_impexpprofiles_file
                self.dPower_ImpExpProfiles = self.get_dPower_ImpExpProfiles()
        else:
            self.dPower_ImpExpHubs = None
            self.dPower_ImpExpProfiles = None

        if not do_not_merge_single_node_buses:
            self.merge_single_node_buses()

    def copy(self):
        return CaseStudy(example_folder=self.example_folder, do_not_merge_single_node_buses=True,
                         dPower_Parameters=self.dPower_Parameters.copy(), dPower_BusInfo=self.dPower_BusInfo.copy(),
                         dPower_Network=self.dPower_Network.copy(), dPower_ThermalGen=self.dPower_ThermalGen.copy(),
                         dPower_RoR=self.dPower_RoR.copy(), dPower_VRES=self.dPower_VRES.copy(), dPower_Demand=self.dPower_Demand.copy(),
                         dPower_Inflows=self.dPower_Inflows.copy(), dPower_VRESProfiles=self.dPower_VRESProfiles.copy())

    def get_dGlobal_Parameters(self):
        dGlobal_Parameters = pd.read_excel(self.example_folder + self.global_parameters_file, skiprows=[0, 1])
        dGlobal_Parameters = dGlobal_Parameters.drop(dGlobal_Parameters.columns[0], axis=1)
        dGlobal_Parameters = dGlobal_Parameters.set_index('Sectors')

        # Transform to make it easier to access values
        dGlobal_Parameters = dGlobal_Parameters.drop(dGlobal_Parameters.columns[1:], axis=1)  # Drop all columns but "Value" (rest is just for information in the Excel)
        dGlobal_Parameters = dict({(parameter_name, parameter_value["Value"]) for parameter_name, parameter_value in dGlobal_Parameters.iterrows()})  # Transform into dictionary

        return dGlobal_Parameters

    def get_dPower_Parameters(self):
        dPower_Parameters = pd.read_excel(self.example_folder + self.power_parameters_file, skiprows=[0, 1])
        dPower_Parameters = dPower_Parameters.drop(dPower_Parameters.columns[0], axis=1)
        dPower_Parameters = dPower_Parameters.dropna(how="all")
        dPower_Parameters = dPower_Parameters.set_index('General')

        self.yesNo_to_bool(dPower_Parameters, ['pEnableChDisPower', 'pFixStInterResToIniReserve', 'pEnablePowerImportExport'])

        # Transform to make it easier to access values
        dPower_Parameters = dPower_Parameters.drop(dPower_Parameters.columns[1:], axis=1)  # Drop all columns but "Value" (rest is just for information in the Excel)
        dPower_Parameters = dict({(parameter_name, parameter_value["Value"]) for parameter_name, parameter_value in dPower_Parameters.iterrows()})  # Transform into dictionary

        # Value adjustments
        dPower_Parameters["pMaxAngleDCOPF"] = dPower_Parameters["pMaxAngleDCOPF"] * np.pi / 180  # Convert angle from degrees to radians
        dPower_Parameters["pSBase"] *= 1e-3
        dPower_Parameters["pENSCost"] *= 1e-3

        return dPower_Parameters

    @staticmethod
    def yesNo_to_bool(df: pd.DataFrame, columns_to_be_changed: list[str]):
        for column in columns_to_be_changed:
            match df.loc[column, "Value"]:
                case "Yes":
                    df.loc[column, "Value"] = 1
                case "No":
                    df.loc[column, "Value"] = 0
                case _:
                    raise ValueError(f"Value for {column} must be either 'Yes' or 'No'.")
        return df

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

        dPower_Network["FixedCostEUR"] = dPower_Network["FixedCost"].fillna(0) * dPower_Network["FxChargeRate"].fillna(0)
        dPower_Network["Pmax"] *= 1e-3

        dPower_Network = dPower_Network.set_index(['i', 'j'])
        return dPower_Network

    def get_dPower_ThermalGen(self):
        dPower_ThermalGen = pd.read_excel(self.example_folder + self.power_thermalgen_file, skiprows=[0, 1, 3, 4, 5])
        dPower_ThermalGen = dPower_ThermalGen.drop(dPower_ThermalGen.columns[0], axis=1)
        dPower_ThermalGen = dPower_ThermalGen.rename(columns={dPower_ThermalGen.columns[0]: "g", dPower_ThermalGen.columns[1]: "tec", dPower_ThermalGen.columns[2]: "i"})
        dPower_ThermalGen = dPower_ThermalGen.set_index('g')
        dPower_ThermalGen = dPower_ThermalGen[(dPower_ThermalGen["ExisUnits"] > 0) | (dPower_ThermalGen["EnableInvest"] > 0)]  # Filter out all generators that are not existing and not investable

        dPower_ThermalGen['pSlopeVarCostEUR'] = (dPower_ThermalGen['OMVarCost'] * 1e-3 +
                                                 dPower_ThermalGen['SlopeVarCost'] * 1e-3 * dPower_ThermalGen['FuelCost'])

        dPower_ThermalGen['pInterVarCostEUR'] = dPower_ThermalGen['InterVarCost'] * 1e-6 * dPower_ThermalGen['FuelCost']
        dPower_ThermalGen['pStartupCostEUR'] = dPower_ThermalGen['StartupCost'] * 1e-6 * dPower_ThermalGen['FuelCost']
        dPower_ThermalGen['MaxInvest'] = dPower_ThermalGen.apply(lambda x: 1 if x['EnableInvest'] == 1 and x['ExisUnits'] == 0 else 0, axis=1)
        dPower_ThermalGen['RampUp'] *= 1e-3
        dPower_ThermalGen['RampDw'] *= 1e-3
        dPower_ThermalGen['MaxProd'] *= 1e-3
        dPower_ThermalGen['MinProd'] *= 1e-3
        dPower_ThermalGen['InvestCostEUR'] = dPower_ThermalGen['InvestCost'] * 1e-3 * dPower_ThermalGen['MaxProd']  # InvestCost is scaled here (1e-3), scaling of MaxProd happens above

        # Fill NaN values with 0 for MinUpTime and MinDownTime
        dPower_ThermalGen['MinUpTime'] = dPower_ThermalGen['MinUpTime'].fillna(0)
        dPower_ThermalGen['MinDownTime'] = dPower_ThermalGen['MinDownTime'].fillna(0)

        # Check that both MinUpTime and MinDownTime are integers and raise error if not
        if not dPower_ThermalGen.MinUpTime.dtype == np.int64:
            raise ValueError("MinUpTime must be an integer for all entries.")
        if not dPower_ThermalGen.MinDownTime.dtype == np.int64:
            raise ValueError("MinDownTime must be an integer for all entries.")
        dPower_ThermalGen['MinUpTime'] = dPower_ThermalGen['MinUpTime'].astype(int)
        dPower_ThermalGen['MinDownTime'] = dPower_ThermalGen['MinDownTime'].astype(int)

        return dPower_ThermalGen

    def get_dPower_RoR(self):
        dPower_RoR = self.read_generator_data(self.example_folder + self.power_ror_file)

        dPower_RoR['InvestCostEUR'] = dPower_RoR['MaxProd'] * 1e-3 * (dPower_RoR['InvestCostPerMW'] * 1e-3 + dPower_RoR['InvestCostPerMWh'] * 1e-3 * dPower_RoR['Ene2PowRatio'])
        dPower_RoR['MaxProd'] *= 1e-3
        return dPower_RoR

    def get_dPower_VRES(self):
        dPower_VRES = self.read_generator_data(self.example_folder + self.power_vres_file)
        if "MinProd" not in dPower_VRES.columns:
            dPower_VRES['MinProd'] = 0

        dPower_VRES['InvestCostEUR'] = dPower_VRES['InvestCost'] * 1e-3 * dPower_VRES['MaxProd'] * 1e-3
        dPower_VRES['MaxProd'] *= 1e-3
        dPower_VRES['OMVarCost'] *= 1e-3
        return dPower_VRES

    def get_dPower_Storage(self):
        dPower_Storage = self.read_generator_data(self.example_folder + self.power_storage_file)
        dPower_Storage['pOMVarCostEUR'] = dPower_Storage['OMVarCost'] * 1e-3
        dPower_Storage['IniReserve'] = dPower_Storage['IniReserve'].fillna(0)
        dPower_Storage['MinReserve'] = dPower_Storage['MinReserve'].fillna(0)
        dPower_Storage['MinProd'] = dPower_Storage["MinProd"].fillna(0)
        dPower_Storage['InvestCostEUR'] = dPower_Storage['MaxProd'] * 1e-3 * (dPower_Storage['InvestCostPerMW'] * 1e-3 + dPower_Storage['InvestCostPerMWh'] * 1e-3 * dPower_Storage['Ene2PowRatio'])
        dPower_Storage['MaxProd'] *= 1e-3
        dPower_Storage['MaxCons'] *= 1e-3
        return dPower_Storage

    def get_dPower_Demand(self):
        dPower_Demand = pd.read_excel(self.example_folder + self.power_demand_file, skiprows=[0, 1, 3, 4, 5])
        dPower_Demand = dPower_Demand.drop(dPower_Demand.columns[0], axis=1)
        dPower_Demand = dPower_Demand.rename(columns={dPower_Demand.columns[0]: "rp", dPower_Demand.columns[1]: "i"})
        dPower_Demand = dPower_Demand.melt(id_vars=['rp', 'i'], var_name='k', value_name='Demand')
        dPower_Demand = dPower_Demand.set_index(['rp', 'k', 'i'])
        dPower_Demand = dPower_Demand * 1e-3
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

    def get_dPower_WeightsRP(self):
        dPower_WeightsRP = pd.read_excel(self.example_folder + self.power_weightsrp_file, skiprows=[0, 1, 3, 4, 5])
        dPower_WeightsRP = dPower_WeightsRP.drop(dPower_WeightsRP.columns[0], axis=1)
        dPower_WeightsRP = dPower_WeightsRP.rename(columns={dPower_WeightsRP.columns[0]: "rp", dPower_WeightsRP.columns[1]: "Weight"})
        dPower_WeightsRP = dPower_WeightsRP.set_index('rp')
        return dPower_WeightsRP

    def get_dPower_WeightsK(self):
        dPower_WeightsK = pd.read_excel(self.example_folder + self.power_weightsk_file, skiprows=[0, 1, 3, 4, 5])
        dPower_WeightsK = dPower_WeightsK.drop(dPower_WeightsK.columns[0], axis=1)
        dPower_WeightsK = dPower_WeightsK.rename(columns={dPower_WeightsK.columns[0]: "k", dPower_WeightsK.columns[1]: "Weight"})
        dPower_WeightsK = dPower_WeightsK.set_index('k')
        return dPower_WeightsK

    def get_dPower_Hindex(self):
        dPower_Hindex = pd.read_excel(self.example_folder + self.power_hindex_file, skiprows=[0, 1, 2, 3, 4])
        dPower_Hindex = dPower_Hindex.drop(dPower_Hindex.columns[0], axis=1)
        dPower_Hindex = dPower_Hindex.rename(columns={dPower_Hindex.columns[0]: "p", dPower_Hindex.columns[1]: "rp", dPower_Hindex.columns[2]: "k"})
        dPower_Hindex = dPower_Hindex.set_index(['p', 'rp', 'k'])
        return dPower_Hindex

    def get_dPower_ImpExpHubs(self):
        dPower_ImpExpHubs = pd.read_excel(self.example_folder + self.power_impexphubs_file, skiprows=[0, 1, 3, 4, 5])
        dPower_ImpExpHubs = dPower_ImpExpHubs.drop(dPower_ImpExpHubs.columns[0], axis=1)
        dPower_ImpExpHubs = dPower_ImpExpHubs.set_index(['hub', 'i'])

        # Validate that all values for "Import Type" and "Export Type" == [Imp/ExpFix or Imp/ExpMax]
        errors = dPower_ImpExpHubs[~dPower_ImpExpHubs['Import Type'].isin(['ImpFix', 'ImpMax'])]
        if len(errors) > 0:
            raise ValueError(f"'Import Type' must be 'ImpFix' or 'ImpMax'. Please check: \n{errors}\n")
        errors = dPower_ImpExpHubs[~dPower_ImpExpHubs['Export Type'].isin(['ExpFix', 'ExpMax'])]
        if len(errors) > 0:
            raise ValueError(f"'Export Type' must be 'ExpFix' or 'ExpMax'. Please check: \n{errors}\n")

        # Validate that for each hub, all connections have the same Import Type and Export Type
        errors = dPower_ImpExpHubs.groupby('hub').agg({'Import Type': 'nunique', 'Export Type': 'nunique'})
        errors = errors[(errors['Import Type'] > 1) | (errors['Export Type'] > 1)]
        if len(errors) > 0:
            raise ValueError(f"Each hub must have the same Import Type (Fix or Max) and the same Export Type (Fix or Max) for each connection. Please check: \n{errors.index}\n")

        # Adjust values
        dPower_ImpExpHubs["Pmax Import"] *= 1e-3
        dPower_ImpExpHubs["Pmax Export"] *= 1e-3

        return dPower_ImpExpHubs

    def get_dPower_ImpExpProfiles(self):
        with warnings.catch_warnings(action="ignore", category=UserWarning):  # Otherwise there is a warning regarding data validation in the Excel-File (see https://stackoverflow.com/questions/53965596/python-3-openpyxl-userwarning-data-validation-extension-not-supported)
            dPower_ImpExpProfiles = pd.read_excel(self.example_folder + self.power_impexpprofiles_file, skiprows=[0, 1, 3, 4, 5], sheet_name='Power ImpExpProfiles')
        dPower_ImpExpProfiles = dPower_ImpExpProfiles.drop(dPower_ImpExpProfiles.columns[0], axis=1)
        dPower_ImpExpProfiles = dPower_ImpExpProfiles.melt(id_vars=['hub', 'rp', 'Type'], var_name='k', value_name='Value')

        # Validate that each multiindex is only present once
        dPower_ImpExpProfiles = dPower_ImpExpProfiles.set_index(['hub', 'rp', 'k', 'Type'])
        if not dPower_ImpExpProfiles.index.is_unique:
            raise ValueError(f"Indices for Imp-/Export values must be unique (i.e., no two entries for the same hub, rp, Type and k). Please check these indices: {dPower_ImpExpProfiles.index[dPower_ImpExpProfiles.index.duplicated(keep=False)]}")

        # Validate that all values for "Type" == [ImpExp, Price]
        dPower_ImpExpProfiles = dPower_ImpExpProfiles.reset_index().set_index(['hub', 'rp', 'k'])
        errors = dPower_ImpExpProfiles[~dPower_ImpExpProfiles['Type'].isin(['ImpExp', 'Price'])]
        if len(errors) > 0:
            raise ValueError(f"'Type' must be 'ImpExp' or 'Price'. Please check: \n{errors}\n")

        # Create combined table (with one row for each hub, rp and k)
        dPower_ImpExpProfiles = dPower_ImpExpProfiles.pivot(columns="Type", values="Value")
        dPower_ImpExpProfiles.columns.name = None  # Fix name of columns/indices (which are altered through pivot)

        # Adjust values
        dPower_ImpExpProfiles["ImpExp"] *= 1e-3

        # Check that Pmax of ImpExpConnections can handle the maximum import and export (for those connections that are ImpFix or ExpFix)
        max_import = dPower_ImpExpProfiles[dPower_ImpExpProfiles["ImpExp"] >= 0]["ImpExp"].groupby("hub").max()
        max_export = -dPower_ImpExpProfiles[dPower_ImpExpProfiles["ImpExp"] <= 0]["ImpExp"].groupby("hub").min()

        pmax_sum_by_hub = self.dPower_ImpExpHubs.groupby('hub').agg({'Pmax Import': 'sum', 'Pmax Export': 'sum', 'Import Type': 'first', 'Export Type': 'first'})
        import_violations = max_import[(max_import > pmax_sum_by_hub['Pmax Import']) & (pmax_sum_by_hub['Import Type'] == 'ImpFix')]
        export_violations = max_export[(max_export > pmax_sum_by_hub['Pmax Export']) & (pmax_sum_by_hub['Export Type'] == 'ExpFix')]

        if not import_violations.empty:
            error_information = pd.concat([import_violations, pmax_sum_by_hub['Pmax Import']], axis=1)  # Concat Pmax information and maximum import
            error_information = error_information[error_information["ImpExp"].notna()]  # Only show rows where there is a violation
            error_information = error_information.rename(columns={"ImpExp": "Max Import from Profiles", "Pmax Import": "Sum of Pmax Import from Hub Definition"})  # Rename columns for readability
            error_information *= 1e3  # Convert back to input format
            raise ValueError(f"At least one hub has ImpFix imports which exceed the sum of Pmax of all connections. Please check: \n{error_information}\n")

        if not export_violations.empty:
            error_information = pd.concat([export_violations, pmax_sum_by_hub['Pmax Export']], axis=1)  # Concat Pmax information and maximum export
            error_information = error_information[error_information["ImpExp"].notna()]  # Only show rows where there is a violation
            error_information = error_information.rename(columns={"ImpExp": "Max Export from Profiles", "Pmax Export": "Sum of Pmax Export from Hub Definition"})  # Rename columns for readability
            error_information *= 1e3  # Convert back to input format
            raise ValueError(f"At least one hub has ExpFix exports which exceed the sum of Pmax of all connections. Please check: \n{error_information}\n")

        return dPower_ImpExpProfiles

    # Function to read generator data
    @staticmethod
    def read_generator_data(file_path):
        d_generator = pd.read_excel(file_path, skiprows=[0, 1, 3, 4, 5])
        d_generator = d_generator.drop(d_generator.columns[0], axis=1)
        d_generator = d_generator[(d_generator["ExisUnits"] > 0) | (d_generator["EnableInvest"] > 0)]  # Filter out all generators that are not existing and not investable
        d_generator = d_generator.rename(columns={d_generator.columns[0]: "g", d_generator.columns[1]: "tec", d_generator.columns[2]: "i"})
        d_generator = d_generator.set_index('g')
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
