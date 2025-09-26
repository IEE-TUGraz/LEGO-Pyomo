import shutil
import warnings

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from InOutModule import ExcelReader
from InOutModule.ExcelWriter import ExcelWriter
from InOutModule.printer import Printer
from LEGO.helpers.CompareModels import compareModels, ModelTypeForComparison

printer = Printer.getInstance()


def test_deterministicVsExtensiveWithNoScenarios(tmp_path):
    """
    Test if the MPS files of a deterministic model and an extensive form model with no scenarios are equal.
    :param tmp_path: Temporary path for the test (provided by pytest).
    :return: None
    """
    mps_equal = compareModels(ModelTypeForComparison.DETERMINISTIC, "data/example", True,
                              ModelTypeForComparison.EXTENSIVE_FORM, "data/example", True, remove_scenario_prefix2=True, tmp_folder_path=tmp_path)

    assert mps_equal


def test_deterministicVsExtensiveWithTwoEqualScenarios(tmp_path):
    """
    Test if the MPS files of a deterministic model and an extensive form model with two equal scenarios are equal.
    :param tmp_path: Temporary path for the test (provided by pytest).
    :return: None
    """
    data_folder = "data/example"

    # Copy the data folder to a temporary path
    tmp_path_originalData = str(tmp_path / "originalData")
    shutil.copytree(data_folder, tmp_path_originalData)

    # Deactivate use of storage for this test. TODO: Implement storage handling (once read and write methods are implemented)
    printer.warning("TODO: Activate use of storage for this test")
    workbook = load_workbook(filename=tmp_path_originalData + "/Power_Parameters.xlsx")
    sheet = workbook.active
    sheet["C27"] = "No"
    workbook.save(filename=tmp_path_originalData + "/Power_Parameters.xlsx")

    tmp_path_scenarioData = str(tmp_path / "scenarioData")
    shutil.copytree(tmp_path_originalData, tmp_path_scenarioData)

    ew = ExcelWriter("InOutModule/TableDefinitions.xml")

    combinations = [
        ("Power_Hindex", f"{tmp_path_scenarioData}/Power_Hindex.xlsx", ExcelReader.get_Power_Hindex, ew.write_Power_Hindex),
        ("Power_WeightsRP", f"{tmp_path_scenarioData}/Power_WeightsRP.xlsx", ExcelReader.get_Power_WeightsRP, ew.write_Power_WeightsRP),
        ("Power_WeightsK", f"{tmp_path_scenarioData}/Power_WeightsK.xlsx", ExcelReader.get_Power_WeightsK, ew.write_Power_WeightsK),
        ("Power_BusInfo", f"{tmp_path_scenarioData}/Power_BusInfo.xlsx", ExcelReader.get_Power_BusInfo, ew.write_Power_BusInfo),
        ("Power_Network", f"{tmp_path_scenarioData}/Power_Network.xlsx", ExcelReader.get_Power_Network, ew.write_Power_Network),
        ("Power_Demand", f"{tmp_path_scenarioData}/Power_Demand.xlsx", ExcelReader.get_Power_Demand, ew.write_Power_Demand),
        ("Power_ThermalGen", f"{tmp_path_scenarioData}/Power_ThermalGen.xlsx", ExcelReader.get_Power_ThermalGen, ew.write_Power_ThermalGen),
        ("Power_VRES", f"{tmp_path_scenarioData}/Power_VRES.xlsx", ExcelReader.get_Power_VRES, ew.write_VRES),
        ("Power_VRESProfiles", f"{tmp_path_scenarioData}/Power_VRESProfiles.xlsx", ExcelReader.get_Power_VRESProfiles, ew.write_VRESProfiles),
        # TODO: Check for Inflows & Storage
    ]

    for excel_definition_id, file_path, read, write in combinations:
        data = read(file_path, True, True)

        # Copy all entries, but with column 'Scenario' set to 'ScenarioB'
        index_names = data.index.names
        data = data.reset_index()
        data_copy = data.copy()
        data_copy['scenario'] = 'ScenarioB'
        data = pd.concat([data, data_copy], ignore_index=True, sort=False)
        data = data.set_index(index_names)

        # Write the modified data back to the Excel file
        write(data, tmp_path_scenarioData)

    # Add ScenarioB to the Global_Scenarios.xlsx
    data = ExcelReader.get_Global_Scenarios(f"{tmp_path_scenarioData}/Global_Scenarios.xlsx", True, True)
    data.loc['ScenarioB'] = [np.nan, None, 1.0, "Example Scenario B added by pytest", "Scenarios"]
    ew.write_Global_Scenarios(data, tmp_path_scenarioData)

    mps_equal = compareModels(ModelTypeForComparison.DETERMINISTIC, tmp_path_originalData, True,
                              ModelTypeForComparison.EXTENSIVE_FORM, tmp_path_scenarioData, True,
                              skip_comparison_overall=True, tmp_folder_path=tmp_path)  # TODO: Adjust implementation so that variables and constraints for ScenarioA can still be compared

    assert mps_equal
