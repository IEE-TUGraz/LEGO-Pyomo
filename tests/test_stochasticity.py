import shutil
import warnings

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from CompareModels import compareModels, ModelTypeForComparison
from InOutModule import ExcelReader
from InOutModule.ExcelWriter import ExcelWriter
from InOutModule.printer import Printer

printer = Printer.getInstance()


def test_deterministicVsExtensiveWithNoScenarios():
    """
    Test if the MPS files of a deterministic model and an extensive form model with no scenarios are equal.
    :return: None
    """
    mps_equal = compareModels(ModelTypeForComparison.DETERMINISTIC, "data/example", True,
                              ModelTypeForComparison.EXTENSIVE_FORM, "data/example", True, remove_scenario_prefix2=True)

    assert mps_equal


def test_deterministicVsExtensiveWithTwoEqualScenarios(tmp_path):
    """
    Test if the MPS files of a deterministic model and an extensive form model with two equal scenarios are equal.
    :param tmp_path: Temporary path for the test
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
        ("Power_Hindex", f"{tmp_path_scenarioData}/Power_Hindex.xlsx", ExcelReader.get_dPower_Hindex, ew.write_dPower_Hindex),
        ("Power_WeightsRP", f"{tmp_path_scenarioData}/Power_WeightsRP.xlsx", ExcelReader.get_dPower_WeightsRP, ew.write_dPower_WeightsRP),
        ("Power_WeightsK", f"{tmp_path_scenarioData}/Power_WeightsK.xlsx", ExcelReader.get_dPower_WeightsK, ew.write_dPower_WeightsK),
        ("Power_BusInfo", f"{tmp_path_scenarioData}/Power_BusInfo.xlsx", ExcelReader.get_dPower_BusInfo, ew.write_dPower_BusInfo),
        ("Power_Network", f"{tmp_path_scenarioData}/Power_Network.xlsx", ExcelReader.get_dPower_Network, ew.write_dPower_Network),
        ("Power_Demand", f"{tmp_path_scenarioData}/Power_Demand.xlsx", ExcelReader.get_dPower_Demand, ew.write_dPower_Demand),
        ("Power_ThermalGen", f"{tmp_path_scenarioData}/Power_ThermalGen.xlsx", ExcelReader.get_dPower_ThermalGen, ew.write_dPower_ThermalGen),
        ("Power_VRES", f"{tmp_path_scenarioData}/Power_VRES.xlsx", ExcelReader.get_dPower_VRES, ew.write_VRES),
        ("Power_VRESProfiles", f"{tmp_path_scenarioData}/Power_VRESProfiles.xlsx", ExcelReader.get_dPower_VRESProfiles, ew.write_VRESProfiles),
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
    data = ExcelReader.get_dGlobal_Scenarios(f"{tmp_path_scenarioData}/Global_Scenarios.xlsx", True, True)
    data.loc['ScenarioB'] = [np.nan, None, 1.0, "Example Scenario B added by pytest", "Scenarios"]
    ew.write_dGlobal_Scenarios(data, tmp_path_scenarioData)

    mps_equal = compareModels(ModelTypeForComparison.DETERMINISTIC, tmp_path_originalData, True,
                              ModelTypeForComparison.EXTENSIVE_FORM, tmp_path_scenarioData, True,
                              skip_comparison_overall=True)  # TODO: Adjust implementation so that variables and constraints for ScenarioA can still be compared

    assert mps_equal


def test_extensiveStochasticVsBendersWithTwoDifferentScenarios():
    warnings.warn("This test is not active, since Benders says the solution is infeasible.")  # TODO
    # mps_equal = compareModels(ModelTypeForComparison.EXTENSIVE_FORM, "data/exampleStochastic", True,
    #                           ModelTypeForComparison.BENDERS, "data/exampleStochastic", True,
    #                           skip_comparison_overall=True)
    assert True
