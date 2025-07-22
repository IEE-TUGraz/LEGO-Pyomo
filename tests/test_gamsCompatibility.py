import shutil

from openpyxl import load_workbook

from CompareModels import compareModels, ModelTypeForComparison
from InOutModule.printer import Printer

printer = Printer.getInstance()


def test_comparisonExampleAgainstGAMS(tmp_path):
    """
    Compares the GAMS result against the Pyomo result.
    :param tmp_path: Temporary path for the test (provided by pytest).
    :return: None
    """
    mps_equal = compareModels(ModelTypeForComparison.DETERMINISTIC, "data/example", True,
                              ModelTypeForComparison.GAMS, "data/example", True, tmp_folder_path=tmp_path)

    assert mps_equal


def test_comparisonExampleSOCPAgainstGAMS(tmp_path):
    """
    Compares the GAMS result against the Pyomo result for the SOCP example.
    :param tmp_path: Temporary path for the test (provided by pytest).
    :return: None
    """
    data_folder = "data/example"

    # Copy the data folder to a temporary path
    tmp_path_originalData = str(tmp_path / "originalData")
    shutil.copytree(data_folder, tmp_path_originalData)

    # Switch solver (to enable quadratic constraints)
    workbook = load_workbook(filename=tmp_path_originalData + "/Global_Parameters.xlsx")
    sheet = workbook.active
    sheet["C5"] = "gurobi"  # Set solver
    workbook.save(filename=tmp_path_originalData + "/Global_Parameters.xlsx")

    # Activate SOCP in Power Parameters
    workbook = load_workbook(filename=tmp_path_originalData + "/Power_Parameters.xlsx")
    sheet = workbook.active
    sheet["C37"] = "Yes"  # Enable SOCP
    workbook.save(filename=tmp_path_originalData + "/Power_Parameters.xlsx")

    # Use SOCP for all lines
    workbook = load_workbook(filename=tmp_path_originalData + "/Power_Network.xlsx")
    sheet = workbook.active
    for i in range(8, sheet.max_row + 1):
        sheet[f"O{i}"] = "SOCP"  # Set all lines to use SOCP
    workbook.save(filename=tmp_path_originalData + "/Power_Network.xlsx")

    mps_equal = compareModels(ModelTypeForComparison.DETERMINISTIC, tmp_path_originalData, True,
                              ModelTypeForComparison.GAMS, tmp_path_originalData, True, tmp_folder_path=tmp_path)

    assert mps_equal
