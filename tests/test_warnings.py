import pandas as pd
import pytest

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer

printer = Printer.getInstance()

scenario_a = "ScenarioA"

network_duplicate_lines_different_c = [
    (
        [
            ("Node_1", "Node_6", "c1", scenario_a),  # network_entries
            ("Node_2", "Node_3", "c1", scenario_a),
            ("Node_2", "Node_6", "c1", scenario_a),
        ],
        False,  # if warning expected
    ),
    (
        [
            ("Node_2", "Node_3", "c1", scenario_a),
            ("Node_2", "Node_3", "c2", scenario_a),
            ("Node_2", "Node_3", "c3", scenario_a),
        ],
        True,
    ),
    (
        [
            ("Node_1", "Node_6", "c1", scenario_a),
            ("Node_6", "Node_1", "c2", scenario_a),
        ],
        True,
    ),
    (
        [
            ("Node_2", "Node_3", "c1", scenario_a),
            ("Node_2", "Node_3", "c2", "ScenarioB"),
        ],
        False,
    ),

]

network_duplicate_lines_same_c = [
    (
        [
            ("Node_1", "Node_6", "c1", scenario_a),  # network_entries
            ("Node_1", "Node_7", "c1", scenario_a),
        ],
        False,  # if error expected
    ),
    (
        [
            ("Node_1", "Node_2", "c1", scenario_a),
            ("Node_1", "Node_2", "c1", scenario_a),
            ("Node_1", "Node_2", "c1", scenario_a),
        ],
        True,
    ),
    (
        [
            ("Node_1", "Node_6", "c1", scenario_a),
            ("Node_6", "Node_1", "c1", scenario_a),
        ],
        True,
    ),
    (
        [
            ("Node_1", "Node_2", "c1", scenario_a),
            ("Node_1", "Node_2", "c1", "ScenarioB"),
        ],
        False,
    ),
]


def create_case_study(storage, pEnableChDisPower, network_entries=None):
    """
    Creates a small case study.
    :param storage: Name of the storage unit.
    :param pEnableChDisPower: Whether to avoid simultaneous charging and discharging.
    :param network_entries: Network line entries as (i, j, c, scenario) tuples.
    :return: Case study.
    """
    rp = "rp01"
    ks = ["k0001", "k0002"]
    bus = "Node_1"
    scenario = scenario_a
    network_entries = [] if (network_entries is None) else network_entries

    network_lines = []
    network_scenarios = []
    for network_entry in network_entries:
        i, j, c, scenario_name = network_entry
        network_lines.append((i, j, c))
        network_scenarios.append(scenario_name)

    dGlobal_Parameters = {
        "pSolver": "highs",
        "pEnableRMIP": 0,
        "pPowerScalingFactor": 1,
        "pCostScalingFactor": 1,
        "pMovWindow": 1,
    }

    dPower_Parameters = {
        "pEnableThermalGen": True,
        "pEnableVRES": False,
        "pEnableStorage": True,
        "pEnablePowerImportExport": False,
        "pEnableSoftLineLoadLimits": False,
        "pEnableSOCP": False,
        "pEnableChDisPower": pEnableChDisPower,
        "p2ndResUp": 0,
        "p2ndResDW": 0,
        "pSBase": 1,
        "pENSCost": 10000,
        "pLOLCost": 10000,
        "pMaxAngleDCOPF": 180,
        "pFixStInterResToIniReserve": False,
        "pReprPeriodEdgeHandlingUnitCommitment": "cyclic",
        "pReprPeriodEdgeHandlingRamping": "cyclic",
        "pReprPeriodEdgeHandlingIntraDayStorage": "cyclic",
        "is": None,
    }

    dPower_Hindex = pd.DataFrame(data={"scenario": scenario}, index=pd.MultiIndex.from_tuples([(k.replace("k", "h"), rp, k) for k in ks], names=["p", "rp", "k"]))

    dPower_WeightsK = pd.DataFrame(data={"pWeight_k": 1}, index=pd.Index(ks, name="k"))

    buses = [bus]
    for entry in network_entries:
        for network_bus in entry[:2]:
            if network_bus not in buses:
                buses.append(network_bus)

    dPower_BusInfo = pd.DataFrame({"z": ["TestZone"] * len(buses), "zoi": [1] * len(buses)}, index=pd.Index(buses, name="i"))

    dPower_Demand = pd.DataFrame(
        {
            "value": [50 if demand_bus == bus else 0 for k in ks for demand_bus in buses],
            "scenario": scenario,
        },
        index=pd.MultiIndex.from_tuples([(rp, k, demand_bus) for k in ks for demand_bus in buses], names=["rp", "k", "i"]),
    )

    dPower_Network = pd.DataFrame(
        columns=["excl", "id", "pRline", "pXline", "pBcline", "pAngle", "pRatio", "pPmax", "pEnableInvest", "pFOMCost", "pInvestCost", "pTecRepr", "scenario"],
        data=[
            (None, None, 0, 1, 0, 0, 1, 100, 0, 0, 0, "TP", network_scenario)
            for network_scenario in network_scenarios
        ],
        index=pd.MultiIndex.from_tuples(network_lines, names=["i", "j", "c"]),
    )

    dPower_ThermalGen = pd.DataFrame(columns=["excl", "tec", "i", "ExisUnits", "MaxProd", "MinProd", "RampUp", "RampDw", "MinUpTime", "MinDownTime", "Qmax", "Qmin", "FuelCost", "Efficiency", "CommitConsumption", "OMVarCost", "StartupConsumption", "EFOR", "EnableInvest", "InvestCost"],
                                     data=[(None, "Gas", bus, 1, 60, 60, 60, 60, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0)],
                                     index=pd.Index(["FixedGenerator"], name="g"))

    dPower_Storage = pd.DataFrame(
        {
            "tec": ["BESS"],
            "i": [bus],
            "ExisUnits": [1],
            "MaxProd": [1],
            "MinProd": [0],
            "MaxCons": [1],
            "DisEffic": [0.9],
            "ChEffic": [1],
            "Qmax": [0],
            "Qmin": [0],
            "MinReserve": [0],
            "IniReserve": [1],
            "IsLDES": [0],
            "OMVarCost": [1],
            "EnableInvest": [0],
            "MaxInvest": [0],
            "InvestCostPerMW": [0],
            "InvestCostPerMWh": [0],
            "Ene2PowRatio": [1],
        },
        index=pd.Index([storage], name="g"),
    )
    cs = CaseStudy(
        "",
        do_not_merge_single_node_buses=True,
        dGlobal_Parameters=dGlobal_Parameters,
        dPower_Parameters=dPower_Parameters,
        dPower_BusInfo=dPower_BusInfo,
        dPower_Demand=dPower_Demand,
        dPower_Hindex=dPower_Hindex,
        dPower_Storage=dPower_Storage,
        dPower_ThermalGen=dPower_ThermalGen,
        dPower_Network=dPower_Network,
        dPower_WeightsK=dPower_WeightsK,
    )

    return cs


@pytest.mark.parametrize("network_entries, expected_warning_displayed", network_duplicate_lines_different_c)
def test_network_duplicate_lines_different_c(tmp_path, network_entries, expected_warning_displayed):
    """
    Tests if a warning is triggered when network lines use the same i, j entries with a different c value.
    :param tmp_path: Temporary path for the test (provided by pytest).
    :param network_entries: Network line entries.
    :param expected_warning_displayed: Whether the warning is expected.
    :return: None
    """
    log_path = str(tmp_path / "test_warning.log")
    printer.set_logfile(log_path)
    printer.information(f"Logging to {log_path}")

    storage = "TestStorage"
    create_case_study(storage, False, network_entries)

    with open(log_path) as log_file:
        log_content = log_file.read()

    warning_displayed = "Warning: Parallel network lines found in (at least) scenario " in log_content

    assert warning_displayed == expected_warning_displayed


@pytest.mark.parametrize("network_entries, expected_error_displayed", network_duplicate_lines_same_c)
def test_network_duplicate_lines_same_c(tmp_path, network_entries, expected_error_displayed):
    """
    Tests if an error is triggered when network lines use the same i, j, entries with the same c value.
    :param tmp_path: Temporary path for the test (provided by pytest).
    :param network_entries: Network line entries.
    :param expected_error_displayed: Whether the error is expected.
    :return: None
    """
    log_path = str(tmp_path / "test_warning.log")
    printer.set_logfile(log_path)
    printer.information(f"Logging to {log_path}")

    storage = "TestStorage"
    error_message = ""
    try:
        create_case_study(storage, False, network_entries)
    except ValueError as error:
        error_message = str(error)

    error_displayed = ("Duplicate network line found in (at least) scenario " in error_message and
                       "If the lines should be parallel, assign different 'c' for each parallel line." in error_message)

    assert error_displayed == expected_error_displayed
