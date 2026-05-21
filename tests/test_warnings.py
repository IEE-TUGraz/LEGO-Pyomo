import pandas as pd
import pytest

import SQLiteWriter
from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO

printer = Printer.getInstance()

charge_discharge_warning_combinations = [
    (True, False),  # constraint enabled, no warning expected
    (False, True),  # constraint disabled, warning expected
]


def create_case_study(storage, pEnableChDisPower):
    """
    Creates a small case study.
    :param storage: Name of the storage unit.
    :param pEnableChDisPower: Whether to avoid simultaneous charging and discharging.
    :return: Case study.
    """
    rp = "rp01"
    ks = ["k0001", "k0002"]
    bus = "Node_1"
    scenario = "ScenarioA"

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

    dPower_BusInfo = pd.DataFrame({"z": ["TestZone"], "zoi": [1]}, index=pd.Index([bus], name="i"))

    dPower_Demand = pd.DataFrame(
        {"value": 50, "scenario": scenario},
        index=pd.MultiIndex.from_tuples([(rp, k, bus) for k in ks], names=["rp", "k", "i"]),
    )

    dPower_Network = pd.DataFrame(columns=["excl", "id", "pRline", "pXline", "pBcline", "pAngle", "pRatio", "pPmax", "pEnableInvest", "pFOMCost", "pInvestCost", "pTecRepr", "scenario"], index=pd.MultiIndex.from_tuples([], names=["i", "j", "c"]))

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


@pytest.mark.parametrize("pEnableChDisPower, expected_warning_displayed", charge_discharge_warning_combinations)
def test_simultaneous_charge_discharge_warning(tmp_path, pEnableChDisPower, expected_warning_displayed):
    """
    Tests if the simultaneous storage charge/discharge warning is printed when expected.
    :param tmp_path: Temporary path for the test (provided by pytest).
    :param pEnableChDisPower: Whether to avoid simultaneous charging and discharging.
    :param expected_warning_displayed: Whether the warning is expected in the log.
    :return: None
    """
    log_path = str(tmp_path / "test_warning.log")
    printer.set_logfile(log_path)
    printer.information(f"Logging to {log_path}")

    storage = "TestStorage"
    cs = create_case_study(storage, pEnableChDisPower)
    lego = LEGO(cs)
    model, _ = lego.build_model()

    lego.solve_model()

    SQLiteWriter.model_to_sqlite(model, str(tmp_path / "test_warning.sqlite"))

    with open(log_path) as log_file:
        log_content = log_file.read()

    warning_displayed = f"Warning: Storage unit {storage} charges and discharges simultaneously in" in log_content

    assert warning_displayed == expected_warning_displayed
