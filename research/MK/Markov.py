import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
import math
import os
import shutil
import sqlite3
import time
import typing
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from rich_argparse import RichHelpFormatter

from InOutModule import ExcelReader, SQLiteWriter, Utilities
from InOutModule.CaseStudy import CaseStudy
from InOutModule.ExcelWriter import ExcelWriter
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO
from LEGO.LEGOUtilities import add_UnitCommitmentSlack_And_FixVariables, getUnitCommitmentSlackCost, markov_summand, markov_sum

########################################################################################################################
# Setup
########################################################################################################################

printer = Printer.getInstance()
printer.set_width(300)

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)


def write_results(model, file_prefix: str, no_sqlite: bool):
    if not no_sqlite:
        sqlite_timer = time.time()
        sqlite_file = f"{file_prefix}.sqlite"
        printer.information(f"Writing model to SQLite database: {sqlite_file}")
        SQLiteWriter.model_to_sqlite(model, sqlite_file)
        printer.information(f"Writing model to SQLite database took {time.time() - sqlite_timer:.2f} seconds")


def _load_unit_commitment_from_sqlite(sqlite_file: str, case_label: str) -> pd.DataFrame:
    """Load unit commitment data from a sqlite file and return a DataFrame with case/rp/k/g index."""
    cnx = sqlite3.connect(sqlite_file)
    # Rename 'thermalGenerators' -> 'g' to normalize index names across tables
    g_rename = {"thermalGenerators": "g"}
    vCommit = pd.read_sql("SELECT * FROM vCommit", cnx).rename(columns={"values": "vCommit", **g_rename})
    vStartup = pd.read_sql("SELECT * FROM vStartup", cnx).rename(columns={"values": "vStartup", **g_rename})
    vShutdown = pd.read_sql("SELECT * FROM vShutdown", cnx).rename(columns={"values": "vShutdown", **g_rename})
    vGenP = pd.read_sql("SELECT * FROM vGenP", cnx).rename(columns={"values": "vGenP"})
    vPNS = pd.read_sql("SELECT * FROM vPNS", cnx).rename(columns={"values": "vPNS"})
    vEPS = pd.read_sql("SELECT * FROM vEPS", cnx).rename(columns={"values": "vEPS"})
    pDemandP = pd.read_sql("SELECT * FROM pDemandP", cnx).rename(columns={"values": "pDemandP"})
    pMinUpTime = pd.read_sql("SELECT * FROM pMinUpTime", cnx).rename(columns={"values": "pMinUpTime", **g_rename})
    pMinDownTime = pd.read_sql("SELECT * FROM pMinDownTime", cnx).rename(columns={"values": "pMinDownTime", **g_rename})
    cnx.close()

    idx = ["rp", "k", "g"]
    # Only keep thermal generators (vCommit index) — vGenP includes all generators
    df = vCommit.set_index(idx)
    df = df.join(vStartup.set_index(idx)["vStartup"])
    df = df.join(vShutdown.set_index(idx)["vShutdown"])
    df = df.join(vGenP.set_index(idx)["vGenP"])

    # Aggregate demand, PNS, EPS across nodes (i) per rp/k
    pDemandP_agg = pDemandP.groupby(["rp", "k"])["pDemandP"].sum()
    vPNS_agg = vPNS.groupby(["rp", "k"])["vPNS"].sum()
    vEPS_agg = vEPS.groupby(["rp", "k"])["vEPS"].sum()
    df = df.join(pDemandP_agg, on=["rp", "k"])
    df = df.join(vPNS_agg, on=["rp", "k"])
    df = df.join(vEPS_agg, on=["rp", "k"])

    # Join generator-level parameters
    df = df.join(pMinUpTime.set_index("g"), on="g")
    df = df.join(pMinDownTime.set_index("g"), on="g")

    df["case"] = case_label
    df = df.reset_index().set_index(["case", "rp", "k", "g"])
    return df


def plot_unit_commitment(sqlite_files: typing.List[str], case_labels: typing.List[str], case_study_folder: str, number_of_hours: int = 24 * 7, start_hour: int = 1):
    """
    Plot the unit commitment from sqlite result files.
    :param sqlite_files: List of paths to SQLite result files
    :param case_labels: List of case labels corresponding to each sqlite file
    :param case_study_folder: Path to folder containing Power_Hindex file
    :param number_of_hours: Number of hours to plot (default: 24 * 7 = 168)
    :param start_hour: Start hour for the plot (default: 1)
    """
    plt.rcParams['figure.dpi'] = 300

    frames = [_load_unit_commitment_from_sqlite(f, label) for f, label in zip(sqlite_files, case_labels)]
    df = pd.concat(frames)

    # Get original mapping from Power_Hindex
    hindex = ExcelReader.get_Power_Hindex(case_study_folder + "Power_Hindex.xlsx")
    hindex = hindex.reset_index()
    hindex["p_int"] = hindex["p"].str.extract(r'(\d+)').astype(int)
    hindex["rp_int"] = hindex["rp"].str.extract(r'(\d+)').astype(int)
    hindex["k_int"] = hindex["k"].str.extract(r'(\d+)').astype(int)

    # Filter the dataframe to only include the relevant hours
    hindex = hindex.loc[(hindex["p_int"] >= start_hour) & (hindex["p_int"] <= start_hour + number_of_hours - 1)]

    index = [i + 1 for i in range(len(hindex))]
    nr_cases = len(df.index.get_level_values("case").unique())

    fig, axs = plt.subplots(nr_cases, len(df.index.get_level_values("g").unique()), figsize=(6 * len(df.index.get_level_values("g").unique()), 2 * nr_cases))

    for i, case in enumerate(df.index.get_level_values("case").unique()):
        for j, g in enumerate(df.index.get_level_values("g").unique()):

            data_vGenP = {}
            data_bar_startup = {}
            data_bar_shutdown = {}
            data_bar_min_uptime_height = {}
            data_bar_min_downtime_bottom = {}
            data_demand = {}
            data_vPNS = {}
            data_vEPS = {}
            data_vCommit = {}

            for counter, (_, row) in enumerate(hindex.iterrows()):
                counter += 1
                rp = row["rp"] if case != "Truth " else "rp01"
                k = row["k"] if case != "Truth " else row["p"].replace("h", "k")
                data_vGenP[counter] = df.loc[case, rp, k, g]["vGenP"]
                data_vCommit[counter] = df.loc[case, rp, k, g]["vCommit"]
                data_bar_startup[counter] = df.loc[case, rp, k, g]["vStartup"]
                data_bar_shutdown[counter] = df.loc[case, rp, k, g]["vShutdown"]
                data_demand[counter] = df.loc[case, rp, k, g]["pDemandP"]
                data_vPNS[counter] = df.loc[case, rp, k, g]["vPNS"]
                data_vEPS[counter] = df.loc[case, rp, k, g]["vEPS"]

            for counter, (_, row) in enumerate(hindex.iterrows()):
                counter += 1
                data_bar_min_uptime_height[counter] = sum([data_bar_startup[a] for a in [counter - b for b in range(0, int(df.loc[case, rp, k, g]["pMinUpTime"] - 1)) if counter - b > 0]])
                data_bar_min_downtime_bottom[counter] = 1 - sum([data_bar_shutdown[a] for a in [counter - b for b in range(0, int(df.loc[case, rp, k, g]["pMinDownTime"] - 1)) if counter - b > 0]])

            axs2 = axs[i].twinx()
            axs2.set_title(f"{case}")
            axs2.set_ylim(0, 3)
            axs2.bar(index, data_bar_startup.values(), color="green", alpha=0.5, bottom=[list(data_vCommit.values())[-1]] + list(data_vCommit.values())[:-1], width=1, label="Startup")
            axs2.bar(index, data_bar_shutdown.values(), color="red", alpha=0.5, bottom=data_vCommit.values(), width=1, label="Shutd.")
            axs2.plot(index, data_vCommit.values(), color="gray", alpha=0.5, label="Commit", linewidth=1.5)
            axs2.set_ylabel("Startup / Shutdown", color="black")

            axs2.bar(index, data_bar_min_uptime_height.values(), color="green", alpha=0.2, width=1)
            axs2.bar(index, bottom=data_bar_min_downtime_bottom.values(), height=[1 - x for x in data_bar_min_downtime_bottom.values()], color="red", alpha=0.2, width=1)

            axs2.hlines(y=1, xmin=0, xmax=len(data_bar_shutdown.values()), color="gray", linestyle=(0, (1, 1)), alpha=0.5)
            axs2.set_yticks([0, 1], ["0", "1"])
            axs2.legend(loc='lower right', fontsize='x-small')

            # Plot demand on second y-axis, add PNS and EPS
            axs[i].set_ylim(-1, 1)
            axs[i].plot(index, data_demand.values(), color="blue", alpha=0.3, label="Demand")
            axs[i].plot(index, data_vGenP.values(), color="black", alpha=0.3, label="Prod.")

            axs[i].bar(index, data_vPNS.values(), color="orange", alpha=0.3, label="PNS", bottom=data_vGenP.values())
            axs[i].bar(index, data_vEPS.values(), color="purple", alpha=0.3, label="EPS", bottom=data_demand.values())
            axs[i].legend(loc='upper right', fontsize='x-small')

            axs[i].hlines(y=0, xmin=0, xmax=len(data_bar_shutdown.values()), color="gray", linestyle=(0, (1, 1)), alpha=0.5)
            axs[i].set_ylabel("Generation / Demand", color="black")
            axs[i].set_yticks([0, 0.5, 1], ["0.0", "0.5", "1.0"])

            # Set ticks and vertical lines
            index_labels = []
            index_positions = []
            axvline_thick_positions = []
            axvline_thin_positions = []
            for x in index:
                if x == index[0]:
                    index_labels.append(x + start_hour - 1)
                    index_positions.append(x)
                    if (x + start_hour - 2) % 24 == 0:
                        axvline_thick_positions.append(x)
                    else:
                        axvline_thin_positions.append(x)
                elif x == index[-1]:
                    index_labels.append(x + start_hour - 1)
                    index_positions.append(x)
                    if (x + start_hour - 2) % 24 == 0:
                        axvline_thick_positions.append(x)
                    else:
                        axvline_thin_positions.append(x)
                elif (x + start_hour - 2) % 24 == 0:
                    axvline_thick_positions.append(x)
                    if abs(x - index[0]) > 2 and abs(x - index[-1]) > 2:
                        index_labels.append(x + start_hour - 1)
                        index_positions.append(x)

            axs[i].set_xticks(index_positions)
            axs[i].set_xticklabels(index_labels)
            for x in axvline_thick_positions:
                axs[i].axvline(x=x, color="gray", linestyle="--", alpha=0.5)
            for x in axvline_thin_positions:
                axs[i].axvline(x=x, color="gray", linestyle="-", alpha=0.2)

    plt.tight_layout()

    # Save plot using a naming scheme matching the sqlite files
    base = os.path.splitext(sqlite_files[0])[0]  # e.g. "MK-datadata_markov-NoEnf"
    label_suffix = f"-{case_labels[0].strip().replace('.', '').replace(' ', '')}"
    plot_prefix = base[:-len(label_suffix)] if base.endswith(label_suffix) else base
    plot_file = f"{plot_prefix}-unit_commitment.png"
    plt.savefig(plot_file)
    printer.information(f"Saved unit commitment plot to '{plot_file}'")

    plt.show()


def _add_push_markov_constraints(lego: LEGO, thermalGeneratorRelaxed: dict):
    """Add ePushMarkov variables and constraints to a LEGO model (Markov-Strict only).

    These constraints force vStartup and vShutdown to be either 0 or the maximum they
    can be due to MinUp/DownTime, ensuring correct push behavior across representative
    period boundaries.
    """
    model = lego.model
    transition_matrix = lego.cs.rpTransitionMatrixRelativeFrom

    # Variables
    model.vU0 = pyo.Var(model.rp, model.k, model.thermalGenerators, domain=pyo.Binary, doc="Binary variable to indicate that vStartup is 0")
    model.vUX = pyo.Var(model.rp, model.k, model.thermalGenerators, domain=pyo.Binary, doc="Binary variable to indicate that vStartup is X (the maximum it can be due to MinDownTime)")
    model.vD0 = pyo.Var(model.rp, model.k, model.thermalGenerators, domain=pyo.Binary, doc="Binary variable to indicate that vShutdown is 0")
    model.vDY = pyo.Var(model.rp, model.k, model.thermalGenerators, domain=pyo.Binary, doc="Binary variable to indicate that vShutdown is Y (the maximum it can be due to MinUpTime)")

    model.pushMarkovCounter = pyo.Set(initialize=range(1, 11 + 1))
    model.ePushMarkov = pyo.Constraint(model.rp, model.k, model.thermalGenerators, model.pushMarkovCounter,
                                       doc="Constraints to force vStartup and vShutdown to be either 0 or the maximum it can be due to MinUp/DownTime")

    for t in model.thermalGenerators:
        if model.pMinDownTime[t] == 1 and model.pMinUpTime[t] == 1:
            continue
        is_relaxed = thermalGeneratorRelaxed.get(t, False)
        for k in model.k:
            if model.k.ord(k) > max(model.pMinDownTime[t], model.pMinUpTime[t]):
                break
            for rp in model.rp:
                if model.k.ord(k) == 1:
                    prev_commit = markov_summand(model.rp, rp, False, model.k.prevw(k), model.vCommit, transition_matrix, t)
                else:
                    prev_commit = model.vCommit[rp, model.k.prev(k), t]

                X = 1 - prev_commit - markov_sum(model.rp, rp, model.k, model.k.ord(k) - model.pMinDownTime[t] + 1, model.k.ord(k), model.vShutdown, transition_matrix, t)
                model.ePushMarkov[rp, k, t, 1] = (model.vStartup[rp, k, t] <= 1 - model.vU0[rp, k, t])
                model.ePushMarkov[rp, k, t, 2] = (model.vStartup[rp, k, t] <= X + (1 - model.vUX[rp, k, t]))
                model.ePushMarkov[rp, k, t, 3] = (model.vStartup[rp, k, t] >= X - (1 - model.vUX[rp, k, t]))
                model.ePushMarkov[rp, k, t, 4] = (model.vU0[rp, k, t] <= model.vDY[rp, k, t] + (1 - prev_commit))
                model.ePushMarkov[rp, k, t, 5] = (model.vUX[rp, k, t] <= model.vD0[rp, k, t])

                Y = prev_commit - markov_sum(model.rp, rp, model.k, model.k.ord(k) - model.pMinUpTime[t] + 1, model.k.ord(k), model.vStartup, transition_matrix, t)
                model.ePushMarkov[rp, k, t, 6] = (model.vShutdown[rp, k, t] <= 1 - model.vD0[rp, k, t])
                model.ePushMarkov[rp, k, t, 7] = (model.vShutdown[rp, k, t] <= Y + (1 - model.vDY[rp, k, t]))
                model.ePushMarkov[rp, k, t, 8] = (model.vShutdown[rp, k, t] >= Y - (1 - model.vDY[rp, k, t]))
                model.ePushMarkov[rp, k, t, 9] = (model.vD0[rp, k, t] <= model.vUX[rp, k, t] + prev_commit)
                model.ePushMarkov[rp, k, t, 10] = (model.vDY[rp, k, t] <= model.vU0[rp, k, t])
                model.ePushMarkov[rp, k, t, 11] = (1 <= model.vU0[rp, k, t] / 2 + model.vUX[rp, k, t] + model.vD0[rp, k, t] / 2 + model.vDY[rp, k, t])

                # Deactivate constraints for relaxed generators
                if is_relaxed:
                    for i in model.pushMarkovCounter:
                        model.ePushMarkov[rp, k, t, i].deactivate()


def execute_case_studies(case_study_path: str, no_sqlite: bool = False,
                         calculate_regret: bool = False, relax_percentage: float = 0, skip_truth: bool = False,
                         enable_strict_markov: bool = False, save_mps: bool = False, invest_regret: bool = False) -> typing.Tuple[typing.List[str], typing.List[str]]:
    ########################################################################################################################
    # Data input from case study
    ########################################################################################################################

    # Load case study from Excels
    printer.information(f"Loading case study from '{case_study_path}'")
    start_time = time.time()
    cs = CaseStudy(case_study_path, clip_method="none", clip_value=0)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    # Create varied case studies
    start_time = time.time()
    printer.information(f"Creating varied case studies")
    cs_notEnforced = cs.copy()
    cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "notEnforced"
    cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "notEnforced"
    cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingIntraDayStorage"] = "notEnforced"

    cs_cyclic = cs_notEnforced.copy()
    cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "cyclic"
    cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "cyclic"
    cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingIntraDayStorage"] = "cyclic"

    cs_markov = cs_notEnforced.copy()
    cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "markov"
    cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "markov"
    cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingIntraDayStorage"] = "markov"

    if enable_strict_markov:
        cs_markov_strict = cs_notEnforced.copy()
        cs_markov_strict.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "markov"
        cs_markov_strict.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "markov"
        cs_markov_strict.dPower_Parameters["pReprPeriodEdgeHandlingIntraDayStorage"] = "markov"
    printer.information(f"Creating varied case studies took {time.time() - start_time:.2f} seconds")

    # Create "truth" case study for comparison
    if skip_truth:
        printer.information(f"Skipping truth case study as requested")
    else:
        start_time = time.time()
        printer.information(f"Creating truth case study (full-hourly)")
        cs_truth = cs.to_full_hourly_model(inplace=False)  # Create a full hourly model (which copies from notEnforced)
        printer.information(f"Creating truth case study (full-hourly) took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    printer.information(f"Building the LEGO models for adjustments")  # Note this is actually faster (1.5-2x) than copying already built models to re-use them
    lego_models = {"NoEnf.": LEGO(cs_notEnforced), "Cyclic": LEGO(cs_cyclic), "Markov": LEGO(cs_markov)}
    if enable_strict_markov:
        lego_models["Markov-Strict"] = LEGO(cs_markov_strict)
    if not skip_truth:
        lego_models["Truth "] = LEGO(cs_truth)
    for name, lego in lego_models.items():
        _, build_time = lego.build_model()
        printer.information(f"Building model for case study '{name}' took {build_time:.2f} seconds")
    printer.information(f"Building the LEGO models took {time.time() - start_time:.2f} seconds overall")

    if save_mps:
        for case_name, lego in lego_models.items():
            mps_file = f"{case_name.replace('.', '')}.mps"
            printer.information(f"Saving MPS file for case study '{case_name}' to '{mps_file}'")
            lego.model.write(mps_file)

    thermalGeneratorRelaxed = {}
    if relax_percentage == 0:
        printer.information(f"Not relaxing any unit commitment variables, all thermal generators stay binary")
        count_relaxed = 0
    else:
        thermalGenerators = cs.dPower_ThermalGen.copy()
        start_time = time.time()
        printer.information(f"Relaxing {relax_percentage * 100:.1f}% of unit commitment variables for thermal generators")
        count_relaxed = math.ceil(len(thermalGenerators.index) * relax_percentage)

        printer.information(f"Relaxing {count_relaxed} thermal generator(s), keeping {len(thermalGenerators.index) - count_relaxed} binary")
        thermalGenerators["MinUpDownTime-Sum"] = thermalGenerators["MinUpTime"] + thermalGenerators["MinDownTime"]
        thermalGenerators.sort_values(by=["MinUpDownTime-Sum"], inplace=True)

        thermalGeneratorRelaxed = {}
        for i, t in enumerate(thermalGenerators.index):
            thermalGeneratorRelaxed[t] = True if i < count_relaxed else False

        printer.information(f"Relaxing {count_relaxed} thermal generators: {[g for g in thermalGenerators.index if thermalGeneratorRelaxed[g]]}")
        for case_name, lego in lego_models.items():
            # Relax unit commitment variables for selected generators
            for g in lego.model.thermalGenerators:
                if thermalGeneratorRelaxed[g]:
                    for rp in lego.model.rp:
                        for k in lego.model.k:
                            lego.model.vCommit[rp, k, g].domain = pyo.PercentFraction
                            lego.model.vStartup[rp, k, g].domain = pyo.PercentFraction
                            lego.model.vShutdown[rp, k, g].domain = pyo.PercentFraction
        printer.information(f"Relaxing {count_relaxed} thermal generators took {time.time() - start_time:.2f} seconds")

    if enable_strict_markov:
        _add_push_markov_constraints(lego_models["Markov-Strict"], thermalGeneratorRelaxed)

    # Build identifier parts for sqlite filenames (similar to TR/ID naming convention)
    identifier_parts = [f"data{case_study_path.rstrip('/').replace('/', '_').replace(' ', '')}"]
    if count_relaxed > 0:
        identifier_parts.append(f"relaxed{count_relaxed}")
    identifier = "-".join(identifier_parts)

    sqlite_files, sqlite_labels = execute_case_study(lego_models, identifier, no_sqlite, calculate_regret, skip_truth, invest_regret)

    return sqlite_files, sqlite_labels


def execute_case_study(lego_models: typing.Dict[str, LEGO], case_name: str, no_sqlite: bool, calculate_regret: bool, skip_truth: bool, invest_regret: bool = False) -> typing.Tuple[typing.List[str], typing.List[str]]:
    ########################################################################################################################
    # Evaluation
    ########################################################################################################################
    results = []
    sqlite_files = []
    sqlite_labels = []
    truth_objective = None

    if not skip_truth:
        truth_lego = lego_models["Truth "]

    for edgeHandlingType, lego in lego_models.items():
        printer.information(f"\n\n{'=' * 60}\n{edgeHandlingType}\n{'=' * 60}")
        model = lego.model

        # Solve model
        optimizer = pyo.SolverFactory('gurobi_persistent')
        optimizer.set_instance(model)
        start_time = time.time()
        result = optimizer.solve(tee=True)
        objective_value = pyo.value(model.objective) if result.solver.termination_condition == pyo.TerminationCondition.optimal else -1
        timing_solving = time.time() - start_time
        work_time = optimizer._solver_model.Work
        printer.information(f"Solving model took {timing_solving:.2f} seconds ({work_time:.2f} work units)")

        if edgeHandlingType == "Truth " and result.solver.termination_condition == pyo.TerminationCondition.optimal:
            truth_objective = objective_value

        file_prefix = f"MK-{case_name}-{edgeHandlingType.strip().replace('.', '').replace(' ', '')}"
        write_results(lego.model, file_prefix, no_sqlite)

        match result.solver.termination_condition:
            case pyo.TerminationCondition.optimal:
                printer.success("Optimal solution found")
            case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                printer.error(f"Model is {result.solver.termination_condition}, logging infeasible constraints:")
                log_infeasible_constraints(model)
            case _:
                printer.warning("Solver terminated with condition:", result.solver.termination_condition)

        # Count binary variables within all variables
        variables = list(model.component_objects(pyo.Var))
        counter_binaries = 0
        for v in variables:
            indices = [i for i in v]
            for i in indices:
                if v[i].domain == pyo.Binary:
                    counter_binaries += 1

        if not no_sqlite:
            sqlite_files.append(f"{file_prefix}.sqlite")
            sqlite_labels.append(edgeHandlingType)

        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            if calculate_regret and edgeHandlingType != "Truth " and not skip_truth:
                regret_lego = truth_lego.copy()

                add_UnitCommitmentSlack_And_FixVariables(regret_lego, model, lego.cs.dPower_Hindex, lego.cs.dPower_ThermalGen, lego.cs.dPower_Parameters["pENSCost"])

                # Re-solve the model
                printer.information("Re-solving model with fixed variables for regret calculation")
                regret_result, regret_timing_solving, regret_objective_value = regret_lego.solve_model(already_solved_ok=True)
                printer.information(f"Solving regret model took {regret_timing_solving:.2f} seconds")

                write_results(regret_lego.model, f"{file_prefix}-regret", no_sqlite)

                match regret_result.solver.termination_condition:
                    case pyo.TerminationCondition.optimal:
                        printer.success("Optimal solution found")
                    case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                        printer.error(f"Model is {regret_result.solver.termination_condition}, logging infeasible constraints:")
                        log_infeasible_constraints(regret_lego.model)
                    case _:
                        printer.warning("Solver terminated with condition:", regret_result.solver.termination_condition)

            if invest_regret and edgeHandlingType != "Truth " and not skip_truth:
                printer.information(f"Calculating invest-regret for '{edgeHandlingType}': fixing vGenInvest into truth model")
                invest_regret_lego = truth_lego.copy()

                # Fix vGenInvest to the values from the edge-handling model
                for g in invest_regret_lego.model.g:
                    invest_regret_lego.model.vGenInvest[g].value = model.vGenInvest[g].value
                    invest_regret_lego.model.vGenInvest[g].fixed = True

                # Re-solve the truth model with fixed investments
                printer.information("Re-solving truth model with fixed vGenInvest for invest-regret calculation")
                invest_regret_result, invest_regret_timing, invest_regret_objective = invest_regret_lego.solve_model(already_solved_ok=True)
                printer.information(f"Solving invest-regret model took {invest_regret_timing:.2f} seconds")

                write_results(invest_regret_lego.model, f"{file_prefix}-invest-regret", no_sqlite)

                match invest_regret_result.solver.termination_condition:
                    case pyo.TerminationCondition.optimal:
                        printer.success(f"Optimal invest-regret solution: {invest_regret_objective:.4f}")
                    case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
                        printer.error(f"Invest-regret model is {invest_regret_result.solver.termination_condition}, logging infeasible constraints:")
                        log_infeasible_constraints(invest_regret_lego.model)
                    case _:
                        printer.warning("Invest-regret solver terminated with condition:", invest_regret_result.solver.termination_condition)

        entry = {
            "Case": f"{case_name}-{edgeHandlingType}",
            "Objective": objective_value if result.solver.termination_condition == pyo.TerminationCondition.optimal else -1,
            "Objective Regret": -1 if not calculate_regret else (regret_objective_value - getUnitCommitmentSlackCost(regret_lego, lego.cs.dPower_ThermalGen, lego.cs.dPower_Parameters["pENSCost"]) if regret_result.solver.termination_condition == pyo.TerminationCondition.optimal and edgeHandlingType != "Truth " else -1),
            "Correction Cost": -1 if not calculate_regret else (getUnitCommitmentSlackCost(regret_lego, lego.cs.dPower_ThermalGen, lego.cs.dPower_Parameters["pENSCost"]) if edgeHandlingType != "Truth " else -1),
            "Solution": result.solver.termination_condition,
            # "Build Time": timing_building,
            "Solve Time": timing_solving,
            "Work Time": work_time,
            "# Variables Overall": model.nvariables(),
            "# Binary Variables": counter_binaries,
            "# Constraints": model.nconstraints(),
            "PNS regr.": -1 if not calculate_regret else (sum(regret_lego.model.vPNS[rp, k, i].value if regret_lego.model.vPNS[rp, k, i].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for i in regret_lego.model.i) if edgeHandlingType != "Truth " else -1),
            "EPS regr.": -1 if not calculate_regret else (sum(regret_lego.model.vEPS[rp, k, i].value if regret_lego.model.vEPS[rp, k, i].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for i in regret_lego.model.i) if edgeHandlingType != "Truth " else -1),
            "Commit Correction +": -1 if not calculate_regret else (sum(regret_lego.model.vCommitCorrectHigher[rp, k, t].value if regret_lego.model.vCommitCorrectHigher[rp, k, t].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for t in regret_lego.model.thermalGenerators) if edgeHandlingType != "Truth " else -1),
            "Commit Correction -": -1 if not calculate_regret else (sum(regret_lego.model.vCommitCorrectLower[rp, k, t].value if regret_lego.model.vCommitCorrectLower[rp, k, t].value is not None else 0 for rp in regret_lego.model.rp for k in regret_lego.model.k for t in regret_lego.model.thermalGenerators) if edgeHandlingType != "Truth " else -1),
            "vGenP": sum(model.vGenP[rp, k, g].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vGenP[rp, k, g].value is not None else 0 for rp in model.rp for k in model.k for g in model.g),
            "vCommit": sum(model.vCommit[rp, k, g].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vCommit[rp, k, g].value is not None else 0 for rp in model.rp for k in model.k for g in model.thermalGenerators),
            "vStartup": sum(model.vStartup[rp, k, g].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vStartup[rp, k, g].value is not None else 0 for rp in model.rp for k in model.k for g in model.thermalGenerators),
            "vShutdown": sum(model.vShutdown[rp, k, g].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vShutdown[rp, k, g].value is not None else 0 for rp in model.rp for k in model.k for g in model.thermalGenerators),
            "vPNS": sum(model.vPNS[rp, k, i].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vPNS[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i),
            "vEPS": sum(model.vEPS[rp, k, i].value * model.pWeight_rp[rp] * model.pWeight_k[k] if model.vEPS[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i),
            "vGenInvest": sum(model.vGenInvest[g].value if model.vGenInvest[g].value is not None else 0 for g in model.g),
            "Invest Regret Obj.": -1 if not invest_regret or edgeHandlingType == "Truth " else (invest_regret_objective if invest_regret_result.solver.termination_condition == pyo.TerminationCondition.optimal else -1),
            "Invest Regret": -1 if not invest_regret or edgeHandlingType == "Truth " else ((invest_regret_objective - truth_objective) if truth_objective is not None and invest_regret_result.solver.termination_condition == pyo.TerminationCondition.optimal else -1),
            "model": model
        }
        ddict = defaultdict(int)
        for g, tec in model.gtec:
            ddict[f"vGenInvest[{tec}]"] += model.vGenInvest[g].value if model.vGenInvest[g].value is not None else 0
        for k, v in ddict.items():
            entry[k] = v
        results.append(entry)

        # Write entry to solutions logfile
        log_file = printer.get_logfile().replace(".log", "-solutions.csv")
        if os.path.exists(log_file):
            with open(log_file, "a") as f:
                f.write(",".join([f"{v}" for v in entry.values()]) + "\n")
        else:
            with open(log_file, "w") as f:
                f.write(",".join(entry.keys()) + "\n")
                f.write(",".join([f"{v}" for v in entry.values()]) + "\n")

    values = ["Case", "Objective", "Solve Time", "vGenP", "vCommit", "vStartup", "vShutdown", "vPNS", "vEPS", "Objective Regret", "Invest Regret"]
    table = []
    for v in values:
        column = [v]
        for result in results:
            value = result[v]
            if isinstance(value, float):
                value = f"{value:.2f}"
            elif isinstance(value, int):
                value = f"{value:d}"
            else:
                value = f"{value}"
            column.append(value)
        table.append(column)

    for i in range(len(table[0])):
        printer.information(" | ".join(f"{table[j][i]:{">" if i != 0 else ""}{max(len(table[j][i2]) for i2 in range(len(table[j])))}}" for j in range(len(table))))

    return sqlite_files, sqlite_labels


def copy_files_non_recursive(src_folder: str, dst_folder: str):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for item in os.listdir(src_folder):
        s = os.path.join(src_folder, item)
        d = os.path.join(dst_folder, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)


def main(caseStudyFolder: str, plot: bool = False, debug: bool = False, no_sqlite: bool = False, calculate_regret: bool = False,
         relax_percentage: float = 0.0, skip_truth: bool = False,
         clusters: int = 1, cluster_stepsize: int = 1, cluster_steps: int = 0,
         shorten_until_k: int | None = None, shift: int = 0, stretch_demand: float = 1,
         reuse_inputfiles: bool = False, enable_strict_markov: bool = False, save_mps: bool = False, invest_regret: bool = False):
    ew = ExcelWriter()

    for folder in caseStudyFolder.split(","):
        try:
            if not folder.endswith("/"):
                folder += "/"
            folder_name = os.path.basename(os.path.normpath(folder))

            if shorten_until_k is not None:
                printer.information(f"Shortening case study to k<={shorten_until_k}")
                new_folder = folder + f"untilK{shorten_until_k}/"
                if reuse_inputfiles and os.path.exists(new_folder):
                    printer.information(f"Reusing already shortened case study in '{new_folder}'")
                    folder = new_folder
                else:
                    copy_files_non_recursive(folder, new_folder)  # Copy original data to new folder
                    folder = new_folder
                    printer.information(f"Copied original case study to '{folder}'")

                    cs = CaseStudy(folder, do_not_scale_units=True)
                    printer.information(f"Case study loaded, now shortening")
                    cs = cs.filter_timesteps("k0001", f"k{shorten_until_k:04}")
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    printer.information(f"Shortened, now writing to '{folder}'")
                    ew.write_caseStudy(cs, folder)
                    printer.information(f"Saved shortened case study to '{folder}'")

            if shift != 0:
                printer.information(f"Shifting case study by {shift} hours")
                new_folder = folder + f"shift{shift}/"
                if reuse_inputfiles and os.path.exists(new_folder):
                    printer.information(f"Reusing already shifted case study in '{new_folder}'")
                    folder = new_folder
                else:
                    copy_files_non_recursive(folder, new_folder)  # Copy original data
                    folder = new_folder
                    printer.information(f"Copied original case study to '{folder}'")

                    cs = CaseStudy(folder, do_not_scale_units=True)
                    printer.information(f"Case study loaded, now shifting")
                    cs = cs.shift_ks(shift)
                    printer.information(f"Shifted by {shift}")
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    ew.write_caseStudy(cs, folder)
                    printer.information(f"Wrote shifted case study to '{folder}'")

            if stretch_demand != 1.0:
                printer.information(f"Stretching demand by factor {stretch_demand}")
                new_folder = folder + f"stretchDemand{stretch_demand:.2f}/"
                if reuse_inputfiles and os.path.exists(new_folder):
                    printer.information(f"Reusing already demand-stretched case study in '{new_folder}'")
                    folder = new_folder
                else:
                    copy_files_non_recursive(folder, new_folder)  # Copy original data
                    folder = new_folder
                    printer.information(f"Copied original case study to '{folder}'")

                    cs = CaseStudy(folder, do_not_scale_units=True)
                    printer.information(f"Case study loaded, now stretching demand for each bus around center")
                    center = cs.dPower_Demand.groupby("i")["value"].mean()
                    scaler = 1 + (stretch_demand - 1) / 2
                    for rp, k, i in cs.dPower_Demand.index:
                        cs.dPower_Demand.at[(rp, k, i), "value"] = center[i] + (cs.dPower_Demand.at[(rp, k, i), "value"] - center[i]) * scaler

                    # Fail if any of the values is negative
                    if (cs.dPower_Demand["value"] < 0).any():
                        to_clip = cs.dPower_Demand[cs.dPower_Demand['value'] < 0]
                        printer.warning(f"Stretching demand by factor {stretch_demand} leads to negative demand values, clipping {to_clip.shape[0]} values for {len(to_clip.index.get_level_values('i').unique())} nodes to 0")
                        printer.warning(f"Clipping for nodes: {", ".join([f"{i} for {to_clip[to_clip.index.get_level_values("i") == i].shape[0]} values" for i in to_clip.index.get_level_values('i').unique().tolist()])}")
                        cs.dPower_Demand["value"] = cs.dPower_Demand["value"].clip(lower=0)

                    printer.information(f"Stretched demand by factor {stretch_demand}")
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    ew.write_caseStudy(cs, folder)
                    printer.information(f"Wrote demand-stretched case study to '{folder}'")

            clusters = list(range(clusters, clusters + cluster_steps * cluster_stepsize + 1, cluster_stepsize))
            for cluster in clusters:
                cluster_folder = folder
                if cluster > 1:
                    cluster_folder = cluster_folder + f"{cluster} clusters/"
                    if reuse_inputfiles and os.path.exists(cluster_folder):
                        printer.information(f"Reusing already clustered case study in '{cluster_folder}'")
                    else:
                        copy_files_non_recursive(folder, cluster_folder)  # Copy original data to new folder

                        cs = CaseStudy(cluster_folder, do_not_scale_units=True)
                        cs_clustered = Utilities.apply_kmedoids_aggregation(cs, cluster)
                        ew.write_caseStudy(cs_clustered, cluster_folder)

                    printer.set_logfile(f"markov-{folder_name}-{cluster}clusters.log")
                else:
                    printer.set_logfile(f"markov-{folder_name}.log")

                printer.information(f"Loading case study from '{cluster_folder}'")
                printer.information(f"Logfile: '{printer.get_logfile()}'")

                sqlite_files, case_labels = execute_case_studies(cluster_folder, no_sqlite, calculate_regret, relax_percentage, skip_truth, enable_strict_markov, save_mps, invest_regret)

                if plot and sqlite_files:
                    printer.information(f"Plotting unit commitment from sqlite files: {sqlite_files}")
                    plot_unit_commitment(sqlite_files, case_labels, cluster_folder, 6 * 24, 1)
        except Exception as e:
            printer.error(f"Exception while executing case study '{folder}': {e}")
            if debug:
                raise e
            else:
                printer.console.print_exception()
                printer.error(f"Continuing with next case study")

    printer.success("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare edge-handling for given case-study", formatter_class=RichHelpFormatter)
    parser.add_argument("caseStudyFolder", type=str, help="Path to folder containing data for LEGO model. Can be a comma-separated list of multiple folders (executed after each other)")
    parser.add_argument("--plot", action="store_true", help="Plot unit commitment results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode where exceptions are passed on")
    parser.add_argument("--no-sqlite", action="store_true", help="Do not save results to SQLite database")
    parser.add_argument("--calculate-regret", action="store_true", help="Calculate regret by re-solving the truth model with fixed unit commitment from the other models (can take a while)")
    parser.add_argument("--relax-percentage", type=float, default=0, help="Percentage of thermal-generators to be relaxed (default: 0 = no relaxation, all binary)")
    parser.add_argument("--skip-truth", action="store_true", help="Skip solving the truth model")
    parser.add_argument("--clusters", type=int, default=1, help="Number of clusters (default: 1, i.e., no clustering)")
    parser.add_argument("--cluster-stepsize", type=int, default=1, help="If in-/decreasing number of clusters should be used (default: 1, leave cluster-steps default to not use in-/decreasing number of clusters)")
    parser.add_argument("--cluster-steps", type=int, default=0, help="Number of steps for in-/decreasing number of clusters (default: 0, i.e., leave clusters as given)")
    parser.add_argument("--shorten-until-k", type=int, default=None, help="Shorten the case study to only consider k=1..N (for faster testing), e.g., 24 for one day, 168 for one week")
    parser.add_argument("--shift", type=int, default=0, help="Shift the time series by N hours (for testing purposes), e.g., 15 to shift by 15 hours")
    parser.add_argument("--stretch-demand", type=float, default=1.0, help="Stretch the demand by a factor (for testing purposes), e.g., 1.1 to increase max of demand by 5% and decrease min by 5%")
    parser.add_argument("--reuse-inputfiles", action="store_true", help="Reuse input files (e.g., after shortening) instead of copying them to a new folder")
    parser.add_argument("--enable-strict-markov", action="store_true", help="Also execute the strict Markov variant (with push constraints active)")
    parser.add_argument("--save-mps", action="store_true", help="Save MPS files for each case study")
    parser.add_argument("--invest-regret", action="store_true", help="Calculate invest-regret: fix vGenInvest from each edge-handling model into the truth model and compare objectives")
    args = parser.parse_args()

    kwargs = vars(args)

    main(**kwargs)
