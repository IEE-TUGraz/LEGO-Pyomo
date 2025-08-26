import argparse
import logging
import os
import time
import pandas as pd

import pyomo.environ as pyo
from pyomo.core import NameLabeler
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds
from rich_argparse import RichHelpFormatter

from InOutModule import SQLiteWriter, ExcelWriter
from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.LEGO import LEGO, ModelType
from InOutModule import TabSepReader

printer = Printer.getInstance()
# define a logger file
Printer.set_logfile(printer, "RINGS_log.txt",)

# Set up logging so that infeasible constraints are logged by pyomo
logger = logging.getLogger("pyomo")
logger.setLevel("INFO")

# Parse command line arguments and automatically check for correct usage
parser = argparse.ArgumentParser(description="Starts LEGO for given case study", formatter_class=RichHelpFormatter)


# Check if given string path is a directory
def directory_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise argparse.ArgumentTypeError(f"Directory path not valid: '{string}'")


parser.add_argument("caseStudyDirectory", type=directory_path, help="Path to folder containing data for LEGO model")
parser.add_argument("modelType", default=ModelType.DETERMINISTIC, type=lambda s: ModelType[s], choices=list(ModelType), nargs="?", help="ModelType of first model")
args = parser.parse_args()

# create a dataframe for the case studies with different time intervals
df_case_studies = pd.DataFrame(columns=["invervall","solvetime","objval","totalPVgen","totalCurtailment","totalImport", "totalDemand"])

# set values for the time intervals
df_case_studies["invervall"] = [1]

for index, row in df_case_studies.iterrows():
    # Load case study
    printer.information(f"Loading case study from '{args.caseStudyDirectory}'")
    start_time = time.time()
    # load settings for RINGS specific data
    rings_data_folder = os.path.join("data", "rings")
    rings_settings = TabSepReader.read_data_settings(os.path.join(rings_data_folder, "DataSettings.yaml"))

    rings_settings["aggregation"]["intervall"] = row["invervall"]

    # load dataframes for RINGS specific data
    df_VRES_profiles = TabSepReader.get_VRES_profiles(rings_data_folder, rings_settings)
    df_power_demand = TabSepReader.get_dPower_Demand(rings_data_folder, rings_settings)

    # get the length of the data
    num_elements = df_VRES_profiles.shape[0]
    printer.information(f"Number of elements in time: {num_elements}")

    # load helper data
    df_ImpExpLim = TabSepReader.create_imp_exp_data(num_elements)
    df_hindex = TabSepReader.create_consecutive_hindex(num_elements)
    df_kWeights = TabSepReader.create_kWeights(num_elements)

    # write the dataframes to a excel in a temp folder
    output_folder = os.path.join("temp", "rings")
    os.makedirs(output_folder, exist_ok=True)

    df_VRES_profiles.to_excel(os.path.join(output_folder, "VRES_profiles.xlsx"))
    df_power_demand.to_excel(os.path.join(output_folder, "power_demand.xlsx"))
    df_ImpExpLim.to_excel(os.path.join(output_folder, "ImpExp_limits.xlsx"))
    df_hindex.to_excel(os.path.join(output_folder, "H_index.xlsx"))
    df_kWeights.to_excel(os.path.join(output_folder, "k_weights.xlsx"))

    cs = CaseStudy(args.caseStudyDirectory, dPower_VRESProfiles=df_VRES_profiles, dPower_Demand=df_power_demand, dPower_ImpExpProfiles=df_ImpExpLim, dPower_Hindex=df_hindex, dPower_WeightsK=df_kWeights)
    #cs = cs.filter_timesteps("k3005", "k3005")

    lego = LEGO(cs)
    printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

    # Build LEGO model
    printer.information("Building LEGO model")
    model, timing = lego.build_model(model_type=args.modelType)
    printer.information(f"Building LEGO model took {timing:.2f} seconds")

    with open("temp/RINGS_model.txt", 'w') as output_file:
        model.pprint(output_file)

    # Solve LEGO model
    printer.information("Solving LEGO model")
    results, timing, objective_value = lego.solve_model(model_type=args.modelType)
    printer.information(f"Solving LEGO model took {timing:.2f} seconds")

    df_case_studies.at[index, "solvetime"] = timing # add also other times (model building)!!

    with open("temp/RINGS_model_solved.txt", 'w') as output_file:
        model.pprint(output_file)

    logger = logging.getLogger('pyomo.util.infeasible')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler('pyomo_infeasible.log', mode='w'))

    # Ensure there is a handler attached
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)

    match results.solver.termination_condition:
        case pyo.TerminationCondition.optimal:
            match args.modelType:
                case ModelType.DETERMINISTIC:
                    printer.success(f"Optimal solution: {pyo.value(model.objective):.4f}")
                case ModelType.EXTENSIVE_FORM:
                    printer.success(f"Optimal solution: {lego._extensive_form.get_objective_value():.4f}")
                case _:
                    printer.warning(f"Model type {args.modelType} not fully tested yet, no objective value reported.")
        case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
            printer.error(f"Model returned as {results.solver.termination_condition}, logging infeasible constraints:")
            log_infeasible_constraints(model, log_expression=True)
            log_infeasible_bounds(model)
        case _:
            printer.warning(f"Solver terminated with condition: {results.solver.termination_condition}")


    df_case_studies.at[index, "objval"] = pyo.value(model.objective)
    df_case_studies.at[index, "totalPVgen"] = sum(model.vGenP[rp, k, g].value for rp in model.rp for k in model.k for g in model.vresGenerators) * model.pWeight_k[model.k.first()]
    df_case_studies.at[index, "totalCurtailment"] = sum(model.vCurtailment[rp, k, g].value for rp in model.rp for k in model.k for g in model.vresGenerators) * model.pWeight_k[model.k.first()]
    df_case_studies.at[index, "totalImport"] = sum(model.vImpExp[rp, k, hub, i].value for rp in model.rp for k in model.k for hub in model.hubs for i in model.i if (hub, i) in model.hubConnections) * model.pWeight_k[model.k.first()]
    df_case_studies.at[index, "totalDemand"] = sum(model.pDemandP[rp, k, i] for rp in model.rp for k in model.k for i in model.i) * model.pWeight_k[model.k.first()]

    # write a warning when the value of PNS or of ENS is larger than eps
    total_PNS = sum(model.vPNS[rp, k, i].value for rp in model.rp for k in model.k for i in model.i)
    total_EPS = sum(model.vEPS[rp, k, i].value for rp in model.rp for k in model.k for i in model.i)

    eps = 1e-5
    if total_PNS > eps:
        printer.warning(f"Power not supplied value {total_PNS} exceeds threshold {eps}")

    if total_EPS > eps:
        printer.warning(f"Excess power supplied value {total_EPS} exceeds threshold {eps}")

    #SQLiteWriter.model_to_sqlite(model, "model.sqlite")
    ExcelWriter.model_to_excel(model, os.path.join("data","rings_base_example","results","results_intv" + str(rings_settings["aggregation"]["intervall"]) + ".xlsx"))
    #model.write("model.mps", io_options={'labeler': NameLabeler()})

df_case_studies.to_excel("data/rings_base_example/results/case_studies.xlsx")