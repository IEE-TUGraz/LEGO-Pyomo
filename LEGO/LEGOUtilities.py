import functools
import typing
import zipfile
from pathlib import Path
from typing import Union, List

import matplotlib.pyplot as plt
import pandas as pd
import py7zr
import pyomo.environ as pyo

from InOutModule import ExcelReader
from LEGO import LEGO


# Returns a list of elements from a set, starting from 'first_index' and ending at 'last_index' (both inclusive) without wrapping around
def set_range_non_cyclic(set: pyo.Set, first_index: int, last_index: int):
    if not (1 <= first_index <= last_index <= len(set)):
        raise ValueError(f"Please select first and last index so that '1 < first_index <= last_index <= len(set)' holds (got: 1 < {first_index} <= {last_index} <= {len(set)})")

    current_index = set.at(first_index)
    result = [current_index]
    for i in range(first_index + 1, last_index + 1):  # Start from first_index + 1 since we already have the first element and go to last_index + 1 since range() is exclusive
        current_index = set.next(current_index)
        result.append(current_index)

    return result


# Returns a list of elements from a set, starting from 'first_index' and ending at 'last_index' (both inclusive) with wrapping around
def set_range_cyclic(set: pyo.Set, first_index: int, last_index: int):
    if first_index > len(set) or first_index < -len(set):
        raise ValueError(f"'first_index' must be <= len(set) and >= -len(set) (got {first_index} and {len(set)})")
    elif last_index < 1:
        raise ValueError(f"'last_index' must be greater than 1 (got {last_index})")

    while first_index < 1:
        first_index += len(set)  # Wrap around if first_index is negative (or zero)

    if first_index > last_index:
        last_index += len(set)  # Wrap around if last_index is smaller than first_index

    current_index = set.at(first_index)
    result = [current_index]
    for i in range(first_index + 1, last_index + 1):  # Start from first_index + 1 since we already have the first element and go to last_index + 1 since range() is exclusive
        current_index = set.nextw(current_index)
        result.append(current_index)

    return result


def markov_summand(rp_set: pyo.Set, rp_target: str, from_target_to_others: bool, k: str, relevant_variable: pyo.Var, transition_matrix: pd.DataFrame, *other_indices: str) -> pyo.Expression:
    """
    Calculate the summand of a variable using Markov-Chains for a given transition matrix
    :param pyo.Set rp_set: Set of representative periods (e.g., model.rp)
    :param str rp_target: Current representative period (e.g., "rp01")
    :param bool from_target_to_others: If True, the summand is calculated from the target rp to the others, otherwise from the others to the target
    :param str k: Current timestep (e.g., "k0001")
    :param pyo.Var relevant_variable: Variable to sum up (e.g., model.vCommit)
    :param pd.DataFrame transition_matrix: Transition matrix between representative periods
    :param str other_indices: Additional indices for the relevant_variable (will be used as 'relevant_variable[rp_current_value, k, *other_indices]')
    :return: Expression for the summand of the relevant_variable multiplied by corresponding transition probabilities
    """
    summand = 0
    safety_check = 0
    for rp in rp_set:  # Iterate over all representative periods
        i = rp_target if from_target_to_others else rp
        j = rp if from_target_to_others else rp_target
        if transition_matrix.at[i, j] > 0:  # Only consider transitions with a probability > 0
            summand += transition_matrix.at[i, j] * relevant_variable[rp, k, *other_indices]
            safety_check += transition_matrix.at[i, j]
    if safety_check != 1:
        raise ValueError(f"Transition matrix is not correctly defined - sum of transition probabilities for representative period {rp_target} is not 1 (but {safety_check})")
    return summand


def markov_sum(rp_set: pyo.Set, rp_target: str, k_set: pyo.Set, k_start_index: int, k_end_index: int, relevant_variable: pyo.Var, transition_matrix: pd.DataFrame, *other_indices: str) -> pyo.Expression:
    """
    Calculate the sum of a variable using Markov-Chains for a given transition matrix
    :param pyomo.environ.Set rp_set: Set of representative periods (e.g., model.rp)
    :param str rp_target: Current representative period (e.g., "rp01")
    :param pyomo.environ.Set k_set: Set of timesteps (e.g., model.k)
    :param int k_start_index: Start timestep index (e.g., "1")
    :param int k_end_index: End timestep index (e.g., "6")
    :param pyomo.environ.Var relevant_variable: Variable to sum up (e.g., model.vCommit)
    :param pd.DataFrame transition_matrix: Transition matrix between representative periods
    :keyword other_indices: Additional indices for the relevant_variable (will be used as 'relevant_variable[rp_current_value, k, *other_indices]')
    :return: Expression for the sum of the relevant_variable multiplied by corresponding transition probabilities
    """
    expression = 0
    for k in set_range_cyclic(k_set, k_start_index, k_end_index):
        if k_set.ord(k) > k_end_index:  # k is still beyond edge towards previous periods
            expression += markov_summand(rp_set, rp_target, False, k, relevant_variable, transition_matrix, *other_indices)
        else:
            expression += relevant_variable[rp_target, k, *other_indices]
    return expression


# Dictionary to store which functions have been executed for the given object
execution_safety_dict = {}


# Decorator to check that function has not been executed and add it to executionSafetyList
def safetyCheck_AddElementDefinitionsAndBounds(func):
    @functools.wraps(func)  # Preserve the original function's name
    def wrapper(*args, **kwargs):
        # Check that function has not already been executed and add it to dictionary
        execution_safety_list = execution_safety_dict[id(args[0])] = [] if id(args[0]) not in execution_safety_dict else execution_safety_dict[id(args[0])]

        fullFuncName = func.__module__ + '.' + func.__name__
        if fullFuncName not in execution_safety_list:
            execution_safety_list.append(fullFuncName)  # Add function to execution list
        else:
            raise RuntimeError(f"Function {fullFuncName} has already been executed, current execution log: {execution_safety_list}")

        # Store variable list before the call
        variables = set(args[0].component_objects(pyo.Var, active=True))

        # Call the function
        first_stage_variables, second_stage_variables = func(*args, **kwargs)

        # Make sure that all new variables are assigned either to first_stage or second_stage
        new_variables = set(args[0].component_objects(pyo.Var, active=True)) - variables

        for var in first_stage_variables:
            if var not in new_variables:
                raise ValueError(f"Variable {var} was assigned to first_stage but has not been added to the model")
            new_variables.remove(var)

        for var in second_stage_variables:
            if var not in new_variables:
                raise ValueError(f"Variable {var} was assigned to second_stage but has not been added to the model")
            new_variables.remove(var)

        if len(new_variables) > 0:
            raise ValueError(f"Some variables were added with function '{fullFuncName}' but are neither assigned to first_stage nor second_stage.\nPlease check: {[str(v) for v in new_variables]}")

        # Return first stage variables
        return first_stage_variables

    return wrapper


# Decorator to check that all required functions have been executed before executing the function
# Also checks that the function has not already been executed
# required_functions: List of function names that need to have been executed before this function (without the file path)
def safetyCheck_addConstraints(required_functions: list[typing.Callable]):
    def decorator(func):
        @functools.wraps(func)  # Preserve the original function's name
        def wrapper(*args, **kwargs):
            # Check if all required functions have been executed
            execution_safety_list = execution_safety_dict[id(args[0])] = [] if id(args[0]) not in execution_safety_dict else execution_safety_dict[id(args[0])]
            fileName = func.__module__
            fullFuncName = fileName + '.' + func.__name__
            required_functions_adapted = [fileName + '.' + func_name.__name__ for func_name in required_functions]

            # Check if all required functions have been executed
            missing_functions = []
            for func_name in required_functions_adapted:
                if func_name not in execution_safety_list:
                    missing_functions.append(func_name)

            if len(missing_functions) > 0:
                raise RuntimeError(f"Not all required functions for calling {fullFuncName} have been executed\n"
                                   f"Missing following function(s):\n"
                                   f"{missing_functions}\n"
                                   f"----------------------------------------\n"
                                   f"Full list of executed functions: \n"
                                   f"{execution_safety_list}")
            elif fullFuncName in execution_safety_list:
                raise RuntimeError(f"Function {fullFuncName} has already been executed, current execution log: {execution_safety_list}")
            else:
                execution_safety_list.append(fullFuncName)  # Add function to execution list

                # Call the function
                first_stage_objective = func(*args, **kwargs)

                return first_stage_objective

        return wrapper

    return decorator


def plot_unit_commitment(unit_commitment_result_file: str, case_study_folder: str, number_of_hours: int = 24 * 7, start_hour: int = 1):
    """
    Plot the unit commitment of a given output file
    :param unit_commitment_result_file: Path to Excel-File containing unit commitment results
    :param case_study_folder: Path to folder containing Power_Hindex file
    :param number_of_hours: Number of hours to plot (default: 24 * 7 = 168)
    :param start_hour: Start hour for the plot (default: 1)
    :return: Nothing (shows plot)
    """
    plt.rcParams['figure.dpi'] = 300  # Set resolution of the plot
    df = pd.read_excel(unit_commitment_result_file)
    df = df.set_index(["case", "rp", "k", "g"])

    # Get original mapping from Power_Hindex
    hindex = ExcelReader.get_dPower_Hindex(case_study_folder + "Power_Hindex.xlsx")
    hindex = hindex.reset_index()
    hindex["p_int"] = hindex["p"].str.extract(r'(\d+)').astype(int)  # Extract the integer part of the "p" column
    hindex["rp_int"] = hindex["rp"].str.extract(r'(\d+)').astype(int)  # Extract the integer part of the "rp" column
    hindex["k_int"] = hindex["k"].str.extract(r'(\d+)').astype(int)  # Extract the integer part of the "k" column

    # Filter the dataframe to only include the relevant hours
    hindex = hindex.loc[(hindex["p_int"] >= start_hour) & (hindex["p_int"] <= start_hour + number_of_hours - 1)]

    # Plot the data
    index = [i + 1 for i in range(len(hindex))]

    fig, axs = plt.subplots(len(df.index.get_level_values("case").unique()), len(df.index.get_level_values("g").unique()), figsize=(6 * len(df.index.get_level_values("g").unique()), 2 * len(df.index.get_level_values("case").unique())))
    for i, case in enumerate(df.index.get_level_values("case").unique()):
        for j, g in enumerate(df.index.get_level_values("g").unique()):

            data_plot = {}
            data_bar_startup = {}
            data_bar_shutdown = {}
            data_bar_min_uptime_height = {}
            data_bar_min_downtime_bottom = {}
            data_plot_demand = {}
            for counter, (_, row) in enumerate(hindex.iterrows()):
                counter += 1
                rp = row["rp"] if case != "Truth " and "regret" not in case else "rp01"
                k = row["k"] if case != "Truth " and "regret" not in case else row["p"].replace("h", "k")
                data_plot[counter] = df.loc[case, rp, k, g]["vCommit"]
                data_bar_startup[counter] = df.loc[case, rp, k, g]["vStartup"]
                data_bar_shutdown[counter] = df.loc[case, rp, k, g]["vShutdown"]
                data_plot_demand[counter] = df.loc[case, rp, k, g]["pDemandP"]

            for counter, (_, row) in enumerate(hindex.iterrows()):
                counter += 1
                data_bar_min_uptime_height[counter] = sum([data_bar_startup[a] for a in [counter - b for b in range(0, int(df.loc[case, rp, k, g]["pMinUpTime"] - 1)) if counter - b > 0]])
                data_bar_min_downtime_bottom[counter] = 1 - sum([data_bar_shutdown[a] for a in [counter - b for b in range(0, int(df.loc[case, rp, k, g]["pMinDownTime"] - 1)) if counter - b > 0]])

            axs[i, j].set_ylim(-0.05, 1.05)
            axs[i, j].plot(index, data_plot.values(), color="black", alpha=0.3)
            axs[i, j].bar(index, data_bar_startup.values(), color="green", alpha=0.5, bottom=[list(data_plot.values())[-1]] + list(data_plot.values())[:-1], width=1)
            axs[i, j].bar(index, data_bar_shutdown.values(), color="red", alpha=0.5, bottom=data_plot.values(), width=1)
            axs[i, j].bar(index, data_bar_min_uptime_height.values(), color="green", alpha=0.2, width=1)
            axs[i, j].bar(index, bottom=data_bar_min_downtime_bottom.values(), height=[1 - x for x in data_bar_min_downtime_bottom.values()], color="red", alpha=0.2, width=1)
            axs[i, j].set_title(f"{case} - {g}")

            # Plot demand on second y-axis
            axs2 = axs[i, j].twinx()
            axs2.plot(index, data_plot_demand.values(), color="blue", alpha=0.3)

            # Set ticks and vertical lines
            index_labels = []
            index_positions = []
            axvline_thick_positions = []
            axvline_thin_positions = []
            for x in index:
                if x == index[0]:  # First index
                    index_labels.append(x + start_hour - 1)
                    index_positions.append(x)
                    if (x + start_hour - 2) % 24 == 0:  # If it's the start of a new day, add a thick line, else a thin line
                        axvline_thick_positions.append(x)
                    else:
                        axvline_thin_positions.append(x)
                elif x == index[-1]:  # Last index
                    index_labels.append(x + start_hour - 1)
                    index_positions.append(x)
                    if (x + start_hour - 2) % 24 == 0:  # If it's the start of a new day, add a thick line, else a thin line
                        axvline_thick_positions.append(x)
                    else:
                        axvline_thin_positions.append(x)
                elif (x + start_hour - 2) % 24 == 0:
                    axvline_thick_positions.append(x)  # Every 24th index (i.e., every day)
                    if abs(x - index[0]) > 2 and abs(x - index[-1]) > 2:  # Every 24th index (i.e., every day), if distance to first and last is big enough (to not overlap)
                        index_labels.append(x + start_hour - 1)
                        index_positions.append(x)

            axs[i, j].set_xticks(index_positions)
            axs[i, j].set_xticklabels(index_labels)
            for x in axvline_thick_positions:
                axs[i, j].axvline(x=x, color="gray", linestyle="--", alpha=0.5)
            for x in axvline_thin_positions:
                axs[i, j].axvline(x=x, color="gray", linestyle="-", alpha=0.2)

    plt.tight_layout()
    plt.show()


def add_UnitCommitmentSlack_And_FixVariables(regret_lego: LEGO, original_model: pyo.Model, hindex_df: pd.DataFrame, thermalGen_df: pd.DataFrame, PNS_cost: float):
    """
    Adds unit commitment slack variables and constraints to the provided model. This function modifies
    the provided `regret_lego` model by introducing slack variables for startup and shutdown
    correction, adds corresponding constraints and adjusts the objective function to include penalties
    for these deviations.

    :param regret_lego: Model that will be adjusted with new variables, constraints, and adapted
        objective to include unit commitment slack corrections.
    :param original_model: The reference model containing the original startup and shutdown variable
        values, which are being corrected by the slack adjustments integrated into `regret_lego`.
    :param hindex_df: DataFrame containing the mapping of hourly data indices (`h`) to problem indexes (`k`
        and `rp`) for aligning model input data.
    :param thermalGen_df: DataFrame containing attributes related to thermal generators, especially maximum
        production (`MaxProd`).
    :param PNS_cost: A float representing the cost multiplier for power not served (PNS), used as part of the
        penalty in the objective function.
    :return: The function does not return any value. Changes are made directly to the `regret_lego` model.
    """

    # Define helper functions to convert between hourly and problem indices
    def hourly_k_to_rp(k: str, hindex_df: pd.DataFrame) -> str:
        return hindex_df.loc[k.replace("k", "h")]["rp"]

    def hourly_k_to_k(k: str, hindex_df: pd.DataFrame) -> str:
        return hindex_df.loc[k.replace("k", "h")]["k"]

    # Add variables for commit correction
    regret_lego.model.vCommitCorrectHigher = pyo.Var(regret_lego.model.rp, regret_lego.model.k, regret_lego.model.thermalGenerators, doc='Commit correction towards 1 for thermal generator g', domain=pyo.PercentFraction)
    regret_lego.model.vCommitCorrectLower = pyo.Var(regret_lego.model.rp, regret_lego.model.k, regret_lego.model.thermalGenerators, doc='Commit correction towards 0 for thermal generator g', domain=pyo.PercentFraction)

    # Get and adjust Power_hindex from case study
    hindex_df = hindex_df.copy()
    hindex_df = hindex_df.reset_index()
    hindex_df = hindex_df.set_index("p")

    # Add correction constraint which also fixes the commit variable
    regret_lego.model.eCommitCorrect = pyo.Constraint(regret_lego.model.rp, regret_lego.model.k, regret_lego.model.thermalGenerators, doc='Commit correction for thermal generator g',
                                                      rule=lambda m, rp, k, t: m.vCommit[rp, k, t] == (original_model.vCommit[hourly_k_to_rp(k, hindex_df), hourly_k_to_k(k, hindex_df), t].value if not original_model.vCommit[hourly_k_to_rp(k, hindex_df), hourly_k_to_k(k, hindex_df), t].stale else 0) + m.vCommitCorrectHigher[rp, k, t] - m.vCommitCorrectLower[rp, k, t])

    # Add penalty for correcting the commit variable
    regret_lego.model.objective += sum(sum(sum((regret_lego.model.vCommitCorrectHigher[rp, k, t] + regret_lego.model.vCommitCorrectLower[rp, k, t]) * regret_lego.model.pWeight_rp[rp] for rp in regret_lego.model.rp) * regret_lego.model.pWeight_k[k] for k in regret_lego.model.k) * thermalGen_df.loc[t]["MaxProd"] * PNS_cost for t in regret_lego.model.thermalGenerators)


def getUnitCommitmentSlackCost(lego: LEGO, thermalGen_df: pd.DataFrame, PNS_cost: float) -> float:
    """
    Calculate the unit commitment slack cost based on corrections for startup and
    shutdown values in the model.

    :param lego: LEGO model object containing the Pyomo model and associated
        parameters.
    :param thermalGen_df: DataFrame containing thermal generator data,
        including "MaxProd" values representing maximum production levels.
    :param PNS_cost: Cost associated with Power Not Supplied.
    :return: The total unit commitment slack cost weighted by the given factors.
    """
    return sum(sum(sum((pyo.value(lego.model.vCommitCorrectHigher[rp, k, t]) + pyo.value(lego.model.vCommitCorrectLower[rp, k, t])) * lego.model.pWeight_rp[rp] for rp in lego.model.rp) * lego.model.pWeight_k[k] for k in lego.model.k) * thermalGen_df.loc[t]["MaxProd"] * PNS_cost for t in lego.model.thermalGenerators)


def decompress_mps_file(mps_file_path: Union[str, Path]) -> str:
    """
    Decompresses a .mps.7z file to extract the corresponding .mps file.

    Args:
        mps_file_path: Path to the .mps file (the function will look for .mps.7z)

    Returns:
        str: Path to the extracted .mps file

    Raises:
        FileNotFoundError: If the .mps.7z file doesn't exist
        OSError: If decompression fails
    """
    mps_path = Path(mps_file_path).resolve()
    seven_zip_path = mps_path.with_suffix(mps_path.suffix + ".7z")

    # If .mps file already exists, assume it's already decompressed
    if mps_path.exists():
        return str(mps_path)

    # Check 7z file exists
    if not seven_zip_path.exists():
        raise FileNotFoundError(f"Compressed file not found: {seven_zip_path}")

    try:
        with py7zr.SevenZipFile(seven_zip_path, mode='r') as archive:
            # Extract to the parent directory
            archive.extract(path=mps_path.parent)

        if not mps_path.exists():
            raise FileNotFoundError(f"Expected .mps file not found after extraction: {mps_path}")

        return str(mps_path)

    except zipfile.BadZipFile as e:
        raise zipfile.BadZipFile(f"Corrupted zip file: {zip_path}") from e


def compress_mps_file(mps_file_path: Union[str, Path], remove_original: bool = True) -> str:
    """
    Compresses a .mps file to a .mps.7z file.

    Args:
        mps_file_path: Path to the .mps file to compress
        remove_original: Whether to remove the original .mps file after compression

    Returns:
        str: Path to the created .mps.7z file

    Raises:
        FileNotFoundError: If the .mps file doesn't exist
    """
    mps_path = Path(mps_file_path)

    if not mps_path.exists():
        raise FileNotFoundError(f"MPS file not found: {mps_path}")

    seven_zip_path = Path(str(mps_path) + ".7z")

    try:
        # Compress using py7zr
        with py7zr.SevenZipFile(seven_zip_path, 'w') as archive:
            archive.write(mps_path, mps_path.name)

        if remove_original and mps_path.exists():
            mps_path.unlink()

        return str(seven_zip_path)

    except Exception as e:
        # Clean up partially created 7z file on error
        if seven_zip_path.exists():
            seven_zip_path.unlink()
        raise OSError(f"Compression failed: {str(e)}")


class MPSFileManager:
    """
    Context manager that automatically handles MPS file compression/decompression using 7zip.

    Usage:
        # Single file - returns string path directly
        with MPSFileManager('path1.mps') as mps_file:
            # Use the decompressed file (mps_file is a string)
            pass

        # Multiple files - returns list of paths
        with MPSFileManager(['path1.mps', 'path2.mps']) as mps_files:
            # Use the decompressed files (mps_files is a list)
            pass

        # Files are automatically cleaned up when exiting the context
    """

    def __init__(self, mps_file_paths: Union[str, Path, List[Union[str, Path]]],
                 remove_decompressed_on_exit: bool = True):
        # Handle single file or list of files
        if isinstance(mps_file_paths, (str, Path)):
            self.mps_file_paths = [Path(mps_file_paths)]
            self.is_single_file = True
        else:
            self.mps_file_paths = [Path(p) for p in mps_file_paths]
            self.is_single_file = False

        self.remove_decompressed_on_exit = remove_decompressed_on_exit
        self.extracted_files = []
        self.originally_compressed = []  # Files that were compressed and we decompressed

    def __enter__(self):
        for mps_path in self.mps_file_paths:
            seven_zip_path = mps_path.with_suffix(mps_path.suffix + '.7z')

            # Check if file was originally compressed (.7z exists, .mps doesn't)
            if seven_zip_path.exists() and not mps_path.exists():
                # File is compressed - decompress it for use
                self.originally_compressed.append(mps_path)
                extracted_path = decompress_mps_file(mps_path)
                self.extracted_files.append(Path(extracted_path))
            elif mps_path.exists():
                # File is already uncompressed - use as is
                self.extracted_files.append(mps_path)
            else:
                raise FileNotFoundError(f"Neither {mps_path} nor {seven_zip_path} exists")

        # Return single path or list based on input type
        extracted_paths = [str(f) for f in self.extracted_files]
        if self.is_single_file:
            return extracted_paths[0]
        else:
            return extracted_paths

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.remove_decompressed_on_exit:
            for mps_path in self.originally_compressed:
                if mps_path.exists():
                    mps_path.unlink()  # Delete the decompressed file
