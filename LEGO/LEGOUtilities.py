import functools
import typing

import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo


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
def addToExecutionLog(func):
    @functools.wraps(func)  # Preserve the original function's name
    def wrapper(*args, **kwargs):
        # Check that function has not already been executed and add it to dictionary
        execution_safety_list = execution_safety_dict[id(args[0])] = [] if id(args[0]) not in execution_safety_dict else execution_safety_dict[id(args[0])]

        fullFuncName = func.__module__ + '.' + func.__name__
        if fullFuncName not in execution_safety_list:
            execution_safety_list.append(fullFuncName)  # Add function to execution list
        else:
            raise RuntimeError(f"Function {fullFuncName} has already been executed, current execution log: {execution_safety_list}")

        # Call the function
        func(*args, **kwargs)

    return wrapper


# Decorator to check that all required functions have been executed before executing the function
# Also checks that the function has not already been executed
# required_functions: List of function names that need to have been executed before this function (without the file path)
def checkExecutionLog(required_functions: list[typing.Callable]):
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
                func(*args, **kwargs)

        return wrapper

    return decorator


def plot_unit_commitment(output_file_path: str):
    """
    Plot the unit commitment of a given output file
    :param output_file_path: Path to Excel-File
    :return: Nothing (shows plot)
    """
    plt.rcParams['figure.dpi'] = 300  # Set resolution of the plot
    df = pd.read_excel(output_file_path)

    # Fix indices for "Truth" case
    if "Truth" in df["case"].unique():
        counter = 0
        for i, row in df[df["case"] == "Truth"].iterrows():
            df.loc[i, "rp"] = df.loc[counter, "rp"]
            df.loc[i, "k"] = df.loc[counter, "k"]
            counter += 1

    # Plot the data
    index = [f"{rp}{k}" for rp in df["rp"].unique() for k in df["k"].unique()]

    def prev(i: str, n: int = 1):
        current_index = index.index(i)
        return index[current_index - n]

    fig, axs = plt.subplots(len(df["case"].unique()), len(df["g"].unique()), figsize=(6 * len(df["g"].unique()), 2 * len(df["case"].unique())))
    for i, case in enumerate(df["case"].unique()):
        for j, g in enumerate(df["g"].unique()):

            data_plot = {}
            data_bar_startup = {}
            data_bar_shutdown = {}
            data_bar_min_uptime_height = {}
            data_bar_min_downtime_bottom = {}
            data_plot_demand = {}
            for rp in df["rp"].unique():
                for k in df["k"].unique():
                    data_plot[f"{rp}{k}"] = df[(df["case"] == case) & (df["g"] == g) & (df["rp"] == rp) & (df["k"] == k)]["vCommit"].values[0]
                    data_bar_startup[f"{rp}{k}"] = df[(df["case"] == case) & (df["g"] == g) & (df["rp"] == rp) & (df["k"] == k)]["vStartup"].values[0]
                    data_bar_shutdown[f"{rp}{k}"] = df[(df["case"] == case) & (df["g"] == g) & (df["rp"] == rp) & (df["k"] == k)]["vShutdown"].values[0]
                    data_plot_demand[f"{rp}{k}"] = df[(df["case"] == case) & (df["g"] == g) & (df["rp"] == rp) & (df["k"] == k)]["pDemandP"].values[0]

            for rp in df["rp"].unique():
                for k in df["k"].unique():
                    data_bar_min_uptime_height[f"{rp}{k}"] = sum([data_bar_startup[f"{a}"] for a in [prev(f"{rp}{k}", b) for b in range(1, df[(df["case"] == case) & (df["g"] == g) & (df["rp"] == rp) & (df["k"] == k)]["pMinUpTime"].values[0])]])
                    data_bar_min_downtime_bottom[f"{rp}{k}"] = 1 - sum([data_bar_shutdown[f"{a}"] for a in [prev(f"{rp}{k}", b) for b in range(1, df[(df["case"] == case) & (df["g"] == g) & (df["rp"] == rp) & (df["k"] == k)]["pMinDownTime"].values[0])]])

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
            axs[i, j].set_xticks(index[::24] + [index[-1]])  # Set x-axis ticks to only show every 24th value
            for x in index[::24]:  # Create vertical lines for each 24th x-tick
                axs[i, j].axvline(x=x, color="gray", linestyle="--", alpha=0.5)
            axs[i, j].axvline(x=index[-1], color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
