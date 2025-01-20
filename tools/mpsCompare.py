import re
import typing
from collections import OrderedDict

from pulp import LpProblem

from tools.printer import Printer

printer = Printer.getInstance()
printer.set_width(240)


def normalize_variable_name(string: str) -> str:
    # Replace first _ with ( and last _ with )
    string = string.replace("_", "(", 1)
    string = string[::-1].replace("_", ")", 1)[::-1]

    return string


# Normalize constraints by
# 1. Replacing all names with actual names in the model
# 2. Sorting the constraint by name
# 3. Normalizing all factors based on the constant
def normalize_constraints(model, constraints_to_skip: list[str] = None, constraints_to_keep: list[str] = None, coefficients_to_skip: list[str] = None) -> typing.Dict[str, OrderedDict[str, str]]:
    original_names = {str(b): a.replace("(", "[").replace(")", "]") for a, b in model[0].items()}
    constraints = {}

    # Sanity checks
    # If skip and keep are set, raise error
    if constraints_to_skip and constraints_to_keep:
        raise ValueError("constraints_to_skip and constraints_to_keep cannot be set at the same time")

    constraints_to_skip = [] if constraints_to_skip is None else constraints_to_skip
    constraints_to_keep = [] if constraints_to_keep is None else constraints_to_keep
    coefficients_to_skip = [] if coefficients_to_skip is None else coefficients_to_skip

    # Loop through all constraints
    for name, constraint in model[1].constraints.items():

        # Skip constraint if it contains any of the strings in constraints_to_skip
        skip_constraint = False
        for c in constraints_to_skip:
            if c in name:
                skip_constraint = True
                break
        if skip_constraint:
            continue

        # Only continue with constraints from constraints_to_keep (if it is set)
        if constraints_to_keep:
            keep_constraint = False
            for c in constraints_to_keep:
                if c in name:
                    keep_constraint = True
                    break
            if not keep_constraint:
                continue

        original_constraint_dict = constraint.toDict()
        result_constraint_dict = {}

        # Try to fix name of constraints for pyomo mps-export
        if "c_e_" in name:
            name = (name.replace("c_e_", "")[:-2] + ")")
        if "c_l_" in name:
            name = (name.replace("c_l_", "")[:-2] + ")")
        if "c_u_" in name:
            name = (name.replace("c_u_", "")[:-2] + ")")

        name = "(".join(name.rsplit("_", 1))

        # Replace name & normalize factors by constant
        constant = original_constraint_dict['constant']
        for coefficient in original_constraint_dict['coefficients']:
            if original_names[coefficient['name']] in result_constraint_dict:
                raise ValueError(f"Coefficient {original_names[coefficient['name']]} found twice in constraint {name}.\n"
                                 f"Full constraint: {constraint}")

            sorted_coefficient = sort_indices(original_names[coefficient['name']])  # Sort indices alphabetically

            # Skip coefficient if it is in coefficients_to_skip
            if sorted_coefficient in coefficients_to_skip:
                continue

            if constant != 0:
                result_constraint_dict[sorted_coefficient] = coefficient['value'] / constant
            else:
                result_constraint_dict[sorted_coefficient] = coefficient['value']

        # Skip constraints without coefficients (which can happen due to coefficients_to_skip)
        if len(result_constraint_dict) == 0:
            continue

        # Create result dictionary
        orderedDict = OrderedDict()

        # Add sense as human-readable string (it is either '<=' [-1], '=' [0], or '>=' [1]) and adjust if constant is negative
        if 'sense' in orderedDict:
            raise ValueError("'sense' already in orderedDict")
        match original_constraint_dict['sense']:
            case -1:
                orderedDict['sense'] = '<=' if constant >= 0 else '>='
            case 0:
                orderedDict['sense'] = '='
            case 1:
                orderedDict['sense'] = '>=' if constant >= 0 else '<='

        # Add constant to the dictionary (as 1 because we divided all factors by the constant (unless it's 0 to begin with))
        if 'constant' in orderedDict:
            raise ValueError("'constant' already in orderedDict")
        if constant != 0:
            orderedDict['constant'] = 1
        else:
            orderedDict['constant'] = 0

        # Add sorted coefficients to the dictionary
        orderedDict.update(sorted(result_constraint_dict.items()))

        constraints[name] = orderedDict

    return constraints


def sort_indices(coefficient: str) -> str:
    regex_result = re.findall(r"(\w*)\[([^]]*)", coefficient)
    indices_groups = regex_result[0] if len(regex_result) > 0 else [""]  # No indices found
    if len(indices_groups) > 2:
        raise ValueError(f"More than one index group found in {coefficient}")
    if len(indices_groups) == 1:  # No indices found
        indices = ""
    else:
        indices = ",".join(sorted([i.strip() for i in indices_groups[1].split(",")]))  # Sort indices alphabetically
    sorted_coefficient = f"{indices_groups[0]}[{indices}]"
    return sorted_coefficient


# Sort constraints by number of coefficients
def sort_constraints(constraints: typing.Dict[str, OrderedDict[str, str]]) -> OrderedDict[int, OrderedDict[str, OrderedDict[str, str]]]:
    constraint_dicts: OrderedDict[int, OrderedDict[str, OrderedDict[str, str]]] = OrderedDict()

    for constraint_name, constraint in constraints.items():
        if len(constraint) not in constraint_dicts:
            constraint_dicts[len(constraint)] = OrderedDict()  # Initialize dictionary

        if constraint_name in constraint_dicts[len(constraint)]:
            raise ValueError(f"Constraint {constraint_name} already in dictionary")

        constraint_dicts[len(constraint)][constraint_name] = constraint  # Add constraint to dictionary
    return OrderedDict(sorted(constraint_dicts.items()))


# Compare two lists of constraints where coefficients are already normalized (i.e. sorted by name and all factors are divided by the constant)
def compare_constraints(constraints1: typing.Dict[str, OrderedDict[str, str]], constraints2: typing.Dict[str, OrderedDict[str, str]], precision: float = 1e-12,
                        constraints_to_enforce_from2: list[str] = None, print_additional_information=False) -> dict[str, int]:
    # Sort constraints by number of coefficients
    constraint_dicts1 = sort_constraints(constraints=constraints1)
    constraint_dicts2 = sort_constraints(constraints=constraints2)

    counter_perfect_total = 0
    counter_partial_total = 0
    counter_missing1_total = 0
    counter_missing2_total = 0

    # Loop through all constraints in first list and for each through all constraints in the second list
    counter_perfectly_matched_constraints = 0
    for length, constraint_dict1 in constraint_dicts1.items():
        if length not in constraint_dicts2:
            if counter_perfectly_matched_constraints > 0:
                printer.success(f"{counter_perfectly_matched_constraints} constraints matched perfectly")
                counter_perfect_total += counter_perfectly_matched_constraints
                counter_perfectly_matched_constraints = 0
            printer.error(f"No constraints of length {length} in second model, skipping comparison for {len(constraint_dict1)} constraints, e.g. {list(constraint_dict1.keys())[0]}", hard_wrap_chars="[...]")
            counter_missing2_total += len(constraint_dict1)
            continue

        for constraint_name1, constraint1 in constraint_dict1.items():
            status = "Potential match"
            for constraint_name2, constraint2 in constraint_dicts2[length].items():
                status = "Potential match"

                partial_match_coefficients1 = {}
                partial_match_coefficients2 = {}
                for coefficient_name1, coefficient_value1 in constraint1.items():
                    if coefficient_name1 not in constraint2:
                        status = "Coefficient name mismatch"
                        break

                    if ((isinstance(coefficient_value1, int) or isinstance(coefficient_value1, float)) and
                            (isinstance(constraint2[coefficient_name1], int) or isinstance(constraint2[coefficient_name1], float))):  # If both values are numeric
                        coefficient_value1 = abs(float(coefficient_value1))
                        coefficient_value2 = abs(float(constraint2[coefficient_name1]))
                        if coefficient_value1 == 0:
                            if abs(coefficient_value2) > precision:  # If coefficient_value1 == 0, check if coefficient_value2 is sufficiently small
                                status = "Coefficient values differ"
                                partial_match_coefficients1[coefficient_name1] = coefficient_value1
                                partial_match_coefficients2[coefficient_name1] = coefficient_value2
                        elif abs((coefficient_value1 - coefficient_value2) / coefficient_value1) > precision:
                            status = "Coefficient values differ"
                            partial_match_coefficients1[coefficient_name1] = coefficient_value1
                            partial_match_coefficients2[coefficient_name1] = coefficient_value2
                    else:  # If one or both values are not numeric, check equality
                        if coefficient_value1 != constraint2[coefficient_name1]:
                            status = "Coefficient values differ"
                            partial_match_coefficients1[coefficient_name1] = coefficient_value1
                            partial_match_coefficients2[coefficient_name1] = constraint2[coefficient_name1]

                match status:
                    case "Potential match":
                        status = "Perfect match"
                        counter_perfectly_matched_constraints += 1
                        if print_additional_information:
                            printer.information(f"Perfect match found for constraint {constraint_name1}: {constraint1}")
                        constraint_dicts2[length].pop(constraint_name2)
                        break
                    case "Coefficient values differ":
                        if counter_perfectly_matched_constraints > 0:
                            printer.success(f"{counter_perfectly_matched_constraints} constraints matched perfectly")
                            counter_perfect_total += counter_perfectly_matched_constraints
                            counter_perfectly_matched_constraints = 0
                        printer.warning(f"Found partial match (factors differ by more than {precision * 100}%):")
                        printer.information(f"model1 {constraint_name1}: {partial_match_coefficients1}", hard_wrap_chars="[...]")
                        printer.information(f"model2 {constraint_name2}: {partial_match_coefficients2}", hard_wrap_chars="[...]")
                        constraint_dicts2[length].pop(constraint_name2)
                        counter_partial_total += 1
                        break
                    case "Coefficient name mismatch":
                        continue
                    case _:
                        raise ValueError("Unknown status")

            if status != "Perfect match" and status != "Coefficient values differ":
                if counter_perfectly_matched_constraints > 0:
                    printer.success(f"{counter_perfectly_matched_constraints} constraints matched perfectly")
                    counter_perfect_total += counter_perfectly_matched_constraints
                    counter_perfectly_matched_constraints = 0
                printer.error(f"No match for {constraint_name1}: {constraint1}", hard_wrap_chars=f"[... {len(constraint1)} total]")
                counter_missing1_total += 1

            if counter_perfectly_matched_constraints > 0 and counter_perfectly_matched_constraints % 500 == 0:
                printer.information(f"{counter_perfectly_matched_constraints} constraints matched perfectly, continue to count...")

    if counter_perfectly_matched_constraints > 0:
        printer.success(f"{counter_perfectly_matched_constraints} constraints matched perfectly")
        counter_perfect_total += counter_perfectly_matched_constraints

    if constraints_to_enforce_from2 is not None:
        for enforced_constraint_name in constraints_to_enforce_from2:
            for length, constraint_dict2 in constraint_dicts2.items():
                for constraint_name2, constraint2 in constraint_dict2.items():
                    if enforced_constraint_name in constraint_name2:
                        printer.error(f"Missing enforced constraint {constraint_name2}: {constraint2}", hard_wrap_chars=f"[... {len(constraint2)} total]")
                        counter_missing1_total += 1

    return {"perfect": counter_perfect_total, "partial": counter_partial_total, "missing in model 1": counter_missing1_total, "missing in model 2": counter_missing2_total}


# Compare two lists of variables where coefficients are already normalized
# Returns a list of variables that are fixed to 0 and missing in the second list
def compare_variables(vars1, vars2, precision: float = 1e-12) -> (dict[str, int], list[str]):
    counter = 0
    counter_perfect_total = 0
    counter_partial_total = 0
    counter_missing1_total = 0
    counter_missing2_total = 0
    vars2 = vars2.copy()
    vars_fixed_to_zero = []  # Variables that are fixed to 0, so if they are missing it is ok
    for v in vars1:
        found = False
        bounds_differ = False
        for v2 in vars2:
            if v[0] == v2[0]:
                found = True
                if v[1] is None or v2[1] is None:
                    if v[1] != v2[1]:
                        bounds_differ = True
                        break
                elif v[1] == 0:
                    if abs(v2[1]) > precision:
                        bounds_differ = True
                        break
                elif abs((v[1] - v2[1]) / v[1]) > precision:
                    bounds_differ = True
                    break

                if v[2] is None or v2[2] is None:
                    if v[2] != v2[2]:
                        bounds_differ = True
                        break
                elif v[2] == 0:
                    if abs(v2[2]) > precision:
                        bounds_differ = True
                        break
                elif abs((v[2] - v2[2]) / v[2]) > precision:
                    bounds_differ = True
                    break
                break  # Found a match so we can break
        if not found:
            if v[1] == 0 and v[2] == 0:  # Variable is missing, but is fixed to 0, so that's ok
                counter += 1
                vars_fixed_to_zero.append(v[0])
            else:
                if counter > 0:
                    printer.success(f"{counter} variables matched perfectly")
                    counter_perfect_total += counter
                    counter = 0
                counter_missing2_total += 1
                printer.warning(f"Variable not found in model2: {v}")
        elif bounds_differ:
            vars2.remove(v2)
            if counter > 0:
                printer.success(f"{counter} variables matched perfectly")
                counter_perfect_total += counter
                counter = 0
            counter_partial_total += 1
            printer.error(f"Variable bounds differ: model1: {v} | model2: {v2}")
        else:
            counter += 1
            vars2.remove(v2)
        if counter > 0 and counter % 500 == 0:
            printer.information(f"{counter} variables matched perfectly, continue to count...")
    if counter > 0:
        printer.success(f"{counter} variables matched perfectly")

    if len(vars2) > 0:
        printer.error(f"Variables missing in model 1: {', '.join([v[0] for v in vars2])}", hard_wrap_chars=f"[... {len(vars2)} total]")
        counter_missing1_total += len(vars2)

    vars_fixed_to_zero = [sort_indices(v.replace("(", "[").replace(")", "]")) for v in vars_fixed_to_zero]  # Adjust indexing-style and sort indices alphabetically
    if len(vars_fixed_to_zero) > 0:
        printer.information(f"Variables missing in list2, but fixed to 0: {', '.join(vars_fixed_to_zero)}", hard_wrap_chars=f"[... {len(vars_fixed_to_zero)} total]")

    counter_perfect_total += counter

    return {"perfect": counter_perfect_total, "partial": counter_partial_total, "missing in model 1": counter_missing1_total, "missing in model 2": counter_missing2_total}, vars_fixed_to_zero


def normalize_objective(model, coefficients_to_skip: list[str] = None) -> dict[str, float]:
    if coefficients_to_skip is None:
        coefficients_to_skip = []

    normalized_objective = {}

    original_names = {str(b): a.replace("(", "[").replace(")", "]") for a, b in model[0].items()}

    for name, value in model[1].objective.items():
        skip_coefficient = False
        for c in coefficients_to_skip:
            if c in original_names[str(name)]:
                skip_coefficient = True
                break
        if skip_coefficient:
            continue

        sorted_name = sort_indices(original_names[str(name)])
        normalized_objective[sorted_name] = value

    return normalized_objective


def compare_objectives(objective1, objective2, precision: float = 1e-12) -> dict[str, int]:
    objective2 = objective2.copy()
    counter_perfect_matches = 0
    partial_matches = []
    coefficients_missing_in_model1 = []
    coefficients_missing_in_model2 = []

    for name1, value1 in objective1.items():
        found = False
        for name2, value2 in objective2.items():
            if name1 == name2:
                if abs((value1 - value2) / value1) > precision:
                    partial_matches.append(f"{name1}: {value1} != {value2}")
                else:
                    counter_perfect_matches += 1
                    if counter_perfect_matches % 500 == 0:
                        printer.information(f"{counter_perfect_matches} coefficients matched perfectly, continue to count...")
                objective2.pop(name2)
                found = True
                break
        if not found:
            coefficients_missing_in_model2.append(f"{name1}: {value1}")

    for name2, value2 in objective2.items():
        coefficients_missing_in_model1.append(f"{name2}: {value2}")

    printer.success(f"{counter_perfect_matches} coefficients of objective matched perfectly")
    if len(partial_matches) > 0:
        printer.error(f"Partial matches found:")
        for match in partial_matches:
            printer.warning(f"Partial: {match}", prefix="")
    if len(coefficients_missing_in_model1) > 0:
        printer.error(f"Coefficients missing in model1:")
        for missing in coefficients_missing_in_model1:
            printer.warning(f"Missing in 1: {missing}", prefix="")
    if len(coefficients_missing_in_model2) > 0:
        printer.error(f"Coefficients missing in model2:")
        for missing in coefficients_missing_in_model2:
            printer.warning(f"Missing in 2: {missing}", prefix="")

    return {"perfect": counter_perfect_matches, "partial": len(partial_matches), "missing in model 1": len(coefficients_missing_in_model1), "missing in model 2": len(coefficients_missing_in_model2)}


def compare_mps(file1, file2, check_vars=True, check_constraints=True, print_additional_information=False,
                constraints_to_skip_from1=None, constraints_to_keep_from1=None, coefficients_to_skip_from1=None,
                constraints_to_skip_from2=None, constraints_to_keep_from2=None, coefficients_to_skip_from2=None, constraints_to_enforce_from2=None):
    # Load MPS files
    model1 = LpProblem.fromMPS(file1)
    model2 = LpProblem.fromMPS(file2)

    comparison_results = {}

    # Variables
    if check_vars:
        vars1 = {(normalize_variable_name(v.name), v.lowBound, v.upBound) for v in model1[1].variables() if v.name not in coefficients_to_skip_from1}
        vars2 = {(v.name, v.lowBound, v.upBound) for v in model2[1].variables() if v.name not in coefficients_to_skip_from2}

        comparison_results['variables'], vars_fixed_to_zero = compare_variables(vars1, vars2)
        coefficients_to_skip_from1.extend(vars_fixed_to_zero)  # Add variables that are fixed to 0 to the list of coefficients to skip

    # Constraints
    if check_constraints:
        # If any of enforce is in skip, raise error
        if constraints_to_skip_from2 and len(constraints_to_enforce_from2) != len(set(constraints_to_enforce_from2).difference(constraints_to_skip_from2)):
            raise ValueError(f"constraints_to_skip_from2 contains elements of constraints_to_enforce_from2: {set(constraints_to_enforce_from2).difference(constraints_to_skip_from2)}")

        # If any of enforce is not in keep, raise error
        if constraints_to_keep_from2 and len(constraints_to_enforce_from2) != len(set(constraints_to_enforce_from2).intersection(constraints_to_keep_from2)):
            raise ValueError(f"constraints_to_keep_from2 is missing elements of constraints_to_enforce_from2: {set(constraints_to_keep_from2).difference(constraints_to_enforce_from2)}")

        constraints1 = normalize_constraints(model1, constraints_to_skip=constraints_to_skip_from1, constraints_to_keep=constraints_to_keep_from1, coefficients_to_skip=coefficients_to_skip_from1)
        constraints2 = normalize_constraints(model2, constraints_to_skip=constraints_to_skip_from2, constraints_to_keep=constraints_to_keep_from2, coefficients_to_skip=coefficients_to_skip_from2)

        # Check if constraints are the same
        comparison_results['constraints'] = compare_constraints(constraints1, constraints2, constraints_to_enforce_from2=constraints_to_enforce_from2, print_additional_information=print_additional_information)

    # Objective
    objective1 = normalize_objective(model1, coefficients_to_skip=coefficients_to_skip_from1)
    objective2 = normalize_objective(model2, coefficients_to_skip=coefficients_to_skip_from2)
    comparison_results['coefficients of objective'] = compare_objectives(objective1, objective2)

    # Print results
    printer.information("\n   ---------   \n\nResults of MPS Comparison:")
    max_key = 0
    max_digits_perfect = 0
    max_digits_partial = 0
    max_digits_missing1 = 0
    max_digits_missing2 = 0
    for key, value in comparison_results.items():
        max_key = max(max_key, len(key))
        max_digits_perfect = max(max_digits_perfect, len(str(value["perfect"])))
        max_digits_partial = max(max_digits_partial, len(str(value["partial"])))
        max_digits_missing1 = max(max_digits_missing1, len(str(value["missing in model 1"])))
        max_digits_missing2 = max(max_digits_missing2, len(str(value["missing in model 2"])))

    all_perfect = True
    for key, value in comparison_results.items():
        text = f"{key:<{max_key}}: "

        text += f"[green]" if value['perfect'] > 0 else f"[yellow]"
        text += f"Perfect: {value['perfect']:>{max_digits_perfect}}"
        text += f"[/green] | " if value['perfect'] > 0 else f"[/yellow] | "

        text += f"[yellow]" if value['partial'] > 0 else f"[green]"
        text += f"Partial: {value['partial']:>{max_digits_partial}}"
        text += f"[/yellow] | " if value['partial'] > 0 else f"[/green] | "

        text += f"[red]" if value['missing in model 1'] > 0 else "[green]"
        text += f"Missing in 1: {value['missing in model 1']:>{max_digits_missing1}}"
        text += f"[/red] | " if value['missing in model 1'] > 0 else f"[/green] | "

        text += f"[red]" if value['missing in model 2'] > 0 else "[green]"
        text += f"Missing in 2: {value['missing in model 2']:>{max_digits_missing2}}"
        text += f"[/red]" if value['missing in model 2'] > 0 else f"[/green]"

        printer.information(text)

        all_perfect = all_perfect and value['partial'] == 0 and value['missing in model 1'] == 0 and value['missing in model 2'] == 0

    if all_perfect:
        printer.success("All checks passed, no missing or partially matching elements found!")
