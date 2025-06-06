import cplex
import re
import typing
from collections import OrderedDict

from tools.printer import Printer

printer = Printer.getInstance()
printer.set_width(240)

# # Might be needed for full symmetry to compare quadratic objectives in the future
# from collections import defaultdict
#
# def symmetrize_quadratic(obj_quad_raw):
#     obj_quad = defaultdict(float)
#     for (i, j), coeff in obj_quad_raw.items():
#         obj_quad[tuple(sorted((i, j)))] += coeff if i == j else coeff / 2
#     return dict(obj_quad)

def load_mps(filepath):

    model = cplex.Cplex()
    model.set_results_stream(None)  # suppress CPLEX output
    model.set_log_stream(None)
    model.set_warning_stream(None)
    model.set_error_stream(None)
    model.read(filepath)

    return model

def get_model_data(model):
    var_names = model.variables.get_names()
    lower_bounds = model.variables.get_lower_bounds()
    upper_bounds = model.variables.get_upper_bounds()

    obj_linear = dict(zip(var_names, model.objective.get_linear()))
    obj_quadratic = {
        (var_names[i], var_names[j]): coeff
        for i, j, coeff in model.objective.get_quadratic()
    }
    obj_sense = model.objective.sense

    constraint_names_linear = model.linear_constraints.get_names()
    rhs_linear = dict(zip(constraint_names_linear, model.linear_constraints.get_rhs()))
    lin_sense = dict(zip(constraint_names_linear, model.linear_constraints.get_senses()))
    matrix_linear = {
        constraint_names_linear[i]: {
            var_names[ind]: val
            for ind, val in zip(model.linear_constraints.get_rows(i).ind,
                                model.linear_constraints.get_rows(i).val)
        }
        for i in range(model.linear_constraints.get_num())
    }

    constraint_names_quadratic = model.quadratic_constraints.get_names()
    rhs_quadratic = dict(zip(constraint_names_quadratic, model.quadratic_constraints.get_rhs()))
    quad_sense = {name: "E" for name in constraint_names_quadratic}
    matrix_quadratic = {
        constraint_names_quadratic[i]: {
            (var_names[ind1], var_names[ind2]): val
            for ind1, ind2, val in zip(
                model.quadratic_constraints.get_quadratic_components(i).ind1,
                model.quadratic_constraints.get_quadratic_components(i).ind2,
                model.quadratic_constraints.get_quadratic_components(i).val
            )
        }
        for i in range(model.quadratic_constraints.get_num())
    }

    data = {
        "variables": var_names,
        "bounds": {
            "lower": lower_bounds,
            "upper": upper_bounds,
        },
        "obj": {
            "linear": obj_linear,
            "quadratic": obj_quadratic,
            "sense": obj_sense,
        },
        "matrix": {
            "linear": matrix_linear,
            "quadratic": matrix_quadratic,
        },
        "rhs": {
            "linear": rhs_linear,
            "quadratic": rhs_quadratic,
        },
        "sense": {
            "linear": lin_sense,
            "quadratic": quad_sense,
        },
    }

    return data

def get_fixed_zero_variables(data, precision: float = 1e-12) -> list[str]:
    """
    Return a list of normalized variable names that are fixed to zero
    (lower bound == upper bound == 0, within given precision).
    """
    var_names = data["variables"]              # list of variable names
    lower_bounds = data["bounds"]["lower"]     # list of lower bounds (matching var_names order)
    upper_bounds = data["bounds"]["upper"]     # list of upper bounds (matching var_names order)

    fixed_to_zero = []
    for i, var in enumerate(var_names):
        lb = lower_bounds[i] if i < len(lower_bounds) else None
        ub = upper_bounds[i] if i < len(upper_bounds) else None

        if lb == ub == 0 or (lb is not None and ub is not None and abs(lb) < precision and abs(ub) < precision):
            norm_var = normalize_variable_names(str(var))
            fixed_to_zero.append(norm_var)

    return fixed_to_zero


def normalize_variable_names(var_name: str) -> str:
    """
    Normalize a single variable name and return both the normalized
    name and the original. Automatically casts to string if needed.
    """
    var_name = var_name.replace('[', '(').replace(']', ')')
    normalized_var = var_name
    return normalized_var


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
    """
    Sorts and organizes the indices within a coefficient string in alphabetical order.

    If the coefficient string does not contain indices (i.e., it doesn't match the
    pattern `name(index1,index2,...)`), the original string is returned unchanged.

    :param coefficient: The input coefficient string with optional indices.
    :type coefficient: str
    :return: A string with sorted indices, or the original string if no indices are present.
    :rtype: str
    """
    regex_result = re.findall(r"(\w+)\(([^)]*)\)", coefficient)

    # If no match found, return the original string
    if not regex_result:
        return coefficient

    if len(regex_result) > 1:
        raise ValueError(f"More than one index group found in {coefficient}")

    name, indices_str = regex_result[0]
    if not indices_str.strip():
        return coefficient  # No indices, return original

    indices = sorted(i.strip() for i in indices_str.split(",") if i.strip())
    return f"{name}[{','.join(indices)}]"


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
                coefficients_off_by_minus_one = 0  # Counter for coefficients that differ by a factor of -1
                for coefficient_name1, coefficient_value1 in constraint1.items():
                    if coefficient_name1 not in constraint2:
                        status = "Coefficient name mismatch"
                        break

                    if ((isinstance(coefficient_value1, int) or isinstance(coefficient_value1, float)) and
                            (isinstance(constraint2[coefficient_name1], int) or isinstance(constraint2[coefficient_name1], float))):  # If both values are numeric
                        coefficient_value1 = float(coefficient_value1)
                        coefficient_value2 = float(constraint2[coefficient_name1])
                        if coefficient_value1 == 0:
                            if abs(coefficient_value2) > precision:  # If coefficient_value1 == 0, check if coefficient_value2 is sufficiently small
                                status = "Coefficient values differ"
                                partial_match_coefficients1[coefficient_name1] = coefficient_value1
                                partial_match_coefficients2[coefficient_name1] = coefficient_value2
                        elif abs((coefficient_value1 - coefficient_value2) / coefficient_value1) > precision:
                            if abs(((coefficient_value1 * -1) - coefficient_value2) / coefficient_value1) <= precision:  # Check if it's just a sign difference
                                coefficients_off_by_minus_one += 1
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
                        # Check if all coefficients only differ by a factor of -1 and the sense is opposite -> Then they are equivalent
                        if (coefficients_off_by_minus_one == len(constraint1) - 2 and  # -2 because we have the constant and the sense
                                ((constraint1['sense'] == '>=' and constraint2['sense'] == '<=') or
                                 (constraint1['sense'] == '<=' and constraint2['sense'] == '>=') or
                                 (constraint1['sense'] == '=' and constraint2['sense'] == '='))):
                            status = "Perfect match"
                            counter_perfectly_matched_constraints += 1
                            if print_additional_information:
                                printer.information(f"Perfect match (differing by factor -1) found for constraint {constraint_name1}: \nmodel1: {constraint1}\n model2: {constraint2}")
                        else:
                            if counter_perfectly_matched_constraints > 0:
                                printer.success(f"{counter_perfectly_matched_constraints} constraints matched perfectly")
                                counter_perfect_total += counter_perfectly_matched_constraints
                                counter_perfectly_matched_constraints = 0
                            printer.warning(f"Found partial match (factors differ by more than {precision * 100}%):")
                            printer.information(f"model1 {constraint_name1}: {partial_match_coefficients1}", hard_wrap_chars="[...]")
                            printer.information(f"model2 {constraint_name2}: {partial_match_coefficients2}", hard_wrap_chars="[...]")
                            counter_partial_total += 1
                        constraint_dicts2[length].pop(constraint_name2)
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
# Returns a list of variables that are missing in the second list
def compare_variables(vars1, vars2, vars_fixed_to_zero=None, precision: float = 1e-12) -> dict[str, int]:
    counter = 0
    counter_perfect_total = 0
    counter_partial_total = 0
    counter_missing1_total = 0
    counter_missing2_total = 0
    vars_fixed_to_zero = set(vars_fixed_to_zero or [])

    vars2 = vars2.copy()

    for v in vars1:
        found = False
        bounds_differ = False
        for v2 in vars2:
            if v[0] == v2[0]:  # We see if the variable names match
                found = True
                if v[1] is None or v2[1] is None: # If one of the bounds is None, we compare if the other one is also none
                    if v[1] != v2[1]:   # If one bound is None and the other is not, they differ
                        bounds_differ = True
                        break
                elif v[1] == 0: # If the lower bound is 0, we check if the other lower bound is sufficiently small
                    if abs(v2[1]) > precision:
                        bounds_differ = True
                        break
                elif abs((v[1] - v2[1]) / v[1]) > precision: # If the first lower bound is not 0 or None we check if the relative difference is within the precision
                    bounds_differ = True
                    break
                # Same for upper bounds
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
                break  # Found a match, exit inner loop

        if not found: # If a variable is not found in model 2
            if v[0] in vars_fixed_to_zero: # If the variable is fixed to zero, we can ignore it
                counter += 1
            else:
                if counter > 0: # If we have counted some perfect matches, we print them
                    printer.success(f"{counter} variables matched perfectly")
                    counter_perfect_total += counter
                    counter = 0
                counter_missing2_total += 1
                printer.warning(f"Variable not found in model2: {v}") # Print variable that is missing in model 2
        elif bounds_differ: # If the variable is found but the bounds differ
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

    if len(vars2) > 0: # If there are still variables left in model 2 that were not found in model 1
        printer.error(
            f"Variables missing in model 1: {', '.join([v[0] for v in vars2])}",
            hard_wrap_chars=f"[... {len(vars2)} total]"
        )
        counter_missing1_total += len(vars2)

    if vars_fixed_to_zero:
        info_list = [sort_indices(v.replace("(", "[").replace(")", "]")) for v in vars_fixed_to_zero]
        printer.information(
            f"Variables missing in list2, but fixed to 0: {', '.join(info_list)}",
            hard_wrap_chars=f"[... {len(info_list)} total]"
        )

    counter_perfect_total += counter

    return {
        "perfect": counter_perfect_total,
        "partial": counter_partial_total,
        "missing in model 1": counter_missing1_total,
        "missing in model 2": counter_missing2_total
    }


def normalize_objective(
        model_data,
        vars_fixed_to_zero: set[str] = None,
        coefficients_to_skip: list[str] = None,
        zero_tol: float = 1e-15
) -> dict[str, float]:
    """
    Normalize the linear objective coefficients in a CPLEX model dictionary. This function processes
    the provided model representation, skipping certain coefficients based on user-defined
    criteria, and excludes variables fixed to zero. It creates a normalized mapping of the
    objective function for further analysis.

    :param model_data: A dictionary representing the model data which includes the linear objective
        coefficients.
    :param coefficients_to_skip: A list of substrings; any variable containing these substrings will
        not be included in the normalized objective. Defaults to None.
    :param vars_fixed_to_zero: A set of variable names that are fixed to zero and should be skipped
        in the normalization process. Defaults to an empty set.
    :param zero_tol: A tolerance value; coefficients with an absolute value equal to or less than this
        threshold are ignored. Defaults to 1e-15.
    :return: A dictionary where keys are the normalized variable names, and values are their corresponding
        objective coefficients derived from the input model data.
    """
    if coefficients_to_skip is None:
        coefficients_to_skip = []
    if vars_fixed_to_zero is None:
        vars_fixed_to_zero = set()

    normalized_objective_lin = {}
    skipped_vars = []

    for original_obj_name_linear, value in model_data["obj"]["linear"].items():
        obj_name_lin_clean = normalize_variable_names(original_obj_name_linear)
        sorted_obj_name_lin = sort_indices(obj_name_lin_clean)
        # Skip if variable is fixed to zero
        if obj_name_lin_clean in vars_fixed_to_zero:
            skipped_vars.append(original_obj_name_linear)
            continue

        normalized_objective_lin[sorted_obj_name_lin] = value

    if skipped_vars:
        max_preview = 10
        print(f"Skipping {len(skipped_vars)} fixed-to-zero variables for coefficients.")
        print("Examples:")
        for var in skipped_vars[:max_preview]:
            print(f"  {var}")
        if len(skipped_vars) > max_preview:
            print(f"  ... and {len(skipped_vars) - max_preview} more.")

    return normalized_objective_lin

def compare_objectives(objective1, objective2, precision: float = 1e-12) -> dict[str, int]:
    objective2 = objective2.copy()
    counter_perfect_matches = 0
    partial_matches = []
    coefficients_missing_in_model1 = []
    coefficients_missing_in_model2 = []

    for name1, value1 in objective1.items():
        found = False
        if name1 in objective2:
            value2 = objective2[name1]
            rel_diff = abs((value1 - value2) / value1) if value1 != 0 else abs(value2)
            if rel_diff > precision:
                partial_matches.append(f"{name1}: {value1} != {value2}")
            else:
                counter_perfect_matches += 1
                if counter_perfect_matches % 500 == 0:
                    print(f"{counter_perfect_matches} coefficients matched perfectly...")

            del objective2[name1]
            found = True

        if not found:
            coefficients_missing_in_model2.append(f"{name1}: {value1}")

    for name2, value2 in objective2.items():
        coefficients_missing_in_model1.append(f"{name2}: {value2}")

    # Summary logging
    print(f"[✔] {counter_perfect_matches} objective coefficients matched perfectly.")
    if partial_matches:
        print("[⚠] Partial mismatches found:")
        for entry in partial_matches:
            print(f"    • {entry}")
    if coefficients_missing_in_model1:
        print("[✘] Coefficients missing in model 1:")
        for entry in coefficients_missing_in_model1:
            print(f"    • {entry}")
    if coefficients_missing_in_model2:
        print("[✘] Coefficients missing in model 2:")
        for entry in coefficients_missing_in_model2:
            print(f"    • {entry}")

    return {
        "perfect": counter_perfect_matches,
        "partial": len(partial_matches),
        "missing in model 1": len(coefficients_missing_in_model1),
        "missing in model 2": len(coefficients_missing_in_model2)
    }
def compare_mps(file1, file1_isPyomoFormat: bool, file2, file2_isPyomoFormat: bool, check_vars=True, check_constraints=True, print_additional_information=False,
                constraints_to_skip_from1=None, constraints_to_keep_from1=None, coefficients_to_skip_from1=None,
                constraints_to_skip_from2=None, constraints_to_keep_from2=None, coefficients_to_skip_from2=None, constraints_to_enforce_from2=None):
    # Safety before more expensive operations start
    if check_constraints:
        # If any of enforce is in skip, raise error
        if len(constraints_to_skip_from2) and len(constraints_to_enforce_from2) != len(set(constraints_to_enforce_from2).difference(constraints_to_skip_from2)):
            raise ValueError(f"constraints_to_skip_from2 contains elements of constraints_to_enforce_from2: {set(constraints_to_enforce_from2).difference(constraints_to_skip_from2)}")

        # If any of enforce is not in keep, raise error
        if len(constraints_to_keep_from2) and len(constraints_to_enforce_from2) != len(set(constraints_to_enforce_from2).intersection(constraints_to_keep_from2)):
            raise ValueError(f"constraints_to_keep_from2 is missing elements of constraints_to_enforce_from2: {set(constraints_to_keep_from2).difference(constraints_to_enforce_from2)}")

        if len(constraints_to_skip_from1) and len(constraints_to_keep_from1):
            printer.warning("Ignoring 'constraints_to_skip_from1' since 'constraints_to_keep_from1' is set!")
            constraints_to_skip_from1 = None

        if len(constraints_to_skip_from2) and len(constraints_to_keep_from2):
            printer.warning("Ignoring 'constraints_to_skip_from2' since 'constraints_to_keep_from2' is set!")
            constraints_to_skip_from2 = None

    # Load MPS files
    model1 = get_model_data(load_mps(file1)) # Pyomo data
    model2 = get_model_data(load_mps(file2)) # GAMS data

    comparison_results = {}

    # Variables
    if check_vars:
        vars1 = set(
            (normalize_variable_names(var), model1["bounds"]["lower"][i], model1["bounds"]["upper"][i])
            for i, var in enumerate(model1["variables"])
        )
        vars2 = set(
            (normalize_variable_names(var), model2["bounds"]["lower"][i], model2["bounds"]["upper"][i])
            for i, var in enumerate(model2["variables"])
        )

        vars_fixed_to_zero = get_fixed_zero_variables(model1) # Only the Pyomo model writes fixed to zero variables, GAMS doesn't write them at all, so those can be ignored

        comparison_results["variables"] = compare_variables(
            vars1, vars2, vars_fixed_to_zero=vars_fixed_to_zero
        )


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
    vars_fixed_to_zero = get_fixed_zero_variables(model1)
    objective1 = normalize_objective(model1, vars_fixed_to_zero = vars_fixed_to_zero, coefficients_to_skip=coefficients_to_skip_from1)
    objective2 = normalize_objective(model2, vars_fixed_to_zero = vars_fixed_to_zero,coefficients_to_skip=coefficients_to_skip_from2)
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
