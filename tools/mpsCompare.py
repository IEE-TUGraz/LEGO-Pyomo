import cplex
import re
import typing
from collections import OrderedDict

from tools.printer import Printer

printer = Printer.getInstance()
printer.set_width(240)

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
    # values = model.solution.get_values() # Only works if the model can be solved
    # var_values = dict(zip(var_names, values))
    # print(var_values)
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

def get_fixed_zero_variables(
    data,
    precision: float = 1e-12,
    coefficients_to_skip: list[str] | None = None,
) -> list[str]:
    """
    Return a list of *normalised* names that are fixed to zero
    (lb == ub == 0, within precision) and do **not** match any
    substring in `coefficients_to_skip`.
    """
    coefficients_to_skip = coefficients_to_skip or []

    fixed_to_zero = []
    for i, var in enumerate(data["variables"]):
        lb = data["bounds"]["lower"][i]
        ub = data["bounds"]["upper"][i]

        if (lb == ub == 0) or (
            lb is not None and ub is not None and
            abs(lb) < precision and abs(ub) < precision
        ):
            norm = normalize_variable_names(str(var))
            if any(s in norm for s in coefficients_to_skip):
                continue
            fixed_to_zero.append(norm)

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
def normalize_constraints(
    data,
    constraints_to_skip: list[str] = None,
    constraints_to_keep: list[str] = None,
    coefficients_to_skip: list[str] = None,
) -> typing.Tuple[dict[str, OrderedDict[str, float]], dict[str, float], dict[str, str]]:
    """
    Normalize linear constraints:
    - Filters constraints and coefficients
    - Converts all RHS to ±1 (if non-zero), adjusts coefficients accordingly
    - Flips inequality direction if RHS is negative
    - Sorts variables in constraints

    Returns:
        constraints: {normalized_constraint_name: OrderedDict of {normalized_variable_name: coeff}}
        rhs: {normalized_constraint_name: float}
        sense: {normalized_constraint_name: "<=", ">=", "="}
    """
    constraints_to_skip = constraints_to_skip or []
    constraints_to_keep = constraints_to_keep or []
    coefficients_to_skip = coefficients_to_skip or []

    if constraints_to_skip and constraints_to_keep:
        raise ValueError("constraints_to_skip and constraints_to_keep cannot be set at the same time")

    sense_map = {"L": "<=", "G": ">=", "E": "="}
    reverse_sense = {"<=": ">=", ">=": "<=", "=": "="}

    constraints = OrderedDict()
    normalized_rhs = {}
    normalized_sense = {}

    for orig_name, coeffs in data["matrix"]["linear"].items():
        name = orig_name

        # Skip or keep
        if any(skip in name for skip in constraints_to_skip):
            continue
        if constraints_to_keep and not any(keep in name for keep in constraints_to_keep):
            continue

        # Remove prefixes and suffixes
        for prefix in ["c_e_", "c_l_", "c_u_"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name.endswith("_"):
            name = name[:-1]
        if "_" in name:
            parts = name.rsplit("_", 1)
            if parts[1].isdigit():
                name = f"{parts[0]}({parts[1]})"

        rhs = data["rhs"]["linear"].get(orig_name, 0.0)
        sense_code = data["sense"]["linear"].get(orig_name, "E")
        symbol = sense_map.get(sense_code, sense_code)

        if rhs == 0.0:
            scale = 1.0
            adjusted_rhs = 0.0
            symbol = symbol
        else:
            scale = 1.0 / abs(rhs)
            adjusted_rhs = 1.0
            symbol = symbol
            if rhs < 0:
                symbol = reverse_sense[symbol]


        normalized_coeffs = OrderedDict()
        for var, val in coeffs.items():
            if any(skip in var for skip in coefficients_to_skip):
                continue

            normalized_var = normalize_variable_names(var)

            sorted_var = sort_indices(normalized_var)
            normal_var = val * scale
            if rhs < 0:
                normal_var *= -1  # Flip variable if RHS is negative

            normalized_coeffs[sorted_var] = normal_var

        normalized_coeffs = OrderedDict(
            sorted((normalize_variable_names(k), v) for k, v in normalized_coeffs.items())
        )
        normalized_name = normalize_variable_names(name)
        constraints[normalized_name] = normalized_coeffs
        # Store normalized RHS and sense but not needed right now
        normalized_rhs[normalized_name] = adjusted_rhs
        normalized_sense[normalized_name] = symbol

    return constraints, normalized_rhs, normalized_sense

def normalize_quadratic_constraints(data,
    constraints_to_skip: list[str] = None,
    constraints_to_keep: list[str] = None,
    coefficients_to_skip: list[str] = None,
) -> typing.Tuple[dict[str, OrderedDict[str, float]], dict[str, float], dict[str, str]]:

    constraints_to_skip = constraints_to_skip or []
    constraints_to_keep = constraints_to_keep or []
    coefficients_to_skip = coefficients_to_skip or []

    if constraints_to_skip and constraints_to_keep:
        raise ValueError("Only one of constraints_to_skip or constraints_to_keep can be set.")

    sense_map = {"L": "<=", "G": ">=", "E": "="}
    reverse_sense = {"<=": ">=", ">=": "<=", "=": "="}

    quad_constraints = OrderedDict()
    quad_normalized_rhs = {}
    quad_normalized_sense = {}

    for quad_orig_name, quad_coeffs in data["matrix"]["quadratic"].items():
        quad_name = quad_orig_name

        if any(skip in quad_name for skip in constraints_to_skip):
            continue
        if constraints_to_keep and not any(keep in quad_name for keep in constraints_to_keep):
            continue

        # Normalize constraint name
        for prefix in ["c_e_", "c_l_", "c_u_"]:
            if quad_name.startswith(prefix):
                quad_name = quad_name[len(prefix):]
        if quad_name.endswith("_"):
            quad_name = quad_name[:-1]
        if "_" in quad_name:
            parts = quad_name.rsplit("_", 1)
            if parts[1].isdigit():
                quad_name = f"{parts[0]}({parts[1]})"

        quad_rhs = data['rhs']['quadratic'].get(quad_name, 0.0)
        quad_sense_code = data['sense']["quadratic"].get(quad_name, "E")
        quad_symbol = sense_map.get(quad_sense_code, quad_sense_code)

        if quad_rhs == 0:
            scale = 1.0
            adjusted_rhs = 0.0
            quad_symbol = quad_symbol
        else:
            scale = 1.0 / abs(quad_rhs)
            adjusted_rhs = 1.0
            quad_symbol = quad_symbol
            if quad_rhs < 0:
                quad_symbol = reverse_sense[quad_symbol]

        quad_normalized_coeffs = OrderedDict()
        for (v1, v2), val in quad_coeffs.items():
            if any(skip in v1 for skip in coefficients_to_skip) or any(skip in v2 for skip in coefficients_to_skip):
                continue


            quad_v1_norm = normalize_variable_names(v1)
            quad_v2_norm= normalize_variable_names(v2)
            quad_v1_norm = sort_indices(quad_v1_norm)
            quad_v2_norm = sort_indices(quad_v2_norm)
            key = tuple(sorted((quad_v1_norm, quad_v2_norm)))
            quad_normalized_coeffs[key] = val * scale



        quad_normalized_name = normalize_variable_names(quad_name)
        quad_constraints[quad_normalized_name] = quad_normalized_coeffs

        # Store normalized RHS and sense but not needed right now
        quad_normalized_rhs[quad_normalized_name] = adjusted_rhs
        quad_normalized_sense[quad_normalized_name] = quad_symbol

    return quad_constraints, quad_normalized_rhs, quad_normalized_sense

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
    # regex_result = re.findall(r"(\w+)\(([^)]*)\)", coefficient)
    #
    # # If no match found, return the original string
    # if not regex_result:
    #     return coefficient
    #
    # if len(regex_result) > 1:
    #     raise ValueError(f"More than one index group found in {coefficient}")
    #
    # name, indices_str = regex_result[0]
    # if not indices_str.strip():
    #     return coefficient  # No indices, return original
    #
    # indices = sorted(i.strip() for i in indices_str.split(",") if i.strip())
    #return f"{name}({','.join(indices)})"
    return coefficient

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


# Compare two lists of constraints where coefficients are already normalized (i.e. sorted by length and all factors are divided by the constant)
def compare_linear_constraints(
    constraints1: dict[str, OrderedDict[str, float]],
    constraints2: dict[str, OrderedDict[str, float]],
    vars_fixed_to_zero: set[str],
    precision: float = 1e-12,
    constraints_to_enforce_from2: list[str] = None,
    print_additional_information: bool = False,
) -> dict[str, int]:
    """
    Compare two sets of normalized constraints, optionally filtering out
    variables fixed to zero and enforcing specific constraints to be present.

    :param constraints1: Constraints from Pyomo Model
    :param constraints2: Constraints from GAMS Model
    :param vars_fixed_to_zero: Set of variable names to eliminate from constraints
    :param precision: Threshold for comparing floating-point values
    :param constraints_to_enforce_from2: List of constraint name substrings that must be present in GAMS Model
    :param print_additional_information: Flag to print detailed debug info
    :return: Dictionary with counts of perfect, partial, and missing matches
    """

    # Remove coefficients for variables fixed to zero
    def clear_linear_constraints(constraints):
        cleaned = {}
        for cname, coeffs in constraints.items():
            new_coeffs = OrderedDict(
                (var, val)
                for var, val in coeffs.items()
                if var not in vars_fixed_to_zero and abs(val) >= precision
            )
            cleaned[cname] = new_coeffs
        return cleaned

    cleaned_constraints1_raw = clear_linear_constraints(constraints1)
    cleaned_constraints2_raw = clear_linear_constraints(constraints2)

    removed1 = [k for k, v in cleaned_constraints1_raw.items() if len(v) == 0]
    removed2 = [k for k, v in cleaned_constraints2_raw.items() if len(v) == 0]

    printer.information(f"Removed {len(removed1)} constraints of 0 length from Pyomo Model after eliminating fixed-to-zero vars.")
    if len(removed1) > 0:
        printer.information("Example constraints removed from Pyomo Model: " + ", ".join(removed1[:5]))

    printer.information(f"Removed {len(removed2)} constraints of 0 length from GAMS Model after eliminating fixed-to-zero vars.")
    if len(removed2) > 0:
        printer.information("Example constraints removed from GAMS Model: " + ", ".join(removed2[:5]))

    cleaned_constraints1 = {k: v for k, v in cleaned_constraints1_raw.items() if len(v) > 0}
    cleaned_constraints2 = {k: v for k, v in cleaned_constraints2_raw.items() if len(v) > 0}

    constraint_dicts1 = sort_constraints(cleaned_constraints1)
    constraint_dicts2 = sort_constraints(cleaned_constraints2)


    counter_perfect_total = 0
    counter_partial_total = 0
    counter_missing1_total = 0
    counter_missing2_total = 0

    for length, group1 in constraint_dicts1.items():
        if length not in constraint_dicts2:
            counter_missing2_total += len(group1)
            if print_additional_information:
                print(f"No constraints of length {length} in GAMS Model; skipping {len(group1)} constraints.")
            continue

        group2 = constraint_dicts2[length]
        matched_in_group2 = set()

        for cname1, c1 in group1.items():
            # Skip constraints that are all zero (i.e., no coefficients left after removing fixed-to-zero vars)
            if all(abs(v) < precision for v in c1.values()):
                continue

            matched = False
            for cname2, c2 in group2.items():
                if cname2 in matched_in_group2:
                    continue
                if all(abs(v) < precision for v in c2.values()):
                    continue
                if set(c1.keys()) != set(c2.keys()):
                    continue

                coefficients_off_by_minus_one = 0
                partial_mismatches1 = {}
                partial_mismatches2 = {}
                status = "Potential match"

                for var in c1.keys():
                    val1, val2 = float(c1[var]), float(c2[var])

                    if abs(val1) < precision:
                        if abs(val2) > precision:
                            status = "Coefficient values differ"
                            partial_mismatches1[var] = val1
                            partial_mismatches2[var] = val2
                            break
                    else:
                        rel_diff = abs((val1 - val2) / val1)
                        if rel_diff > precision:
                            rel_diff_sign_flip = abs(((val1 * -1) - val2) / val1)
                            if rel_diff_sign_flip <= precision:
                                coefficients_off_by_minus_one += 1
                            else:
                                status = "Coefficient values differ"
                                partial_mismatches1[var] = val1
                                partial_mismatches2[var] = val2
                                break

                if status == "Potential match":
                    matched = True
                    counter_perfect_total += 1
                    matched_in_group2.add(cname2)
                    break
                elif status == "Coefficient values differ":
                    matched = True
                    counter_partial_total += 1
                    matched_in_group2.add(cname2)
                    if print_additional_information:
                        print(f"Partial mismatch: {cname1} vs {cname2}")
                        print(f"  Pyomo Model coefficients: {partial_mismatches1}")
                        print(f"  GAMS Model coefficients: {partial_mismatches2}")
                    break

            if not matched:
                counter_missing2_total += 1
                if print_additional_information:
                    print(f"No match for constraint: {cname1}")

        unmatched_2 = len(group2) - len(matched_in_group2)
        counter_missing1_total += unmatched_2

        if unmatched_2 > 0 and print_additional_information:
            print(f"{unmatched_2} constraints of length {length} in GAMS Model unmatched for in Pyomo" )

            # Print missing constraint names or keys
            missing_constraints = [key for key in group2 if key not in matched_in_group2]
            print("Example unmatched constraints of GAMS missing in Pyomo:")
            for key in missing_constraints[:50]:

                print(f"  - {key}")

    extra_lengths_in_model2 = set(constraint_dicts2.keys()) - set(constraint_dicts1.keys())
    for length in extra_lengths_in_model2:
        unmatched_group = {
            name: coeffs for name, coeffs in constraint_dicts2[length].items()
            if not all(abs(v) < precision for v in coeffs.values())
        }
        num_unmatched = len(unmatched_group)
        counter_missing1_total += num_unmatched
        if num_unmatched > 0 and print_additional_information:
            print(f"{num_unmatched} constraints in GAMS Model of length {length} missing in Pyomo Model.")

    if constraints_to_enforce_from2:
        for enforced_name in constraints_to_enforce_from2:
            found = any(enforced_name in c_name for group in constraint_dicts2.values() for c_name in group.keys())
            if not found:
                counter_missing1_total += 1
                if print_additional_information:
                    print(f"Enforced constraint missing in GAMS Model: {enforced_name}")

    return {
        "perfect": counter_perfect_total,
        "partial": counter_partial_total,
        "missing in Pyomo Model": counter_missing1_total,
        "missing in GAMS Model": counter_missing2_total,
    }

def compare_quadratic_constraints(
    quad_constraints1: dict[str, dict[tuple[str, str], float]],
    quad_constraints2: dict[str, dict[tuple[str, str], float]],
    vars_fixed_to_zero: set[str],
    precision: float = 1e-12,
    constraints_to_enforce_from2: list[str] = None,
    print_additional_information: bool = False,
) -> dict[str, int]:
    """
    Compare two sets of quadratic constraints, ignoring variables fixed to zero.
    Each constraint is a dict mapping (var1, var2) -> coefficient.
    """

    def clear_quad_constraints(
        quad_constraints: dict[str, dict[tuple[str, str], float]]
    ) -> tuple[dict[str, dict[tuple[str, str], float]], list[str]]:
        cleaned = {}
        removed = []
        for cname, qdict in quad_constraints.items():
            q_clean = {}
            for (v1, v2), coeff in qdict.items():
                if v1 in vars_fixed_to_zero or v2 in vars_fixed_to_zero:
                    continue
                if abs(coeff) < precision:
                    continue
                # Canonicalize key
                key = tuple(sorted((v1, v2)))
                q_clean[key] = q_clean.get(key, 0.0) + coeff
            if q_clean:
                cleaned[cname] = q_clean
            else:
                removed.append(cname)
        return cleaned, removed

    if constraints_to_enforce_from2 is None:
        constraints_to_enforce_from2 = []

    cleaned1, removed1 = clear_quad_constraints(quad_constraints1)
    cleaned2, removed2 = clear_quad_constraints(quad_constraints2)

    printer.information(f"Removed {len(removed1)} quadratic constraints of 0 length from Pyomo Model after eliminating fixed-to-zero vars.")
    printer.information(f"Removed {len(removed2)} quadratic constraints of 0 length from GAMS Model after eliminating fixed-to-zero vars.")


    names1 = set(cleaned1.keys())
    names2 = set(cleaned2.keys())

    counter_perfect = 0
    counter_partial = 0
    counter_missing1 = 0
    counter_missing2 = 0

    for cname in names1:
        if cname not in cleaned2:
            counter_missing2 += 1
            if print_additional_information:
                print(f"[✘] Constraint {cname} missing in GAMS Model.")
            continue

        q1 = cleaned1[cname]
        q2 = cleaned2[cname]

        if set(q1.keys()) != set(q2.keys()):
            counter_partial += 1
            if print_additional_information:
                missing_keys = set(q1.keys()).symmetric_difference(q2.keys())
                print(f"[⚠] Constraint {cname}: mismatched variable pairs: {missing_keys}")
            continue

        mismatch_found = False

        for key in q1:
            val1 = q1[key]
            val2 = q2[key]
            denom = max(abs(val1), abs(val2), precision)  # prevent zero division
            rel_diff = abs(val1 - val2) / denom
            if rel_diff > precision:
                mismatch_found = True
                if print_additional_information:
                    print(f"[⚠] Constraint {cname}: {key} -> {val1} ≠ {val2}")
                break

        if mismatch_found:
            counter_partial += 1
        else:
            counter_perfect += 1

    for cname in names2 - names1:
        counter_missing1 += 1
        if print_additional_information:
            print(f"[✘] Quadratic constraint {cname} missing in Pyomo Model.")

    print(f"[✔] {counter_perfect} quadratic constraints matched perfectly.")
    if counter_partial:
        print(f"[⚠] {counter_partial} quadratic constraints partially mismatched.")
    if counter_missing1:
        print(f"[✘] {counter_missing1} quadratic constraints missing in Pyomo Model.")
    if counter_missing2:
        print(f"[✘] {counter_missing2} quadratic constraints missing in GAMS Model.")

    return {
        "perfect": counter_perfect,
        "partial": counter_partial,
        "missing in Pyomo Model": counter_missing1,
        "missing in GAMS Model": counter_missing2,
    }


# Compare two lists of variables where coefficients are already normalized
# Returns a list of variables that are missing in the second list
def compare_variables(vars1, vars2, vars_fixed_to_zero=None, precision: float = 1e-12,print_additional_information: bool = False) -> dict[str, int]:
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

        if not found: # If a variable is not found in GAMS Model
            if v[0] in vars_fixed_to_zero: # If the variable is fixed to zero, we can ignore it
                counter += 1
            else:
                if counter > 0: # If we have counted some perfect matches, we print them
                    printer.success(f"{counter} variables matched perfectly")
                    counter_perfect_total += counter
                    counter = 0
                counter_missing2_total += 1
                if print_additional_information:
                    printer.warning(f"Variable not found in GAMS Model: {v}") # Print variable that is missing in GAMS Model
        elif bounds_differ: # If the variable is found but the bounds differ
            vars2.remove(v2)
            if counter > 0:
                printer.success(f"{counter} variables matched perfectly")
                counter_perfect_total += counter
                counter = 0
            counter_partial_total += 1
            if print_additional_information:
             printer.error(f"Variable bounds differ: Pyomo Model: {v} | GAMS Model: {v2}")
        else:
            counter += 1
            vars2.remove(v2)

        if counter > 0 and counter % 500 == 0:
            printer.information(f"{counter} variables matched perfectly, continue to count...")

    if counter > 0:
        printer.success(f"{counter} variables matched perfectly")

    if len(vars2) > 0:  # If there are still variables left in GAMS Model that were not found in Pyomo Model
        if print_additional_information:
            missing_var_names = [v[0] for v in vars2]
            formatted_var_list = ", ".join(missing_var_names)
            printer.error(
                f"Variables missing in Pyomo model ({len(vars2)} total): {formatted_var_list}"
            )

        counter_missing1_total += len(vars2)

    if vars_fixed_to_zero:
        info_list = [str(v) for v in vars_fixed_to_zero]
        preview = ", ".join(info_list[:5])
        total = len(info_list)
        printer.information(
            f"Variables missing in GAMS model, but fixed to 0: {preview}"
            + (f", ... [{total} total]" if total > 5 else "")
        )

    counter_perfect_total += counter

    return {
        "perfect": counter_perfect_total,
        "partial": counter_partial_total,
        "missing in Pyomo Model": counter_missing1_total,
        "missing in GAMS Model": counter_missing2_total
    }


def normalize_objective(
    model_data,
    vars_fixed_to_zero: set[str] | None = None,
    coefficients_to_skip: list[str] | None = None,
    zero_tol: float = 1e-15,
) -> tuple[dict, dict]:
    """
    Build a dict {normalised-var-name: coeff} while
    • dropping vars fixed to zero
    • dropping vars whose name contains any substring in `coefficients_to_skip`
    • dropping coefficients whose magnitude ≤ zero_tol
    """
    vars_fixed_to_zero = vars_fixed_to_zero or set()
    coefficients_to_skip = coefficients_to_skip or []
    norm_lin = {}
    norm_quad = {}
    skipped = []
    skipped_quadratic = []
    # Normalize linear objective coefficients
    for raw_name, val in model_data["obj"]["linear"].items():
        if abs(val) <= zero_tol:
            continue

        norm_name = sort_indices(normalize_variable_names(raw_name))
        if any(substr in norm_name for substr in coefficients_to_skip):
            skipped.append(raw_name)
            continue
        if norm_name in vars_fixed_to_zero:
            skipped.append(raw_name)
            continue


        norm_lin[sort_indices(norm_name)] = val
    if skipped: print(f"Skipped {len(skipped)} vars in linear objective (0-fixed or substring match)")
    # Normalize quadratic objective coefficients

    for (raw_name1, raw_name2), value in model_data["obj"]["quadratic"].items():
        if abs(value) <= zero_tol:
            continue

        name1 = normalize_variable_names(raw_name1)
        name2 = normalize_variable_names(raw_name2)

        if any(substr in name1 for substr in coefficients_to_skip) or any(substr in name2 for substr in coefficients_to_skip):
            skipped_quadratic.append((raw_name1, raw_name2))
            continue
        if name1 in vars_fixed_to_zero and name2 in vars_fixed_to_zero:
            skipped_quadratic.append((raw_name1, raw_name2))
            continue
        if skipped_quadratic: print(f"Skipped {len(skipped_quadratic)} quadratic objective (0-fixed or substring match)")

        norm1 = sort_indices(name1)
        norm2 = sort_indices(name2)
        var_pair = tuple(sorted([norm1, norm2]))

        norm_quad[var_pair] = value



    return norm_lin, norm_quad


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
        print("[✘] Coefficients missing in Pyomo Model:")
        for entry in coefficients_missing_in_model1:
            print(f"    • {entry}")
    if coefficients_missing_in_model2:
        print("[✘] Coefficients missing in GAMS Model:")
        for entry in coefficients_missing_in_model2:
            print(f"    • {entry}")

    return {
        "perfect": counter_perfect_matches,
        "partial": len(partial_matches),
        "missing in Pyomo Model": len(coefficients_missing_in_model1),
        "missing in GAMS Model": len(coefficients_missing_in_model2)
    }


def compare_mps(file1, file1_isPyomoFormat: bool, file2, file2_isPyomoFormat: bool, check_vars=True, check_constraints=True, check_quadratic_constraints = True, check_objectives = True, print_additional_information=False,
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
        vars1 = set()
        for i, var in enumerate(model1["variables"]):
            norm_var = normalize_variable_names(var)
            if coefficients_to_skip_from1 and any(skip in norm_var for skip in coefficients_to_skip_from1):
                continue
            vars1.add((norm_var, model1["bounds"]["lower"][i], model1["bounds"]["upper"][i]))

        vars2 = set()
        for i, var in enumerate(model2["variables"]):
            norm_var = normalize_variable_names(var)
            if coefficients_to_skip_from2 and any(skip in norm_var for skip in coefficients_to_skip_from2):
                continue
            vars2.add((norm_var, model2["bounds"]["lower"][i], model2["bounds"]["upper"][i]))

        vars_fixed_to_zero = get_fixed_zero_variables(model1) # Only the Pyomo model writes fixed to zero variables, GAMS doesn't write them at all, so those can be ignored

        comparison_results["variables"] = compare_variables(
            vars1, vars2, vars_fixed_to_zero=vars_fixed_to_zero,print_additional_information=print_additional_information
        )


    # Constraints
    if check_constraints:
        # If any of enforce is in skip, raise error
        if constraints_to_skip_from2 and len(constraints_to_enforce_from2) != len(set(constraints_to_enforce_from2).difference(constraints_to_skip_from2)):
            raise ValueError(f"constraints_to_skip_from2 contains elements of constraints_to_enforce_from2: {set(constraints_to_enforce_from2).difference(constraints_to_skip_from2)}")

        # If any of enforce is not in keep, raise error
        if constraints_to_keep_from2 and len(constraints_to_enforce_from2) != len(set(constraints_to_enforce_from2).intersection(constraints_to_keep_from2)):
            raise ValueError(f"constraints_to_keep_from2 is missing elements of constraints_to_enforce_from2: {set(constraints_to_keep_from2).difference(constraints_to_enforce_from2)}")

        constraints1, _, _ = normalize_constraints(model1, constraints_to_skip=constraints_to_skip_from1, constraints_to_keep=constraints_to_keep_from1, coefficients_to_skip=coefficients_to_skip_from1)
        constraints2, _, _ = normalize_constraints(model2, constraints_to_skip=constraints_to_skip_from2, constraints_to_keep=constraints_to_keep_from2, coefficients_to_skip=coefficients_to_skip_from2)

        vars_fixed_to_zero_raw = get_fixed_zero_variables(model1)
        vars_fixed_to_zero = {
            sort_indices(normalize_variable_names(str(v)))
            for v in vars_fixed_to_zero_raw
        }
        # Check if constraints are the same
        comparison_results['constraints'] = compare_linear_constraints(constraints1, constraints2,vars_fixed_to_zero =vars_fixed_to_zero, constraints_to_enforce_from2=constraints_to_enforce_from2, print_additional_information=print_additional_information)

    if check_quadratic_constraints:
        # If any of enforce is in skip, raise error
        if constraints_to_skip_from2 and len(constraints_to_enforce_from2) != len(set(constraints_to_enforce_from2).difference(constraints_to_skip_from2)):
            raise ValueError(f"constraints_to_skip_from2 contains elements of constraints_to_enforce_from2: {set(constraints_to_enforce_from2).difference(constraints_to_skip_from2)}")

        # If any of enforce is not in keep, raise error
        if constraints_to_keep_from2 and len(constraints_to_enforce_from2) != len(set(constraints_to_enforce_from2).intersection(constraints_to_keep_from2)):
            raise ValueError(f"constraints_to_keep_from2 is missing elements of constraints_to_enforce_from2: {set(constraints_to_keep_from2).difference(constraints_to_enforce_from2)}")

        quad_constraints1, _, _ = normalize_quadratic_constraints(model1, constraints_to_skip=constraints_to_skip_from1, constraints_to_keep=constraints_to_keep_from1, coefficients_to_skip=coefficients_to_skip_from1)
        quad_constraints2, _, _ = normalize_quadratic_constraints(model2, constraints_to_skip=constraints_to_skip_from2, constraints_to_keep=constraints_to_keep_from2, coefficients_to_skip=coefficients_to_skip_from2)

        vars_fixed_to_zero_raw = get_fixed_zero_variables(model1)
        vars_fixed_to_zero = {
            sort_indices(normalize_variable_names(str(v)))
            for v in vars_fixed_to_zero_raw
        }
        # Check if constraints are the same
        comparison_results['quadratic_constraints'] = compare_quadratic_constraints(quad_constraints1, quad_constraints2,vars_fixed_to_zero =vars_fixed_to_zero, constraints_to_enforce_from2=constraints_to_enforce_from2, print_additional_information=print_additional_information)
    # Objective
    if check_objectives:
        vars_fixed_to_zero = get_fixed_zero_variables(model1)

        objective1, quad_objective1 = normalize_objective(
            model1,
            vars_fixed_to_zero=vars_fixed_to_zero,
            coefficients_to_skip=coefficients_to_skip_from1
        )

        objective2, quad_objective2 = normalize_objective(
            model2,
            vars_fixed_to_zero=vars_fixed_to_zero,
            coefficients_to_skip=coefficients_to_skip_from2
        )

        comparison_results['coefficients of objective'] = compare_objectives(objective1, objective2)
        comparison_results['quadratic coefficients of objective'] = compare_objectives(quad_objective1, quad_objective2)

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
        max_digits_missing1 = max(max_digits_missing1, len(str(value["missing in Pyomo Model"])))
        max_digits_missing2 = max(max_digits_missing2, len(str(value["missing in GAMS Model"])))

    all_perfect = True
    for key, value in comparison_results.items():
        text = f"{key:<{max_key}}: "

        text += f"[green]" if value['perfect'] > 0 else f"[yellow]"
        text += f"Perfect: {value['perfect']:>{max_digits_perfect}}"
        text += f"[/green] | " if value['perfect'] > 0 else f"[/yellow] | "

        text += f"[yellow]" if value['partial'] > 0 else f"[green]"
        text += f"Partial: {value['partial']:>{max_digits_partial}}"
        text += f"[/yellow] | " if value['partial'] > 0 else f"[/green] | "

        text += f"[red]" if value['missing in Pyomo Model'] > 0 else "[green]"
        text += f"Missing in Pyomo: {value['missing in Pyomo Model']:>{max_digits_missing1}}"
        text += f"[/red] | " if value['missing in Pyomo Model'] > 0 else f"[/green] | "

        text += f"[red]" if value['missing in GAMS Model'] > 0 else "[green]"
        text += f"Missing in GAMS: {value['missing in GAMS Model']:>{max_digits_missing2}}"
        text += f"[/red]" if value['missing in GAMS Model'] > 0 else f"[/green]"

        printer.information(text)

        all_perfect = all_perfect and value['partial'] == 0 and value['missing in Pyomo Model'] == 0 and value['missing in GAMS Model'] == 0

    if all_perfect:
        printer.success("All checks passed, no missing or partially matching elements found!")
