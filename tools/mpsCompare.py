import re
import typing
from collections import OrderedDict

from pulp import LpProblem

from tools.printer import Printer

printer = Printer.getInstance()
printer.set_width(180)


def remove_underscore(string: str) -> str:
    # Regular expression to find all occurrences of '[0-9]_' in a string
    regex_replace = re.compile(r"(\d)(_)")

    return regex_replace.sub(r"\1,", string)


# Normalize constraints by
# 1. Replacing all names with actual names in the model
# 2. Sorting the constraint by name
# 3. Normalizing all factors based on the constant
def normalize_constraints(model, constraints_to_skip: list[str] = None, constraints_to_keep: list[str] = None, coefficients_to_skip: list[str] = None):
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
            name = (name.replace("c_e_", "")[:-2] + ")").replace("_", "(")

        # Replace name & normalize factors by constant
        constant = original_constraint_dict['constant']
        for coefficient in original_constraint_dict['coefficients']:
            if original_names[coefficient['name']] in result_constraint_dict:
                raise ValueError(f"Coefficient {original_names[coefficient['name']]} found twice in constraint {name}.\n"
                                 f"Full constraint: {constraint}")

            indices_groups = re.findall(r"(\w*)\[([^]]*)", original_names[coefficient['name']])[0]
            if len(indices_groups) > 2:
                raise ValueError(f"More than one index group found in {original_names[coefficient['name']]}")
            if len(indices_groups) == 1:  # No indices found
                indices = ""
            else:
                indices = ",".join(sorted([i.strip() for i in indices_groups[1].split(",")]))  # Sort indices alphabetically

            sorted_coefficient = f"{indices_groups[0]}[{indices}]"

            if constant != 0:
                result_constraint_dict[sorted_coefficient] = coefficient['value'] / constant
            else:
                result_constraint_dict[sorted_coefficient] = coefficient['value']

        # Create result dictionary
        orderedDict = OrderedDict()

        # Add name to the dictionary
        if 'name' in orderedDict:
            raise ValueError("'name' already in orderedDict")
        orderedDict['name'] = name

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

        # Remove coefficients from the skip list
        for coefficient_name in coefficients_to_skip:
            orderedDict.pop(coefficient_name, None)

        constraints[name] = orderedDict

    return constraints


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
                        constraints_to_enforce_from2: list[str] = None,
                        print_additional_information=False) -> bool:
    # Sort constraints by number of coefficients
    constraint_dicts1 = sort_constraints(constraints=constraints1)
    constraint_dicts2 = sort_constraints(constraints=constraints2)

    # Loop through all constraints in first list and for each through all constraints in the second list
    counter_perfectly_matched_constraints = 0
    for length, constraint_dict1 in constraint_dicts1.items():
        if length not in constraint_dicts2:
            print(f"No constraints of length {length} in second model, skipping comparison for {len(constraint_dict1)} constraints")
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
                            printer.information(f"{counter_perfectly_matched_constraints} constraints matched perfectly")
                            counter_perfectly_matched_constraints = 0
                        printer.information(f"Found partial match (factors differ by more than {precision * 100}%):")
                        printer.information(f"{constraint_name1}: {partial_match_coefficients1}", hard_wrap_chars="[...]")
                        printer.information(f"{constraint_name2}: {partial_match_coefficients2}", hard_wrap_chars="[...]")
                        constraint_dicts2[length].pop(constraint_name2)
                        break
                    case "Coefficient name mismatch":
                        continue
                    case _:
                        raise ValueError("Unknown status")

            if status != "Perfect match" and status != "Coefficient values differ":
                if counter_perfectly_matched_constraints > 0:
                    printer.information(f"{counter_perfectly_matched_constraints} constraints matched perfectly")
                    counter_perfectly_matched_constraints = 0
                printer.error(f"No match found for constraint {constraint_name1}: {constraint1}", hard_wrap_chars=f"[... {len(constraint1)} total]")
    if counter_perfectly_matched_constraints > 0:
        printer.information(f"{counter_perfectly_matched_constraints} constraints matched perfectly")

    if constraints_to_enforce_from2 is not None:
        for enforced_constraint_name in constraints_to_enforce_from2:
            for length, constraint_dict2 in constraint_dicts2.items():
                for constraint_name2, constraint2 in constraint_dict2.items():
                    if enforced_constraint_name in constraint_name2:
                        printer.error(f"Missing enforced constraint {constraint_name2}: {constraint2}", hard_wrap_chars=f"[... {len(constraint2)} total]")
    return False


def compare_mps(file1, file2, check_vars=True, check_constraints=True, print_additional_information=False,
                constraints_to_skip_from1=None, constraints_to_keep_from1=None, coefficients_to_skip_from1=None,
                constraints_to_skip_from2=None, constraints_to_keep_from2=None, coefficients_to_skip_from2=None, constraints_to_enforce_from2=None):
    # Load MPS files
    model1 = LpProblem.fromMPS(file1)
    model2 = LpProblem.fromMPS(file2)

    # Variables
    if check_vars:
        vars1 = {(remove_underscore(v.name), v.lowBound, v.upBound) for v in model1[1].variables()}
        vars2 = {(remove_underscore(v.name), v.lowBound, v.upBound) for v in model2[1].variables()}

        list1 = [v[0] for v in vars1]
        list2 = [v[0] for v in vars2]

        for v in vars1:
            found = False
            for v2 in vars2:
                if v[0] == v2[0]:
                    if v[1] == v2[1] and v[2] == v2[2]:
                        found = True
                        break
                    else:
                        printer.error("Variable bounds differ:", v, v2)
                        break
            if not found:
                printer.error("Variable not found in model2:", v)

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
        constraint_check_result = compare_constraints(constraints1, constraints2, constraints_to_enforce_from2=constraints_to_enforce_from2, print_additional_information=print_additional_information)

    # Objectives
    obj1 = model1[1].objective
    obj2 = model2[1].objective

    print("Objectives differ:", obj1 != obj2)
