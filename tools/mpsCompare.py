import re
import typing
from collections import OrderedDict

from pulp import LpProblem


def remove_underscore(string: str) -> str:
    # Regular expression to find all occurrences of '[0-9]_' in a string
    regex_replace = re.compile(r"(\d)(_)")

    return regex_replace.sub(r"\1,", string)


# Normalize constraints by
# 1. Replacing all names with actual names in the model
# 2. Sorting the constraint by name
# 3. Normalizing all factors based on the constant
def normalize_constraints(model):
    original_names = {str(b): a.replace("(", "[").replace(")", "]") for a, b in model[0].items()}
    constraints = {}

    # Loop through all constraints
    for name, constraint in model[1].constraints.items():
        original_constraint_dict = constraint.toDict()
        result_constraint_dict = {}

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

        constraints[name] = orderedDict

    return constraints


# Sort constraints by number of coefficients
def sort_constraints_by_length(constraints: typing.Dict[str, OrderedDict[str, str]], coefficients_to_skip: list[str] = None) -> OrderedDict[int, OrderedDict[str, OrderedDict[str, str]]]:
    constraint_dicts: OrderedDict[int, OrderedDict[str, OrderedDict[str, str]]] = OrderedDict()

    if coefficients_to_skip is None:
        coefficients_to_skip = []

    for constraint_name, constraint in constraints.items():
        for coefficient_name in coefficients_to_skip:
            constraint.pop(coefficient_name, None)  # Remove coefficient if it is in the skip list

        if len(constraint) not in constraint_dicts:
            constraint_dicts[len(constraint)] = OrderedDict()  # Initialize dictionary

        if constraint_name in constraint_dicts[len(constraint)]:
            raise ValueError(f"Constraint {constraint_name} already in dictionary")

        constraint_dicts[len(constraint)][constraint_name] = constraint  # Add constraint to dictionary
    return OrderedDict(sorted(constraint_dicts.items()))


# Compare two lists of constraints where coefficients are already normalized (i.e. sorted by name and all factors are divided by the constant)
def compare_constraints(constraints1: typing.Dict[str, OrderedDict[str, str]], constraints2: typing.Dict[str, OrderedDict[str, str]]):
    coefficients_to_skip_from1 = ["name"]
    coefficients_to_skip_from2 = ["name",
                                  "v2ndResDW", "vGenP1"]

    # Sort constraints by number of coefficients
    constraint_dicts1 = sort_constraints_by_length(constraints1, coefficients_to_skip_from1)
    constraint_dicts2 = sort_constraints_by_length(constraints2, coefficients_to_skip_from2)

    # Loop through all constraints in first list and for each through all constraints in the second list
    for length, constraint_dict1 in constraint_dicts1.items():
        if length not in constraint_dicts2:
            print(f"No constraints of length {length} in second model, skipping comparison for {len(constraint_dict1)} constraints")
            continue

        for constraint_name1, constraint1 in constraint_dict1.items():
            status = "Potential match"
            for constraint_name2, constraint2 in constraint_dicts2[length].items():
                status = "Potential match"

                for coefficient_name1, coefficient_value1 in constraint1.items():
                    if coefficient_name1 not in constraint2:
                        status = "Coefficient name mismatch"
                        break
                    elif coefficient_value1 != constraint2[coefficient_name1]:
                        status = "Coefficient values differ"

                match status:
                    case "Potential match":
                        status = "Perfect match"
                        print("Found perfect match")
                        constraint_dicts2[length].pop(constraint_name2)
                        break
                    case "Coefficient values differ":
                        print("Found partial match (factors differ)")
                        print(f"Constraint 1: {constraint1}")
                        print(f"Constraint 2: {constraint2}")
                        constraint_dicts2[length].pop(constraint_name2)
                        break
                    case "Coefficient name mismatch":
                        continue
                    case _:
                        raise ValueError("Unknown status")

            if status != "Perfect match" and status != "Coefficient values differ":
                print(f"No match found for constraint {constraint_name1}: {constraint1}")
    return None


def compare_mps(file1, file2, check_vars=True, check_constraints=True):
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
                        print("Variable bounds differ:", v, v2)
                        break
            if not found:
                print("Variable not found in model2:", v)

    # Constraints
    if check_constraints:
        constraints1 = normalize_constraints(model1)
        constraints2 = normalize_constraints(model2)

        # Check if constraints are the same
        constraint_check_result = compare_constraints(constraints1, constraints2)

    # Objectives
    obj1 = model1[1].objective
    obj2 = model2[1].objective

    print("Objectives differ:", obj1 != obj2)
