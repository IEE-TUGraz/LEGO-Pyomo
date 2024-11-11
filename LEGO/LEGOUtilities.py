import functools
import typing


# Turns "k0001" into 1, "k0002" into 2, etc.
def k_to_int(k: str):
    return int(k[1:])


# Turns 1 into "k0001", 2 into "k0002", etc.
def int_to_k(i: int, digits: int = 4):
    return f"k{i:0{digits}d}"


# Turns "rp01" into 1, "rp02" into 2, etc.
def rp_to_int(rp: str):
    return int(rp[2:])


# Turns 1 into "rp01", 2 into "rp02", etc.
def int_to_rp(i: int, digits: int = 2):
    return f"rp{i:0{digits}d}"


# Decorator to check that function has not been executed and add it to executionSafetyList
def addExecutionLog(func):
    @functools.wraps(func)  # Preserve the original function's name and docstring
    def wrapper(*args, **kwargs):
        # Check that function has not already been executed and add it to dictionary
        execution_safety_list = args[0]._executionSafetyList
        fullFuncName = func.__module__ + '.' + func.__name__
        if fullFuncName not in execution_safety_list:
            execution_safety_list.append(fullFuncName)  # Set the function's key to True
            print(f"Function {fullFuncName} has been executed, current execution safety: {execution_safety_list}")
        else:
            raise RuntimeError(f"Function {fullFuncName} has already been executed, current execution safety: {execution_safety_list}")

        # Call the function
        func(*args, **kwargs)

    return wrapper


# Decorator to check that all required functions have been executed before executing the function
# Also checks that the function has not already been executed
# required_functions: List of function names that need to have been executed before this function (without the file path)
def checkExecutionLog(required_functions: list[typing.Callable]):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check if all required functions have been executed
            execution_safety_list = args[0]._executionSafetyList
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
                raise RuntimeError(f"Function {fullFuncName} has already been executed, current execution safety: {execution_safety_list}")
            else:
                execution_safety_list.append(fullFuncName)  # Set the function's key to True
                print(f"Function {fullFuncName} has been executed, current execution safety: {execution_safety_list}")

                # Call the function
                func(*args, **kwargs)

        return wrapper

    return decorator
