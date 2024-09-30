from pyomo import environ as pyo
from rich.console import Console


class Printer:
    __instance = None

    @staticmethod
    def getInstance():
        if Printer.__instance is None:
            Printer(Console(width=80))
        return Printer.__instance

    def __init__(self, console):
        if Printer.__instance is not None:
            raise Exception("Printer is a singleton but got initialized twice")

        Printer.__instance = self
        self.console = console

    # Definition of standard output for errors
    def error(self, text: str, prefix="Error: "):
        self.console.print(f"[red]{prefix}{text}[/red]")
        return None

    # Definition of standard output for warnings
    def warning(self, text: str, prefix="Warning: "):
        self.console.print(f"[yellow]{prefix}{text}[/yellow]")
        return None

    # Definition of standard output for notes
    def note(self, text: str, prefix="Note: "):
        self.console.print(f"[yellow]{prefix}{text}[/yellow]")
        return None

    # Definition of standard output for success
    def success(self, text="", prefix=""):
        self.console.print(f"[green]{prefix}{text}[/green]")
        return None

    # Definition of standard output for information
    def information(self, text: str, prefix=""):
        self.console.print(f"{prefix}{text}")
        return None


# Helper function to pretty-print the values of a Pyomo indexed variable within zone of interest
def pprint_var(var, zoi, index_positions: list = None, decimals: int = 2):
    if index_positions is None:
        index_positions = [0]

    key_list = ["Key"]
    lower_list = ["Lower"]
    value_list = ["Value"]
    upper_list = ["Upper"]
    fixed_list = ["Fixed"]
    stale_list = ["Stale"]
    domain_list = ["Domain"]

    for index in var:
        # check if at least one index is in zone of interest
        if not any(i in zoi for i in index):
            continue
        key_list.append(str(index))
        lower_list.append(f"{var[index].lb:.2f}" if var[index].has_lb() else str(var[index].lb))
        value_list.append(f"{pyo.value(var[index]):.2f}" if not var[index].value is None else str(var[index].value))
        upper_list.append(f"{var[index].ub:.2f}" if var[index].has_ub() else str(var[index].ub))
        fixed_list.append(str(var[index].fixed))
        stale_list.append(str(var[index].stale))
        domain_list.append(str(var[index].domain.name))

    key_spacer = len(max(key_list, key=len))
    lower_spacer = len(max(lower_list, key=len))
    value_spacer = len(max(value_list, key=len))
    upper_spacer = len(max(upper_list, key=len))
    fixed_spacer = len(max(fixed_list, key=len))
    stale_spacer = len(max(stale_list, key=len))
    domain_spacer = len(max(domain_list, key=len))

    print(f"{var.name} : {var.doc}")
    print(f"    Size={len(var)}, In Zone of Interest={len(key_list) - 1}, Index={var.index_set()}")

    # Iterate over all lists and print the values
    for i in range(len(value_list)):
        print(f"    {key_list[i]:>{key_spacer}} : {lower_list[i]:>{lower_spacer}} : {value_list[i]:>{value_spacer}} : {upper_list[i]:>{upper_spacer}} : {fixed_list[i]:>{fixed_spacer}} : {stale_list[i]:>{stale_spacer}} : {domain_list[i]:>{domain_spacer}}")
