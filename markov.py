import logging
import time

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints

from LEGO.CaseStudy import CaseStudy
from LEGO.LEGO import LEGO
from tools.printer import Printer

########################################################################################################################
# Setup
########################################################################################################################

printer = Printer.getInstance()
printer.console.width = 180

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)

########################################################################################################################
# Data input from case study
########################################################################################################################

# Load case study from Excels
printer.information(f"Loading case study from '{"data/markov/"}'")
start_time = time.time()
cs_notEnforced = CaseStudy("data/markov/")
printer.information(f"Loading case study took {time.time() - start_time:.2f} seconds")

# Create varied case studies
start_time = time.time()
cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "notEnforced"
cs_notEnforced.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "notEnforced"

cs_cyclic = cs_notEnforced.copy()
cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "cyclic"
cs_cyclic.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "cyclic"

cs_markov = cs_notEnforced.copy()
cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingUnitCommitment"] = "markov"
cs_markov.dPower_Parameters["pReprPeriodEdgeHandlingRamping"] = "markov"

lego_models = [("NoEnf.", LEGO(cs_notEnforced)), ("Cyclic", LEGO(cs_cyclic)), ("Markov", LEGO(cs_markov))]
printer.information(f"Creating varied case studies took {time.time() - start_time:.2f} seconds")

########################################################################################################################
# Evaluation
########################################################################################################################

optimizer = SolverFactory("gurobi")

results = []

df = pd.DataFrame()
for caseName, lego in lego_models:
    printer.information(f"\n\n{'=' * 60}\n{caseName}\n{'=' * 60}")

    model, timing_building = lego.build_model()
    printer.information(f"Building model took {timing_building:.2f} seconds")

    result, timing_solving = lego.solve_model(optimizer)
    printer.information(f"Solving model took {timing_solving:.2f} seconds")

    match result.solver.termination_condition:
        case pyo.TerminationCondition.optimal:
            printer.success("Optimal solution found")
        case pyo.TerminationCondition.infeasible | pyo.TerminationCondition.unbounded:
            printer.error(f"Model is {result.solver.termination_condition}, logging infeasible constraints:")
            log_infeasible_constraints(model)
        case _:
            printer.warning("Solver terminated with condition:", result.solver.termination_condition)

    # Count binary variables within all variables
    variables = list(model.component_objects(pyo.Var))
    counter_binaries = 0
    for v in variables:
        indices = [i for i in v]
        for i in indices:
            if v[i].domain == pyo.Binary:
                counter_binaries += 1

    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        for x in [pd.Series({"case": caseName, "rp": i[0], "k": i[1], "g": i[2], "vCommit": pyo.value(model.vCommit[i]), "vGenP1": pyo.value(model.vGenP1[i])}) for i in list(model.vCommit)]:
            df = pd.concat([df, x], axis=1)

    results.append({
        "Case": caseName,
        "Objective": pyo.value(model.objective) if result.solver.termination_condition == pyo.TerminationCondition.optimal else -1,
        "Solution": result.solver.termination_condition,
        "Build Time": timing_building,
        "Solve Time": timing_solving,
        "# Variables Overall": model.nvariables(),
        "# Binary Variables": counter_binaries,
        "# Constraints": model.nconstraints(),
        "PNS": sum(model.vPNS[rp, k, i].value if model.vPNS[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i),
        "EPS": sum(model.vEPS[rp, k, i].value if model.vEPS[rp, k, i].value is not None else 0 for rp in model.rp for k in model.k for i in model.i)
    })

    print(df.head())

printer.information("Case   |  Objective  | Solution | Build Time | Solve Time | # Variables Overall | # Binary Variables | # Constraints | PNS     | EPS    |")
for result in results:
    printer.information(f"{result['Case']} | {result['Objective']:11.2f} | {result['Solution']}  | {result['Build Time']:10.2f} | {result['Solve Time']:10.2f} | {result['# Variables Overall']:>19} | {result['# Binary Variables']:>18} | {result['# Constraints']:>13} | {result['PNS']:>7.2f} | {result['EPS']:>7.2f}")

df.T.to_excel("markov.xlsx")

printer.success("Done")
