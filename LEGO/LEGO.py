import contextlib
import copy
import enum
import logging
import time
import typing

import pyomo.environ as pyo
import pyomo.opt.results.results_
from pyomo.core import TransformationFactory

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.LEGOUtilities import reset_execution_safety_dict, set_range_non_cyclic
from LEGO.modules import storage, power, secondReserve, importExport, softLineLoadLimits, thermalGen, vres, selfSufficiency, heat

printer = Printer.getInstance()


def mpi_rank() -> int:
    """This process's MPI rank, or 0 when not launched under mpiexec.

    mpi-sppy already initialises mpi4py; a plain `python` process has a world
    communicator of size 1 (rank 0), so every rank-gated branch runs normally in
    the non-MPI case. Under `mpiexec -n K` the decomposition (subproblem build +
    solve) is split across ranks, but only rank 0 holds the master solution."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except Exception:
        return 0


@contextlib.contextmanager
def _suppress_relaxed_integer_warning():
    """Silence Pyomo's harmless 'Implicitly replacing ... _relaxed_integer_vars'
    warning emitted when mpi-sppy's L-shaped relaxes subproblems that _build_model
    (with pEnableRMIP) already relaxed. Filters ONLY that exact message on the
    'pyomo.core' logger, so genuine 'implicitly replacing' warnings for other
    components still surface."""
    pyomo_logger = logging.getLogger("pyomo.core")

    def _drop_relaxed_integer(record):
        return "_relaxed_integer_vars" not in record.getMessage()

    pyomo_logger.addFilter(_drop_relaxed_integer)
    try:
        yield
    finally:
        pyomo_logger.removeFilter(_drop_relaxed_integer)


class ModelType(enum.Enum):
    """
    Enum-like class to represent the type of model.
    """
    DETERMINISTIC = "deterministic"
    EXTENSIVE_FORM = "extensive_form"
    BENDERS = "benders"
    PROGRESSIVE_HEDGING = "progressive_hedging"


class LEGO:
    def __init__(self, cs: CaseStudy = None, model: pyo.Model = None, results=None):
        self.cs: CaseStudy = cs
        self.model: typing.Optional[pyo.Model] = model
        self.results: typing.Optional[pyomo.opt.results.results_.SolverResults] = results
        self.timings = {"model_building": -1.0, "model_solving": -1.0}
        self.solver_name = None
        self._extensive_form = None  # Used for the ExtensiveForm model type, not used in other model types
        self.scenario_models = None  # {scenario_name: solved sub-problem model}; populated for BENDERS

    def build_model(self, already_existing_ok: bool = False, model_type: ModelType = ModelType.DETERMINISTIC, solver_name: str = None) -> (pyo.Model, float):
        if not already_existing_ok and self.model is not None:
            raise RuntimeError("Model already exists, please set already_existing_ok to True if that's intentional")

        start_time = time.time()
        match model_type:
            case ModelType.DETERMINISTIC:
                if solver_name is not None:
                    printer.warning(f"Solver name {solver_name} provided for 'build_model', but not used when building deterministic model type. Make sure to provide it when solving the model instead.")
                model = _build_model(self.cs)
                self.model = model
            case ModelType.EXTENSIVE_FORM:
                from mpisppy.opt.ef import ExtensiveForm

                scenario_names = self.cs.dGlobal_Scenarios.index.tolist()

                if solver_name is None:
                    solver_name = self.cs.dGlobal_Parameters["pSolver"]  # Use the solver name from the case study parameters if not provided
                elif self.cs.dGlobal_Parameters["pSolver"] != solver_name:
                    printer.warning(f"Solver name {solver_name} does not match the one used in the case study ({self.cs.dGlobal_Parameters['pSolver']}) - using {solver_name}")
                options = {
                    "solver": solver_name,
                }

                ef = ExtensiveForm(options, scenario_names, _scenario_creator, scenario_creator_kwargs={"full_case_study": self.cs})
                self._extensive_form = ef
                self.model = ef.ef
            case ModelType.BENDERS | ModelType.PROGRESSIVE_HEDGING:
                raise RuntimeError(f"Model type {model_type} can not be built seperately, it is built using the 'solve_model' method")
            case _:
                raise RuntimeError(f"Model type {model_type} not implemented (yet?)")

        stop_time = time.time()
        self.timings["model_building"] = stop_time - start_time
        self.timings["model_solving"] = -1.0
        self.results = None
        self.solver_name = solver_name

        return self.model, self.timings["model_building"]

    def solve_model(self, model_type: ModelType = ModelType.DETERMINISTIC, solver_name: str = None, already_solved_ok=False) -> (pyomo.opt.results.results_.SolverResults, float, float):
        if not already_solved_ok and self.results is not None:
            raise RuntimeError("Model already solved, please set already_solved_ok to True if that's intentional")

        if solver_name is None:
            solver_name = self.cs.dGlobal_Parameters["pSolver"]  # Use the solver name from the case study parameters if not provided
        elif self.cs.dGlobal_Parameters["pSolver"] != solver_name:
            printer.warning(f"Solver name {solver_name} does not match the one used in the case study ({self.cs.dGlobal_Parameters['pSolver']}) - using {solver_name}")

        # Add suffixes to request dual values from solver (skip for decomposition methods that don't have a single root model)
        if self.model is not None and not hasattr(self.model, 'dual'):
            self.model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        start_time = time.time()
        self.work_units = None  # Initialize work_units
        match model_type:
            case ModelType.DETERMINISTIC:
                # Use persistent solver for Gurobi to access work units
                if solver_name.lower() in ['gurobi', 'gurobi_persistent']:
                    optimizer = pyo.SolverFactory('gurobi_persistent')
                    optimizer.set_instance(self.model)
                    results = optimizer.solve(tee=True, load_solutions=True)
                    objective_value = pyo.value(self.model.objective) if results.solver.termination_condition == pyo.TerminationCondition.optimal else -1
                    # Extract work units from Gurobi model
                    try:
                        self.work_units = optimizer._solver_model.Work
                    except Exception as e:
                        printer.warning(f"Could not extract work units from Gurobi: {e}")
                        self.work_units = None
                else:
                    optimizer = pyo.SolverFactory(solver_name)
                    results = optimizer.solve(self.model, tee=True, load_solutions=True)
                    objective_value = pyo.value(self.model.objective) if results.solver.termination_condition == pyo.TerminationCondition.optimal else -1
            case ModelType.EXTENSIVE_FORM:
                if solver_name != self.solver_name:
                    raise RuntimeError(f"Optimizer name {solver_name} does not match the one used to build the model ({self.solver_name}), please use the same optimizer name when solving using the 'ExtensiveForm' model type")
                start_time = time.time()

                solver_options = {
                    "BarHomogeneous": 1,  # robust homogeneous self-dual barrier (Gurobi's own suggestion)
                    "NumericFocus": 2,  # prioritize numerical stability (try 3 if it still breaks)
                    "Method": 2,  # barrier only — stop wasting cores on concurrent simplex
                    "ScaleFlag": 2,  # aggressive geometric scaling for the wide coefficient range
                    "Crossover": 0,  # return the interior solution directly, skip crossover
                }

                results = self._extensive_form.solve_extensive_form(solver_options= solver_options,tee=True)
                stop_time = time.time()
                objective_value = self._extensive_form.get_objective_value() if results.solver.termination_condition == pyo.TerminationCondition.optimal else -1

                # variables = self.model.get_root_solution()
                # for (var_name, var_val) in variables.items():
                # print(var_name, var_val)
            case ModelType.BENDERS:
                printer.warning("Benders decomposition NOT FULLY TESTED YET, MIGHT HAVE SOME ISSUES OR BUGS!")
                from mpisppy.opt.lshaped import LShapedMethod

                scenario_names = self.cs.dGlobal_Scenarios.index.tolist()
                # Supplying ANY valid_eta_lb flips mpi-sppy's compute_eta_bound to False, which
                # makes create_subproblem SKIP a standalone set_instance()+solve of every
                # subproblem (lshaped.py) — that duplicate persistent build of each large
                # subproblem was the dominant cost. We deliberately do NOT reuse a self-computed
                # standalone optimum here: mpi-sppy's eta_s is the *probability-weighted,
                # second-stage-only* cost (lshaped.py multiplies each 2nd-stage coef by
                # _mpisppy_probability), whereas our scenario objective is unweighted and includes
                # first-stage cost — so that number is not a valid eta bound and could cut off the
                # optimum. A sufficiently negative constant is always a valid lower bound (it can
                # never bind above the optimum, regardless of weighting); the only price is possibly
                # a few extra Benders iterations, which are cheap warm-started re-solves. Make it
                # more negative if any scenario's weighted recourse cost could fall below this.
                LOOSE_ETA_LB = -1e3
                eta_lb = {s: LOOSE_ETA_LB for s in scenario_names}
                options = {
                    "root_solver": "gurobi_persistent",
                    "sp_solver": "gurobi_persistent",
                    # Sub-problems are large pure LPs (~315k rows each). Barrier (Gurobi's default
                    # concurrent solve) is far faster on them than simplex from scratch. Dual
                    # simplex would warm-start in principle, but mpi-sppy adds/removes the
                    # first-stage-fixing constraints every iteration (benders_cuts.py), so the basis
                    # cannot carry over and we'd pay a cold simplex start each time (~3x slower here).
                    # Default crossover still yields exact vertex duals for valid cuts.
                    "sp_solver_options": {"Threads": 14},
                    "tol": 1e-6,  # default is 1e-8 — too tight; causes non-improving cuts to keep firing
                    "max_iter": 200,
                    "valid_eta_lb": eta_lb,  # -> compute_eta_bound=False -> skips the duplicate per-subproblem build
                }
                start_time = time.time()
                with _suppress_relaxed_integer_warning():
                    ls = LShapedMethod(options, scenario_names, _scenario_creator,
                                       scenario_creator_kwargs={"full_case_study": self.cs})
                    results = ls.lshaped_algorithm()
                stop_time = time.time()

                # Keep the solved per-scenario sub-problem models for result export. mpi-sppy
                # splits the scenarios across ranks, so each rank holds only its local ones;
                # after convergence each model carries the full solution (first-stage fixed to
                # the optimum + second-stage optimal, with variable values loaded).
                self.scenario_models = dict(ls.local_scenarios)

                # variables = ls.gather_var_values_to_rank0()
                # for ((scen_name, var_name), var_value) in variables.items():
                #   print(scen_name, var_name, var_value)

                # Under mpiexec only rank 0 (the master) gets a populated SolverResults from
                # lshaped_algorithm(); worker ranks return None (they only solved their
                # subproblems), so the bound/objective extraction must be rank-0 only.
                if mpi_rank() == 0:
                    lower_bound = results.json_repn()['Problem'][0]['Lower bound']
                    upper_bound = results.json_repn()['Problem'][0]['Upper bound']
                    spread = upper_bound - lower_bound
                    rel_gap_pct = (spread / upper_bound * 100) if upper_bound not in (0, None) else float('nan')
                    printer.warning(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}, spread: {spread:.2f} | {rel_gap_pct:.2f}%)")

                    # LShapedMethod has no `.objective` attribute; use the best feasible (upper) bound from the results object.
                    # For a minimization problem this is the best incumbent objective found; at convergence it equals the lower bound.
                    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                        objective_value = upper_bound
                        printer.warning("Reporting upper bound (best feasible) as objective value for BENDERS.")
                    else:
                        objective_value = -1
                else:
                    objective_value = -1

            case ModelType.PROGRESSIVE_HEDGING:
                printer.warning("Progressive Hedging NOT FULLY TESTED YET, MIGHT HAVE SOME ISSUES OR BUGS!")
                from mpisppy.opt.ph import PH

                scenario_names = self.cs.dGlobal_Scenarios.index.tolist()
                options = {
                    "solver_name": solver_name,
                    "PHIterLimit": 50,
                    "defaultPHrho": 10,
                    "convthresh": 1e-7,
                    "verbose": False,
                    "display_progress": True,
                    "display_timing": True,
                    "iter0_solver_options": dict(),
                    "iterk_solver_options": dict(),
                }
                start_time = time.time()
                ph = PH(options, scenario_names, _scenario_creator, scenario_creator_kwargs={"full_case_study": self.cs})
                results = ph.ph_main()
                stop_time = time.time()

                objective_value = pyo.value(ph.objective) if results.solver.termination_condition == pyo.TerminationCondition.optimal else -1

                # variables = ph.gather_var_values_to_rank0()
                # for ((scen_name, var_name), var_value) in variables.items():
                #   print(scen_name, var_name, var_value)
            case _:
                raise RuntimeError(f"Model type {model_type} not implemented yet")

        stop_time = time.time()

        self.timings["model_solving"] = stop_time - start_time
        self.results = results

        eps = 1e-5
        try:
            total_PNS = sum(pyo.value(self.model.vPNS[rp, k, i]) for rp in self.model.rp for k in self.model.k for i in self.model.i)
            total_EPS = sum(pyo.value(self.model.vEPS[rp, k, i]) for rp in self.model.rp for k in self.model.k for i in self.model.i)
            if total_PNS > eps:
                printer.warning(f"Power not supplied value {total_PNS} exceeds threshold {eps}")
            if total_EPS > eps:
                printer.warning(f"Excess power supplied value {total_EPS} exceeds threshold {eps}")
        except Exception as e:
            printer.warning(f"Could not check slack variables automatically: {e}")

        return results, self.timings["model_solving"], objective_value

    def get_number_of_variables(self, dont_multiply_by_indices=False) -> int:
        # Check if pyomo-implementation is the same as this "manual" one
        assert self.model.nvariables() == len(list(self.model.component_objects(pyo.Var, active=True))), "Check implementation of lego.get_number_of_variables()"

        if dont_multiply_by_indices:  # Only count the number of variables, not multiplied by the number of indices
            return len(list(self.model.component_objects(pyo.Var, active=True)))
        else:  # Iterate through variables and sum up each individual variable
            return sum([len(x) for x in self.model.component_objects(pyo.Var, active=True)])
        pass

    def get_number_of_constraints(self, dont_multiply_by_indices=False) -> int:
        # Check if pyomo-implementation is the same as this "manual" one
        assert self.model.nconstraints() == len(list(self.model.component_objects(pyo.Constraint, active=True))), "Check implementation of lego.get_number_of_constraints()"

        if dont_multiply_by_indices:  # Only count the number of constraints, not multiplied by the number of indices
            return len(list(self.model.component_objects(pyo.Constraint, active=True)))
        else:  # Iterate through constraints and sum up each individual constraint
            return sum([len(x) for x in self.model.component_objects(pyo.Constraint, active=True)])

    def copy(self):
        return copy.deepcopy(self)


# Clone given model and fix specified variables to values from another model
def build_from_clone_with_fixed_results(model_to_be_cloned: pyo.Model, model_with_fixed_results: pyo.Model, variables_to_fix: list[str]) -> LEGO:
    model_new = model_to_be_cloned.clone()

    # Fix variables to values from model_with_fixed_results
    for var_name in variables_to_fix:
        var = getattr(model_with_fixed_results, var_name)
        new_var = getattr(model_new, var_name)
        for index in var:
            new_var[index].fix(pyo.value(var[index].value))

    return LEGO(model=model_new)


def _scenario_creator(scenario_name: str, full_case_study: CaseStudy) -> pyo.ConcreteModel:
    """
    Creates a scenario based on the given scenario name. Used for mpi-sppy.
    :param scenario_name: Name of the scenario to create.
    :return: A pyomo ConcreteModel object for the given scenario.
    """
    import mpisppy.utils.sputils as sputils

    model = _build_model(full_case_study.filter_scenario(scenario_name))
    sputils.attach_root_node(model, model.first_stage_objective, model.first_stage_varlist)
    # in _scenario_creator — correct probability (was inverted)
    model._mpisppy_probability = (
            full_case_study.dGlobal_Scenarios.loc[scenario_name, "relativeWeight"]
            / full_case_study.dGlobal_Scenarios.loc[:, "relativeWeight"].sum()
    )
    return model


def _build_model(cs: CaseStudy) -> pyo.ConcreteModel:
    """
    Builds a pyomo ConcreteModel based on the given CaseStudy object.
    :param cs: The CaseStudy object to build the model from.
    :return: A pyomo ConcreteModel object.
    """
    model = pyo.ConcreteModel()
    reset_execution_safety_dict(model)  # Reset the execution safety dict to ensure that decorators work correctly
    model.objective = pyo.Objective(doc='Total production cost (Objective Function)', sense=pyo.minimize, expr=0.0)  # Initialize objective function

    # Initialize first_stage variables and objective required for stochasticity.
    # Note: add_element_definitions_and_bounds is wrapped by
    # safetyCheck_AddElementDefinitionsAndBounds, which validates that every new variable is
    # assigned to exactly one stage and returns ONLY the first-stage variables, so `+=` here
    # correctly extends first_stage_varlist with a flat list of the nonanticipative vars.
    model.first_stage_varlist = []
    model.first_stage_objective = 0.0

    # Element definitions
    model.first_stage_varlist += power.add_element_definitions_and_bounds(model, cs)
    if cs.dPower_Parameters["pEnableThermalGen"]:
        model.first_stage_varlist += thermalGen.add_element_definitions_and_bounds(model, cs)
    if cs.dPower_Parameters["pEnableVRES"]:
        model.first_stage_varlist += vres.add_element_definitions_and_bounds(model, cs)
    if cs.dPower_Parameters["pEnableStorage"]:
        model.first_stage_varlist += storage.add_element_definitions_and_bounds(model, cs)

    if cs.dPower_Parameters["p2ndResUp"] > 0.0 or cs.dPower_Parameters["p2ndResDW"] > 0.0:
        model.first_stage_varlist += secondReserve.add_element_definitions_and_bounds(model, cs)

    if cs.dPower_Parameters["pEnablePowerImportExport"]:
        model.first_stage_varlist += importExport.add_element_definitions_and_bounds(model, cs)
    if cs.dPower_Parameters["pEnableSoftLineLoadLimits"]:
        model.first_stage_varlist += softLineLoadLimits.add_element_definitions_and_bounds(model, cs)

    if cs.dGlobal_Parameters["pEnableSelfSufficiency"]:
        model.first_stage_varlist += selfSufficiency.add_element_definitions_and_bounds(model, cs)

    if cs.dGlobal_Parameters["pEnableHeat"]:
        model.first_stage_varlist += heat.add_element_definitions_and_bounds(model, cs)

    # Helper Sets for zone of interest
    model.zoi_i = pyo.Set(doc="Buses in zone of interest", initialize=cs.dPower_BusInfo.loc[cs.dPower_BusInfo["zoi"] == 1].index.tolist(), within=model.i)

    # Add constraints
    model.first_stage_objective += power.add_constraints(model, cs)
    if cs.dPower_Parameters["pEnableThermalGen"]:
        model.first_stage_objective += thermalGen.add_constraints(model, cs)
    if cs.dPower_Parameters["pEnableVRES"]:
        model.first_stage_objective += vres.add_constraints(model, cs)
    if cs.dPower_Parameters["pEnableStorage"]:
        model.first_stage_objective += storage.add_constraints(model, cs)

    if cs.dGlobal_Parameters["pEnableSelfSufficiency"]:
        model.first_stage_objective += selfSufficiency.add_constraints(model, cs)

    if cs.dPower_Parameters["p2ndResUp"] > 0.0 or cs.dPower_Parameters["p2ndResDW"] > 0.0:
        model.first_stage_objective += secondReserve.add_constraints(model, cs)

    if cs.dPower_Parameters["pEnablePowerImportExport"]:
        model.first_stage_objective += importExport.add_constraints(model, cs)
    if cs.dPower_Parameters["pEnableSoftLineLoadLimits"]:
        model.first_stage_objective += softLineLoadLimits.add_constraints(model, cs)

    if cs.dGlobal_Parameters["pEnableHeat"]:
        model.first_stage_objective += heat.add_constraints(model, cs)

    if cs.dGlobal_Parameters["pEnableRMIP"]:
        TransformationFactory('core.relax_integer_vars').apply_to(model)  # Relaxes all integer variables to continuous variables

    return model


def addToSet(model: pyo.ConcreteModel, set_name: str, values: iter) -> None:
    """
    Adds values to a set in the model. If the set does not exist, it raises an error.
    :param model: The model to which the set belongs.
    :param set_name: Name of the set to add values to.
    :param values: Values to add to the set.
    :return: None
    """
    if not hasattr(model, set_name):
        raise RuntimeError(f"Set {set_name} does not exist in model, please add it first")
    else:
        for i in values:
            model.component(set_name).add(i)


def addToParameter(model: pyo.ConcreteModel, parameter_name: str, values: iter, doc: str = None, indices: list[object] = None, overwrite=False) -> None:
    """
    Adds values to a parameter in the model. If the parameter does not exist, it creates it.
    If the parameter exists, it updates the values.
    :param model: The model to which the parameter belongs.
    :param parameter_name: Name of the parameter to add or update.
    :param values: Values to add or update in the parameter.
    :param doc: Documentation string for the parameter.
    :param indices: Indices for the parameter.
    :param overwrite: If True, it overwrites existing values in the parameter.
    :return: None
    """
    if not hasattr(model, parameter_name):  # Check if parameter exists
        if not doc:
            raise RuntimeError(f"Parameter {parameter_name} does not exist in model, but no doc string was provided")
        elif not indices:
            raise RuntimeError(f"Parameter {parameter_name} does not exist in model, but no indices were provided")
        else:
            model.add_component(parameter_name, pyo.Param(*indices, initialize=values, doc=doc, domain=pyo.Reals))  # Add set which is not present yet
    else:
        current_values = model.component(parameter_name).extract_values()  # Get current values
        if not doc:
            doc = model.component(parameter_name).doc
        if not indices:
            indices = [model.component(parameter_name).index_set()]
        if not overwrite:  # Check if any value would be overwritten
            for k, v in values.items():
                if k in current_values.keys():
                    raise RuntimeError(f"Value for {k} already exists in parameter {parameter_name}, but overwrite=False")

        model.del_component(parameter_name)  # Delete parameter
        current_values.update(values)  # Update values with new values
        model.add_component(parameter_name, pyo.Param(*indices, initialize=current_values, doc=doc, domain=pyo.Reals))  # Add parameter as new parameter