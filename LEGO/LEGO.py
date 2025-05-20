import copy
import time
import typing

import pyomo.environ as pyo
import pyomo.opt.results.results_
from pyomo.core import TransformationFactory

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO.modules import storage, power, secondReserve, importExport, softLineLoadLimits

printer = Printer.getInstance()


class LEGO:
    def __init__(self, cs: CaseStudy = None, model: pyo.Model = None, results=None):
        self.cs: CaseStudy = cs
        self.model: typing.Optional[pyo.Model] = model
        self.results: typing.Optional[pyomo.opt.results.results_.SolverResults] = results
        self.timings = {"model_building": -1.0, "model_solving": -1.0}

    def build_model(self, already_existing_ok=False) -> (pyo.Model, float):
        if not already_existing_ok and self.model is not None:
            raise RuntimeError("Model already exists, please set already_existing_ok to True if that's intentional")

        start_time = time.time()
        model = _build_model(self.cs)
        self.model = model

        stop_time = time.time()
        self.timings["model_building"] = stop_time - start_time
        self.timings["model_solving"] = -1.0
        self.results = None

        return self.model, self.timings["model_building"]

    # Returns the objective value of this model (either overall or within the zone of interest)
    def get_objective_value(self, zoi: bool):
        return get_objective_value(self.model, zoi)

    def solve_model(self, optimizer=None, already_solved_ok=False) -> {pyomo.opt.results.results_.SolverResults, float}:
        if not already_solved_ok and self.results is not None:
            raise RuntimeError("Model already solved, please set already_solved_ok to True if that's intentional")

        if optimizer is None:
            optimizer = pyo.SolverFactory("gurobi")

            if self.cs.dGlobal_Parameters["pEnableRMIP"]:
                TransformationFactory('core.relax_integer_vars').apply_to(self.model)  # Relaxes all integer variables to continuous variables

        start_time = time.time()
        results = optimizer.solve(self.model)
        stop_time = time.time()

        self.timings["model_solving"] = stop_time - start_time
        self.results = results

        return results, self.timings["model_solving"]

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


def get_objective_value(model: pyo.Model, zoi: bool):
    # This is calculated in any case to make sure that the objective is calculated correctly - please ALWAYS update this AND the calculation below whenever something changes in the objective function
    result_overall = (sum(pyo.value(model.pInterVarCost[g]) * sum(pyo.value(model.bUC[g, :, :])) +  # Fixed cost of thermal generators
                          pyo.value(model.pStartupCost[g]) * sum(pyo.value(model.bStartup[g, :, :])) +  # Startup cost of thermal generators
                          pyo.value(model.pSlopeVarCost[g]) * sum(pyo.value(model.vGenP[g, :, :])) for g in model.thermalGenerators) +  # Production cost of thermal generators
                      sum(pyo.value(model.pProductionCost[g]) * sum(pyo.value(model.vGenP[g, :, :])) for g in model.vresGenerators) +
                      sum(pyo.value(model.pProductionCost[g]) * sum(pyo.value(model.vGenP[g, :, :])) for g in model.rorGenerators) +
                      sum(pyo.value(model.pOMVarCost[g]) * sum(pyo.value(model.vGenP[g, :, :])) for g in model.storageUnits) +
                      (sum(pyo.value(model.vSlack_DemandNotServed[:, :, :])) + sum(pyo.value(model.vSlack_OverProduction[:, :, :]))) * pyo.value(model.pSlackPrice))

    if (abs(result_overall - pyo.value(model.objective)) / pyo.value(model.objective)) > 1e-12:
        raise RuntimeError(f"Check calculation of objective value, something is off: {result_overall} != {pyo.value(model.objective)}")
    if not zoi:
        return result_overall
    else:
        result_zoi = (sum(pyo.value(model.pInterVarCost[g]) * sum(pyo.value(model.bUC[g, :, :])) +  # Fixed cost of thermal generators
                          pyo.value(model.pStartupCost[g]) * sum(pyo.value(model.bStartup[g, :, :])) +  # Startup cost of thermal generators
                          pyo.value(model.pSlopeVarCost[g]) * sum(pyo.value(model.vGenP[g, :, :])) for g in (model.thermalGenerators & model.zoi_g)) +  # Production cost of thermal generators
                      #                                                                    0.0 for g in (model.thermalGenerators & model.zoi_g)) +  # Production cost of thermal generators -> Removed variable cost to test TODO: Discuss with Sonja & Diego
                      # sum(pyo.value(model.pProductionCost[g]) * sum(pyo.value(model.vGenP[g, :, :])) for g in model.vresGenerators) +
                      # sum(pyo.value(model.pProductionCost[g]) * sum(pyo.value(model.vGenP[g, :, :])) for g in model.rorGenerators) +
                      sum(sum(pyo.value(model.vSlack_DemandNotServed[:, :, i])) + sum(pyo.value(model.vSlack_OverProduction[:, :, i])) for i in model.zoi_i) * pyo.value(model.pSlackPrice))

        return result_zoi


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


def _build_model(cs: CaseStudy) -> pyo.ConcreteModel:
    """
    Builds a pyomo ConcreteModel based on the given CaseStudy object.
    :param cs: The CaseStudy object to build the model from
    :return: A pyomo ConcreteModel object
    """
    model = pyo.ConcreteModel()

    # Element definitions
    power.add_element_definitions_and_bounds(model, cs)
    if cs.dPower_Parameters["pEnableStorage"]:
        storage.add_element_definitions_and_bounds(model, cs)
    secondReserve.add_element_definitions_and_bounds(model, cs)
    if cs.dPower_Parameters["pEnablePowerImportExport"]:
        importExport.add_element_definitions_and_bounds(model, cs)
    if cs.dPower_Parameters["pEnableSoftLineLoadLimits"]:
        softLineLoadLimits.add_element_definitions_and_bounds(model, cs)

    # Helper Sets for zone of interest
    model.zoi_i = pyo.Set(doc="Buses in zone of interest", initialize=cs.dPower_BusInfo.loc[cs.dPower_BusInfo["zoi"] == 1].index.tolist(), within=model.i)
    model.zoi_g = pyo.Set(doc="Generators in zone of interest", initialize=[g for g in model.g for i in model.i if (g, i) in model.gi], within=model.g)

    # Add constraints
    power.add_constraints(model, cs)
    if cs.dPower_Parameters["pEnableStorage"]:
        storage.add_constraints(model, cs)
    secondReserve.add_constraints(model, cs)
    if cs.dPower_Parameters["pEnablePowerImportExport"]:
        importExport.add_constraints(model, cs)
    if cs.dPower_Parameters["pEnableSoftLineLoadLimits"]:
        softLineLoadLimits.add_constraints(model, cs)

    return model


def addToSet(model: pyo.ConcreteModel, set_name: str, values: iter) -> None:
    """
    Adds values to a set in the model. If the set does not exist, it raises an error.
    :param model: The model to which the set belongs
    :param set_name: Name of the set to add values to
    :param values: Values to add to the set
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
    :param model: The model to which the parameter belongs
    :param parameter_name: Name of the parameter to add or update
    :param values: Values to add or update in the parameter
    :param doc: Documentation string for the parameter
    :param indices: Indices for the parameter
    :param overwrite: If True, it overwrites existing values in the parameter
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
