import typing

import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGOUtilities, LEGO

printer = Printer.getInstance()


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> typing.Tuple[list[pyo.Var], list[pyo.Var]]:
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.vresGenerators = pyo.Set(doc='Variable renewable energy sources', initialize=cs.dPower_VRES.index.tolist())
    LEGO.addToSet(model, "g", model.vresGenerators)
    LEGO.addToSet(model, "gi", cs.dPower_VRES.reset_index().set_index(['g', 'i']).index)
    LEGO.addToSet(model, "tec", cs.dPower_VRES['tec'].unique().tolist())
    LEGO.addToSet(model, "gtec", cs.dPower_VRES.reset_index().set_index(['g', 'tec']).index)

    # Parameters
    LEGO.addToParameter(model, "pOMVarCost", cs.dPower_VRES['OMVarCost'])
    LEGO.addToParameter(model, "pEnabInv", cs.dPower_VRES['EnableInvest'])
    LEGO.addToParameter(model, "pMaxInvest", cs.dPower_VRES['MaxInvest'])
    LEGO.addToParameter(model, "pInvestCost", cs.dPower_VRES['InvestCostEUR'])
    LEGO.addToParameter(model, "pMaxProd", cs.dPower_VRES['MaxProd'])
    LEGO.addToParameter(model, "pMinProd", cs.dPower_VRES['MinProd'])
    LEGO.addToParameter(model, "pExisUnits", cs.dPower_VRES['ExisUnits'])

    LEGO.addToParameter(model, 'pMaxGenQ', cs.dPower_VRES['Qmax'])
    LEGO.addToParameter(model, 'pMinGenQ', cs.dPower_VRES['Qmin'])

    dCapacityFactor = cs.dPower_VRESProfiles["value"]
    ror_with_spillage = []  # List of ror generators that have spillage (i.e., inflow > maximum production)
    for g in model.vresGenerators:
        if g in cs.dPower_Inflows.index.get_level_values("g"):
            if g in cs.dPower_VRESProfiles.index.get_level_values("g"):
                raise ValueError(f"Generator '{g}' has both VRES profiles and inflows defined - please provide only one of them.")

            capacityFactors = cs.dPower_Inflows.loc[(slice(None), slice(None), g), 'value'] / model.pMaxProd[g]
            if capacityFactors.max() > 1.0:
                capacityFactors.loc[(slice(None), slice(None), g), 'value'] = capacityFactors.loc[(slice(None), slice(None), g), 'value'].clip(upper=1.0)  # If inflows exceed maximum production, forced spillage occurs and we need to clip the values
                ror_with_spillage.append(g)
            dCapacityFactor = pd.concat([dCapacityFactor, capacityFactors], axis=0)
        elif g not in cs.dPower_VRESProfiles.index.get_level_values("g"):
            raise ValueError(f"Generator '{g}' does not have VRES profiles or inflows defined - please provide one of them.")

    if len(ror_with_spillage):
        printer.warning(f"The following generators have inflows that exceed maximum production - it got capped to 1: {ror_with_spillage}")
    model.pCapacityFactors = pyo.Param(model.rp, model.k, model.vresGenerators, initialize=dCapacityFactor, doc="Capacity factor of VRES generators (from VRES profiles and inflows)")

    # Variables
    model.vCurtailment = pyo.Var(model.rp, model.k, model.vresGenerators, doc="Curtailment of VRES generators", bounds=(0, None))
    second_stage_variables.append(model.vCurtailment)

    for g in model.vresGenerators:
        for rp in model.rp:
            for k in model.k:
                maximumProduction = model.pMaxProd[g] * (model.pExisUnits[g] + (model.pMaxInvest[g] * model.pEnabInv[g])) * model.pCapacityFactors[rp, k, g]
                model.vCurtailment[rp, k, g].setub(maximumProduction)
                model.vGenP[rp, k, g].setub(maximumProduction)

    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    def eReMaxProd_rule(model, rp, k, r):
        return model.vGenP[rp, k, r] + model.vCurtailment[rp, k, r] == model.pMaxProd[r] * (model.pExisUnits[r] + model.vGenInvest[r]) * model.pCapacityFactors[rp, k, r]

    model.eReMaxProd = pyo.Constraint(model.rp, model.k, model.vresGenerators, rule=eReMaxProd_rule)

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = 0.0

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
