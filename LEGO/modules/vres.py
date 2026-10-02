import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGOUtilities, LEGO

printer = Printer.getInstance()


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.vresGenerators = pyo.Set(doc='Variable renewable energy sources', initialize=cs.dPower_VRES.index.tolist())
    LEGO.addToSet(model, "g", model.vresGenerators)
    LEGO.addToSet(model, "gi", cs.dPower_VRES.reset_index().set_index(['g', 'i']).index)
    model.pvGenerators = pyo.Set(doc='PV generators',initialize=cs.dPower_VRES.loc[cs.dPower_VRES['tec'] == 'Solar'].index.tolist())
    model.windGenerators = pyo.Set(doc='Wind generators',initialize=cs.dPower_VRES.loc[cs.dPower_VRES['tec'] == 'Wind'].index.tolist())

    # Parameters
    model.pCurtailmentLimit = pyo.Param( initialize=0.4,doc="Curtailment limit for PV generators")

    LEGO.addToParameter(model, "pOMVarCost", cs.dPower_VRES['OMVarCost'])
    LEGO.addToParameter(model, "pEnabInv", cs.dPower_VRES['EnableInvest'])
    LEGO.addToParameter(model, "pMaxInvest", cs.dPower_VRES['MaxInvest'])
    for g in model.g:
        if model.pEnabInv[g] * model.pMaxInvest[g] == 0:
            model.vGenInvest[g].fix(0)  # Ensure that max investment is exactly zero if investment is disabled

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
                capacityFactors.loc[(slice(None), slice(None), g)] = capacityFactors.loc[(slice(None), slice(None), g)].clip(upper=1.0)  # If inflows exceed maximum production, forced spillage occurs and we need to clip the values
                ror_with_spillage.append(g)
            dCapacityFactor = pd.concat([dCapacityFactor, capacityFactors], axis=0)
        elif g not in cs.dPower_VRESProfiles.index.get_level_values("g"):
            raise ValueError(f"Generator '{g}' does not have VRES profiles or inflows defined - please provide one of them.")

    if len(ror_with_spillage):
        printer.warning(f"The following generators have inflows that exceed maximum production - it got capped to 1: {ror_with_spillage}")
    model.pCapacityFactors = pyo.Param(model.rp, model.k, model.vresGenerators, initialize=dCapacityFactor, doc="Capacity factor of VRES generators (from VRES profiles and inflows)")

    # Variables
    model.vCurtailment = pyo.Var(model.rp, model.k, model.vresGenerators, doc="Curtailment of PV generators", bounds=(0, None))
    second_stage_variables.append(model.vCurtailment)

    # Pre-compute base capacity per generator and maximum curtailment
    for g in model.vresGenerators:
        base_cap = model.pMaxProd[g] * (model.pExisUnits[g] + model.pMaxInvest[g] * model.pEnabInv[g])
        for rp in model.rp:
            for k in model.k:
                maximumProduction = base_cap * model.pCapacityFactors[rp, k, g]
                model.vGenP[rp, k, g].setub(maximumProduction)
                if g in model.pvGenerators:
                    model.vCurtailment[rp, k, g].setub(maximumProduction)


    # NOTE: Return both first and second stage variables as a safety measure - only the first_stage_variables will actually be returned (rest will be removed by the decorator)
    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    def eReMaxProd_rule(model, rp, k, r):
        available = model.pMaxProd[r] * (model.pExisUnits[r] + model.vGenInvest[r]) * model.pCapacityFactors[rp, k, r]
        if r in model.pvGenerators:
            return model.vGenP[rp, k, r] + model.vCurtailment[rp, k, r] == available
        return model.vGenP[rp, k, r] == available  # non-PV: must-take, no curtailment

    def eMaxCurtailment_rule(model, rp, k, r):
        if r in model.pvGenerators:
            maximum_Curtailment = model.pMaxProd[r] * (model.pExisUnits[r] + model.vGenInvest[r]) * model.pCurtailmentLimit
            return model.vCurtailment[rp, k, r] <= maximum_Curtailment
        return pyo.Constraint.Skip  # Skip for non-PV generators

    def ePeakshaving_rule(model, rp, k, r):
        # Only the share of available power above (1 - L) * installed capacity may be curtailed
        if r in model.pvGenerators:
            shaveable_share = max(0.0, pyo.value(model.pCapacityFactors[rp, k, r]) - (1 - pyo.value(model.pCurtailmentLimit)))
            installed_cap = model.pMaxProd[r] * (model.pExisUnits[r] + model.vGenInvest[r])
            if shaveable_share == 0:
                return model.vCurtailment[rp, k, r] == 0
            return model.vCurtailment[rp, k, r] <= installed_cap * shaveable_share
        return pyo.Constraint.Skip  # Skip for non-PV generators

    model.eReMaxProd = pyo.Constraint(model.rp, model.constraintsActiveK, model.vresGenerators, rule=eReMaxProd_rule)

    model.eReMaxProd = pyo.Constraint(model.rp, model.constraintsActiveK, model.vresGenerators, rule=eReMaxProd_rule)
    model.eMaxCurtailment = pyo.Constraint(model.rp, model.constraintsActiveK, model.vresGenerators, rule=eMaxCurtailment_rule)
    model.ePeakshaving = pyo.Constraint(model.rp, model.constraintsActiveK, model.vresGenerators, rule=ePeakshaving_rule)
    if cs.dPower_Parameters["pEnableSOCP"]:
        model.eSOCP_QMaxOut_RES = pyo.Constraint(model.rp, model.constraintsActiveK, model.vresGenerators, doc="Max reactive power output of generator unit", rule=lambda m, rp, k, g: (m.vGenQ[rp, k, g] <= m.pMaxGenQ[g] * model.pCapacityFactors[rp, k, g]) if m.pMaxGenQ[g] != 0 and (m.pExisUnits[g] > 0 or m.pEnabInv[g] == 1) else pyo.Constraint.Skip)

    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = 0.0
    second_stage_objective = 0.0

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
