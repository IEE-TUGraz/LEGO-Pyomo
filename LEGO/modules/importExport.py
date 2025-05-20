import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy):
    # Sets
    model.hubs = pyo.Set(doc='Import/Export hubs', initialize=cs.dPower_ImpExpHubs.index.unique(level=0))
    model.hubConnections = pyo.Set(doc='Nodes connected to hub', initialize=cs.dPower_ImpExpHubs.index, within=model.hubs * model.i)

    # Parameters
    model.pHubConnectionPmaxImport = pyo.Param(model.hubConnections, doc='Maximum import power at hub connection', initialize=cs.dPower_ImpExpHubs['Pmax Import'])
    model.pHubConnectionPmaxExport = pyo.Param(model.hubConnections, doc='Maximum export power at hub connection', initialize=cs.dPower_ImpExpHubs['Pmax Export'])
    model.pImpExpPrice = pyo.Param(model.rp, model.k, model.hubs, doc='Imp-/Export price at hub', initialize=cs.dPower_ImpExpProfiles['Price'].reorder_levels(["rp", "k", "hub"]))
    model.pImpExpValue = pyo.Param(model.rp, model.k, model.hubs, doc='Imp-/Export value at hub', initialize=cs.dPower_ImpExpProfiles['ImpExp'].reorder_levels(["rp", "k", "hub"]))
    model.pImpType = pyo.Param(model.hubs, doc='Import type at hub', initialize=cs.dPower_ImpExpHubs['Import Type'].groupby('hub').first(), within=pyo.Any)
    model.pExpType = pyo.Param(model.hubs, doc='Export type at hub', initialize=cs.dPower_ImpExpHubs['Export Type'].groupby('hub').first(), within=pyo.Any)

    # Variables
    def vImpExp_bounds(model, rp, k, hub, i):
        if model.pImpExpValue[rp, k, hub] >= 0:  # Only importing allowed
            return 0, min(model.pImpExpValue[rp, k, hub], model.pHubConnectionPmaxImport[hub, i])
        else:  # Only exporting allowed
            return max(model.pImpExpValue[rp, k, hub], -model.pHubConnectionPmaxExport[hub, i]), 0

    model.vImpExp = pyo.Var(model.rp, model.k, model.hubConnections, doc='Import/Export at hub connection', bounds=vImpExp_bounds)


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    # Enforce ImpFix/ImpMax and ExpFix/ExpMax
    def eImpExp_rule(model, rp, k, hub):
        if model.pImpExpValue[rp, k, hub] >= 0:  # Only importing allowed
            if model.pImpType[hub] == 'ImpFix':
                return sum(model.vImpExp[rp, k, hub, i] for i in model.i if (hub, i) in model.hubConnections) == model.pImpExpValue[rp, k, hub]
            elif model.pImpType[hub] == 'ImpMax':
                return sum(model.vImpExp[rp, k, hub, i] for i in model.i if (hub, i) in model.hubConnections) <= model.pImpExpValue[rp, k, hub]
            else:
                raise ValueError(f"Unknown import type for hub '{hub}': '{model.pImpType[hub]}'")
        else:  # Only exporting allowed
            if model.pExpType[hub] == 'ExpFix':
                return sum(model.vImpExp[rp, k, hub, i] for i in model.i if (hub, i) in model.hubConnections) == model.pImpExpValue[rp, k, hub]
            elif model.pExpType[hub] == 'ExpMax':
                return sum(model.vImpExp[rp, k, hub, i] for i in model.i if (hub, i) in model.hubConnections) >= model.pImpExpValue[rp, k, hub]
            else:
                raise ValueError(f"Unknown export type for hub '{hub}': '{model.pExpType[hub]}'")

    model.eImpExp = pyo.Constraint(model.rp, model.k, model.hubs, doc='Imp-/Export sum at each hub', rule=eImpExp_rule)

    # Add import/export to power-balance of each node in each hour
    for rp in model.rp:
        for k in model.k:
            for hub, i in model.hubConnections:
                model.eDC_BalanceP_expr[rp, k, i] += model.vImpExp[rp, k, hub, i]

    # Add import/export cost/revenues to total cost
    model.objective.expr += sum(model.vImpExp[rp, k, hub, i] * model.pImpExpPrice[rp, k, hub] for rp in model.rp for k in model.k for hub, i in model.hubConnections)
