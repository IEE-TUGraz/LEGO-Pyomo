import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(lego: LEGO):
    # Sets
    lego.model.hubs = pyo.Set(doc='Import/Export hubs', initialize=lego.cs.dPower_ImpExpHubs.index.unique(level=0))
    lego.model.hubConnections = pyo.Set(doc='Nodes connected to hub', initialize=lego.cs.dPower_ImpExpHubs.index, within=lego.model.hubs * lego.model.i)

    # Parameters
    lego.model.pHubConnectionPmaxImport = pyo.Param(lego.model.hubConnections, doc='Maximum import power at hub connection', initialize=lego.cs.dPower_ImpExpHubs['Pmax Import'])
    lego.model.pHubConnectionPmaxExport = pyo.Param(lego.model.hubConnections, doc='Maximum export power at hub connection', initialize=lego.cs.dPower_ImpExpHubs['Pmax Export'])
    lego.model.pImpExpPrice = pyo.Param(lego.model.rp, lego.model.k, lego.model.hubs, doc='Imp-/Export price at hub', initialize=lego.cs.dPower_ImpExpProfiles['Price'].reorder_levels(["rp", "k", "hub"]))
    lego.model.pImpExpValue = pyo.Param(lego.model.rp, lego.model.k, lego.model.hubs, doc='Imp-/Export value at hub', initialize=lego.cs.dPower_ImpExpProfiles['ImpExp'].reorder_levels(["rp", "k", "hub"]))
    lego.model.pImpType = pyo.Param(lego.model.hubs, doc='Import type at hub', initialize=lego.cs.dPower_ImpExpHubs['Import Type'].groupby('hub').first(), within=pyo.Any)
    lego.model.pExpType = pyo.Param(lego.model.hubs, doc='Export type at hub', initialize=lego.cs.dPower_ImpExpHubs['Export Type'].groupby('hub').first(), within=pyo.Any)

    # Variables
    def vImpExp_bounds(model, rp, k, hub, i):
        if model.pImpExpValue[rp, k, hub] >= 0:  # Only importing allowed
            return 0, min(model.pImpExpValue[rp, k, hub], model.pHubConnectionPmaxImport[hub, i])
        else:  # Only exporting allowed
            return max(lego.model.pImpExpValue[rp, k, hub], -model.pHubConnectionPmaxExport[hub, i]), 0

    lego.model.vImpExp = pyo.Var(lego.model.rp, lego.model.k, lego.model.hubConnections, doc='Import/Export at hub connection', bounds=vImpExp_bounds)


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):
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

    lego.model.eImpExp = pyo.Constraint(lego.model.rp, lego.model.k, lego.model.hubs, doc='Imp-/Export sum at each hub', rule=eImpExp_rule)

    # Add import/export to power-balance of each node in each hour
    for rp in lego.model.rp:
        for k in lego.model.k:
            for hub, i in lego.model.hubConnections:
                lego.model.eDC_BalanceP_expr[rp, k, i] += lego.model.vImpExp[rp, k, hub, i]

    # Add import/export cost/revenues to total cost
    lego.model.objective.expr += sum(lego.model.vImpExp[rp, k, hub, i] * lego.model.pImpExpPrice[rp, k, hub] for rp in lego.model.rp for k in lego.model.k for hub, i in lego.model.hubConnections)
