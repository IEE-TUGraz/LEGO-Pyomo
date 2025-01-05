import pyomo.environ as pyo

from LEGO import LEGO, LEGOUtilities


@LEGOUtilities.addToExecutionLog
def add_element_definitions_and_bounds(lego: LEGO):
    # Sets
    lego.model.importNodes = pyo.Set(doc='Import nodes', initialize=lego.cs.dPower_ImpExp[lego.cs.dPower_ImpExp.index.isin(['ImportFix', 'ImportMax'], level=2)].index.unique(level="i").tolist(), within=lego.model.i)
    lego.model.exportNodes = pyo.Set(doc='Export nodes', initialize=lego.cs.dPower_ImpExp[lego.cs.dPower_ImpExp.index.isin(['ExportFix', 'ExportMax'], level=2)].index.unique(level="i").tolist(), within=lego.model.i)

    # Variables
    lego.model.vImport = pyo.Var(lego.model.rp, lego.model.k, lego.model.importNodes, doc='Import at node i', bounds=(0, None))
    lego.model.vExport = pyo.Var(lego.model.rp, lego.model.k, lego.model.exportNodes, doc='Export at node i', bounds=(0, None))
    for rp in lego.model.rp:
        for k in lego.model.k:
            for i in lego.model.importNodes:
                if any(lego.cs.dPower_ImpExp.index.isin([(i, rp, 'ImportFix', k)])):
                    lego.model.vImport[rp, k, i].fix(lego.cs.dPower_ImpExp.loc[(i, rp, 'ImportFix', k)].iloc[0])
                elif any(lego.cs.dPower_ImpExp.index.isin([(i, rp, 'ImportMax', k)])):
                    lego.model.vImport[rp, k, i].setub(lego.cs.dPower_ImpExp.loc[(i, rp, 'ImportMax', k)].iloc[0])
                else:
                    lego.model.vImport[rp, k, i].fix(0)

                if any(lego.cs.dPower_ImpExp.index.isin([(i, rp, 'ExportFix', k)])):
                    lego.model.vExport[rp, k, i].fix(lego.cs.dPower_ImpExp.loc[(i, rp, 'ExportFix', k)].iloc[0])
                elif any(lego.cs.dPower_ImpExp.index.isin([(i, rp, 'ExportMax', k)])):
                    lego.model.vExport[rp, k, i].setub(lego.cs.dPower_ImpExp.loc[(i, rp, 'ExportMax', k)].iloc[0])
                else:
                    lego.model.vExport[rp, k, i].fix(0)

    # Parameters
    lego.model.pImpExpPrice = pyo.Param(lego.model.rp, lego.model.k, lego.model.importNodes | lego.model.exportNodes, doc='Imp-/Export price at node i',
                                        initialize=lego.cs.dPower_ImpExp[lego.cs.dPower_ImpExp.index.isin(['Price'], level=2)].droplevel("Type").reorder_levels(["rp", "k", "i"]))


@LEGOUtilities.checkExecutionLog([add_element_definitions_and_bounds])
def add_constraints(lego: LEGO):
    # Add import/export to power-balance of each node in each hour
    for rp in lego.model.rp:
        for k in lego.model.k:
            for i in lego.model.importNodes:
                lego.model.eDC_BalanceP_expr[rp, k, i] += lego.model.vImport[rp, k, i]
            for i in lego.model.exportNodes:
                lego.model.eDC_BalanceP_expr[rp, k, i] -= lego.model.vExport[rp, k, i]

    # Add import/export cost/revenues to total cost
    lego.model.objective.expr += sum(sum(sum(lego.model.vImport[rp, k, i] * lego.model.pImpExpPrice[rp, k, i] for rp in lego.model.rp) for k in lego.model.k) for i in lego.model.importNodes)
    lego.model.objective.expr -= sum(sum(sum(lego.model.vExport[rp, k, i] * lego.model.pImpExpPrice[rp, k, i] for rp in lego.model.rp) for k in lego.model.k) for i in lego.model.exportNodes)
