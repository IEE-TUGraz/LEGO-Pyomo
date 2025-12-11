import pyomo.environ as pyo
import numpy as np

from InOutModule.printer import Printer
printer = Printer.getInstance()

def check_exactness_of_socp_solution(lego, results):

    """Check the exactness of the SOCP solution after solving the model.

    Args:
        lego (LEGO): The LEGO instance containing the model.
        results: The results object returned by the solver.
    """
    model = lego.model
    if not lego.cs.dPower_Parameters['pEnableSOCP']:
        return  # SOCP not enabled, no need to check exactness

    printer.information("\nChecking exactness of SOCP solution...")

    # check if constraint eSOCP_VoltageDrop_rule is in the model
    
    max_deviation = 0.0
    if hasattr(model, "eSOCP_VoltageDrop"):
        for rp in model.rp:
            for k in model.constraintsActiveK:
                for (i, j, c) in model.la:
                    ui = pyo.value(model.vSOCP_ui[rp, k, i])
                    lij = pyo.value(model.vSOCP_lij[rp, k, i, j, c])
                    vLineP = pyo.value(model.vLineP[rp, k, i, j, c])
                    vLineQ = pyo.value(model.vLineQ[rp, k, i, j, c])

                    if ui is None or vLineP is None or lij is None or vLineQ is None:
                        continue  # Skip if any value is None

                    deviation = ui * lij - (vLineP **2 + vLineQ **2)
                    if deviation > max_deviation:
                        max_deviation = deviation

    elif hasattr(model, "eSOCP_ExiLinePij"):
        for rp in model.rp:
            for k in model.constraintsActiveK:
                for (i, j, c) in model.la:
                    cii = pyo.value(model.vSOCP_cii[rp, k, i])
                    cjj = pyo.value(model.vSOCP_cii[rp, k, j])
                    sij = pyo.value(model.vSOCP_sij[rp, k, i, j])
                    cij = pyo.value(model.vSOCP_cij[rp, k, i, j])

                    if cii is None or cjj is None or sij is None or cij is None:
                        continue  # Skip if any value is None

                    deviation = cii * cjj - (sij **2 + cij **2)
                    if deviation > max_deviation:
                        max_deviation = deviation
    else:
        printer.warning("No matching SOCP constraints found to choose AC-OPF implementation..")
        return

    tolerance = 1e-6  # Define a tolerance level for exactness
    if max_deviation > tolerance:
        printer.warning(f"SOCP solution is not exact! Maximum deviation: {max_deviation}\n")
    else:
        printer.information(f"SOCP solution is exact within the defined tolerance. {max_deviation}\n")