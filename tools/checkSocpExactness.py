import pyomo.environ as pyo
import numpy as np

def check_exactness_of_socp_solution(cs, results):

    """Check the exactness of the SOCP solution after solving the model.

    Args:
        cs (LEGO): The LEGO instance containing the model.
        results: The results object returned by the solver.
    """
    model = cs.model
    if not cs.case_study.dPower_Parameters['pEnableSOCP']:
        return  # SOCP not enabled, no need to check exactness

    printer = cs.printer
    printer.information("Checking exactness of SOCP solution...")

    # check if constraint eSOCP_VoltageDrop_rule is in the model
    
    max_deviation = 0.0
    if hasattr(model, "eSOCP_VoltageDrop_rule"):
        for rp in model.rp:
            for k in model.constraintsActiveK:
                for (i, j, c) in model.la:
                    ui = pyo.value(model.vSOCP_ui[rp, k, i])
                    lij = pyo.value(model.vSOCP_lij[rp, k, i, j, c])
                    vLineP = pyo.value(model.vLineP[rp, k, i, j, c])
                    vLineQ = pyo.value(model.vLineQ[rp, k, i, j, c])

                    if ui is None or vLineP is None or lij is None or vLineQ is None:
                        continue  # Skip if any value is None

                    deviation = abs(ui * lij - (vLineP **2 + vLineQ **2))
                    if deviation > max_deviation:
                        max_deviation = deviation

    elif hasattr(model, "eSOCP_ExiLinePij_rule"):
        for rp in model.rp:
            for k in model.constraintsActiveK:
                for (i, j, c) in model.la:
                    vLineP = pyo.value(model.vLineP[rp, k, i, j, c])
                    vLineQ = pyo.value(model.vLineQ[rp, k, i, j, c])
                    pPmax = pyo.value(model.pPmax[i, j, c]) if (i, j, c) in model.la else pyo.value(model.pPmax[j, i, c])
                    pQmax = pyo.value(model.pQmax[i, j, c]) if (i, j, c) in model.la else pyo.value(model.pQmax[j, i, c])

                    if pPmax is None or vLineP is None or pQmax is None or vLineQ is None:
                        continue  # Skip if any value is None

                    deviation = abs(vLineP ** 2 + vLineQ ** 2 - np.sqrt(pPmax ** 2 + pQmax ** 2))
                    if deviation > max_deviation:
                        max_deviation = deviation
    else:
        printer.warning("No matching SOCP constraints found to choose AC-OPF implementation..")
        return

    tolerance = 1e-5  # Define a tolerance level for exactness
    if max_deviation > tolerance:
        printer.warning(f"SOCP solution is not exact! Maximum deviation: {max_deviation:.6f}")
    else:
        printer.information("SOCP solution is exact within the defined tolerance.")