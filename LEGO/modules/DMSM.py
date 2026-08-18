



import pandas as pd
import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGO, LEGOUtilities

printer = Printer.getInstance()


def _shift_window(model, k, activation_time_param, i):
    """
    Returns the list of timesteps kk that lie up to ActivationTime[i] steps before k (including k
    itself), cyclically within the rp. Used both for the repayment-deadline constraint and for
    computing the bound of the repayment variable.
    """
    horizon = len(model.constraintsActiveK)
    activation_time = int(pyo.value(activation_time_param[i]))
    # Window size: ActivationTime timesteps before k, plus k itself, capped at the full cycle length
    # (if ActivationTime >= horizon, the entire cycle is reachable anyway).
    window_size = min(activation_time + 1, horizon)

    # Normalize first_index (possibly shifted multiple times around the cycle) into the valid range
    # [1, horizon], so set_range_cyclic doesn't break even when ActivationTime is close to or exceeds
    # the window length.
    first_index_raw = model.constraintsActiveK.ord(k) - window_size + 1
    first_index = ((first_index_raw - 1) % horizon) + 1
    last_index = model.constraintsActiveK.ord(k)
    if last_index < first_index:
        last_index += horizon

    return LEGOUtilities.set_range_cyclic(model.constraintsActiveK, first_index, last_index)


def _forward_step(model, k, steps):
    """
    Returns the timestep that lies `steps` steps after k, cyclically within the rp (wraps back to the
    start at the end of the horizon). Used for the start-anchored repayment deadline (target timestep =
    start of an activation cycle + 2*ActivationTime).
    """
    horizon = len(model.constraintsActiveK)
    target_ord = ((model.constraintsActiveK.ord(k) - 1 + steps) % horizon) + 1
    return model.constraintsActiveK.at(target_ord)


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> (list[pyo.Var], list[pyo.Var]):
    first_stage_variables = []
    second_stage_variables = []

    # Parameters
    model.pDSM_pos = pyo.Param(model.rp, model.constraintsActiveK, model.i, initialize=cs.dPower_DSM_pos['value'], default=0.0, doc="Maximum positive DSM reduction potential per node and timestep [p.u. of demand]")
    model.pDSM_neg = pyo.Param(model.rp, model.constraintsActiveK, model.i, initialize=cs.dPower_DSM_neg['value'], default=0.0, doc="Maximum negative DSM increase potential per node and timestep [p.u. of demand]")

    model.pDSM_Ramping_pos = pyo.Param(model.i, initialize=cs.dPower_DSM_Ramping['DSM_Ramping_pos'], default=0.0, doc="Ramp-up delay (in timesteps) after activation before positive DSM can start reducing at all")
    model.pDSM_ActivationTime_pos = pyo.Param(model.i, initialize=cs.dPower_DSM_Ramping['DSM_activation_time_pos'], default=0.0, doc="Maximum number of consecutive timesteps a positive DSM activation cycle may run; together with the 2x rule, also the basis for its repayment deadline")
    model.pDSM_Ramping_neg = pyo.Param(model.i, initialize=cs.dPower_DSM_Ramping['DSM_Ramping_neg'], default=0.0, doc="Ramp-up delay (in timesteps) after activation before negative DSM can start increasing at all")
    model.pDSM_ActivationTime_neg = pyo.Param(model.i, initialize=cs.dPower_DSM_Ramping['DSM_activation_time_neg'], default=0.0, doc="Maximum number of consecutive timesteps a negative DSM activation cycle may run; together with the 2x rule, also the basis for its repayment deadline")

    # Debug output
    sample_rp = list(model.rp)[0]
    sample_k = list(model.constraintsActiveK)[0]
    printer.information("DSM_Ramping / DSM_activation_time eingelesen (erste 10 Knoten):")
    for i in list(model.i)[:10]:
        demand_sample = pyo.value(model.pDemandP[sample_rp, sample_k, i])
        dsm_pos_sample = pyo.value(model.pDSM_pos[sample_rp, sample_k, i])
        dsm_neg_sample = pyo.value(model.pDSM_neg[sample_rp, sample_k, i])
        # printer.information(
        #     f"  {i}: pos: Ramping={pyo.value(model.pDSM_Ramping_pos[i])}, ActivationTime={pyo.value(model.pDSM_ActivationTime_pos[i])}, "
        #     f"pDSM_pos[{sample_rp},{sample_k}]={dsm_pos_sample:.4f} p.u. -> {dsm_pos_sample * demand_sample:.4f} MW | "
        #     f"neg: Ramping={pyo.value(model.pDSM_Ramping_neg[i])}, ActivationTime={pyo.value(model.pDSM_ActivationTime_neg[i])}, "
        #     f"pDSM_neg[{sample_rp},{sample_k}]={dsm_neg_sample:.4f} p.u. -> {dsm_neg_sample * demand_sample:.4f} MW "
        #     f"(pDemandP[{sample_rp},{sample_k}]={demand_sample:.4f} MW)")

    # Variables
    model.vDSM_pos = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Power reduction at bus i through demand-side management", bounds=(0, None))
    second_stage_variables.append(model.vDSM_pos)

    model.vDSM_pos_payback = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Power increase at bus i to pay back a prior DSM reduction", bounds=(0, None))
    second_stage_variables.append(model.vDSM_pos_payback)

    model.vDSM_neg = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Voluntary power increase at bus i through negative demand-side management", bounds=(0, None))
    second_stage_variables.append(model.vDSM_neg)

    model.vDSM_neg_payback = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Power decrease at bus i to pay back a prior negative DSM increase", bounds=(0, None))
    second_stage_variables.append(model.vDSM_neg_payback)

    model.vDSM_pos_Bank = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Outstanding (not yet paid back) positive DSM reduction debt at bus i", bounds=(0, None))
    second_stage_variables.append(model.vDSM_pos_Bank)

    model.vDSM_neg_Bank = pyo.Var(model.rp, model.constraintsActiveK, model.i, doc="Outstanding (not yet paid back) negative DSM increase debt at bus i", bounds=(0, None))
    second_stage_variables.append(model.vDSM_neg_Bank)

    model.bDSM_neg_Mode = pyo.Var(model.rp, model.constraintsActiveK, model.i, domain=pyo.Binary,
        doc="1 if node i is allowed to have an open negative DSM bank at timestep k (the positive bank must then be 0), 0 if an open positive bank is allowed (the negative bank must then be 0)")
    second_stage_variables.append(model.bDSM_neg_Mode)

    model.bDSM_pos_Active = pyo.Var(model.rp, model.constraintsActiveK, model.i, domain=pyo.Binary,
        doc="1 if positive DSM reduction is switched on at this node/timestep (ongoing activation cycle)")
    second_stage_variables.append(model.bDSM_pos_Active)

    model.bDSM_neg_Active = pyo.Var(model.rp, model.constraintsActiveK, model.i, domain=pyo.Binary,
        doc="1 if negative DSM increase is switched on at this node/timestep (ongoing activation cycle)")
    second_stage_variables.append(model.bDSM_neg_Active)

    # Bounds
    # Upper bound per node/timestep for the activation variables (vDSM_pos/vDSM_neg): share pDSM_pos/
    # pDSM_neg of the node's demand in that SAME timestep. Within the ramp-up delay (the first
    # pDSM_Ramping_pos/neg[i] timesteps per rp), DSM cannot be activated yet - the activation variable
    # stays fixed at 0 there, after which the bound jumps to the full value.
    #
    # For the repayment variables, the demand of the repayment timestep itself is NOT the right
    # quantity - the bound instead needs to reflect what was actually available in the timesteps where a
    # reduction/increase actually happened, i.e. the summed potential over the same ActivationTime
    # window that also limits eDSM_pos_MaxDuration below - a safe, never-too-tight bound whose real
    # precision comes from the vDSM_pos_Bank state balance, not from this bound itself.
    for rp in model.rp:
        for k in model.constraintsActiveK:
            for i in model.i:
                max_shift_pos = pyo.value(model.pDSM_pos[rp, k, i]) * pyo.value(model.pDemandP[rp, k, i])
                if model.constraintsActiveK.ord(k) <= pyo.value(model.pDSM_Ramping_pos[i]):
                    model.vDSM_pos[rp, k, i].setub(0)  # still in the ramp-up delay
                    model.bDSM_pos_Active[rp, k, i].setub(0)
                else:
                    model.vDSM_pos[rp, k, i].setub(max_shift_pos)

                window_pos = _shift_window(model, k, model.pDSM_ActivationTime_pos, i)
                max_payback_pos = sum(pyo.value(model.pDSM_pos[rp, kk, i]) * pyo.value(model.pDemandP[rp, kk, i]) for kk in window_pos)
                model.vDSM_pos_payback[rp, k, i].setub(max_payback_pos)
                model.vDSM_pos_Bank[rp, k, i].setub(max_payback_pos)

                max_shift_neg = pyo.value(model.pDSM_neg[rp, k, i]) * pyo.value(model.pDemandP[rp, k, i])
                if model.constraintsActiveK.ord(k) <= pyo.value(model.pDSM_Ramping_neg[i]):
                    model.vDSM_neg[rp, k, i].setub(0)  # still in the ramp-up delay
                    model.bDSM_neg_Active[rp, k, i].setub(0)
                else:
                    model.vDSM_neg[rp, k, i].setub(max_shift_neg)

                window_neg = _shift_window(model, k, model.pDSM_ActivationTime_neg, i)
                max_payback_neg = sum(pyo.value(model.pDSM_neg[rp, kk, i]) * pyo.value(model.pDemandP[rp, kk, i]) for kk in window_neg)
                model.vDSM_neg_payback[rp, k, i].setub(max_payback_neg)
                model.vDSM_neg_Bank[rp, k, i].setub(max_payback_neg)

    return first_stage_variables, second_stage_variables


@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    """
    Tracks DSM activation and repayment per node and direction via a debt state (vDSM_pos_Bank/
    vDSM_neg_Bank), analogous to the storage level in storage.py (eStIntraRes_rule): every activation
    raises the debt, every repayment lowers it, cyclically via constraintsActiveK.prevw(k). Because of
    this cyclic balance, the sum of all activations automatically equals the sum of all repayments per
    rp/node/direction (DSM is a pure load shift) - no separate total-balance constraint is needed.
    """

    # Positive DSM
    def eDSM_pos_BankBalance_rule(model, rp, k, i):
        prev_k = model.constraintsActiveK.prevw(k)
        return model.vDSM_pos_Bank[rp, k, i] == model.vDSM_pos_Bank[rp, prev_k, i] + model.vDSM_pos[rp, k, i] - model.vDSM_pos_payback[rp, k, i]

    model.eDSM_pos_BankBalance = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Cyclic outstanding-debt balance for positive DSM", rule=eDSM_pos_BankBalance_rule)

    # Big-M via the variable's own already-known tight bound, instead of a generic Big-M constant
    def eDSM_pos_ActiveLink_rule(model, rp, k, i):
        return model.vDSM_pos[rp, k, i] <= model.vDSM_pos[rp, k, i].ub * model.bDSM_pos_Active[rp, k, i]

    model.eDSM_pos_ActiveLink = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Positive DSM reduction only allowed when switched on", rule=eDSM_pos_ActiveLink_rule)

    # Window-sum constraint modeled on thermalGen.py's eMinUpTime, but as an upper bound instead of a
    # lower bound (max instead of min consecutive active timesteps)
    def eDSM_pos_MaxDuration_rule(model, rp, k, i):
        window = _shift_window(model, k, model.pDSM_ActivationTime_pos, i)
        return sum(model.bDSM_pos_Active[rp, kk, i] for kk in window) <= model.pDSM_ActivationTime_pos[i]

    model.eDSM_pos_MaxDuration = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Positive DSM reduction may be active for at most ActivationTime consecutive timesteps", rule=eDSM_pos_MaxDuration_rule)

    # start_expr is 1 only on a genuine 0->1 transition of bDSM_pos_Active - detects the start of a new
    # activation cycle without a dedicated start variable. target_k is "start + 2*ActivationTime - 1":
    # the start timestep itself counts as timestep 1 of that total, so e.g. with ActivationTime=2 either
    # 2 timesteps of activation + 2 of repayment, or 1 + 3, are possible (both sum to 2*ActivationTime).
    # If a cycle starts at k, the bank must be fully repaid (0) by target_k.
    def eDSM_pos_RepaymentDeadline_rule(model, rp, k, i):
        activation_time = int(pyo.value(model.pDSM_ActivationTime_pos[i]))
        prev_k = model.constraintsActiveK.prevw(k)
        start_expr = model.bDSM_pos_Active[rp, k, i] - model.bDSM_pos_Active[rp, prev_k, i]
        target_k = _forward_step(model, k, 2 * activation_time - 1)
        return model.vDSM_pos_Bank[rp, target_k, i] <= model.vDSM_pos_Bank[rp, target_k, i].ub * (1 - start_expr)

    model.eDSM_pos_RepaymentDeadline = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Repayment of an activation cycle must be completed at most 2xActivationTime steps after its start", rule=eDSM_pos_RepaymentDeadline_rule)

    # Negative DSM (mirrors the positive-direction constraints above)
    def eDSM_neg_BankBalance_rule(model, rp, k, i):
        prev_k = model.constraintsActiveK.prevw(k)
        return model.vDSM_neg_Bank[rp, k, i] == model.vDSM_neg_Bank[rp, prev_k, i] + model.vDSM_neg[rp, k, i] - model.vDSM_neg_payback[rp, k, i]

    model.eDSM_neg_BankBalance = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Cyclic outstanding-debt balance for negative DSM", rule=eDSM_neg_BankBalance_rule)

    def eDSM_neg_ActiveLink_rule(model, rp, k, i):
        return model.vDSM_neg[rp, k, i] <= model.vDSM_neg[rp, k, i].ub * model.bDSM_neg_Active[rp, k, i]

    model.eDSM_neg_ActiveLink = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Negative DSM increase only allowed when switched on", rule=eDSM_neg_ActiveLink_rule)

    def eDSM_neg_MaxDuration_rule(model, rp, k, i):
        window = _shift_window(model, k, model.pDSM_ActivationTime_neg, i)
        return sum(model.bDSM_neg_Active[rp, kk, i] for kk in window) <= model.pDSM_ActivationTime_neg[i]

    model.eDSM_neg_MaxDuration = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Negative DSM increase may be active for at most ActivationTime consecutive timesteps", rule=eDSM_neg_MaxDuration_rule)

    def eDSM_neg_RepaymentDeadline_rule(model, rp, k, i):
        activation_time = int(pyo.value(model.pDSM_ActivationTime_neg[i]))
        prev_k = model.constraintsActiveK.prevw(k)
        start_expr = model.bDSM_neg_Active[rp, k, i] - model.bDSM_neg_Active[rp, prev_k, i]
        target_k = _forward_step(model, k, 2 * activation_time - 1)
        return model.vDSM_neg_Bank[rp, target_k, i] <= model.vDSM_neg_Bank[rp, target_k, i].ub * (1 - start_expr)

    model.eDSM_neg_RepaymentDeadline = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Repayment of an activation cycle must be completed at most 2xActivationTime steps after its start", rule=eDSM_neg_RepaymentDeadline_rule)

    # Cross-direction exclusivity
    # At most one of the two banks may be open per node/timestep (never both at once), analogous to
    # storage.py's eExclusiveChargeDischarge - a new cycle in the other direction may only start once the
    # running bank is fully repaid to 0. bDSM_neg_Mode switches between the two; each side is bounded by
    # its own already-known tight bound (.ub) instead of a generic Big-M constant.
    def eDSM_pos_Exclusivity_rule(model, rp, k, i):
        return model.vDSM_pos_Bank[rp, k, i] <= model.vDSM_pos_Bank[rp, k, i].ub * (1 - model.bDSM_neg_Mode[rp, k, i])

    model.eDSM_pos_Exclusivity = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Positive DSM bank may only be open when no negative DSM bank is open", rule=eDSM_pos_Exclusivity_rule)

    def eDSM_neg_Exclusivity_rule(model, rp, k, i):
        return model.vDSM_neg_Bank[rp, k, i] <= model.vDSM_neg_Bank[rp, k, i].ub * model.bDSM_neg_Mode[rp, k, i]

    model.eDSM_neg_Exclusivity = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Negative DSM bank may only be open when no positive DSM bank is open", rule=eDSM_neg_Exclusivity_rule)

    # Objective
    first_stage_objective = 0.0
    # DSM cost applies only to activation (not to repayment), for both directions, weighted by
    # pWeight_rp/pWeight_k and a hardcoded cost rate per MW of activation
    second_stage_objective = sum(model.pWeight_rp[rp] *
                                 sum(model.pWeight_k[k] *
                                     sum(model.vDSM_pos[rp, k, i] + model.vDSM_neg[rp, k, i]
                                         for i in model.i)
                                     for k in model.constraintsActiveK)
                                 for rp in model.rp) * 0.00        #  DSM cost rate, hardcoded

    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
