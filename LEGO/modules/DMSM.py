import pyomo.environ as pyo

from InOutModule.CaseStudy import CaseStudy
from LEGO import LEGO, LEGOUtilities


def _shift_window(model, k, activation_time):
    """
    Returns the list of timesteps kk that lie up to `activation_time` steps before k (including k
    itself), cyclically within the rp. Used both for the MaxDuration window-sum constraint and for
    computing the bound of the repayment variable.
    """
    horizon = len(model.constraintsActiveK)
    activation_time = int(activation_time)
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
    start of an activation cycle + Ramping + 2*ActivationTime).
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

    # Ramping/ActivationTime vary by season, not by rp/k directly - plain dicts keyed by (i, season)
    # instead of pyo.Param, since they're only ever used as constant bounds/RHS values (via
    # season_of_k[k] below), never summed symbolically in the objective like pDSM_pos/neg above.
    model.season_of_k = cs.dPower_WeightsK['season'].to_dict()
    model.pDSM_Ramping_pos = cs.dPower_DSM_Ramping['DSM_Ramping_pos'].to_dict()
    model.pDSM_ActivationTime_pos = cs.dPower_DSM_Ramping['DSM_activation_time_pos'].to_dict()
    model.pDSM_Ramping_neg = cs.dPower_DSM_Ramping['DSM_Ramping_neg'].to_dict()
    model.pDSM_ActivationTime_neg = cs.dPower_DSM_Ramping['DSM_activation_time_neg'].to_dict()

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

    # Continuous, not binary: Startup only ever appears as a tightening term in eDSM_pos_RampUpDelay/
    # RepaymentDeadline below, never as a benefit, so cost-minimization always pushes it down to its
    # lower bound - combined with eDSM_pos_StartStop's inequality (pins Startup to 1 at a genuine 0->1
    # transition) and eDSM_pos_StartupLink (forces Startup to 0 whenever Active is 0, including at
    # deactivations, where StartStop's RHS is negative and would otherwise leave Startup unconstrained),
    # it's naturally integral at the optimum without needing an explicit Binary domain.
    model.vDSM_pos_Startup = pyo.Var(model.rp, model.constraintsActiveK, model.i, bounds=(0, 1), doc="1 if a new positive DSM activation cycle starts at this node/timestep")
    second_stage_variables.append(model.vDSM_pos_Startup)

    model.vDSM_neg_Startup = pyo.Var(model.rp, model.constraintsActiveK, model.i, bounds=(0, 1), doc="1 if a new negative DSM activation cycle starts at this node/timestep")
    second_stage_variables.append(model.vDSM_neg_Startup)

    # Bounds
    # Upper bound per node/timestep for the activation variables (vDSM_pos/vDSM_neg): share pDSM_pos/
    # pDSM_neg of the node's demand in that SAME timestep. The ramp-up delay itself (Ramping steps after
    # a cycle's start during which no actual reduction/increase is allowed yet) is enforced separately
    # via eDSM_pos/neg_RampUpDelay below, not through this bound.
    #
    # For the repayment variables, the demand of the repayment timestep itself is NOT the right
    # quantity - the bound instead needs to reflect what was actually available in the timesteps where a
    # reduction/increase actually happened, i.e. the summed potential over the same ActivationTime
    # window that also limits eDSM_pos_MaxDuration below - a safe, never-too-tight bound whose real
    # precision comes from the vDSM_pos_Bank state balance, not from this bound itself.
    for rp in model.rp:
        for k in model.constraintsActiveK:
            season = model.season_of_k[k]
            for i in model.i:
                max_shift_pos = pyo.value(model.pDSM_pos[rp, k, i]) * pyo.value(model.pDemandP[rp, k, i])
                model.vDSM_pos[rp, k, i].setub(max_shift_pos)

                window_pos = _shift_window(model, k, model.pDSM_ActivationTime_pos[i, season])
                max_payback_pos = sum(pyo.value(model.pDSM_pos[rp, kk, i]) * pyo.value(model.pDemandP[rp, kk, i]) for kk in window_pos)
                model.vDSM_pos_payback[rp, k, i].setub(max_payback_pos)
                model.vDSM_pos_Bank[rp, k, i].setub(max_payback_pos)

                max_shift_neg = pyo.value(model.pDSM_neg[rp, k, i]) * pyo.value(model.pDemandP[rp, k, i])
                model.vDSM_neg[rp, k, i].setub(max_shift_neg)

                window_neg = _shift_window(model, k, model.pDSM_ActivationTime_neg[i, season])
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

    # Links activation transitions to a dedicated startup indicator (no separate shutdown variable
    # needed). The >= inequality alone only pins Startup at genuine 0->1 transitions (RHS=+1); at a
    # deactivation (RHS=-1) it leaves Startup unconstrained from below, so it's paired with the
    # StartupLink constraint below (Startup <= Active) to force it to 0 whenever the node isn't active,
    # closing that gap without needing an explicit Binary domain or a separate Shutdown variable.
    def eDSM_pos_StartStop_rule(model, rp, k, i):
        prev_k = model.constraintsActiveK.prevw(k)
        return model.vDSM_pos_Startup[rp, k, i] >= model.bDSM_pos_Active[rp, k, i] - model.bDSM_pos_Active[rp, prev_k, i]

    model.eDSM_pos_StartStop = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Links positive DSM activation transitions to a dedicated startup indicator", rule=eDSM_pos_StartStop_rule)

    def eDSM_pos_StartupLink_rule(model, rp, k, i):
        return model.vDSM_pos_Startup[rp, k, i] <= model.bDSM_pos_Active[rp, k, i]

    model.eDSM_pos_StartupLink = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Positive DSM startup indicator can only be set while the node is active", rule=eDSM_pos_StartupLink_rule)

    # Big-M via the variable's own already-known tight bound, instead of a generic Big-M constant
    def eDSM_pos_ActiveLink_rule(model, rp, k, i):
        return model.vDSM_pos[rp, k, i] <= model.vDSM_pos[rp, k, i].ub * model.bDSM_pos_Active[rp, k, i]

    model.eDSM_pos_ActiveLink = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Positive DSM reduction only allowed when switched on", rule=eDSM_pos_ActiveLink_rule)

    # Window-sum constraint modeled on thermalGen.py's eMinUpTime, but as an upper bound instead of a
    # lower bound (max instead of min consecutive active timesteps). ActivationTime describes only the
    # actually-productive duration (after ramp-up) - the allowed length of the whole Active block is
    # therefore ActivationTime + Ramping, so the ramp-up phase (Active but not yet producing, see
    # eDSM_pos_RampUpDelay) doesn't eat into the productive time.
    def eDSM_pos_MaxDuration_rule(model, rp, k, i):
        season = model.season_of_k[k]
        activation_time = model.pDSM_ActivationTime_pos[i, season]
        ramping = model.pDSM_Ramping_pos[i, season]
        total_allowed_duration = activation_time + ramping
        window = _shift_window(model, k, total_allowed_duration)
        return sum(model.bDSM_pos_Active[rp, kk, i] for kk in window) <= total_allowed_duration

    model.eDSM_pos_MaxDuration = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Positive DSM reduction may be active for at most ActivationTime+Ramping consecutive timesteps", rule=eDSM_pos_MaxDuration_rule)

    # Ramp-up delay: for Ramping steps after a cycle's start (window of the last Ramping steps,
    # including k, containing a Startup), the actual reduction stays at 0 even though bDSM_pos_Active is
    # already 1 - the cycle has "started" but isn't producing yet. Skipped entirely when Ramping==0.
    def eDSM_pos_RampUpDelay_rule(model, rp, k, i):
        ramping = int(model.pDSM_Ramping_pos[i, model.season_of_k[k]])
        if ramping == 0:
            return pyo.Constraint.Skip
        window = LEGOUtilities.set_range_cyclic(model.constraintsActiveK, model.constraintsActiveK.ord(k) - ramping + 1, model.constraintsActiveK.ord(k))
        return model.vDSM_pos[rp, k, i] <= model.vDSM_pos[rp, k, i].ub * (1 - sum(model.vDSM_pos_Startup[rp, k2, i] for k2 in window))

    model.eDSM_pos_RampUpDelay = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Positive DSM cannot actually reduce load until Ramping steps after its activation cycle starts", rule=eDSM_pos_RampUpDelay_rule)

    # target_k is "start + Ramping + 2*ActivationTime - 1": the start timestep itself counts as timestep
    # 1 of that total, and the first Ramping steps produce no reduction at all (see
    # eDSM_pos_RampUpDelay), so they're added on top of the usual activation+repayment budget (e.g. with
    # ActivationTime=2, Ramping=0: 2 timesteps of activation + 2 of repayment, or 1 + 3, both sum to
    # 2*ActivationTime; with Ramping=1, the same budget is available but only starting 1 step later). If
    # a cycle starts at k, the bank must be fully repaid (0) by target_k.
    def eDSM_pos_RepaymentDeadline_rule(model, rp, k, i):
        season = model.season_of_k[k]
        activation_time = int(model.pDSM_ActivationTime_pos[i, season])
        ramping = int(model.pDSM_Ramping_pos[i, season])
        target_k = _forward_step(model, k, ramping + 2 * activation_time - 1)
        return model.vDSM_pos_Bank[rp, target_k, i] <= model.vDSM_pos_Bank[rp, target_k, i].ub * (1 - model.vDSM_pos_Startup[rp, k, i])

    model.eDSM_pos_RepaymentDeadline = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Repayment of an activation cycle must be completed at most 2xActivationTime steps after its start", rule=eDSM_pos_RepaymentDeadline_rule)

    # Negative DSM (mirrors the positive-direction constraints above)
    def eDSM_neg_BankBalance_rule(model, rp, k, i):
        prev_k = model.constraintsActiveK.prevw(k)
        return model.vDSM_neg_Bank[rp, k, i] == model.vDSM_neg_Bank[rp, prev_k, i] + model.vDSM_neg[rp, k, i] - model.vDSM_neg_payback[rp, k, i]

    model.eDSM_neg_BankBalance = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Cyclic outstanding-debt balance for negative DSM", rule=eDSM_neg_BankBalance_rule)

    def eDSM_neg_StartStop_rule(model, rp, k, i):
        prev_k = model.constraintsActiveK.prevw(k)
        return model.vDSM_neg_Startup[rp, k, i] >= model.bDSM_neg_Active[rp, k, i] - model.bDSM_neg_Active[rp, prev_k, i]

    model.eDSM_neg_StartStop = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Links negative DSM activation transitions to a dedicated startup indicator", rule=eDSM_neg_StartStop_rule)

    def eDSM_neg_StartupLink_rule(model, rp, k, i):
        return model.vDSM_neg_Startup[rp, k, i] <= model.bDSM_neg_Active[rp, k, i]

    model.eDSM_neg_StartupLink = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Negative DSM startup indicator can only be set while the node is active", rule=eDSM_neg_StartupLink_rule)

    def eDSM_neg_ActiveLink_rule(model, rp, k, i):
        return model.vDSM_neg[rp, k, i] <= model.vDSM_neg[rp, k, i].ub * model.bDSM_neg_Active[rp, k, i]

    model.eDSM_neg_ActiveLink = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Negative DSM increase only allowed when switched on", rule=eDSM_neg_ActiveLink_rule)

    def eDSM_neg_MaxDuration_rule(model, rp, k, i):
        season = model.season_of_k[k]
        activation_time = model.pDSM_ActivationTime_neg[i, season]
        ramping = model.pDSM_Ramping_neg[i, season]
        total_allowed_duration = activation_time + ramping
        window = _shift_window(model, k, total_allowed_duration)
        return sum(model.bDSM_neg_Active[rp, kk, i] for kk in window) <= total_allowed_duration

    model.eDSM_neg_MaxDuration = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Negative DSM increase may be active for at most ActivationTime+Ramping consecutive timesteps", rule=eDSM_neg_MaxDuration_rule)

    def eDSM_neg_RampUpDelay_rule(model, rp, k, i):
        ramping = int(model.pDSM_Ramping_neg[i, model.season_of_k[k]])
        if ramping == 0:
            return pyo.Constraint.Skip
        window = LEGOUtilities.set_range_cyclic(model.constraintsActiveK, model.constraintsActiveK.ord(k) - ramping + 1, model.constraintsActiveK.ord(k))
        return model.vDSM_neg[rp, k, i] <= model.vDSM_neg[rp, k, i].ub * (1 - sum(model.vDSM_neg_Startup[rp, k2, i] for k2 in window))

    model.eDSM_neg_RampUpDelay = pyo.Constraint(model.rp, model.constraintsActiveK, model.i, doc="Negative DSM cannot actually increase load until Ramping steps after its activation cycle starts", rule=eDSM_neg_RampUpDelay_rule)

    def eDSM_neg_RepaymentDeadline_rule(model, rp, k, i):
        season = model.season_of_k[k]
        activation_time = int(model.pDSM_ActivationTime_neg[i, season])
        ramping = int(model.pDSM_Ramping_neg[i, season])
        target_k = _forward_step(model, k, ramping + 2 * activation_time - 1)
        return model.vDSM_neg_Bank[rp, target_k, i] <= model.vDSM_neg_Bank[rp, target_k, i].ub * (1 - model.vDSM_neg_Startup[rp, k, i])

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
                                 for rp in model.rp) * 0.01        #  DSM cost rate, hardcoded

    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
