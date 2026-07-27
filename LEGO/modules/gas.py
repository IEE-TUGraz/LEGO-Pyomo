import math
import typing

import pandas as pd
import pyomo.environ as pyo
from scipy.constants import speed_of_sound
import numpy as np

from pyomo.environ import *
from InOutModule.CaseStudy import CaseStudy
from InOutModule.printer import Printer
from LEGO import LEGO, LEGOUtilities

printer = Printer.getInstance()


@LEGOUtilities.safetyCheck_AddElementDefinitionsAndBounds
def add_element_definitions_and_bounds(model: pyo.ConcreteModel, cs: CaseStudy) -> typing.Tuple[list[pyo.Var], list[pyo.Var]]:
    # Lists for defining stochastic behavior. First stage variables are common for all scenarios, second stage variables are scenario-specific.
    first_stage_variables = []
    second_stage_variables = []

    # Sets
    model.m = pyo.Set(doc='Gas nodes', initialize=cs.dGas_NodeInfo.index)
    model.l = pyo.Set(doc='Loops/Name', initialize=cs.dGas_Network[cs.dGas_Network["pEnableInvest"] == 0].index.get_level_values('l').unique().tolist())
    model.d = pyo.Set(doc='Pipeline diameter', initialize=cs.dGas_CandDiam.index.tolist() + ['dExis'])
    model.gasNetworkElements = pyo.Set(doc='All gas network elements', initialize=cs.dGas_Network[cs.dGas_Network["pEnableInvest"] == 0].index, within=model.m * model.m * model.l)

    base = cs.dGas_Network[(cs.dGas_Network["pLength"].fillna(0).ne(0)) | (cs.dGas_Network["pDiam"].fillna(0).ne(0))]
    model.gasa = pyo.Set(doc='All gas pipelines', initialize=base.index)
    model.gase = pyo.Set(doc='Existing gas pipelines', initialize=base[base["pEnableInvest"] == 0].index)
    model.gasc = pyo.Set(doc='Candidate gas pipelines', initialize=base[base["pEnableInvest"] == 1].index)

    pipee = []
    pipec = []
    compa = []
    for (m, n, l), row_network in cs.dGas_Network.iterrows():
        if not pd.isna(row_network['pLength']):
            if row_network['pEnableInvest'] == 0:
                pipee.append((m, n, l, 'dExis'))
            else:
                for d, row_canddiam in cs.dGas_CandDiam.iterrows():
                    if row_canddiam['InvGroup'] == row_network['InvGroup']:
                        pipec.append((m, n, l, d))
        else:
            compa.append((m, n, l))

    model.pipee = pyo.Set(doc='Existing pipelines',initialize=pipee, within=model.m * model.m * model.l * model.d)
    model.pipec = pyo.Set(doc='Candidate pipelines',initialize=pipec, within=model.m * model.m * model.l * model.d)

    model.pipea = pyo.Set(doc='All pipelines', initialize=model.pipee | model.pipec, within=model.m * model.m * model.l * model.d)
    model.compa = pyo.Set(doc='All compressors', initialize=compa, within=model.gasNetworkElements)

    model.gs = pyo.Set(doc='Gas sources', initialize=cs.dGas_Source.index)
    model.gsm = pyo.Set(doc='Gas source gs connected to node m', initialize=cs.dGas_Source.reset_index().set_index(["gs", "m"]).index, within=model.gs * model.m)

    # Parameters
    model.pGasDemand = pyo.Param(model.rp, model.k, model.m, initialize=cs.dGas_Demand['value'], doc='Gas demand at node m in representative period rp and timestep k')

    model.pGasMaxProd = pyo.Param(model.gs, initialize=cs.dGas_Source['GasMaxProd'], doc='Maximum production of gas source gs')
    model.pGasMinProd = pyo.Param(model.gs, initialize=cs.dGas_Source['GasMinProd'], doc='Minimum production of gas source gs')
    model.pGasCost = pyo.Param(model.gs, initialize=cs.dGas_Source['GasCost'], doc='Gas cost of gas source gs')

    model.pMaxPress = pyo.Param(model.m, initialize=cs.dGas_NodeInfo['pMaxPress'], doc='Maximum pressure at node m')
    model.pMinPress = pyo.Param(model.m, initialize=cs.dGas_NodeInfo['pMinPress'], doc='Minimum pressure at node m')

    model.pGasType = pyo.Param(model.gasNetworkElements, initialize=cs.dGas_Network.loc[model.gasNetworkElements, 'pGasType'], doc='Gas type of network elements gasNetworkElements', domain=pyo.Any)
    model.pLength = pyo.Param(model.pipea, initialize=lambda mdl, m, n, l, d: cs.dGas_Network.loc[(m, n, l), 'pLength'], doc='Length of pipeline pipea')

    speed_of_sound = 400 # speed of sound in gas in m/s

    gas_mass_unit = "kg"  # "kg", "t"
    gas_time_unit = "s"  # "s", "h"
    gas_pressure_unit = "Pa^2"  # "Pa^2", "bar^2", "MPa^2"

    mass_factor = {"kg": 1, "t": 1000}
    time_factor = {"s": 1, "h": 3600}
    pressure_factor = {"Pa^2": 1, "bar^2": 10 ** 10, "MPa^2": 10 ** 12}

    gas_unit_scale = (
            mass_factor[gas_mass_unit] ** 2
            / (time_factor[gas_time_unit] ** 2 * pressure_factor[gas_pressure_unit])
    )

    cs.dGas_Network['pPipeCharacteristics'] =  gas_unit_scale * (16 * speed_of_sound ** 2 * cs.dGas_Network['pLength'] * (2 * np.log10(cs.dGas_Network['pDiam'] / cs.dGas_Network['pRough']) + 1.138) ** -2 )/ (math.pi ** 2 * cs.dGas_Network['pDiam'] ** 5)
    model.pPipe = pyo.Param(model.pipee, initialize=lambda mdl, m, n, l, d: cs.dGas_Network.loc[(m, n, l), 'pPipeCharacteristics'] , doc='Condensed parameters of pipeline pipea')

    cs.dGas_CandDiam['pCandidateCharacteristicsQuotient'] = ((2 * np.log10(cs.dGas_CandDiam['pDiam'] / cs.dGas_CandDiam['pRough']) + 1.138) ** -2) / (math.pi ** 2 * cs.dGas_CandDiam['pDiam'] ** 5)
    LEGO.addToParameter(model, 'pPipe', {(m, n, l, d): (0.0 if d == 'cand_d0' else (gas_unit_scale * 16 * speed_of_sound ** 2 * cs.dGas_Network.loc[(m, n, l), 'pLength'] * cs.dGas_CandDiam.loc[d, 'pCandidateCharacteristicsQuotient']))  for m, n, l, d in model.pipec}, indices=model.pipec)

    model.pFlowTP = pyo.Param(model.pipea, initialize=lambda model, m, n, l, d: (0.0 if d == 'cand_d0' else ( sqrt((model.pMaxPress[m] ** 2 - model.pMinPress[m] ** 2) / model.pPipe[m, n, l ,d]))), doc='Maximum flow on pipea under transport problem')

    model.pCompRatio= pyo.Param(model.compa, initialize=cs.dGas_Network.loc[model.compa, 'pCompRatio'], doc='Compression ratio of compressor compa')
    model.InvGroup= pyo.Param(model.gasNetworkElements, initialize=cs.dGas_Network.loc[model.gasNetworkElements, 'InvGroup'], doc='Investment group of network elements gasNetworkElements', domain=pyo.Any)

    model.pGasPipeFOMCost = pyo.Param(model.pipee, initialize=lambda model, m, n, l, d: cs.dGas_Network.loc[(m, n, l), 'pFOMCost'], doc='Fixed operation and maintenance cost pipea')
    LEGO.addToParameter(model, 'pGasPipeFOMCost', {(m, n, l, d): cs.dGas_CandDiam.loc[d, 'pFOMCost'] for m, n, l, d in model.pipec}, indices=model.pipec)

    model.pGasPipeInvestCost = pyo.Param(model.pipec, initialize=lambda model, m, n, l, d: cs.dGas_Network.loc[(m, n, l), 'pLength'] * cs.dGas_CandDiam.loc[d,'pInvestCost'], doc='Pipeline investment cost pipec')


    #Variables
    model.vGasProd = pyo.Var(model.rp, model.k, model.gs, doc='Gas production of gas source gs [power]', bounds=lambda model, rp, k, gs: (0, 1))
    second_stage_variables += [model.vGasProd]

    model.vGNS = pyo.Var(model.rp, model.k, model.m, doc='Gas non-supplied [power]', bounds=lambda model, rp, k, m: (0, model.pGasDemand[rp, k, m]))
    second_stage_variables += [model.vGNS]

    model.vGasFlow = pyo.Var(model.rp, model.k, model.gasa, doc='Gas flow in pipeline gasa [power]', bounds=lambda model, rp, k, m, n, l: (None, None))
    second_stage_variables += [model.vGasFlow]

    model.vGasFlowIn = pyo.Var(model.rp, model.k, model.gasa, doc='Gas inflow of pipeline gasa [power]', bounds=lambda model, rp, k, m, n, l: (None, None))
    second_stage_variables += [model.vGasFlowIn]

    model.vGasFlowOut = pyo.Var(model.rp, model.k, model.gasa, doc='Gas outflow of pipeline gasa [power]', bounds=lambda model, rp, k, m, n, l: (None, None))
    second_stage_variables += [model.vGasFlowOut]

    model.vGasCompFlowIn = pyo.Var(model.rp, model.k, model.compa, doc='Gas inflow of compressor compa [power]', bounds=lambda model, rp, k, m, n, l: (None, None))
    second_stage_variables += [model.vGasCompFlowIn]

    model.vGasCompFlowOut = pyo.Var(model.rp, model.k, model.compa, doc='Gas outflow of compressor compa [power]', bounds=lambda model, rp, k, m, n, l: (None, None))
    second_stage_variables += [model.vGasCompFlowOut]

    if cs.dGas_Parameters["pEnableStSt"]:
        model.vGasPressSqr = pyo.Var(model.rp, model.k, model.m, doc='Squared gas pressure at node m [bar]', bounds=lambda model, rp, k, m: (model.pMinPress[m] * model.pMinPress[m], model.pMaxPress[m] * model.pMaxPress[m]))
        second_stage_variables += [model.vGasPressSqr]

    model.vPipelineInvest = pyo.Var(model.pipec, doc='Pipeline investment', domain=pyo.Binary)
    first_stage_variables += [model.vPipelineInvest]

    return first_stage_variables, second_stage_variables



@LEGOUtilities.safetyCheck_addConstraints([add_element_definitions_and_bounds])
def add_constraints(model: pyo.ConcreteModel, cs: CaseStudy):
    # Gas balance for nodes
    def eGas_Balance_rule(model, rp, k, m, gsm, gasa0, gasa1, compa0, compa1):
        return (sum(model.vGasProd[rp, k, gs] for gs in gsm) -  # Production of gas source at node m
                sum(model.vGasFlowOut[rp, k, e] for e in gasa0) +  # Pipeline gas flow from node m to n
                sum(model.vGasFlowIn[rp, k, e] for e in gasa1) -  # Pipeline gas flow from node n to m
                sum(model.vGasCompFlowOut[rp, k, e] for e in compa0) +  # Compressor gas flow from node m to n
                sum(model.vGasCompFlowIn[rp, k, e] for e in compa1) -  # Compressor gas flow from node n to m
                model.pGasDemand[rp, k, m] +  # Demand at node m
                model.vGNS[rp, k, m] )# Slack variable for demand not served


    # Precompute sets for faster access within rules
    gsm = {m: [gs for gs in model.gs if (gs, m) in model.gsm] for m in model.m}  # Gas sources at node m
    gasa0 = {m: [e for e in model.gasa if (e[0] == m)] for m in model.m}  # Pipelines from m to n
    gasa1 = {m: [e for e in model.gasa if (e[1] == m)] for m in model.m}  # Pipelines from n to m
    compa0 = {m: [e for e in model.compa if (e[0] == m)] for m in model.m}  # Compressors from m to n
    compa1 = {m: [e for e in model.compa if (e[1] == m)] for m in model.m}  # Compressors from n to m

    model.eGas_Balance_expr = pyo.Expression(model.rp, model.k, model.m, rule=lambda model, rp, k, m: eGas_Balance_rule(model, rp, k, m, gsm[m], gasa0[m], gasa1[m], compa0[m], compa1[m]))
    model.eGas_Balance = pyo.Constraint(model.rp, model.k, model.m, doc='Gas balance constraint for each node', rule=lambda model, rp, k, m: model.eGas_Balance_expr[rp, k, m] == 0)

    mnl_to_d_dict = {(m, n, l): [d for d in model.d if (m, n, l, d) in model.pipea] for m, n, l in model.gasa}


    def eGasPipeInv_rule(model, m, n, l):
        return (sum(model.vPipelineInvest[m, n, l, d] for d in mnl_to_d_dict[(m, n, l)]) == 1)

    model.eGasPipeInv = pyo.Constraint(model.gasc, doc="Diameter expansion limit for pipeline gasc", rule=eGasPipeInv_rule)


    def eGasAverageFlow_rule(model, rp, k, m, n, l):
        return (model.vGasFlowIn[rp, k, m, n, l] == model.vGasFlowOut[rp, k, m, n, l])

    model.eGasAverageFlow = pyo.Constraint(model.rp, model.k, model.gasa, doc="Average gas flow of pipeline gasa under transport problem and steady-state problem", rule=eGasAverageFlow_rule)

    if cs.dGas_Parameters["pEnableStSt"]:
        def eGasFlowStSt_rule(model, rp, k, m, n, l):
            if (m, n, l) in model.gase:
                return ((model.vGasPressSqr[rp, k, m] - model.vGasPressSqr[rp, k, n]) == sum(model.pPipe[m, n, l, d] for d in mnl_to_d_dict[(m, n, l)]) * model.vGasFlowIn[rp, k, m, n, l] * abs(model.vGasFlowIn[rp, k, m, n, l]))
            else:
                return ((model.vGasPressSqr[rp, k, m] - model.vGasPressSqr[rp, k, n]) == sum((model.pPipe[m, n, l, d] * model.vPipelineInvest[m, n, l, d]) for d in mnl_to_d_dict[(m, n, l)]) *  model.vGasFlowIn[rp, k, m, n, l] * abs(model.vGasFlowIn[rp, k, m, n, l]))

        model.eGasFlowStSt = pyo.Constraint(model.rp, model.k, model.gasa, doc="Steady-state gas flow of pipeline gasa", rule=eGasFlowStSt_rule)


        def eGasCompStSt_rule(model, rp, k, m, n, l):
            return (model.vGasPressSqr[rp, k, n] == model.pCompRatio[m, n, l] ** 2 * model.vGasPressSqr[rp, k, m])

        model.eGasCompStSt = pyo.Constraint(model.rp, model.k, model.compa, doc="Formulation of compressor compa under steady-state gas flow", rule=eGasCompStSt_rule)





    if cs.dGas_Parameters["pEnableTP"]:
        def eGasMinInFlowTP_rule(model, rp, k, m, n, l):
            if (m, n, l) in model.gase:
                return - sum(model.pFlowTP[m, n, l ,d] for d in mnl_to_d_dict[(m, n, l)]) <= model.vGasFlowIn[rp, k, m, n, l]
            else:
                return (- sum((model.pFlowTP[m, n, l, d] * model.vPipelineInvest[m, n, l, d]) for d in mnl_to_d_dict[(m, n, l)])  <= model.vGasFlowIn[rp, k, m, n, l])

        model.eGasMinInFlowTP = pyo.Constraint(model.rp, model.k, model.gasa, doc="Minimum gas inflow of pipeline gasa", rule=eGasMinInFlowTP_rule)

        def eGasMinOutFlowTP_rule(model, rp, k, m, n, l):
            if (m, n, l) in model.gase:
                return - sum(model.pFlowTP[m, n, l ,d] for d in mnl_to_d_dict[(m, n, l)]) <= model.vGasFlowOut[rp, k, m, n, l]
            else:
                return (- sum((model.pFlowTP[m, n, l, d] * model.vPipelineInvest[m, n, l, d]) for d in mnl_to_d_dict[(m, n, l)])  <= model.vGasFlowOut[rp, k, m, n, l])

        model.eGasMinOutFlowTP = pyo.Constraint(model.rp, model.k, model.gasa, doc="Minimum gas outflow of pipeline gasa", rule=eGasMinOutFlowTP_rule)

        def eGasMaxInFlowTP_rule(model, rp, k, m, n, l):
            if (m, n, l) in model.gase:
                return + sum(model.pFlowTP[m, n, l ,d] for d in mnl_to_d_dict[(m, n, l)]) >= model.vGasFlowIn[rp, k, m, n, l]
            else:
                return (+ sum((model.pFlowTP[m, n, l, d] * model.vPipelineInvest[m, n, l, d]) for d in mnl_to_d_dict[(m, n, l)])  >= model.vGasFlowIn[rp, k, m, n, l])

        model.eGasMaxInFlowTP = pyo.Constraint(model.rp, model.k, model.gasa, doc="Maximum gas inflow of pipeline gasa", rule=eGasMaxInFlowTP_rule)

        def eGasMaxOutFlowTP_rule(model, rp, k, m, n, l):
            if (m, n, l) in model.gase:
                return + sum(model.pFlowTP[m, n, l ,d] for d in mnl_to_d_dict[(m, n, l)]) >= model.vGasFlowOut[rp, k, m, n, l]
            else:
                return (+ sum((model.pFlowTP[m, n, l, d] * model.vPipelineInvest[m, n, l, d]) for d in mnl_to_d_dict[(m, n, l)])  >= model.vGasFlowOut[rp, k, m, n, l])

        model.eGasMaxOutFlowTP = pyo.Constraint(model.rp, model.k, model.gasa, doc="Maximum gas outflow of pipeline gasa", rule=eGasMaxOutFlowTP_rule)


    print(0)


    # OBJECTIVE FUNCTION ADJUSTMENT(S)
    first_stage_objective = (sum(model.pGasPipeFOMCost[m, n, l, d] for m, n, l, d in model.pipee) +  # FOM cost of pipeline pipee
                             sum((model.pGasPipeInvestCost[m, n, l, d] + model.pGasPipeFOMCost[m, n, l, d]) * model.vPipelineInvest[m, n, l, d] for m, n, l, d in model.pipec))   # Investment and FOM cost of pipeline pipec
    #                         sum(model.pInvestCost[g] * model.vGenInvest[g] for g in model.g))  # Investment cost of generators
    second_stage_objective = (sum(model.pWeight_rp[rp] *  # Weight of representative periods
                                 sum(model.pWeight_k[k] *  # Weight of time steps
                                     (+ sum(+ model.vGNS[rp, k, m] * model.pENSCost  # Gas not served
                                            for m in model.m)
                                      + sum(+ model.vGasProd[rp, k, gs] * model.pGasCost[gs]  # Production cost of gas source
                                            for gs in model.gs))
                                     for k in model.k)
                                 for rp in model.rp))

    # Adjust objective and return first_stage_objective expression
    model.objective.expr += first_stage_objective + second_stage_objective
    return first_stage_objective
