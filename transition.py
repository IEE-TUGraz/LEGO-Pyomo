import logging

from pyomo.core import NameLabeler

from LEGO.CaseStudy import CaseStudy
from LEGO.LEGO import LEGO
from tools.mpsCompare import compare_mps
from tools.printer import Printer

########################################################################################################################
# Setup
########################################################################################################################

pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.INFO)
printer = Printer.getInstance()

########################################################################################################################
# Data input from case study
########################################################################################################################

if True:
    cs = CaseStudy("data/example/", do_not_merge_single_node_buses=True)
    cs.dPower_Network['Technical Representation'] = 'DC-OPF'

    lego = LEGO(cs)

    #####################################################################################################################
    # Evaluation
    #####################################################################################################################

    model, timing = lego.build_model()
    printer.information(f"Building model took {timing:.2f} seconds")
    model.write("model.mps", io_options={'labeler': NameLabeler()})

constraints_to_skip_from1 = ["eStIntraRes", "eExclusiveChargeDischarge"]
constraints_to_skip_from2 = []
coefficients_to_skip_from1 = ["name"]
coefficients_to_skip_from2 = ["name",
                              "v2ndResDW", "vGenP1"]

compare_mps("model.mps", "originalLEGO.mps", check_vars=False,
            constraints_to_skip_from1=constraints_to_skip_from1, constraints_to_skip_from2=constraints_to_skip_from2,
            coefficients_to_skip_from1=coefficients_to_skip_from1, coefficients_to_skip_from2=coefficients_to_skip_from2)

print("Done")
