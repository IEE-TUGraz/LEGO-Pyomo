import logging

from pyomo.core import NameLabeler, AlphaNumericTextLabeler, CuidLabeler, NumericLabeler, CounterLabeler, CNameLabeler
from pyomo.core.base.label import LPFileLabeler, ShortNameLabeler

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

compare_mps("model.mps", "originalLEGO.mps", check_vars=False)

print("Done")
