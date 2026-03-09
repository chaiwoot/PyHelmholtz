import numpy as np
from .stencils_data import *

__all__ = ["FD"]

class FD:
    def __init__(self, order=2):
        self.order = order

        if order == 2:
            
            self.stc1_onesided = {
                0: stc_12_fw_02
            }

            self.stc1_catalog = {
                0: stc_12_fw_02,
                1: stc_12_ct_11
            }

            self.stc2_catalog = {
                0: stc_22_fw_03,
                1: stc_22_ct_11
            }

            self.pmlstc2_catalog = {
                1: pmlstc_22_ct_11
            }
        
        elif order == 4:

            self.stc1_onesided = {
                0: stc_14_fw_04
            }

            self.stc1_catalog = {
                0: stc_14_fw_04,
                1: stc_14_as_13,
                2: stc_14_ct_22
            }

            self.stc2_catalog = {
                0: stc_24_fw_05,
                1: stc_24_as_14,
                2: stc_24_ct_22,
                3: stc_24_ct_33 # long arm stencil
            }

            self.pmlstc2_catalog = {
                1: pmlstc_24_as_14,
                2: pmlstc_24_as_23,
                3: pmlstc_24_ct_33
            }