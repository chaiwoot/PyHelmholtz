import numpy as np
from .stencils_data import *

__all__ = ["FD"]

class FD:
    """Organizes finite difference (FD) stencil catalogs by derivative order and accuracy order"""

    def __init__(self, order: int = 2):

        """
        Initialize FD stencil catalogs

        Parameter:
            order: accuracy order (2 and 4)
        """

        if order == 2:
            self._initialize_fd2()

        elif order == 4:
            self._initialize_fd4()

        else:
            raise ValueError(f"Unsupported accuracy order: {order}")
        
        self.order = order

    def _initialize_fd2(self) -> None:

        """Setup 2nd-order FD accuracy stencils"""

        # number of cell overlaped between physical and pml domains
        self.noverlap_pml = 1

        # 1st-order derivative stencil catalogs for non-PML method
        self.stc1_onesided = {0: stc_12_fw_02}
        self.stc1_catalog = {0: stc_12_fw_02, 1: stc_12_ct_11}

        # 2nd-order derivative stencil catalogs for non-PML method
        self.stc2_onesided = {0: stc_22_fw_03}
        self.stc2_catalog = {0: stc_22_fw_03, 1: stc_22_ct_11}

        # 2nd-order derivative stencil catalog for PML method
        self.pmlstc2_catalog = {1: pmlstc_22_ct_11}
        
    def _initialize_fd4(self) -> None:
    
        """Setup 4th-order FD accuracy stencils"""

        # number of cell overlaped between physical and pml domains
        self.noverlap_pml = 2

        # 1st-order derivative stencil catalogs for non-PML method
        self.stc1_onesided = {0: stc_14_fw_04}
        self.stc1_catalog = {0: stc_14_fw_04, 1: stc_14_as_13, 2: stc_14_ct_22}

        # 2nd-order derivative stencil catalogs for non-PML method
        self.stc2_onesided = {0: stc_24_fw_05}
        self.stc2_catalog = {0: stc_24_fw_05, 1: stc_24_as_14, 2: stc_24_ct_22}

        # 2nd-order derivative stencil catalog for PML method
        self.pmlstc2_catalog = {1: pmlstc_24_as_14, 2: pmlstc_24_as_23, 3: pmlstc_24_ct_33}
