import numpy as np

# Abbreviations:
# ct = central
# fw = forward
# as = asymmetric

# Naming explanation:
# stencil1d_24_as_23 represents a 2nd-order derivative approximation with 4th-order accuracy using finite differences.
# The asymmetric 1D stencil spans 2 nodes to the left and 3 nodes to the right of the central point.

class FDStencil:

    def __init__(self, nnode, dv_order, idx_stencil, coeff):
        self.nnode = nnode
        self.dv_order = dv_order
        self.idx_stencil = np.array(idx_stencil, dtype=np.int32)
        self.coeff = np.array(coeff, dtype=np.float64)

    # Return all attributes
    def unpack(self):
        return self.nnode, self.dv_order, self.idx_stencil, self.coeff

    # Return a string representation of the FDStencil object
    def __repr__(self):
        return (f"FDStencil(nnode={self.nnode}, "
                f"dv_order={self.dv_order}, "
                f"idx_stencil={self.idx_stencil.tolist()}, "
                f"coeff={self.coeff.tolist()})")

stc_12_ct_11 = FDStencil(
    nnode = 2,
    dv_order = 1,
    idx_stencil = np.array([-1, 1], dtype=np.int32),
    coeff = np.array([-0.5, 0.5])
)

stc_14_ct_22 = FDStencil(
    nnode = 4,
    dv_order = 1,
    idx_stencil = np.array([-2, -1, 1, 2], dtype=np.int32),
    coeff = np.array([1/12, -2/3, 2/3, -1/12])
)

stc_11_fw_01 = FDStencil(
    nnode = 2,
    dv_order = 1,
    idx_stencil = np.array([0, 1], dtype=np.int32),
    coeff = np.array([-1, 1])
)

stc_12_fw_02 = FDStencil(
    nnode = 3,
    dv_order = 1,
    idx_stencil = np.array([0, 1, 2], dtype=np.int32),
    coeff = np.array([-1.5, 2, -0.5])
)

stc_14_as_13 = FDStencil(
    nnode = 5,
    dv_order = 1,
    idx_stencil = np.array([-1, 0, 1, 2, 3], dtype=np.int32),
    coeff = np.array([-0.25, -5/6, 3/2, -0.5, 1/12])
)

stc_14_fw_04 = FDStencil(
    nnode = 5,
    dv_order = 1,
    idx_stencil = np.array([0, 1, 2, 3, 4], dtype=np.int32),
    coeff = np.array([-25/12, 4.0, -3.0, 4/3, -0.25])
)

stc_22_ct_11 = FDStencil(
    nnode = 3,
    dv_order = 2,
    idx_stencil = np.array([-1, 0, 1], dtype=np.int32),
    coeff = np.array([1.0, -2.0, 1.0])
)

stc_22_fw_03 = FDStencil(
    nnode = 4,
    dv_order = 2,
    idx_stencil = np.array([0, 1, 2, 3], dtype=np.int32),
    coeff = np.array([2.0, -5.0, 4.0, -1.0])
)

stc_24_ct_22 = FDStencil(
    nnode = 5,
    dv_order = 2,
    idx_stencil = np.array([-2, -1, 0, 1, 2], dtype=np.int32),
    coeff = np.array([-1/12, 4/3, -2.5, 4/3, -1/12])
)

stc_24_fw_05 = FDStencil(
    nnode = 6, # numerical order = ~4
    dv_order = 2,
    idx_stencil = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
    coeff = np.array([15/4, -77/6, 107/6, -13.0, 61/12, -5/6])
)

stc_24_ct_33 = FDStencil(
    nnode = 7, # numerical order = ~4
    dv_order = 2,
    idx_stencil = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32),
    coeff = np.array([1/576, -3/32, 87/64, -365/144, 87/64, -3/32, 1/576])
)

# numerical order = ~6 (MIT version)
# stc_26_ct_33 = FDStencil(
#     nnode = 7, 
#     dv_order = 2,
#     idx_stencil = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32),
#     coeff = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
# )

# numerical order = ~3 (paper)
# stc_23_as_23 = FDStencil(
#     nnode = 6, 
#     dv_order = 2,
#     idx_stencil = np.array([0, -1, 1, -2, 2, 3], dtype=np.int32),
#     coeff = np.array([-725/288, 773/576, 389/288, -49/576, -53/576, 1/576])
# )

# numerical order = ~5
# stc_25_as_24 = FDStencil(
#     nnode = 7, 
#     dv_order = 2,
#     idx_stencil = np.array([-2, -1, 0, 1, 2, 3, 4], dtype=np.int32),
#     coeff = np.array([-13/180, 19/15, -7/3, 10/9, 1/12, -1/15, 1/90])
# )

# numerical order = ~3 (paper)
# stc_23_as_14 = FDStencil(
#     nnode = 6,
#     dv_order = 2,
#     idx_stencil = np.array([0, -1, 1, 2, 3, 4], dtype=np.int32),
#     coeff = np.array([-275/192, 167/192, 1/32, 77/96, -61/192, 3/64])
# )

# numerical order = ~4.1 (MIT version)
stc_24_as_14 = FDStencil(
    nnode = 6, 
    dv_order = 2,
    idx_stencil = np.array([-1, 0, 1, 2, 3, 4], dtype=np.int32),
    coeff = np.array([5/6, -5/4, -1/3, 7/6, -1/2, 1/12])
)

def get_stencil(dv_order: int, fd_order: int, zone_dir: str, narmlength: int) -> FDStencil:

    """
    get the coefficient from desired derivative order, FD order and length of FD stencil
    """

    stencil_table = {

        ### 1st-order derivative for absorbing domain
        (1, 2, 'abs_ctfw', 1): stc_12_ct_11,
        (1, 2, 'abs_ctfw', 0): stc_12_fw_02,        
        (1, 4, 'abs_ctasfw', 2): stc_14_ct_22,
        (1, 4, 'abs_ctasfw', 1): stc_14_as_13,        
        (1, 4, 'abs_ctasfw', 0): stc_14_fw_04,     
        (1, 1, "abs_fw", 0): stc_11_fw_01,   
        (1, 2, 'abs_fw', 0): stc_12_fw_02,
        (1, 4, 'abs_fw', 0): stc_14_fw_04,
        ### 2nd-order derivative for absorbing domain
        (2, 2, 'abs_ctfw', 1): stc_22_ct_11,
        (2, 2, 'abs_ctfw', 0): stc_22_fw_03,
        (2, 4, 'abs_ctasfw', 3): stc_24_ct_33,
        (2, 4, 'abs_ctasfw', 2): stc_24_ct_22,
        (2, 4, 'abs_ctasfw', 1): stc_24_as_14,
        (2, 4, 'abs_ctasfw', 0): stc_24_fw_05,
        (2, 2, 'abs_fw', 0): stc_22_fw_03,
        (2, 4, 'abs_fw', 0): stc_24_fw_05,
        ### 2nd-order derivative for physical domain
        (2, 2, 'phy_ctas', 1): stc_22_ct_11,
        (2, 4, 'phy_ctas', 3): stc_24_ct_33,
        (2, 4, 'phy_ctas', 2): stc_24_ct_22,
        (2, 4, 'phy_ctas', 1): stc_24_as_14
    }

    stencil = stencil_table.get((dv_order, fd_order, zone_dir, narmlength), None)

    if stencil is None:
        raise Exception('No stencil found in the table.')
    
    return stencil

def get_narmlengthmax(dv_order: int, fd_order: int, zone_dir: str) -> int:

    """
    Get the stencil length based on derivative order, FD order, and zone direction.    

    Note that
    Stencil selection follows the spatial hierarchy: central (ct) → asymmetric (as) → forward (fw),
    corresponding to interior points, near-boundary points, and boundary points, respectively.
    """
    
    narmlengthmax_table = {

        ### 1st-order derivative for absorbing domain
        (1, 1, "abs_fw"): 0,
        (1, 2, 'abs_fw'): 0,
        (1, 2, 'abs_ctfw'): 1,
        (1, 4, 'abs_fw'): 0,
        (1, 4, 'abs_ctasfw'): 2,

        ### 2nd-order derivative for absorbing domain
        (2, 2, 'abs_fw'): 0,
        (2, 2, 'abs_ctfw'): 1,
        (2, 4, 'abs_fw'): 0,
        (2, 4, 'abs_ctasfw'): 3,

        ### 2nd-order derivative for physical domain
        (2, 2, 'phy_ctas'): 1,
        (2, 4, 'phy_ctas'): 3,

    }
    
    nal_max = narmlengthmax_table.get((dv_order, fd_order, zone_dir), None)

    if nal_max is None:
        raise Exception('No stencil found in the table.')

    return nal_max