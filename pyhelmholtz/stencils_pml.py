import numpy as np

class FDPMLStencil:

    def __init__(self, nnode, dv_order, idx_stencil, idx_q, weight_q):
        self.nnode = nnode
        self.dv_order = dv_order
        self.idx_stencil = np.array(idx_stencil, dtype=np.int32)
        self.idx_q = np.array(idx_q, dtype=np.float32)
        self.weight_q = np.array(weight_q, dtype=np.float64)

    # Return all attributes
    def unpack(self):
        return self.nnode, self.dv_order, self.idx_stencil, self.idx_q, self.weight_q

    # Return a string representation of the FDPMLStencil object.
    def __repr__(self):
        return (f"FDPMLStencil(nnode={self.nnode}, "
                f"dv_order={self.dv_order}, "
                f"idx_stencil={self.idx_stencil.tolist()}, "
                f"idx_q={self.idx_q.tolist()}, "
                f"weight_q={self.weight_q.tolist()})")

pmlstc_22_ct_11 = FDPMLStencil(
    nnode = 3,
    dv_order = 2,
    idx_stencil = np.array([-1, 0, 1], dtype=np.int32),
    idx_q = np.array([-0.5, 0.5]),
    weight_q = np.array([
                    [1, 0],
                    [-1, -1],
                    [0, 1]
    ])
)

pmlstc_24_ct_33 = FDPMLStencil(
    nnode = 7,
    dv_order = 2,
    idx_stencil = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32),
    idx_q = np.array([-1.5, -0.5, 0.5, 1.5]),
    weight_q = np.array([
                    [1/576, 0, 0, 0], #-3
                    [-3/64, -3/64, 0, 0], #-2
                    [3/64, 81/64, 3/64, 0], #-1
                    [-1/576, -81/64, -81/64, -1/576], #0
                    [0, 3/64, 81/64, 3/64], #1
                    [0, 0, -3/64, -3/64], #2
                    [0, 0, 0, 1/576] #3
    ])
)

# numerical order = ~3 if q = 1
pmlstc_24_as_23 = FDPMLStencil(
    nnode = 6,
    dv_order = 2,
    idx_stencil = np.array([-2, -1, 0, 1, 2, 3], dtype=np.int32),
    idx_q = np.array([-1.5, -0.5, 0.5, 1.5]),
    weight_q = np.array([
                    [-11/288, -3/64, 0, 0], # 2
                    [17/576, 81/64, 3/64, 0], #-1
                    [1/64, -81/64, -81/64, -1/576], #0
                    [-5/576, 3/64, 81/64, 3/64], #1
                    [1/576, 0, -3/64, -3/64], #2
                    [0, 0, 0, 1/576] #3

    ])
)

# numerical order = ~3 if q = 1
pmlstc_24_as_14 = FDPMLStencil(
    nnode = 6,
    dv_order = 2,
    idx_stencil = np.array([-1, 0, 1, 2, 3, 4], dtype=np.int32),
    idx_q = np.array([-0.5, 0.5, 1.5, 2.5, 3.5]),
    weight_q = np.array([
                    [121/144, 17/576, 0, 0, 0], #-1
                    [-187/288, -51/64, 1/64, 0, -1/576], #0
                    [-11/32, 51/64, -27/64, -5/576, 5/576], #1
                    [55/288, -17/576, 27/64, 15/64, -1/64], #2
                    [-11/288, 0, -1/64, -15/64, -17/576], #3
                    [0, 0, 0, 5/576, 11/288] #4

    ])
)

def get_pmlstencil(dv_order: int, fd_order: int, narmlength: int) -> FDPMLStencil:

    """
    get the coefficient from desired derivative order, FD order and length of FD stencil using PML method
    """

    if fd_order == 2:
        zone_dir = 'abs_ct'

    elif fd_order == 4:
        zone_dir = 'abs_ctas'

    pmlstencil_table = {

        ### 2nd-order derivative for absorbing domain
        (2, 2, 'abs_ct', 1): pmlstc_22_ct_11,

        (2, 4, 'abs_ctas', 3): pmlstc_24_ct_33,
        (2, 4, 'abs_ctas', 2): pmlstc_24_as_23,
        (2, 4, 'abs_ctas', 1): pmlstc_24_as_14
    }
    
    pmlstencil = pmlstencil_table.get((dv_order, fd_order, zone_dir, narmlength), None)

    if pmlstencil is None:
        raise Exception('No pmlstencil found in the table.')

    return pmlstencil

def get_pmlnarmlengthmax(dv_order: int, fd_order: int, zone_dir: str) -> int:

    '''
    Get the PML stencil length based on derivative order, FD order, and zone direction.    

    Note that
    Stencil selection follows the spatial hierarchy: central (ct) → asymmetric (as) → forward (fw),
    corresponding to interior points, near-boundary points, and boundary points, respectively.
    '''

    pmlnarmlengthmax_table = {

        ### 2nd-order derivative for absorbing domain
        (2, 2, 'abs_ct'): 1,
        (2, 4, 'abs_ctas'): 3

    }
    
    nal_max = pmlnarmlengthmax_table.get((dv_order, fd_order, zone_dir), None)

    if nal_max is None:
        raise Exception('No pmlstencil found in the table.')

    return nal_max