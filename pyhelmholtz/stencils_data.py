import numpy as np

class FDStencil:

    def __init__(self, nnode, dv_order, idx_stencil, coeff):
        self.nnode = nnode
        self.dv_order = dv_order
        self.idx_stencil = np.array(idx_stencil, dtype=np.int32)
        self.coeff = np.array(coeff, dtype=np.float64)

    def unpack(self):
        return self.nnode, self.dv_order, self.idx_stencil, self.coeff

    def __repr__(self):
        return (f"FDStencil(nnode={self.nnode}, "
                f"dv_order={self.dv_order}, "
                f"idx_stencil={self.idx_stencil.tolist()}, "
                f"coeff={self.coeff.tolist()})")
    
class FDPMLStencil:

    def __init__(self, nnode, dv_order, idx_stencil, idx_qstencil, weight_q):
        self.nnode = nnode
        self.dv_order = dv_order
        self.idx_stencil = np.array(idx_stencil, dtype=np.int32)
        self.idx_qstencil = np.array(idx_qstencil, dtype=np.float32)
        self.weight_q = np.array(weight_q, dtype=np.float64)
        self.coeff = np.zeros_like(idx_stencil, dtype=np.float64)

    def unpack(self):
        return self.nnode, self.dv_order, self.idx_stencil, self.coeff

    def __repr__(self):
        return (f"FDStencil(nnode={self.nnode}, "
                f"dv_order={self.dv_order}, "
                f"idx_stencil={self.idx_stencil.tolist()}, "
                f"idx_qstencil={self.idx_qstencil.tolist()}, "
                f"weight_q={self.weight_q.tolist()})")

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
    nnode = 6,
    dv_order = 2,
    idx_stencil = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
    coeff = np.array([15/4, -77/6, 107/6, -13.0, 61/12, -5/6])
)

stc_24_ct_33 = FDStencil(
    nnode = 7,
    dv_order = 2,
    idx_stencil = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32),
    coeff = np.array([1/576, -3/32, 87/64, -365/144, 87/64, -3/32, 1/576])
)

stc_24_as_14 = FDStencil(
    nnode = 6, 
    dv_order = 2,
    idx_stencil = np.array([-1, 0, 1, 2, 3, 4], dtype=np.int32),
    coeff = np.array([5/6, -5/4, -1/3, 7/6, -1/2, 1/12])
)

pmlstc_22_ct_11 = FDPMLStencil(
    nnode = 3,
    dv_order = 2,
    idx_stencil = np.array([-1, 0, 1], dtype=np.int32),
    idx_qstencil = np.array([-0.5, 0.5]),
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
    idx_qstencil = np.array([-1.5, -0.5, 0.5, 1.5]),
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

pmlstc_24_as_23 = FDPMLStencil(
    nnode = 6,
    dv_order = 2,
    idx_stencil = np.array([-2, -1, 0, 1, 2, 3], dtype=np.int32),
    idx_qstencil = np.array([-1.5, -0.5, 0.5, 1.5]),
    weight_q = np.array([
                    [-11/288, -3/64, 0, 0], # 2
                    [17/576, 81/64, 3/64, 0], #-1
                    [1/64, -81/64, -81/64, -1/576], #0
                    [-5/576, 3/64, 81/64, 3/64], #1
                    [1/576, 0, -3/64, -3/64], #2
                    [0, 0, 0, 1/576] #3

    ])
)

pmlstc_24_as_14 = FDPMLStencil(
    nnode = 6,
    dv_order = 2,
    idx_stencil = np.array([-1, 0, 1, 2, 3, 4], dtype=np.int32),
    idx_qstencil = np.array([-0.5, 0.5, 1.5, 2.5, 3.5]),
    weight_q = np.array([
                    [121/144, 17/576, 0, 0, 0], #-1
                    [-187/288, -51/64, 1/64, 0, -1/576], #0
                    [-11/32, 51/64, -27/64, -5/576, 5/576], #1
                    [55/288, -17/576, 27/64, 15/64, -1/64], #2
                    [-11/288, 0, -1/64, -15/64, -17/576], #3
                    [0, 0, 0, 5/576, 11/288] #4

    ])
)