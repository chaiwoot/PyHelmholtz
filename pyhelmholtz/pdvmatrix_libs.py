import numpy as np
from scipy.sparse import csr_matrix, diags
from .util import Util
from .stencils import get_stencil, get_narmlengthmax

def diag_matrix(idx: np.ndarray, val1d: np.ndarray, p: int) -> csr_matrix:

    """
    Construct a complex diagonal matrix form with entries val1d[idx]**p
    """
    
    # val2d.shape = (ny, nx) 
    # val1d.shape = nx*ny = n
    # len(idx) <= n

    n = len(val1d) 

    ir = np.copy(idx)
    ic = np.copy(idx)
    val = (val1d[idx])**p

    return csr_matrix((val, (ir, ic)), shape=(n, n), dtype=np.complex128)

def pdv_matrix(stencil: list, idx: np.ndarray, factor_axis: int, factor_oneside: int, n: int) -> csr_matrix:

    """
    Construct a 2D array representing a partial derivative (PDV) matrix
    using the given FD stencil in the specified direction (x- or y-direction) and at the given index position.
    """

    nnode, dv_order, idx_stencil, coeff = stencil.unpack()

    factor_idxshf = factor_axis*factor_oneside
    factor_coeff = 1 # we compute the first derivative in outward direction from the physical domain, so the factor_coeff must be 1.
    
    ir = np.repeat(idx, nnode)
    ic = ir + np.tile(factor_idxshf*idx_stencil, len(idx))
    val = np.tile(factor_coeff*coeff, len(idx))
    
    return csr_matrix((val, (ir, ic)), shape=(n, n), dtype=np.complex128)

def normalize_pdv_matrix(A: csr_matrix, idx1d_abs: np.ndarray, weight1d: np.ndarray) -> csr_matrix:

    """
    Normalize the rows of "A" specified by "idx1d_abs" using their diagonal entries,
    and then multiply those rows by the corresponding weights in "weight1d".
    """

    diagA1d = A.diagonal()
    hbfactor1d = np.ones_like(diagA1d, dtype=np.complex128)
    hbfactor1d[idx1d_abs] = weight1d[idx1d_abs]/diagA1d[idx1d_abs]
    hbmatrix = diags(hbfactor1d, offsets=0, format='csr')
    
    return hbmatrix @ A

def wall_matrix(nx: int, ny: int, nly: int, nphyouter: int) -> csr_matrix:

    """
    Construct a 2D array A that enforces the PEC boundary condition (u = 0) on the boundary of the computational domain.
    """

    n = nx*ny
    _, _, _, idx_boundary = Util.get_idx_zone(nx, ny, nly, nphyouter)

    ir = np.copy(idx_boundary)
    ic = np.copy(idx_boundary)
    val = np.ones_like(idx_boundary)
    
    return csr_matrix((val, (ir, ic)), shape=(n, n), dtype=np.complex128)

def pdv2_operation_matrix(
        nx: int,
        ny: int,
        nly: int,
        nphyouter: int,
        fd_order: int,
        zone_stcs: str,
        axis: str
    ) -> csr_matrix:

    """
    Construct a 2D array representing the unscaled discretized Laplacian using the specified FD stencil
    (e.g., the 2nd-order FD stencil [1, −2, 1]) in the chosen x- or y-direction.
    """

    n = nx*ny
    nexterior = nly + nphyouter    
    narmlengthmax = get_narmlengthmax(2, fd_order, zone_stcs)
    
    mask2d_phy_inner = np.zeros([ny, nx], dtype=np.int32)
    mask2d_phy_inner[nexterior:-nexterior, nexterior:-nexterior] = 1

    # physical domain (axis = x or y)
    if axis == 'x':
        factor_axis = 1
        depth2d_axis, _ = Util.depth2d(nx, ny)
        sign2d_axis, _ = Util.mask2d_sign(nx, ny)
        
    elif axis == 'y':
        factor_axis = nx
        _, depth2d_axis = Util.depth2d(nx, ny)
        _, sign2d_axis = Util.mask2d_sign(nx, ny)

    # nal1d_phy_inner: 2d array with the size of [ny, nx] collects the number of armlength for pdv2 used in "inner" physical domain
    nal1d_phy_inner = (mask2d_phy_inner*sign2d_axis*depth2d_axis).flatten()
    nal1d_phy_inner[np.abs(nal1d_phy_inner)>=narmlengthmax] = narmlengthmax
    
    # indices of "inner" physical domain
    _, idx1d_phy_inner, _, _ = Util.get_idx_zone(nx, ny, nly, nphyouter)

    # nalpdv2_arr: possible number of armlength for pdv2 used in "inner" physical domain
    nalarr = np.unique(nal1d_phy_inner[idx1d_phy_inner])
    nalpdv2_arr = nalarr[np.argsort(abs(nalarr))]

    A_components = []
    for nalpdv2_used in nalpdv2_arr:

        idx = np.where(nal1d_phy_inner == nalpdv2_used)[0]
        
        factor_oneside = np.sign(nalpdv2_used) # factor_oneside = 1 (asfw) or -1 (asbw)
        stencil_used = get_stencil(2, fd_order, zone_stcs, abs(nalpdv2_used))

        A_components.append(pdv_matrix(stencil_used, idx, factor_axis, factor_oneside, n))

    A = sum(A_components)

    return A