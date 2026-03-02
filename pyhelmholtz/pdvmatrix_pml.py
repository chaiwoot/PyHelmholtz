import numpy as np
from scipy.sparse import csr_matrix, diags
from .util import Util
from .stencils_pml import *
from .pdvmatrix_libs import diag_matrix

def convert_idx_to_iw(idx: int, nx: int, axis: str) -> int:

    """
    Convert a 1D index into its (row, column) position in the 2D array.
    """

    iy = int(idx//nx)
    ix = idx - iy*nx

    if axis == 'x':
        return ix
    
    elif axis == 'y':
        return iy

def calculate_coeff(
        q_profile: callable,
        pointinfo: tuple[int, int, int],
        idx: np.ndarray,
        idx_qstencil: np.ndarray,
        weight_q: np.ndarray,
        factor_idxshf: int # factor_axis*factor_oneside    
    ) -> np.ndarray:
    
    """
    Calculate the FD stencil coefficients for the 2nd-order derivative, incorporating the PML method.
    """

    nx, ny, nly = pointinfo

    if np.abs(factor_idxshf) == 1:
        nw, axis = nx, 'x'

    elif np.abs(factor_idxshf) > 1:
        nw, axis = ny, 'y'
    
    # calculate indices position of q
    idx_rep = idx[0] # "idx" contains indices'position in an axis having the same depth 
    iw0 = convert_idx_to_iw(idx_rep, nx, axis)
    iwq = iw0 + np.sign(factor_idxshf)*idx_qstencil

    # calculate depth_q
    if np.sign(factor_idxshf) == 1:
        depth_q = nly - iwq
        depth_q0 = nly - iw0

    elif np.sign(factor_idxshf) == -1:
        depth_q = iwq - (nw-nly-1)
        depth_q0 = iw0 - (nw-nly-1)

    depth_q[depth_q<0] = 0

    if depth_q0 < 0:
        depth_q0 = 0

    # calculate q
    q = q_profile(depth_q, nly)

    # calculate coeff
    coeff = weight_q @ q

    q0 = q_profile(depth_q0, nly)
    coeff = q0*coeff

    return coeff

def pmlpdv_matrix(
        q_profile: callable,
        pmlstencil: list,
        idx: np.ndarray,
        factor_oneside: int,
        pointinfo: tuple[int, int, int],
        axis: str
    ) -> csr_matrix:

    """
    Construct a 2D array that the discretized operation matrix incoparating the PML method
    using the given PML FD stencil in the specified direction (x- or y-direction) and at the given index position.
    """
    
    nnode, dv_order, idx_stencil, idx_q, weight_q = pmlstencil.unpack()
    nx, ny, nly = pointinfo

    if axis == 'x':
        factor_axis = 1

    elif axis == 'y':
        factor_axis = nx

    factor_idxshf = factor_axis*factor_oneside
    factor_coeff = 1

    n = nx*ny

    coeff = calculate_coeff(q_profile, pointinfo, idx, idx_q, weight_q, factor_idxshf)
    
    ir = np.repeat(idx, nnode)
    ic = ir + np.tile(factor_idxshf*idx_stencil, len(idx))
    val = np.tile(factor_coeff*coeff, len(idx))

    return csr_matrix((val, (ir, ic)), shape=(n, n), dtype=np.complex128)

def pml_operation_matrices(
        gridinfo: tuple[int, int, int, float, float],
        fd_order: int,
        q_profile: callable, # q =  1/(1 + 1j*(sigma0/omega)*(x/L)**p)
        wnb1d: np.ndarray
        ) -> csr_matrix:
    
    """
    Construct three 2D array (Axxpml, Ayypml and Apml_wnb2) representing the unscaled discretized Laplacian,
    incoparating the PML method, using the specified FD order and damping profile.
    """

    nxp, nyp, nly, dx, dy = gridinfo
    
    if fd_order == 2:
        lmaxpdv2, nphyouter = 1, 1

    elif fd_order == 4:
        lmaxpdv2, nphyouter = 3, 2

    nx, ny = nxp + 2*nly, nyp + 2*nly
    n = nx*ny
    
    pointinfo = nx, ny, nly

    # pml zone + physical zone (boundary + near boundary)
    idx2d_domain = np.arange(n, dtype=np.int32).reshape([ny, nx])
    idx1d_domain = idx2d_domain.flatten()

    # narm in each axis
    lpdv2_x, lpdv2_y = Util.distance_from_edge(nx), Util.distance_from_edge(ny)
    lpdv2_x[lpdv2_x>lmaxpdv2] = lmaxpdv2
    lpdv2_y[lpdv2_y>lmaxpdv2] = lmaxpdv2

    # signs in each axis
    sign_x, sign_y = Util.mask1d_sign(nx), Util.mask1d_sign(ny)

    Axxpml_cpn, Ayypml_cpn = [], []
    
    # left zone (LB/LT/L)
    for ix in range(1, nly+nphyouter):
        idx = idx2d_domain[1:-1, ix].flatten()
        pmlstencil_used = get_pmlstencil( 2, fd_order, lpdv2_x[ix])
        factor_onesided = 1 # fixed
        Axxpml_cpn.append(pmlpdv_matrix(q_profile, pmlstencil_used, idx, factor_onesided, pointinfo, 'x'))

    # right zone (RB/RT/R)
    for ix in range(nx-nly-nphyouter, nx-1):
        idx = idx2d_domain[1:-1, ix].flatten()
        pmlstencil_used = get_pmlstencil(2, fd_order, lpdv2_x[ix])
        factor_onesided = -1 # fixed
        Axxpml_cpn.append(pmlpdv_matrix(q_profile, pmlstencil_used, idx, factor_onesided, pointinfo, 'x'))

    # middle bottom zone
    for ix in range(nly+nphyouter, nx-nly-nphyouter):
        idx = idx2d_domain[1:nly+nphyouter, ix]
        pmlstencil_used = get_pmlstencil(2, fd_order, lpdv2_x[ix])
        factor_onesided = sign_x[ix]
        Axxpml_cpn.append(pmlpdv_matrix(q_profile, pmlstencil_used, idx, factor_onesided, pointinfo, 'x'))

    # middle top zone
    for ix in range(nly+nphyouter, nx-nly-nphyouter):
        idx = idx2d_domain[-nly-nphyouter:-1, ix]
        pmlstencil_used = get_pmlstencil(2, fd_order, lpdv2_x[ix])
        factor_onesided = sign_x[ix]
        Axxpml_cpn.append(pmlpdv_matrix(q_profile, pmlstencil_used, idx, factor_onesided, pointinfo, 'x'))

    # bottom zone 
    for iy in range(1, nly+nphyouter):
        idx = idx2d_domain[iy, 1:- 1].flatten()
        pmlstencil_used = get_pmlstencil(2, fd_order, lpdv2_y[iy])
        factor_onesided = 1 # fixed
        Ayypml_cpn.append(pmlpdv_matrix(q_profile, pmlstencil_used, idx, factor_onesided, pointinfo, 'y'))

    # top zone
    for iy in range(ny-nly-nphyouter, ny-1):
        idx = idx2d_domain[iy, 1:-1].flatten()
        pmlstencil_used = get_pmlstencil(2, fd_order, lpdv2_y[iy])
        factor_onesided = -1 # fixed
        Ayypml_cpn.append(pmlpdv_matrix(q_profile, pmlstencil_used, idx, factor_onesided, pointinfo, 'y'))

    # middle left zone
    for iy in range(nly+nphyouter, ny-nly-nphyouter):
        idx = idx2d_domain[iy, 1:nly+nphyouter]
        pmlstencil_used = get_pmlstencil(2, fd_order, lpdv2_y[iy])
        factor_onesided = sign_y[iy]
        Ayypml_cpn.append(pmlpdv_matrix(q_profile, pmlstencil_used, idx, factor_onesided, pointinfo, 'y'))

    # middle right zone
    for iy in range(nly+nphyouter, ny-nly-nphyouter):
        idx = idx2d_domain[iy, -nly-nphyouter:-1]
        pmlstencil_used = get_pmlstencil(2, fd_order, lpdv2_y[iy])
        factor_onesided = sign_y[iy]
        Ayypml_cpn.append(pmlpdv_matrix(q_profile, pmlstencil_used, idx, factor_onesided, pointinfo, 'y'))

    Axxpml = sum(Axxpml_cpn)
    Ayypml = sum(Ayypml_cpn)

    idx1d_phy_inner = idx2d_domain[nly+nphyouter:-nly-nphyouter, nly+nphyouter:-nly-nphyouter].flatten()
    _, _, _, idx1d_boundary = Util.get_idx_zone(nx, ny, nly, )
    idx_abs_phyouter = idx1d_domain[~np.isin(idx1d_domain, idx1d_phy_inner)]
    idx_abs_phyouter_wo_boundary = np.setdiff1d(idx_abs_phyouter, idx1d_boundary)    

    Apml_wnb2 = diag_matrix(idx_abs_phyouter_wo_boundary, wnb1d, 2)

    return Axxpml, Ayypml, Apml_wnb2