import numpy as np

from .hhem import hybrid_em_matrices
from .pdvmatrix_libs import diag_matrix, wall_matrix, pdv2_operation_matrix
from .pdvmatrix_pml import pml_operation_matrices

# parameters for Engquist-Majda one-way wave equation
sq2 = np.sqrt(2)
corner_case = 0
const_pdv1 = 1/2
const_pdv2 = 0
const_wnb2 = sq2
corner_config = corner_case, const_pdv1, const_pdv2, const_wnb2

def helmholtz_pml(
        gridinfo: list,
        wnb2d: np.ndarray,
        em_order: int,
        fd_order: int,
        stretching_profile: callable
    ) -> np.ndarray:

    """
    Construct the matrix "A" that A = A_phy_inner + A_abs_nearbd + A_em (or A_wall)
    from given parameters (grid size, wavenumber, order of EM equation, FD order and weight profile),
    incorporating PML or PML+EM method.
    """

    nxp, nyp, nly, dx, dy = gridinfo # unpack variables
    
    nly_boundary = 1 # can be applied PEC or EM
    
    nx, ny = nxp + 2*nly, nyp + 2*nly
    n = nx*ny
    
    if fd_order == 2:
        
        nphyouter = 1
        pz_stcs_pdv2 = 'phy_ctas'    

        if em_order != 0:            
            az_stcs_pdv1 = 'abs_fw' # one-sided stencil
            az_stcs_pdv2 = 'abs_ctfw'

    elif fd_order == 4:

        nphyouter = 2
        pz_stcs_pdv2 = 'phy_ctas'

        if em_order != 0:            
            az_stcs_pdv1 = 'abs_fw' # one-sided stencil
            az_stcs_pdv2 = 'abs_ctasfw'

    if em_order != 0:
        emfd_configs = em_order, fd_order, az_stcs_pdv1, az_stcs_pdv2

    # (inner) interior domain : laplacian matrix
    Axx = pdv2_operation_matrix(nx, ny, nly, nphyouter, fd_order, pz_stcs_pdv2, 'x')
    Ayy = pdv2_operation_matrix(nx, ny, nly, nphyouter, fd_order, pz_stcs_pdv2, 'y')

    # (inner) interior domain: wavenumber matrix
    wnb1d = wnb2d.flatten()
    idx2d_domain = np.arange(n, dtype=np.int32).reshape([ny, nx])
    idx1d_phy_inner = idx2d_domain[nly+nphyouter:-nly-nphyouter, nly+nphyouter:-nly-nphyouter].flatten()
    A_wnb2 = diag_matrix(idx1d_phy_inner, wnb1d, 2)

    # (inner) interior domain: scaled discretized Helmholtz matrix
    A_phy_inner = ((dy/dx)**2)*Axx + Ayy + (dy**2)*A_wnb2

    # (near boundary) absorbing domain + (bd/near bd) interior domain
    # scaled discretized Helmholtz with PML matrix
    Axxpml, Ayypml, Apml_wnb2 = pml_operation_matrices(gridinfo, fd_order, stretching_profile, wnb1d)
    A_abs_nearbd = ((dy/dx)**2)*Axxpml + Ayypml + (dy**2)*Apml_wnb2

    # (boundary) absorbing domain
    if em_order == 0:
        A_wall = wall_matrix(nx, ny, nly, nphyouter)
        A = A_phy_inner + A_abs_nearbd + A_wall

    else:
        nly_boundary = 1
        modgridinfo = nxp+2*(nly-1), nyp+2*(nly-1), nly_boundary, dx, dy
        
        def emweight_profile(layer1d: np.ndarray, nly: int) -> np.ndarray:
            return (layer1d/nly)**2
        
        A_em, _ = hybrid_em_matrices(modgridinfo, emfd_configs, corner_config, wnb1d, emweight_profile)
        A = A_phy_inner + A_abs_nearbd + A_em 

    return A