import numpy as np
from .util import Util
from .pdvmatrix_libs import *

sq2 = np.sqrt(2)
corner_case = 0
const_pdv1 = 1/2
const_pdv2 = 0
const_wnb2 = sq2
corner_config = corner_case, const_pdv1, const_pdv2, const_wnb2

def helmholtz_em(
        gridinfo,
        wnb2d: np.ndarray,
        em_order: int,
        fd_order: int,
        weight_profile = None
    ) -> np.ndarray:

    """
    Construct the matrix "A" that A = A + A_abs_em + A_abs_hh
    from given parameters (grid size, wavenumber, order of EM equation, FD order and weight profile),
    incorporating HEM or EM method.
    """

    NPHYOUTER = 0

    nxp, nyp, nly, dx, dy = gridinfo
    nx, ny = nxp + 2*nly, nyp + 2*nly

    if fd_order == 2:
        pz_stcs_pdv2 = 'phy_ctas'
        az_stcs_pdv1 = 'abs_fw' # one-sided stencil
        az_stcs_pdv2 = 'abs_ctfw'

    elif fd_order == 4:
        pz_stcs_pdv2 = 'phy_ctas'
        az_stcs_pdv1 = 'abs_fw' # one-sided stencil
        az_stcs_pdv2 = 'abs_ctasfw'    

    emfd_configs = em_order, fd_order, az_stcs_pdv1, az_stcs_pdv2

    # interior domain : laplacian matrix
    Axx = pdv2_operation_matrix(nx, ny, nly, NPHYOUTER, fd_order, pz_stcs_pdv2, 'x')
    Ayy = pdv2_operation_matrix(nx, ny, nly, NPHYOUTER, fd_order, pz_stcs_pdv2, 'y')

    # interior domain: wavenumber matrix
    wnb1d = wnb2d.flatten()
    idx1d_phy, _, _, _ = Util.get_idx_zone(nx, ny, nly, NPHYOUTER)
    A_wnb2 = diag_matrix(idx1d_phy, wnb1d, 2)

    # interior domain: scaled discretized Helmholtz matrix
    A_phy = ((dy/dx)**2)*Axx + Ayy + (dy**2)*A_wnb2

    # absorbing domain: scaled EM and Helmholtz matrices
    A_abs_em, A_abs_hh = hybrid_em_matrices(gridinfo, emfd_configs, corner_config, wnb1d, weight_profile)

    # combination
    A = A_phy + A_abs_em + A_abs_hh

    return A

def hybrid_em_matrices(
        gridinfo: tuple[int, int, int, float, float],
        emfd_configs: tuple[int, int, str, str],
        corner_config: tuple[int, float, float, float],
        wnb1d: np.ndarray,
        beta_profile=None
    ) -> tuple[csr_matrix, csr_matrix]:

    """
    Construct the two matrices "A_abs_em" and "A_abs_hh"
    from given parameters (grid size, wavenumber, order of EM equation, FD order and weight profile),
    incorporating HEM or EM method.
    """

    NPHYOUTER = 0
    
    nxp, nyp, nly, dx, dy = gridinfo
    em_order, fd_order, zone_stcs_pdv1, zone_stcs_pdv2 = emfd_configs
    corner_case, scale_pdv1, scale_pdv2, scale_wnb2 = corner_config

    gr = dy/dx
    gr2 = gr**2

    nx, ny = nxp + 2*nly, nyp + 2*nly
    n = nx*ny

    if em_order == 1:
        em_factor = 0 # deactivate T_pdv2

    elif em_order == 2:
        em_factor = 1 # activate T_pdv2
    
    lmaxpdv1 = get_narmlengthmax(1, fd_order, zone_stcs_pdv1)
    lmaxpdv2 = get_narmlengthmax(2, fd_order, zone_stcs_pdv2)
    
    idx1d_domain = np.arange(n, dtype=np.int32)
    index_ix = np.arange(nx, dtype=np.int32)
    index_iy = np.arange(ny, dtype=np.int32)
    
    # length of arm in each axis
    lpdv1_x, lpdv1_y = Util.distance_from_edge(nx), Util.distance_from_edge(ny)
    lpdv1_x[lpdv1_x>lmaxpdv1] = lmaxpdv1
    lpdv1_y[lpdv1_y>lmaxpdv1] = lmaxpdv1

    lpdv2_x, lpdv2_y = Util.distance_from_edge(nx), Util.distance_from_edge(ny)
    lpdv2_x[lpdv2_x>lmaxpdv2] = lmaxpdv2
    lpdv2_y[lpdv2_y>lmaxpdv2] = lmaxpdv2

    # signs in each axis
    sign_x, sign_y = Util.mask1d_sign(nx), Util.mask1d_sign(ny)
    
    # arrays represents labelled absorbing layers
    abslayer2d = Util.get_abslayer2d(nx, ny, nly) 
    abslayer1d = abslayer2d.flatten()

    # correctness the first derivative to the outward-direction derivative
    ow_correction = -1 # ow = outward direction correction

    EM_components, HH_components = [], []

    for k in range(1, nly+1):
        
        ixc1, ixc2, iyc1, iyc2 = Util.get_idx_abs_corners(k, nx, ny, nly)

        # (1/5)a bottom side [fw/bw+pdv2] (fixed iy + varied ix)
        ix_left = np.arange(ixc1+1, lmaxpdv2)
        ix_right = (nx - 1) - ix_left
        list_ix = np.concatenate([ix_left, ix_right])

        for ix in list_ix:

            idx = np.array([ix + nx*iyc1])

            stc_pdv1y = get_stencil(1, fd_order, zone_stcs_pdv1, lpdv1_y[iyc1]) # fixed 
            stc_pdv2x = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_x[ix]) # varied
            stc_pdv2y = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_y[iyc1]) # fixed

            N_pdv1 = pdv_matrix(stc_pdv1y, idx, nx, sign_y[iyc1], n) # d/dy (fixed)
            T_pdv2 = pdv_matrix(stc_pdv2x, idx, 1, sign_x[ix], n) # d2/dx2 (varied)

            W_wnb1 = diag_matrix(idx, wnb1d, 1)
            W_wnb2 = diag_matrix(idx, wnb1d, 2)

            EM = em_factor*(0.5*gr2)*T_pdv2 + ow_correction*(1j*dy)*(W_wnb1 @ N_pdv1) + (dy**2)*W_wnb2
            EM_components.append(EM)
            
            HHy = pdv_matrix(stc_pdv2y, idx, nx, sign_y[iyc1], n) # d2/dy2 (fixed)
            HH = gr2*T_pdv2 + HHy + (dy**2)*W_wnb2                    
            HH_components.append(HH)
        
        # (1/5)b bottom side [ct+pdv2] (fixed iy + varied ix)
        if len(list_ix) == 0:
            ix_middle = index_ix[ixc1+1:-(ixc1+1)]
        else:
            ix_middle = index_ix[lmaxpdv2:-lmaxpdv2]

        idx = ix_middle + nx*iyc1
        
        stc_pdv1y = get_stencil(1, fd_order, zone_stcs_pdv1, lpdv1_y[iyc1]) # fixed
        stc_pdv2x = get_stencil(2, fd_order, zone_stcs_pdv2, lmaxpdv2) # fixed!
        stc_pdv2y = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_y[iyc1]) # fixed

        N_pdv1 = pdv_matrix(stc_pdv1y, idx, nx, sign_y[iyc1], n) # d/dy (fixed)
        T_pdv2 = pdv_matrix(stc_pdv2x, idx, 1, 1, n) # d2/dx2 (fixed) + factor_onesided = 1 (wo l.o.g)

        W_wnb1 = diag_matrix(idx, wnb1d, 1)
        W_wnb2 = diag_matrix(idx, wnb1d, 2)

        EM = em_factor*(0.5*gr2)*T_pdv2 + ow_correction*(1j*dy)*(W_wnb1 @ N_pdv1) + (dy**2)*W_wnb2

        EM_components.append(EM)
        
        HHy = pdv_matrix(stc_pdv2y, idx, nx, sign_y[iyc1], n) # d2/dy2 (fixed)
        HH = gr2*T_pdv2 + HHy + (dy**2)*W_wnb2                    
        HH_components.append(HH)

        # (2/5)a top side [fw/bw+pdv2] (fixed iy + varied ix)
        ix_left = np.arange(ixc1+1, lmaxpdv2)
        ix_right = (nx - 1) - ix_left
        list_ix = np.concatenate([ix_left, ix_right])

        for ix in list_ix:

            idx = np.array([ix + nx*iyc2])

            stc_pdv1y = get_stencil(1, fd_order, zone_stcs_pdv1, lpdv1_y[iyc2]) # fixed
            stc_pdv2x = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_x[ix]) # varied
            stc_pdv2y = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_y[iyc2]) # fixed

            N_pdv1 = pdv_matrix(stc_pdv1y, idx, nx, sign_y[iyc2], n) # d/dy (fixed)
            T_pdv2 = pdv_matrix(stc_pdv2x, idx, 1, sign_x[ix], n) # d2/dx2 (varied)

            W_wnb1 = diag_matrix(idx, wnb1d, 1)
            W_wnb2 = diag_matrix(idx, wnb1d, 2)

            EM = em_factor*(0.5*gr2)*T_pdv2 + ow_correction*(1j*dy)*(W_wnb1 @ N_pdv1) + (dy**2)*W_wnb2
            EM_components.append(EM)
            
            HHy = pdv_matrix(stc_pdv2y, idx, nx, sign_y[iyc2], n) # d2/dy2 (fixed)
            HH = gr2*T_pdv2 + HHy + (dy**2)*W_wnb2                    
            HH_components.append(HH)
        
        # (2/5)b top side [ct+pdv2] (fixed iy + varied ix)
        if len(list_ix) == 0:
            ix_middle = index_ix[ixc1+1:-(ixc1+1)]
        else:
            ix_middle = index_ix[lmaxpdv2:-lmaxpdv2]

        idx = ix_middle + nx*iyc2
        
        stc_pdv1y = get_stencil(1, fd_order, zone_stcs_pdv1, lpdv1_y[iyc2]) # fixed
        stc_pdv2x = get_stencil(2, fd_order, zone_stcs_pdv2, lmaxpdv2) # fixed!
        stc_pdv2y = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_y[iyc2]) # fixed

        N_pdv1 = pdv_matrix(stc_pdv1y, idx, nx, sign_y[iyc2], n) # d/dy (fixed)
        T_pdv2 = pdv_matrix(stc_pdv2x, idx, 1, 1, n) # d2/dx2 (fixed) + factor_onesided = 1 (wo l.o.g)

        W_wnb1 = diag_matrix(idx, wnb1d, 1)
        W_wnb2 = diag_matrix(idx, wnb1d, 2)

        EM = em_factor*(0.5*gr2)*T_pdv2 + ow_correction*(1j*dy)*(W_wnb1 @ N_pdv1) + (dy**2)*W_wnb2
        EM_components.append(EM)
        
        HHy = pdv_matrix(stc_pdv2y, idx, nx, sign_y[iyc2], n) # d2/dy2 (fixed)
        HH = gr2*T_pdv2 + HHy + (dy**2)*W_wnb2                    
        HH_components.append(HH)

        # (3/5)a left side [fw/bw+pdv2] (fixed ix + varied iy)
        iy_bottom = np.arange(iyc1+1, lmaxpdv2)
        iy_top = (ny - 1) - iy_bottom
        list_iy = np.concatenate([iy_bottom, iy_top])

        for iy in list_iy:

            idx = np.array([ixc1 + nx*iy])

            stc_pdv1x = get_stencil(1, fd_order, zone_stcs_pdv1, lpdv1_x[ixc1]) # fixed
            stc_pdv2y = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_y[iy]) # varied
            stc_pdv2x = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_x[ixc1]) # fixed

            N_pdv1 = pdv_matrix(stc_pdv1x, idx, 1, sign_x[ixc1], n) # d/dx (fixed)
            T_pdv2 = pdv_matrix(stc_pdv2y, idx, nx, sign_y[iy], n) # d2/dy2 (varied)

            W_wnb1 = diag_matrix(idx, wnb1d, 1)
            W_wnb2 = diag_matrix(idx, wnb1d, 2)

            EM = em_factor*0.5*T_pdv2 + ow_correction*(1j*dy*gr)*(W_wnb1 @ N_pdv1) + (dy**2)*W_wnb2
            EM_components.append(EM)
            
            HHx = pdv_matrix(stc_pdv2x, idx, 1, sign_x[ixc1], n) # d2/dx2 (fixed)
            HH = T_pdv2 + gr2*HHx + (dy**2)*W_wnb2                   
            HH_components.append(HH)
        
        # (3/5)b left side [ct+pdv2] (fixed ix + varied iy)
        if len(list_iy) == 0:
            iy_middle = index_iy[iyc1+1:-(iyc1+1)]
        else:
            iy_middle = index_iy[lmaxpdv2:-lmaxpdv2]

        idx = ixc1 + nx*iy_middle
        
        stc_pdv1x = get_stencil(1, fd_order, zone_stcs_pdv1, lpdv1_x[ixc1]) # fixed
        stc_pdv2y = get_stencil(2, fd_order, zone_stcs_pdv2, lmaxpdv2) # fixed!
        stc_pdv2x = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_x[ixc1]) # fixed

        N_pdv1 = pdv_matrix(stc_pdv1x, idx, 1, sign_x[ixc1], n) # d/dx (fixed)
        T_pdv2 = pdv_matrix(stc_pdv2y, idx, nx, 1, n) # d2/dy2 (fixed) + factor_onesided = 1 (wo l.o.g)

        W_wnb1 = diag_matrix(idx, wnb1d, 1)
        W_wnb2 = diag_matrix(idx, wnb1d, 2)

        EM = em_factor*0.5*T_pdv2 + ow_correction*(1j*dy*gr)*(W_wnb1 @ N_pdv1) + (dy**2)*W_wnb2
        EM_components.append(EM)
        
        HHx = pdv_matrix(stc_pdv2x, idx, 1, sign_x[ixc1], n) # d2/dx2 (fixed)
        HH = T_pdv2 + gr2*HHx + (dy**2)*W_wnb2                   
        HH_components.append(HH)

        # (4/5)a right side [fw/bw+pdv2] (fixed ix + varied iy)
        iy_bottom = np.arange(iyc1+1, lmaxpdv2)
        iy_top = (ny - 1) - iy_bottom
        list_iy = np.concatenate([iy_bottom, iy_top])

        for iy in list_iy:

            idx = np.array([ixc2 + nx*iy])

            stc_pdv1x = get_stencil(1, fd_order, zone_stcs_pdv1, lpdv1_x[ixc2]) # fixed
            stc_pdv2y = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_y[iy]) # varied
            stc_pdv2x = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_x[ixc2]) # fixed

            N_pdv1 = pdv_matrix(stc_pdv1x, idx, 1, sign_x[ixc2], n) # d/dx (fixed)
            T_pdv2 = pdv_matrix(stc_pdv2y, idx, nx, sign_y[iy], n) # d2/dy2 (varied)

            W_wnb1 = diag_matrix(idx, wnb1d, 1)
            W_wnb2 = diag_matrix(idx, wnb1d, 2)

            EM = em_factor*0.5*T_pdv2 + ow_correction*(1j*dy*gr)*(W_wnb1 @ N_pdv1) + (dy**2)*W_wnb2
            EM_components.append(EM)
            
            HHx = pdv_matrix(stc_pdv2x, idx, 1, sign_x[ixc2], n) # d2/dx2
            HH = T_pdv2 + gr2*HHx + (dy**2)*W_wnb2                   
            HH_components.append(HH)
        
        # (4/5)b right side [ct+pdv2] (fixed ix + varied iy)
        if len(list_iy) == 0:
            iy_middle = index_iy[iyc1+1:-(iyc1+1)]
        else:
            iy_middle = index_iy[lmaxpdv2:-lmaxpdv2]

        idx = ixc2 + nx*iy_middle
        
        stc_pdv1x = get_stencil(1, fd_order, zone_stcs_pdv1, lpdv1_x[ixc2]) # fixed
        stc_pdv2y = get_stencil(2, fd_order, zone_stcs_pdv2, lmaxpdv2) # fixed!
        stc_pdv2x = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_x[ixc2]) # fixed

        N_pdv1 = pdv_matrix(stc_pdv1x, idx, 1, sign_x[ixc2], n) # d/dx (fixed)
        T_pdv2 = pdv_matrix(stc_pdv2y, idx, nx, 1, n) # d2/dy2 (fixed) + factor_onesided = 1 (wo l.o.g)

        W_wnb1 = diag_matrix(idx, wnb1d, 1)
        W_wnb2 = diag_matrix(idx, wnb1d, 2)

        EM = em_factor*0.5*T_pdv2 + ow_correction*(1j*dy*gr)*(W_wnb1 @ N_pdv1) + (dy**2)*W_wnb2
        EM_components.append(EM)
        
        HHx = pdv_matrix(stc_pdv2x, idx, 1, sign_x[ixc2], n) # d2/dx2 (fixed)
        HH = T_pdv2 + gr2*HHx + (dy**2)*W_wnb2                   
        HH_components.append(HH)

        # 5/5 corners
        for ix in np.array([ixc1, ixc2]):
            for iy in np.array([iyc1, iyc2]):

                idx = np.array([ix + nx*iy])

                stc_pdv1x = get_stencil(1, fd_order, zone_stcs_pdv1, lpdv1_x[ix])
                stc_pdv1y = get_stencil(1, fd_order, zone_stcs_pdv1, lpdv1_y[iy])
                stc_pdv2x = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_x[ix])
                stc_pdv2y = get_stencil(2, fd_order, zone_stcs_pdv2, lpdv2_y[iy])

                N_pdv1x = pdv_matrix(stc_pdv1x, idx, 1, sign_x[ix], n)
                N_pdv1y = pdv_matrix(stc_pdv1y, idx, nx, sign_y[iy], n)
                T_pdv2x = pdv_matrix(stc_pdv2x, idx, 1, sign_x[ix], n)
                T_pdv2y = pdv_matrix(stc_pdv2y, idx, nx, sign_y[iy], n)

                W_wnb1 = diag_matrix(idx, wnb1d, 1)
                W_wnb2 = diag_matrix(idx, wnb1d, 2)

                PDV1_vtc = ow_correction*(1j*dy)*(W_wnb1 @ N_pdv1y)
                PDV2_vtc = em_factor*0.5*gr2*T_pdv2x
                
                PDV1_hrz = ow_correction*(1j*dy*gr)*(W_wnb1 @ N_pdv1x)
                PDV2_hrz = em_factor*0.5*T_pdv2y
                
                EM = scale_pdv1*(PDV1_vtc + PDV1_hrz) + corner_case*scale_pdv2*(PDV2_vtc + PDV2_hrz) + scale_wnb2*(dy**2)*W_wnb2
                EM_components.append(EM)

                # Helmholtz equation
                HH = gr2*T_pdv2x + T_pdv2y + (dy**2)*W_wnb2            
                HH_components.append(HH)

        # finish each absorbing layer then sum
        EM_acc, HH_acc = sum(EM_components), sum(HH_components)
        EM_components, HH_components = [], []
        EM_components.append(EM_acc)
        HH_components.append(HH_acc)

    EM, HH = sum(EM_components), sum(HH_components)

    _, _, idx1d_abs, _ = Util.get_idx_zone(nx, ny, nly, NPHYOUTER)
    
    if beta_profile is None:
        beta1d = np.ones_like(abslayer1d)

    else:
        beta1d = beta_profile(abslayer1d, nly)
        
    alpha1d = 1 - beta1d

    # normalize and scale with weight
    EM = normalize_pdv_matrix(EM, idx1d_abs, beta1d)
    HH = normalize_pdv_matrix(HH, idx1d_abs, alpha1d)

    return EM, HH