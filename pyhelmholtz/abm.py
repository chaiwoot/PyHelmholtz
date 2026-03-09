# Authors: Sirawit Inpuak and Chaiwoot Boonyasiriwat

import numpy as np
from abc import ABC, abstractmethod # for abstract base class

from .domain import Domain
from .source import Source
from .fd import FD
from .util import Util

__all__ = ["ABM", "RenLiu", "PML"]

class ABM(ABC):

    def __init__(self, abm, n, damping_profile):
        self.abm = abm                              # type of ABM
        self.n = n                                  # thickness of absorbing layer in unit of cells
        self.damping_profile = damping_profile      # damping profile

    @abstractmethod
    def build_A(self):
        pass

class RenLiu(ABM):

    """
    Ren-Liu hybrid absorbing boundary condition (ABC) of the Engquist-Majda ABC.
    When n = 1, the Engquist-Majda ABC is used.
    When n > 1, the Ren-Liu hybrid ABC is used.
    """

    def __init__(self, abm="EM1", n=1, damping_profile=None):
        super().__init__(abm, n, damping_profile)

        # first-order and second-order Engquist-Majda ABC
        if abm == "EM1" or abm == "EM2":
            self.n = 1 # thickness must be 1 cell

        # hybrid of first-order and second-order Engquist-Majda ABC
        elif abm == "RL1" or abm == "RL2":
            self.n = n # n should be larger than 1 cell to be absorbing boundary layer (ABL)

            if damping_profile == None:
                self.damping_profile =    lambda i, nly: (i/nly)**2

            else:
                self.damping_profile = damping_profile

    # construct the system matrix
    def build_A(self, domain:Domain, source:Source, fd:FD):

        nphy_outer = 0

        k_pad = 2*np.pi*source.freq/domain.v_pad # wavenumber
        k_pad_1d = k_pad.flatten()
        
        idx_int, idx_abs, idx_abs_inner, idx_bd = Util.get_idx(domain.nx_pad, domain.ny_pad, domain.n)

        # system matrix for interior domain (Helmhotlz equation)
        Axx_int = Util.build_Axx(domain.nx_pad, domain.ny_pad, domain.n, nphy_outer, "RenLiu", fd.stc2_catalog)
        Ayy_int = Util.build_Ayy(domain.nx_pad, domain.ny_pad, domain.n, nphy_outer, "RenLiu", fd.stc2_catalog)
        K2_int = Util.diag_matrix(k_pad_1d, idx_int, 2)
        A_int = ((domain.dy/domain.dx)**2)*Axx_int + Ayy_int + (domain.dy**2)*K2_int

        # operation matrices for absorinbing domain (EM or RL methods)
        if domain.n == 1:

            Axn_bd, Ayn_bd = Util.build_An(domain.nx_pad, domain.ny_pad, domain.n, "boundary", fd.stc1_onesided)
            Axn_bd = Util.corner_treatment(Axn_bd, 1/np.sqrt(2), domain.nx_pad, domain.ny_pad, domain.n)
            Ayn_bd = Util.corner_treatment(Ayn_bd, 1/np.sqrt(2), domain.nx_pad, domain.ny_pad, domain.n)

            K1_bd = Util.diag_matrix(k_pad_1d, idx_bd, 1)
            K2_bd = Util.diag_matrix(k_pad_1d, idx_bd, 2)

            An_bd = -1*(domain.dy/domain.dx*Axn_bd + Ayn_bd)        
            A_abs = 1j*domain.dy*K1_bd @ An_bd + (domain.dy**2)*K2_bd

            if self.abm == "EM2":

                Axxt_bd, Ayyt_bd = Util.build_At(domain.nx_pad, domain.ny_pad, domain.n, "boundary", fd.stc2_catalog)
                Axxt_bd = Util.corner_treatment(Axxt_bd, 0, domain.nx_pad, domain.ny_pad, domain.n)
                Ayyt_bd = Util.corner_treatment(Ayyt_bd, 0, domain.nx_pad, domain.ny_pad, domain.n)
                A_abs += 0.5*((domain.dy/domain.dx)**2*Axxt_bd + Ayyt_bd)

            A = A_int + A_abs

        elif domain.n >= 2:

            Axn_abs, Ayn_abs = Util.build_An(domain.nx_pad, domain.ny_pad, domain.n, "absorbing_domain", fd.stc1_onesided)
            Axn_abs = Util.corner_treatment(Axn_abs, 1/np.sqrt(2), domain.nx_pad, domain.ny_pad, domain.n)
            Ayn_abs = Util.corner_treatment(Ayn_abs, 1/np.sqrt(2), domain.nx_pad, domain.ny_pad, domain.n)
            
            K2_abs_inner = Util.diag_matrix(k_pad_1d, idx_abs_inner, 2)
            K1_abs = Util.diag_matrix(k_pad_1d, idx_abs, 1)
            K2_abs = Util.diag_matrix(k_pad_1d, idx_abs, 2)

            Axx_abs_inner = Util.build_Axx_abs_inner(domain.nx_pad, domain.ny_pad, domain.n, fd.stc2_catalog)
            Ayy_abs_inner = Util.build_Ayy_abs_inner(domain.nx_pad, domain.ny_pad, domain.n, fd.stc2_catalog)
            A_abs_hh = ((domain.dy/domain.dx)**2)*Axx_abs_inner + Ayy_abs_inner + (domain.dy**2)*K2_abs_inner    
            A_abs_ow = -1j*domain.dy*K1_abs @ (domain.dy/domain.dx*Axn_abs + Ayn_abs) + (domain.dy**2)*K2_abs

            if self.abm == "RL2":

                Axxt_abs, Ayyt_abs = Util.build_At(domain.nx_pad, domain.ny_pad, domain.n, "absorbing_domain", fd.stc2_catalog)
                Axxt_abs = Util.corner_treatment(Axxt_abs, 0, domain.nx_pad, domain.ny_pad, domain.n)
                Ayyt_abs = Util.corner_treatment(Ayyt_abs, 0, domain.nx_pad, domain.ny_pad, domain.n)
                A_abs_ow += 0.5*((domain.dy/domain.dx)**2*Axxt_abs + Ayyt_abs)

            weight_ow, weight_hh = Util.compute_RL_weights(domain.nx_pad, domain.ny_pad, domain.n, self.damping_profile)

            # linear combination of two equations
            A_abs_ow = Util.scale_rows_by_diagonal(A_abs_ow, idx_abs_inner, weight_ow)
            A_abs_hh = Util.scale_rows_by_diagonal(A_abs_hh, idx_abs_inner, weight_hh)
            A_abs = A_abs_hh + A_abs_ow

            A = A_int + A_abs

        return A

class PML(ABM):

    """
    Boundary condition options:
    PML0 = Standard PML with Dirichlet boundary condition.
    PML1 = Hybrid of PML and first-order Engquist–Majda ABC.
    PML2 = Hybrid of PML and second-order Engquist–Majda ABC.
    """

    def __init__(self, abm="PML0", n=10, damping_profile=None):
        super().__init__(abm, n, damping_profile)

    # construct the system matrix
    def build_A(self, domain:Domain, source:Source, fd:FD):

        if domain.n == 1:
            raise ValueError("n must be greater than or equal to 2.")

        # quadratic damping profile for PML
        if self.damping_profile is None:
            
            m = 2
            h = (domain.dx + domain.dy)/2 # averaged grid spacing
            v_max = (domain.v_pad).max()
            self.Rcoeff = 1e-5 # target reflection coefficient

            sigma_max = -(m+1)*v_max*np.log(self.Rcoeff) / (2*h*self.n)
            omega = 2*np.pi*source.freq
            
            self.damping_profile = lambda j, n: 1 / (1 + 1j*(sigma_max/omega)*(j/(n-1))**m)
    
        k_pad = 2*np.pi*source.freq/domain.v_pad # wavenumber
        k_pad_1d = k_pad.flatten()

        if fd.order == 2:
            nphy_outer = 1

        elif fd.order == 4:
            nphy_outer = 2

        idx_int, idx_abs, idx_abs_inner, idx_bd = Util.get_idx(domain.nx_pad, domain.ny_pad, domain.n)
        idx_nonbd = np.concatenate((idx_int, idx_abs_inner)).astype(int)

        Axx = Util.build_Axx(domain.nx_pad, domain.ny_pad, domain.n, nphy_outer, "PML", fd.stc2_catalog)
        Ayy = Util.build_Ayy(domain.nx_pad, domain.ny_pad, domain.n, nphy_outer, "PML", fd.stc2_catalog)
        K2_nonbd = Util.diag_matrix(k_pad_1d, idx_nonbd, 2)
        A_laplacian = ((domain.dy/domain.dx)**2)*Axx + Ayy

        Axx_pml = Util.build_Axxpml(domain.nx_pad, domain.ny_pad, domain.n, nphy_outer, fd.pmlstc2_catalog, self.damping_profile)
        Ayy_pml = Util.build_Ayypml(domain.nx_pad, domain.ny_pad, domain.n, nphy_outer, fd.pmlstc2_catalog, self.damping_profile)
        A_laplacian_pml = ((domain.dy/domain.dx)**2)*Axx_pml + Ayy_pml

        Axn, Ayn = Util.build_An(domain.nx_pad, domain.ny_pad, domain.n, "boundary", fd.stc1_onesided)
        Axn = Util.corner_treatment(Axn, 1/np.sqrt(2), domain.nx_pad, domain.ny_pad, domain.n)
        Ayn = Util.corner_treatment(Ayn, 1/np.sqrt(2), domain.nx_pad, domain.ny_pad, domain.n)

        K1_bd = Util.diag_matrix(k_pad_1d, idx_bd, 1)

        if self.abm == "PML0":
            A_bd = Util.dirichlet_bc_matrix(domain.nx_pad, domain.ny_pad)

        elif self.abm == "PML1" or self.abm == "PML2":

            K2_bd = Util.diag_matrix(k_pad_1d, idx_bd, 2)
            A_bd = -1j*domain.dy*K1_bd @ (domain.dy/domain.dx*Axn + Ayn) + (domain.dy**2)*K2_bd

            if self.abm == "PML2":

                Axxt, Ayyt = Util.build_At(domain.nx_pad, domain.ny_pad, domain.n, "boundary", fd.stc2_catalog)
                Axxt = Util.corner_treatment(Axxt, 0, domain.nx_pad, domain.ny_pad, domain.n)
                Ayyt = Util.corner_treatment(Ayyt, 0, domain.nx_pad, domain.ny_pad, domain.n)                                
                A_bd += 0.5*((domain.dy/domain.dx)**2*Axxt + Ayyt)

        A = A_laplacian + A_laplacian_pml + (domain.dy**2)*K2_nonbd + A_bd

        return A