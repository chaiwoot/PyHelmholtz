# Authors: Sirawit Inpuak and Chaiwoot Boonyasiriwat

import numpy as np
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod   # for abstract base class

from .domain import Domain
from .fd import FD
from .renliu import system_matrix_rl
from .pml import system_matrix_pml
from .source import Source
from .util import Util

__all__ = ["ABM", "RenLiu", "PML"]

class ABM(ABC):

    def __init__(self, abm, n, damping_profile):
        self.abm = abm                            # type of ABM
        self.n = n                                # thickness of absorbing layer in unit of cells
        self.damping_profile = damping_profile    # damping profile

    @abstractmethod
    def build_A(self):
        pass

class RenLiu(ABM):

    """
    Ren-Liu hybrid absorbing boundary condition (ABC) of the Engquist-Majda ABC.
    When n = 1, the Engquist-Majda ABC is used.
    When n > 1, the Ren-Liu hybrid ABC is used.
    The order of the ABC is specified by setting em_order = 1 or 2.
    """

    def __init__(self, abm="EM2", n=1, damping_profile=None):
        super().__init__(abm, n, damping_profile)
        if abm == "EM1":
            self.n = 1           # thickness must be 1 cell
            self.em_order = 1    # use first-order Engquist-Majda ABC
        elif abm == "EM2":
            self.n = 1           # thickness must be 1 cell
            self.em_order = 2    # use second-order Engquist-Majda ABC
        elif abm == "RL1":
            self.n = n           # n should be larger than 1 cell to be absorbing boundary layer (ABL)
            self.em_order = 1    # use hybrid of first-order Engquist-Majda ABC
            if damping_profile == None:
                self.damping_profile = lambda i, nly: (i/nly)**2
            else:
                self.damping_profile = damping_profile
        elif abm == "RL2":
            self.n = n           # n should be larger than 1 cell to be ABL
            self.em_order = 2    # use hybrid of second-order Engquist-Majda ABC
            if damping_profile == None:
                self.damping_profile = lambda i, nly: (i/nly)**2
            else:
                self.damping_profile = damping_profile

    # construct the system matrix
    def build_A(self, domain:Domain, source:Source, fd:FD):
        gridinfo = domain.nx, domain.ny, domain.n, domain.dx, domain.dy   # grid information
        k_pad = 2*np.pi*source.freq/domain.v_pad                          # wavenumber grid
        A = system_matrix_rl(gridinfo, k_pad, self.em_order, fd.order, self.damping_profile)
        return A

class PML(ABM):

    """
    PML and hybrid of PML and Engquist-Majda ABC.
    When em_order = 0, the standard PML is used.
    When em_order = 1 or 2, the hybrid of PML and ABC is used.
    """

    def __init__(self, abm="PML", n=10, damping_profile=None):
        super().__init__(abm, n, damping_profile)
        if abm == "PML0":
            self.em_order = 0   # Dirichlet BC: u = 0 is used at outer nodes
        elif abm == "PML1":
            self.em_order = 1   # first-order Engquist-Majda ABC is used at outer nodes
        elif abm == "PML2":
            self.em_order = 2   # second-order Engquist-Majda ABC is used at outer nodes

    # construct the system matrix
    def build_A(self, domain:Domain, source:Source, fd:FD):
        # quadratic damping profile for PML
        if self.damping_profile is None:
            m = 2
            v_max = (domain.v_pad).max()
            R = 1e-5                        # target reflection coefficient
            h = (domain.dx + domain.dy)/2   # averaged grid spacing
            sigma_max = -(m+1)*v_max*np.log(R) / (2*h*self.n)
            omega = 2*np.pi*source.freq
            self.damping_profile = lambda j, nly: 1 / (1 + 1j*(sigma_max/omega)*(j/nly)**m)

        gridinfo = domain.nx, domain.ny, domain.n, domain.dx, domain.dy   # grid information
        k_pad = 2*np.pi*source.freq/domain.v_pad                          # wavenumber grid
        A = system_matrix_pml(gridinfo, k_pad, self.em_order, fd.order, self.damping_profile)
        
        return A

# class EM1(ABM):

#     def __init__(self, abm="EM1", n=1, damping_profile=None):
#         super().__init__(abm, n, damping_profile)
#         self.n = 1

#     def build_A(self, domain:Domain, source:Source, fd:FD):
#         em_order = 1
#         k_pad = 2*np.pi*source.freq/domain.v_pad # wavenumber
#         gridinfo = domain.nx, domain.ny, domain.n, domain.dx, domain.dy
#         A = helmholtz_em(gridinfo, k_pad, em_order, fd.order, self.damping_profile)
#         return A

# class EM2(ABM):

#     def __init__(self, abm="EM2", n=1, damping_profile=None):
#         super().__init__(abm, n, damping_profile)
#         self.n = 1

#     def build_A(self, domain:Domain, source:Source, fd:FD):
#         em_order = 2        
#         k_pad = 2*np.pi*source.freq/domain.v_pad # wavenumber
#         gridinfo = domain.nx, domain.ny, domain.n, domain.dx, domain.dy
#         A = helmholtz_em(gridinfo, k_pad, em_order, fd.order, self.damping_profile)
#         return A

# class HEM1(ABM):

#     def __init__(self, abm="HEM1", n=10, damping_profile=lambda i, nly: (i/nly)**2):
#         super().__init__(abm, n, damping_profile)

#     def build_A(self, domain:Domain, source:Source, fd:FD):
#         em_order = 1        
#         k_pad = 2*np.pi*source.freq/domain.v_pad # wavenumber
#         gridinfo = domain.nx, domain.ny, domain.n, domain.dx, domain.dy
#         A = helmholtz_em(gridinfo, k_pad, em_order, fd.order, self.damping_profile)
#         return A

# class HEM2(ABM):

#     def __init__(self, abm="HEM2", n=10, damping_profile=lambda j, nly: (j/nly)**2):
#         super().__init__(abm, n, damping_profile)

#     def build_A(self, domain:Domain, source:Source, fd:FD):
#         em_order = 2
#         k_pad = 2*np.pi*source.freq/domain.v_pad # wavenumber
#         gridinfo = domain.nx, domain.ny, domain.n, domain.dx, domain.dy
#         A = helmholtz_em(gridinfo, k_pad, em_order, fd.order, self.damping_profile)
#         return A

# class PML(ABM):

#     def __init__(self, abm="PML", n=10, damping_profile=None):
#         super().__init__(abm, n, damping_profile)

#     def build_A(self, domain:Domain, source:Source, fd:FD):

#         # default PML damping profile
#         if self.damping_profile is None:
#             m = 2
#             v_max = (domain.v_pad).max()

#             if fd.order == 2:
#                 Rcoeff = 1e-5 # FD2 = 1e-5

#             elif fd.order == 4:
#                 Rcoeff = 1e-7 # FD4 = 1e-7

#             sigma_max = -(m+1)*v_max*np.log(Rcoeff) / (2*domain.h*self.n)
#             omega = 2*np.pi*source.freq
#             self.damping_profile = lambda j, nly: 1 / (1 + 1j*(sigma_max/omega)*(j/nly)**m)

#         em_order = 0 # apply PEC condition at boundary (4 sides + 4 corners)
#         k_pad = 2*np.pi*source.freq/domain.v_pad # wavenumber

#         gridinfo = domain.nx, domain.ny, domain.n, domain.dx, domain.dy
#         A = helmholtz_pml(gridinfo, k_pad, em_order, fd.order, self.damping_profile)

#         return A

# class PMLEM1(ABM):

#     def __init__(self, abm="PMLEM1", n=10, damping_profile=None):
#         super().__init__(abm, n, damping_profile)

#     def build_A(self, domain:Domain, source:Source, fd:FD):

#         # default PML damping profile
#         if self.damping_profile is None:
#             m = 2
#             v_max = (domain.v_pad).max()
#             Rcoeff = 1e-5
#             sigma_max = -(m+1)*v_max*np.log(Rcoeff) / (2*domain.h*self.n)
#             omega = 2*np.pi*source.freq
#             self.damping_profile = lambda j, nly: 1 / (1 + 1j*(sigma_max/omega)*(j/nly)**m)

#         em_order = 1 # apply EM1 condition at boundary (4 sides + 4 corners)
#         k_pad = 2*np.pi*source.freq/domain.v_pad # wavenumber

#         gridinfo = domain.nx, domain.ny, domain.n, domain.dx, domain.dy
#         A = helmholtz_pml(gridinfo, k_pad, em_order, fd.order, self.damping_profile)
        
#         return A

# class PMLEM2(ABM):

#     def __init__(self, abm="PMLEM2", n=10, damping_profile=None):
#         super().__init__(abm, n, damping_profile)

#     def build_A(self, domain:Domain, source:Source, fd:FD):

#         # default PML damping profile
#         if self.damping_profile is None:
#             m = 2
#             v_max = (domain.v_pad).max()
#             Rcoeff = 1e-5
#             h = (domain.dx + domain.dy)/2 # average grid space
#             sigma_max = -(m+1)*v_max*np.log(Rcoeff) / (2*h*self.n)
#             omega = 2*np.pi*source.freq
#             self.damping_profile = lambda j, nly: 1 / (1 + 1j*(sigma_max/omega)*(j/nly)**m)

#         em_order = 2 # apply EM2 condition at boundary (4 sides + 4 corners)
#         k_pad = 2*np.pi*source.freq/domain.v_pad # wavenumber

#         gridinfo = domain.nx, domain.ny, domain.n, domain.dx, domain.dy
#         A = helmholtz_pml(gridinfo, k_pad, em_order, fd.order, self.damping_profile)
        
#         return A
