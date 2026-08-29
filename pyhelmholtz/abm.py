# Authors: Sirawit Inpuak and Chaiwoot Boonyasiriwat

from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_matrix

from .domain import Domain
from .source import Source
from .fd import FD
from .util import Util

__all__ = ["ABM", "RenLiu", "PML"]


def _build_A_interior(domain:Domain, source:Source, fd:FD, noverlap:int) -> csr_matrix:
    
    """
    Assemble Helmholtz equation operator for interior domain

    Constructs discrete Helmholtz operator: (d^2/dx^2 + d^2/dy^2 + k^2)u = 0
    using 2nd-order finite difference stencils

    Parameters
    ----------

    domain : Domain
        computational domain with velocity model
    source : Source
        source information
    fd : FD
        stencil catalog with given FD accuracy order
    noverlap : int
        Overlap region size (0 for non-PML, >0 for PML)   

    Returns
    -------
    csr_matrix
        Interior Helmholtz operator
    """

    hx, hy = domain.hx, domain.hy
    k_1d = (2*np.pi*source.freq/domain.v_2d).flatten() # wavenumber k
    
    domain_grids = domain.nx, domain.ny, domain.n, noverlap
    idx_data = Util.get_idx(domain.nx, domain.ny, domain.n, noverlap, "interior")

    # 2nd-order derivatives in x and y
    Ai_xx = Util.build_system_matrix(domain_grids, idx_data["interior"], fd.stc2_catalog, "x")
    Ai_yy = Util.build_system_matrix(domain_grids, idx_data["interior"], fd.stc2_catalog, "y")

    # wavenumber diagonal matrix: k^2
    Ki2 = Util.diag_matrix(k_1d, idx_data["interior"], 2)

    # discrete Helmholtz operator for interior domain, scaled with hy^2
    # hy^2 * (d^2u/dx^2 + d^2u/dy^2 + k^2 u) = 0
    Ai = ((hy/hx)**2)*Ai_xx + Ai_yy + (hy**2)*Ki2 

    return Ai

# =========== Absorbing Boundary Method Base Class ===========


class ABM(ABC):

    """
    Abstract base class for absorbing boundary methods (ABM) 
        
    Defines interface for constructing discrete system matrices
    for wave equation with various absorbing boundary conditions
    """

    def __init__(self, abm: str, n: int, damping_profile: callable = None) -> None:
        
        """
        Initialize ABM parameters.
        
        Parameters
        ----------
        abm : str
            absorbing boundary method (e.g., "EM2", "RLEM2", "PMLEM2").
        n : int
            transition layer or PML thickness in cells
        damping_profile : callable
            damping profile function
        """

        self.abm = abm                              
        self.n = n                              
        self.damping_profile = damping_profile

    @abstractmethod
    def build_A(self, domain: Domain, source: Source, fd:FD) -> csr_matrix:

        """
        Build complete discrete system matrix "A"
        
        Parameters
        ----------
        domain : Domain
            computational domain
        source : Source
            source information
        fd : FD
            stencil catalog with give FD accuracy order
        
        Returns
        -------
        csr_matrix
            discrete system matrix "A"
        """
        pass  

# =========== Engquist-Majda, Higdon and Ren-Liu Hybrid Absorbing Boundary Conditions (noverlap=0) ===========

class RenLiu(ABM):

    """
    Engquist-Majda, Higdon and Ren-Liu absorbing boundary methods
    
    Supports:
    - Single boundary layer (nc=1): Engquist-Majda ABCs (EM1, EM2) and Higdon ABC (H2)
    - Multiple transition layers (nc>=2): Ren-Liu hybrid ABC (RLEM1, RLEM2, RLH2)
    - Dirichlet boundary (nc=1): zero wavefield at boundary (applied only for PML0 method)
    
    Zone structure:
    - interior: the Helmholtz equation
    - transition: hybrid formulation    
    - boundary: ABC 
    
    Reference
    ----------
    Ren, Z., & Liu, Y. (2013). A hybrid absorbing boundary condition for
    frequency-domain finite-difference modelling. Journal of Geophysics and Engineering
    """

    def __init__(self, abm="EM2", n=1, damping_profile=None) -> None:        

        """
        Initialize Engquist-Majda, Higdon, or Ren-Liu hybrid absorbing boundary methods
        
        Parameters
        ----------
        abm : str
            six variants: "D0", "EM1", "EM2", "H2", "RLEM1", "RLEM2", "RLH2"
        n : int
            n = 1 for boundary operator ("D0", "EM1", "EM2", "H2")
            n >= 2 for RenLiu ("RLEM1", "RLEM2", "RLH2").
        damping_profile : callable
            default: quadratic profile (depth/n)^2
        
        Raises
        ------
        ValueError
            If single-layer boundary operator is used with nc != 1
            If RenLiu variant is used with nc < 2
        """

        super().__init__(abm, n, damping_profile)

        # Validate nc for each method type
        if abm in ["D0", "EM1", "EM2", "H2"]:
            if n != 1:
                raise ValueError(f"{abm} requires n=1 (single boundary layer), got n={n}")
            self.is_boundary_operator = True
            self.is_renliu_hybrid = False
            
        elif abm in ["RLEM1", "RLEM2", "RLH2"]:
            if n < 2:
                raise ValueError(f"{abm} requires n>=2 (transition layers), got n={n}")
            self.is_boundary_operator = False
            self.is_renliu_hybrid = True
            
            # Set default damping profile if not provided
            if damping_profile is None:
                self.damping_profile = lambda i, nly: (i/nly)**2
        else:
            raise ValueError(f"Unknown abm type: {abm}. Use D0/EM1/EM2/H2 or RLEM1/RLEM2/RLH2")

    def _build_A_outermost_boundary(self, domain:Domain, source:Source, fd:FD) -> csr_matrix:

        """
        Construct boundary operator for EM method
                
        Parameters
        ----------
        domain, source, fd : Domain, Source, FD
            problem parameters and FD stencils for the desired accuracy
        
        Returns
        -------
        csr_matrix
            discrete boundary operator matrix "Ab"
        """

        noverlap_fixed = 0

        if self.abm == "D0":
            # Dirichlet boundary condition: wavefield = 0 at boundary for PML0 
            Ab = Util.dirichlet_bc_matrix(domain.nx, domain.ny) 

        elif self.abm in ("EM1", "EM2", "H2"):

            hx, hy = domain.hx, domain.hy
            k_1d = (2*np.pi*source.freq/domain.v_2d).flatten() # wavenumber k

            n_fixed = 1  
            domain_grids = domain.nx, domain.ny, self.n, noverlap_fixed
            idx_data = Util.get_idx(domain.nx, domain.ny, self.n, noverlap_fixed, "boundary")

            # 1st-order normal derivatives: du/dn
            # scaling by -1 compensates for flipping the FD coefficients
            Ab_xn = -1*Util.build_system_matrix(domain_grids, idx_data["boundary_lr"], fd.stc1_onesided, "x")
            Ab_yn = -1*Util.build_system_matrix(domain_grids, idx_data["boundary_bt"], fd.stc1_onesided, "y")

            # wavenumber diagonal matrices: k, k^2
            Kb1 = Util.diag_matrix(k_1d, idx_data["boundary"], 1) 
            Kb2 = Util.diag_matrix(k_1d, idx_data["boundary"], 2)

            if self.abm in ("EM1", "EM2"):

                # corner treatment: scale by 1/sqrt(2) to account for the diagonal normal vector
                Ab_xn = Util.corner_treatment(Ab_xn, 1/np.sqrt(2), domain.nx, domain.ny, n_fixed)
                Ab_yn = Util.corner_treatment(Ab_yn, 1/np.sqrt(2), domain.nx, domain.ny, n_fixed)
    
                Ab_n = (hy/hx)*Ab_xn + Ab_yn # unscaling with grid spacing

                # discrete EM1 operator scaling with hy^2
                # hy^2 * (i*k*du/dn + k^2 u) = 0
                Ab = 1j*hy*Kb1 @ Ab_n + (hy**2)*Kb2  

            if self.abm == "EM2":

                # EM2 adds 2nd-order tangential derivatives: (1/2)*(d^2u/dt^2)
                Ab_xxt = Util.build_system_matrix(domain_grids, idx_data["boundary_bt"], fd.stc2_catalog, "x")
                Ab_yyt = Util.build_system_matrix(domain_grids, idx_data["boundary_lr"], fd.stc2_catalog, "y")

                # apply EM1 at corners of boundary layer
                Ab_xxt = Util.corner_treatment(Ab_xxt, 0, domain.nx, domain.ny, n_fixed)
                Ab_yyt = Util.corner_treatment(Ab_yyt, 0, domain.nx, domain.ny, n_fixed)

                # discrete EM2 operator scaling with hy^2
                # hy^2 * (i*k*du/dn + k^2 u + (1/2)*(d^2u/dt^2)) = 0
                Ab += 0.5*((hy/hx)**2*Ab_xxt + Ab_yyt)

            if self.abm == "H2":
        
                # selected incident angles
                angle1, angle2 = np.deg2rad(0), np.deg2rad(45)                
                
                # 2nd-order one-sided derivatives: d^2u/dn^2
                Ab_xxn = Util.build_system_matrix(domain_grids, idx_data["boundary_lr"], fd.stc2_onesided, "x")
                Ab_yyn = Util.build_system_matrix(domain_grids, idx_data["boundary_bt"], fd.stc2_onesided, "y")

                # corner treatment
                Ab_xn = Util.corner_treatment(Ab_xn, 1, domain.nx, domain.ny, n_fixed)
                Ab_yn = Util.corner_treatment(Ab_yn, 1, domain.nx, domain.ny, n_fixed)
                Ab_xxn = Util.corner_treatment(Ab_xxn, 1, domain.nx, domain.ny, n_fixed)
                Ab_yyn = Util.corner_treatment(Ab_yyn, 1, domain.nx, domain.ny, n_fixed)
                Kb2 = Util.corner_treatment(Kb2, 2, domain.nx, domain.ny, n_fixed)

                # discrete H2 operator scaling with hy^2 
                Ab_n = (hy/hx)*Ab_xn + Ab_yn
                Ab = (hy/hx)**2*Ab_xxn + Ab_yyn
                Ab += -(hy**2)*(np.cos(angle1)*np.cos(angle2))*Kb2
                Ab += -1j*hy*(np.cos(angle1) + np.cos(angle2))*Kb1 @ Ab_n

        return Ab

    def _build_A_transition(self, domain:Domain, source:Source, fd:FD) -> csr_matrix:

        """
        Construct transition zone operator for RL method
        (depth-weighted combination of Helmholtz and EM one-way)
                
        Parameters
        ----------
        domain, source, fd : Domain, Source, FD
            problem parameters and FD stencils for the desired accuracy
        
        Returns
        -------
        csr_matrix
            transition zone operator matrix "At"
        """

        noverlap_fixed = 0
        
        hx, hy = domain.hx, domain.hy
        k_1d = (2*np.pi*source.freq/domain.v_2d).flatten() # wavenumber k
        
        domain_grids = domain.nx, domain.ny, domain.n, noverlap_fixed
        idx_data = Util.get_idx(domain.nx, domain.ny, domain.n, noverlap_fixed, "transition")

        if self.abm in ("RLEM1", "RLEM2", "RLH2"):
            
            # 1st-order normal derivatives: du/dn
            # scaling by -1 compensates for flipping the FD coefficients
            At_xn = -1*Util.build_system_matrix(domain_grids, idx_data["transition_lr"], fd.stc1_onesided, "x")
            At_yn = -1*Util.build_system_matrix(domain_grids, idx_data["transition_bt"], fd.stc1_onesided, "y")           

            # 2nd-order derivatives: d^2u/dx^2, d^2u/dy^2
            At_xx = Util.build_system_matrix(domain_grids, idx_data["transition"], fd.stc2_catalog, "x")
            At_yy = Util.build_system_matrix(domain_grids, idx_data["transition"], fd.stc2_catalog, "y")

            # wavenumber diagonal matrices: k, k^2
            Kt1 = Util.diag_matrix(k_1d, idx_data["transition"], 1)
            Kt2 = Util.diag_matrix(k_1d, idx_data["transition"], 2)

            # discrete Helmholtz operator scaled with hy^2
            # hy^2 * (d^2u/dx^2 + d^2u/dy^2 + k^2 u) = 0
            At_hh = ((hy/hx)**2)*At_xx + At_yy + (hy**2)*Kt2

            if self.abm in ("RLEM1", "RLEM2"):

                # corner treatment: scale by 1/sqrt(2) to account for the diagonal normal vector
                At_xn = Util.corner_treatment(At_xn, 1/np.sqrt(2), domain.nx, domain.ny, domain.n)
                At_yn = Util.corner_treatment(At_yn, 1/np.sqrt(2), domain.nx, domain.ny, domain.n)

                At_n = (hy/hx)*At_xn + At_yn # unscaling with grid spacing

                # discrete EM1 operator scaling with hy^2
                # hy^2 * (i*k*du/dn + k^2 u) = 0
                At_ow = 1j*hy*Kt1 @ At_n + (hy**2)*Kt2

            if self.abm == "RLEM2":

                # RLEM2 add 2nd-order tangential derivatives to EM1 one-way operator
                At_xxt =  Util.build_system_matrix(domain_grids, idx_data["transition_bt"], fd.stc2_catalog, "x")
                At_yyt =  Util.build_system_matrix(domain_grids, idx_data["transition_lr"], fd.stc2_catalog, "y")

                # apply EM1 at corners of transition layer
                At_xxt = Util.corner_treatment(At_xxt, 0, domain.nx, domain.ny, domain.n)
                At_yyt = Util.corner_treatment(At_yyt, 0, domain.nx, domain.ny, domain.n)

                # discrete EM2 operator (scaling with hy^2) applied on transition zone for RLEM2
                # hy^2 * (i*k*du/dn + k^2 u + (1/2)*(d^2u/dt^2)) = 0
                At_ow += 0.5*((hy/hx)**2*At_xxt + At_yyt)

            if self.abm == "RLH2":

                angle1, angle2 = np.deg2rad(0), np.deg2rad(45) # selected incident angles
            
                # 2nd-order one-sided derivatives: d^2u/dn^2
                At_xxn = Util.build_system_matrix(domain_grids, idx_data["transition_lr"], fd.stc2_onesided, "x")
                At_yyn = Util.build_system_matrix(domain_grids, idx_data["transition_bt"], fd.stc2_onesided, "y")

                # corner treatment
                At_xn = Util.corner_treatment(At_xn, 1, domain.nx, domain.ny, domain.n)
                At_yn = Util.corner_treatment(At_yn, 1, domain.nx, domain.ny, domain.n)
                At_xxn = Util.corner_treatment(At_xxn, 1, domain.nx, domain.ny, domain.n)
                At_yyn = Util.corner_treatment(At_yyn, 1, domain.nx, domain.ny, domain.n)
                Kt2 = Util.corner_treatment(Kt2, 2, domain.nx, domain.ny, domain.n)
            
                # add the discrete operator applied on absorbing zone for RLH2
                At_n = (hy/hx)*At_xn + At_yn
                At_ow = (hy/hx)**2*At_xxn + At_yyn
                At_ow += -1j*hy*(np.cos(angle1) + np.cos(angle2))*Kt1 @ At_n
                At_ow += -(hy**2)*(np.cos(angle1)*np.cos(angle2))*Kt2

        # compute depth-dependent weights
        depth1D = Util.depth_from_physical_boundary(domain.nx, domain.ny, domain.n)
        weight_ow = self.damping_profile(depth1D, self.n)
        weight_hh = 1 - weight_ow

        # apply depth-weighted scaling to each operator row of At_ow and At_hh
        At_ow = Util.scale_rows_by_diagonal(At_ow, idx_data["transition"], weight_ow)
        At_hh = Util.scale_rows_by_diagonal(At_hh, idx_data["transition"], weight_hh)

        # assemble transition operator "At"
        At = At_hh + At_ow        

        return At

    def build_A(self, domain:Domain, source:Source, fd:FD) -> csr_matrix:

        """
        Assemble complete system matrix "A"
                
        Returns
        -------
        csr_matrix
            A = A_interior + A_transition + A_boundary
        """

        # discrete Helmholtz operator in interior domain "A_interior"
        noverlap = 0
        A_interior = _build_A_interior(domain, source, fd, noverlap)

        if self.is_boundary_operator:

            A_boundary = self._build_A_outermost_boundary(domain, source, fd)        
            
            # complete system matrix "A"
            A = A_interior + A_boundary 

        elif self.is_renliu_hybrid:
        
            # discrete EM one-way wave equation at boundary "A_boundary"
            if self.abm == "RLEM1":
                bd_operator = RenLiu(abm="EM1")    
            elif self.abm == "RLEM2":
                bd_operator = RenLiu(abm="EM2")
            elif self.abm == "RLH2":
                bd_operator = RenLiu(abm="H2")
            else:
                raise ValueError(f"Unknown Ren-Liu variant: {self.abm}")

            A_boundary = bd_operator._build_A_outermost_boundary(domain, source, fd)

            # discrete Ren-Liu hybrid operator in transition domain "A_transition"
            A_transition = self._build_A_transition(domain, source, fd)

            # complete system matrix "A"
            A = A_interior + A_transition + A_boundary

        return A


# =========== Perfectly Matched Layers (n>1, noverlap>0) ===========

class PML(ABM):

    """
    "Perfectly Matched Layer absorbing boundary condition (PML0/PMLEM1/PMLEM2).
    
    Implements perfectly matched layers using complex coordinate stretching
    to exponentially damp outgoing waves without reflections.
    
    Zone structure:
    - interior: the Helmholtz equation (isolated from PML influences)
    - pml: the Helmholtz with damping in PML region
    - boundary: the Engquist-Majda one-way wave equation or Dirichlet BC
    
    Notes
    -----
    The overlap region (noverlap > 0) ensures gradual transition between
    interior domain and PML zones, reducing reflections at the interface.
    """

    def __init__(self, abm="PML0", n=5, damping_profile=None) -> None:

        """
        Initialize PML absorbing boundary condition.
                
        Parameters
        ----------
        abm : str
            three variants: "PML0", "PMLEM1" and "PMLEM2"
        n : int
            PML thickness in cells (must be >=2)
        damping_profile : callable
            complex damping profile: s(depth) = 1 + i*(sigma_max/omega)*(depth/(n-1))^m for PML stretching
            default: - quadratic damping profile (depth/(n-1))^2 when m = 2
                     - expected normal-incidence reflection coefficient R = 10^(-5)
        
        Raises
        ------
        ValueError
            If n < 2
        """

        super().__init__(abm, n, damping_profile)

        if n == 1:
            raise ValueError("n must be >=2 for PML implementation.")
        
    def _build_A_PML(self, domain:Domain, source:Source, fd:FD) -> csr_matrix:

        """
        Construct PML domain operator with depth-dependent damping in x and y directions       

        Parameters
        ----------
        domain, source, fd : Domain, Source, FD
            problem parameters and FD stencils for the desired accuracy
        
        Returns
        -------
        csr_matrix
            PML operator matrix "Ap"
        """

        # default damping profile for PML if not provided
        if self.damping_profile is None:

            # quadratic PML profile: sigma(j) = sigma_max*(j/n)^m
            m = 2 # polynomial order
            h_avg = (domain.hx + domain.hy)/2 # averaged grid spacing
            v_max = (domain.v_2d).max()
            self.R0 = 1e-5 # target normal-incidence reflection coefficient

            sigma_max = -(m+1)*v_max*np.log(self.R0) / (2*h_avg*(self.n-1))
            omega = 2*np.pi*source.freq

            self.damping_profile = lambda j, n: 1 / (1 + 1j*(sigma_max/omega)*(j/(n-1))**m)
    
        noverlap = fd.noverlap_pml

        hx, hy = domain.hx, domain.hy
        k_1d = (2*np.pi*source.freq/domain.v_2d).flatten() # wavenumber k
        
        domain_grids = domain.nx, domain.ny, domain.n, noverlap
        idx_data = Util.get_idx(domain.nx, domain.ny, domain.n, noverlap, "pml")

        # standard 2nd-order derivatives in x-direction (inside bottom and top of PML zones)
        Ap_xx = Util.build_system_matrix(domain_grids, idx_data["pml_bt_x"], fd.stc2_catalog, "x")

        # standard 2nd-order derivatives in y-direction (inside left and right of PML zones)
        Ap_yy = Util.build_system_matrix(domain_grids, idx_data["pml_lr_y"], fd.stc2_catalog, "y")

        # 2nd-order derivatives with PML in x-direction (inside left and right of PML zones)
        Ap_xxs = Util.build_system_matrix(domain_grids, idx_data["pml_lr_xs"], fd.pmlstc2_catalog, "x", self.damping_profile)

        # 2nd-order derivatives with PML in y-direction (inside bottom and top of PML zones)
        Ap_yys = Util.build_system_matrix(domain_grids, idx_data["pml_bt_ys"], fd.pmlstc2_catalog, "y", self.damping_profile)

        # wavenumber diagonal matrices: k, k^2
        Kp2 = Util.diag_matrix(k_1d, idx_data["pml"], 2)

        # discrete PML-Helmholtz operator scaled with hy^2
        Ap = ((hy/hx)**2)*(Ap_xx + Ap_xxs) + (Ap_yy + Ap_yys) + (hy**2)*Kp2

        return Ap
    
    def build_A(self, domain:Domain, source:Source, fd:FD) -> csr_matrix:

        """
        Assemble complete system matrix "A"
                
        Returns
        -------
        csr_matrix
            A = A_interior + A_pml + A_boundary
        """

        # discrete Helmholtz operator in interior domain "A_interior"
        noverlap = fd.noverlap_pml
        A_interior = _build_A_interior(domain, source, fd, noverlap)

        # discrete EM one-way wave equation at boundary "A_boundary"
        if self.abm == "PML0":
            bd_operator = RenLiu(abm="D0")

        elif self.abm == "PMLEM1":
            bd_operator = RenLiu(abm="EM1")

        elif self.abm == "PMLEM2": 
            bd_operator = RenLiu(abm="EM2")

        elif self.abm == "PMLH2":
            bd_operator = RenLiu(abm="H2")

        A_boundary = bd_operator._build_A_outermost_boundary(domain, source, fd)

        # discrete PML-Helmholtz operator in PML domain "A_pml"
        A_pml = self._build_A_PML(domain, source, fd)

        # complete system matrix "A"
        A = A_interior + A_pml + A_boundary

        return A
