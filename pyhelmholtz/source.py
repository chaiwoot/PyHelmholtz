# Authors: Sirawit Inpuak and Chaiwoot Boonyasiriwat 

from abc import ABC, abstractmethod
import numpy as np
from .domain import Domain

__all__ = ["Source", "PointSource", "PlaneWave"]


class Source(ABC):
    """Abstract base class for sources in the Helmholtz equation"""

    def __init__(
        self,
        freq: float = 2e9, 
        source_type: str = "point_source"
    ) -> None:

        self.freq = freq
        self.source_type = source_type

    @abstractmethod
    def build_b(self, domain: Domain) -> np.ndarray:

        """
        Build the source vector "b" for the linear system "Ax = b"
 
        Parameters:
            domain: computational domain with padding
 
        Returns:
            source vector of shape (nx*ny, )
        """

        pass

    
class PointSource(Source):
    """Point source(s) supporting both single and multiple point sources via array inputs"""

    def __init__(
        self,
        freq: float = 2e9,
        source_type: str = "point_source",
        xs: float | np.ndarray = 0.0,
        ys: float | np.ndarray = 0.0,
        strength: complex | np.ndarray = 1.0
    ) -> None:  

        """
        Initialize point source(s) (scalar or array-based)

        Parameters:
            freq: source frequency in Hz (default: 2x10^9)
            source_type: source type (default: "point_source")
            xs: source(s) in x-coordinate (can be scalar or numpy array)
            ys: source(s) in y-coordinate (can be scalar or numpy array)
            strength: source strength (can be scalar or numpy array)
        """

        super().__init__(freq, source_type)

        self.xs = xs
        self.ys = ys
        self.strength = strength

        self.n_pointsource = len(np.atleast_1d(xs))

    def build_b(self, domain:Domain) -> np.ndarray:

        # convert physical coordinates to grid indices (accounting for padding)
        isx = np.asarray((self.xs-domain.xpmin)/domain.hx).astype(np.int32) + domain.n
        isy = np.asarray((self.ys-domain.ypmin)/domain.hy).astype(np.int32) + domain.n
        isrc = isx + domain.nx*isy

        # create source vector b for the linear system Ax = b
        b = np.zeros(domain.ny*domain.nx, dtype=np.complex128)
        b[isrc] = -(self.strength)

        return b


class PlaneWave(Source):
    """Plane wave source in a specified direction"""

    def __init__(
        self,
        freq: str = 2e9,
        source_type: str = "plane_wave",
        v0: float = 299792458.0,
        strength: complex = 1.0,
        theta: float = 0.0,
        x_zerophase: float = 0.0,
        y_zerophase: float =0.0
    ) -> None:
        
        """
        Initialize a plane wave source.
 
        Parameters:
            freq: source frequency in Hz (default: 2x10^9)
            source_type: source type (default: "plane_wave")
            v0: background wave speed in m/s (default: 299792458.0)
            strength: wave amplitude (default: 1.0)
            theta: propagation direction with respect to x-axis in degrees (default: 0.0)
            x_zerophase: x-coordinate of zero-phase reference point (default: 0.0)
            y_zerophase: y-coordinate of zero-phase reference point (default: 0.0)
 
        Note:
            The incident plane wave is:
            u_inc(x,y) = strength * exp(i*kx*(x-x_zerophase)) * exp(i*ky*(y-y_zerophase)),
            where kx = k0*cos(angle), ky = k0*sin(angle)
        """

        super().__init__(freq, source_type)

        self.v0 = v0
        self.theta = theta
        self.strength = strength
        self.x_zerophase = x_zerophase
        self.y_zerophase = y_zerophase

    def build_b(self, domain:Domain) -> np.ndarray:

        # compute wavenumber components
        self.k0 = 2.*np.pi*self.freq/self.v0
        kx = self.k0*np.cos(np.deg2rad(self.theta))
        ky = self.k0*np.sin(np.deg2rad(self.theta))

        # create padded coordinate grid
        xmin, xmax = domain.xpmin - domain.n*domain.hx, domain.xpmax + domain.n*domain.hx
        ymin, ymax = domain.ypmin - domain.n*domain.hy, domain.ypmax + domain.n*domain.hy

        x = np.linspace(xmin, xmax, domain.nx)
        y = np.linspace(ymin, ymax, domain.ny)

        xx, yy = np.meshgrid(x, y)

        # Compute incident field on computational domain (padded domain)
        term1 = np.exp(1j*kx*(xx - self.x_zerophase))
        term2 = np.exp(1j*ky*(yy - self.y_zerophase))
        self.uinc_2d = self.strength*term1*term2

        # extract physical domain incident field for reference
        self.uincp_2d = (self.uinc_2d)[domain.n:-domain.n, domain.n:-domain.n]

        # compute source vector b for the linear system Ax = b
        omega = 2.*np.pi*self.freq
        b = (-(domain.hy**2)*(omega**2)*((1/domain.v_2d)**2 - (1/self.v0)**2)*self.uinc_2d).flatten()
        self.b = b.astype(np.complex128)

        return self.b
