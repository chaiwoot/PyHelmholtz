# Authors: Sirawit Inpuak and Chaiwoot Boonyasiriwat

from abc import ABC, abstractmethod   # for abstract base class
import numpy as np
from .domain import Domain

class Source(ABC):

    def __init__(self, freq=2e9, source_type="point_source"):
        self.freq = freq # freq = source frequency
        self.source_type = source_type

    @abstractmethod
    def build_b(self):
        pass

# Class for point sources
class PointSource(Source):

    def __init__(self, freq=2e9, source_type="point_source", xs=0., ys=0., strength=1):

        super().__init__(freq, source_type)
        self.xs = xs # (xs,ys) = position of point source
        self.ys = ys
        self.strength = strength

    def build_b(self, domain:Domain, n):

        nx_pad, ny_pad = domain.nx_pad, domain.ny_pad
        h = domain.h
        xs, ys = self.xs, self.ys
        isx, isy = int((xs-domain.xmin)/h) + n, int((ys-domain.ymin)/h) + n

        b = np.zeros([ny_pad, nx_pad])
        b[isy, isx] = -(self.strength)
        b = b.flatten()

        return b

# Class for plane waves
class PlaneWave(Source):

    # c0 = background wave speed
    # theta = propagation direction of plane wave with respect to the x-axis
    def __init__(self, freq=2e9, source_type="plane_wave", c0=299792458.0, strength=1., theta=0., xzp=0., yzp=0.):

        super().__init__(freq, source_type)
        self.c0 = c0
        self.theta = theta
        self.strength = strength
        self.xzp = xzp
        self.yzp = yzp

    def build_b(self, domain:Domain, n):
        
        theta_rad = np.deg2rad(self.theta)
        self.k0 = 2.*np.pi*self.freq/self.c0      # background wavenumber
        kx = self.k0*np.cos(theta_rad)
        ky = self.k0*np.sin(theta_rad)

        xmin_pad, xmax_pad = domain.xmin - n*domain.h, domain.xmax + n*domain.h
        ymin_pad, ymax_pad = domain.ymin - n*domain.h, domain.ymax + n*domain.h

        x_pad = np.linspace(xmin_pad, xmax_pad, domain.nx+2*n, True)
        y_pad = np.linspace(ymin_pad, ymax_pad, domain.ny+2*n, True)

        xx_pad, yy_pad = np.meshgrid(x_pad, y_pad)
        term1 = np.exp(1j*kx*(xx_pad-self.xzp))
        term2 = np.exp(1j*ky*(yy_pad-self.yzp))
        ui = self.strength*term1*term2
        
        omega = 2.*np.pi*self.freq
        self.b = (-(domain.h**2)*(omega**2)*((1/domain.v_pad)**2 - (1/self.c0)**2)*ui).flatten()
        self.ui = ui[n:-n,n:-n]

        return self.b

# for future implementation
# class GaussianBeam(Source):
#     def __init__(self, freq=2e9, source_type="gaussian_beam"):
#         super().__init__(freq, source_type)