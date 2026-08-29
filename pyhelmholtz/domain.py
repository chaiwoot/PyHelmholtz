# Authors: Sirawit Inpuak and Chaiwoot Boonyasiriwat

import numpy as np
from .util import Util

__all__ = ["Domain"]

class Domain:

    """
    2D computational domain for wave simulation
        
    Manages spatial discretization, velocity model, and boundary padding
    """

    def __init__(
        self,
        limits: float | tuple[float, ...] = (-1, 1, -1, 1),
        h: float | tuple[float, float] = 0.01,
        v: float | np.ndarray =299792458,
        positive_downward: bool = False
    ) -> None:

        """
        Initialize computational domain

        Parameters:        
            limits : domain extent
                - scalar: (0, L, 0, L)
                - 2-tuple: (0, Lx, 0, Ly)
                - 4-tuple: (xpmin, xpmax, ypmin, ypmax)
            h: grid spacing
                - scalar: hx = hy = h
                - tuple: (hx, hy)
            v: velocity model
            positive_downward: if True, flip velocity array along y-axis.
        """
        
        self.positive_downward = positive_downward

        # handle grid spacing
        if np.isscalar(h):
            self.hx, self.hy = h, h
            self.h = h

        elif len(h) == 2:
            self.hx, self.hy = h[0], h[1]
            self.h = (h[0]+ h[1]) / 2

        # handle physical domain bounds
        if np.isscalar(limits):
            self.xpmin, self.xpmax, self.ypmin, self.ypmax = 0, limits, 0, limits

        elif len(limits) == 2:
            self.xpmin, self.xpmax, self.ypmin, self.ypmax = 0, limits[0], 0, limits[1]

        elif len(limits) == 4:
            self.xpmin, self.xpmax, self.ypmin, self.ypmax = limits

        # compute physical grid dimensions
        self.nxp = round((self.xpmax - self.xpmin)/self.hx) + 1
        self.nyp = round((self.ypmax - self.ypmin)/self.hy) + 1
        self.xp = np.linspace(self.xpmin, self.xpmax, self.nxp)
        self.yp = np.linspace(self.ypmin, self.ypmax, self.nyp)

        # initialize velocity model
        if np.isscalar(v):
            self.vp_2d = v*np.ones([self.nyp, self.nxp])

        else:

            if self.positive_downward:
                self.vp_2d = np.flipud(np.array(v)) 

            else:
                self.vp_2d = np.array(v) 
                
            if len(self.vp_2d.shape) == 2:

                self.nyp, self.nxp = v.shape
                self.xpmin, self.ypmin = 0., 0.
                self.xpmax, self.ypmax = (self.nxp-1)*self.hx, (self.nyp-1)*self.hy

                self.xp = np.linspace(self.xpmin, self.xpmax, self.nxp)
                self.yp = np.linspace(self.ypmin, self.ypmax, self.nyp)

            else:
                raise ValueError("Velocity model must be a 2D array.")

    def pad_velocity(self, n: int) -> None:

        # pad velocity array with n cells in all directions (left, right, bottom and top)
        self.v_2d = Util.pad_array2d(self.vp_2d, n)
        self.n = n
        self.nx = self.nxp + 2*n
        self.ny = self.nyp + 2*n

    def add_circle(
        self,
        center: tuple[float, float],
        radius: float,
        velocity_value: float
    ) -> None :

        """
        Embed circular region with specified velocity

        Parameters:
            center : circle center coordinates (xc, yc)
            radius : circle radius             
            velocity_value: velocity value inside circle
        """

        xc, yc = center
        xx, yy = np.meshgrid(self.xp, self.yp)
        xx, yy = xx.flatten(), yy.flatten()
        distance_sq = (xx-xc)**2 + (yy-yc)**2

        vp_1d = self.vp_2d.flatten()
        vp_1d[distance_sq<=radius*radius] = velocity_value
        self.vp_2d = vp_1d.reshape([self.nyp, self.nxp])

    def add_rectangle(
        self,
        bounding_box: tuple[float, float, float, float],
        velocity_value: float
    ) -> None:
            
        """
        Embed rectangular region with specified velocity
        
        Parameters:
        bounding_box: rectangle coordinates (x1, x2, y1, y2) (tuple)
        velocity_value: velocity value inside rectangle (float)
        """
        x1, x2, y1, y2 = bounding_box

        # convert physical coordinates to grid indices
        ix1, ix2 = round((x1-self.xpmin)/self.hx), round((x2-self.xpmin)/self.hx)
        iy1, iy2 = round((y1-self.ypmin)/self.hy), round((y2-self.ypmin)/self.hy)

        self.vp_2d[iy1:iy2+1, ix1:ix2+1] = velocity_value

    def is_homogeneous(self):

        # check if domain has uniform velocity.
        return len(np.unique(self.vp_2d)) == 1
