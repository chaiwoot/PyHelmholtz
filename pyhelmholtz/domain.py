# Authors: Sirawit Inpuak and Chaiwoot Boonyasiriwat

import numpy as np
from .util import Util

class Domain:
    def __init__(self, limits=(-1, 1, -1, 1), h=0.01, v=299792458, positive_downward=False):
        self.positive_downward = positive_downward
        if np.isscalar(h):
            self.dx, self.dy = h, h
            self.h = h
        elif len(h) == 2:
            self.dx, self.dy = h[0], h[1]
            self.h = (h[0]+ h[1]) / 2
    
        if np.isscalar(limits):
            # xmin = ymin = 0, xmax = ymax = limits
            self.xmin, self.xmax, self.ymin, self.ymax = 0, limits, 0, limits
        elif len(limits) == 2:
            # limits = (xmax,ymax), xmin = ymin = 0
            self.xmin, self.xmax, self.ymin, self.ymax = 0, limits[0], 0, limits[1]
        elif len(limits) == 4:
            # limits = (xmin,xmax,ymin,ymax)
            self.xmin, self.xmax, self.ymin, self.ymax = limits

        self.nx = int((self.xmax - self.xmin)/self.dx) + 1
        self.ny = int((self.ymax - self.ymin)/self.dy) + 1
        self.x = np.linspace(self.xmin, self.xmax, self.nx, True)
        self.y = np.linspace(self.ymin, self.ymax, self.ny, True)
        
        if np.isscalar(v):
            self.v = v*np.ones((self.ny, self.nx))
        else:
            if self.positive_downward:
                self.v = np.flipud(np.array(v))  # in case the input v is a collection, not ndarray
            else:
                self.v = np.array(v)  # in case the input v is a collection, not ndarray
                
            if len(self.v.shape) == 2:
                self.ny, self.nx = self.v.shape
                self.xmin, self.ymin = 0., 0.
                self.xmax, self.ymax = (self.nx-1)*h, (self.ny-1)*h
                self.x = np.linspace(self.xmin, self.xmax, self.nx, True)
                self.y = np.linspace(self.ymin, self.ymax, self.ny, True)
            else:
                raise Exception("v must be a 2D array!")

    def pad_velocity(self, n): # pad the velocity array v with n cells in all directions

        self.v_pad = Util.pad_array2d(self.v, n)
        self.n = n
        self.nx_pad = self.nx+2*n
        self.ny_pad = self.ny+2*n

    # Added by Chaiwoot on Dec 9, 2025
    # modified code from the removed method add_shape()
    def add_circle(self, center, radius, vel):
            xc, yc = center
            xx, yy = np.meshgrid(self.x, self.y)
            xx, yy = xx.flatten(), yy.flatten()
            distance_sq = (xx-xc)**2 + (yy-yc)**2
            self.v = self.v.flatten()
            self.v[distance_sq<=radius*radius] = vel
            self.v = self.v.reshape(self.ny, self.nx)

    # Added by Chaiwoot on Dec 9, 2025
    # code from the removed method add_shape()
    def add_rectangle(self, bounding_box, vel):
            x1, x2, y1, y2 = bounding_box
            ix1 = int((x1-self.xmin)/self.h)
            ix2 = int((x2-self.xmin)/self.h)
            iy1 = int((y1-self.ymin)/self.h)
            iy2 = int((y2-self.ymin)/self.h)
            self.v[iy1:iy2+1, ix1:ix2+1] = vel

    # This method checks if the domain is homogeneous
    # added by Chaiwoot on Nov 30, 2025
    def is_homogeneous(self):
        return len(np.unique(self.v)) == 1

# renamed to indicate that it is a private function
# modified by Chaiwoot on Nov 28, 2025
def _build_circle_object(
        domaininfo: tuple[float, float, float, float, int, int],
        objectinfo: tuple[float, float, float, float, float]       
    ) -> np.ndarray:

    # build an curved object with smooth pixel technique
    xmin, xmax, ymin, ymax, nx, ny = domaininfo
    xc, yc, radius, valmat, valbg = objectinfo
    dx, dy = (xmax-xmin)/(nx-1), (ymax-ymin)/(ny-1)
    xnew = np.linspace(xmin-0.5*dx, xmax+0.5*dx, nx+1)
    ynew = np.linspace(ymin-0.5*dy, ymax+0.5*dy, ny+1)

    xnew2d, ynew2d = np.meshgrid(xnew, ynew)
    xnew1d, ynew1d = xnew2d.flatten(), ynew2d.flatten()
    dist1d = np.sqrt((xnew1d-xc)**2 + (ynew1d-yc)**2)
    
    mat2d_ = valbg*np.ones([ny+1, nx+1])
    mat1d_ = mat2d_.flatten()            
    mat1d_[dist1d<=radius] = valmat
    mat2d_ = mat1d_.reshape(ny+1, nx+1)
    
    mat2d = _average_matrix(mat2d_)

    return mat2d

# renamed to indicate that it is a private function
# modified by Chaiwoot on Nov 28, 2025
def _average_matrix(A1: np.ndarray) -> np.ndarray:
    ny, nx = A1.shape    
    A2 = np.zeros([ny-1, nx-1])
    A2[0:ny-1, 0:nx-1] = 0.25*(A1[0:ny-1, 0:nx-1] + A1[0:ny-1, 1:nx] + A1[1:ny, 0:nx-1] + A1[1:ny, 1:nx])
    return A2

# added by Chaiwoot on Nov 28, 2025
__all__ = ["Domain"]