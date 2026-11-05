# Authors: Sirawit Inpuak and Chaiwoot Boonyasiriwat

import numpy as np
from .abm import *
from .domain import *
from .source import *
from .fd import *
from .util import *
from .solver import *

LIGHT_SPEED = 299792458.0 # light speed in vacuum

class Helmholtz:
    
    def __init__(
        self,
        domain = Domain(),
        source = PointSource(),
        abm = RenLiu(),
        fd = FD()
    ):

        self.domain = domain
        self.source = source
        self.abm = abm
        self.fd = fd

        self.domain.pad_velocity(self.abm.n)

    # this method forms the matrices A and b
    def build(self):
        
        A = self.abm.build_A(self.domain, self.source, self.fd)
        b = self.source.build_b(self.domain)
        self.A = A

        return (A, b)

    # This method solve the linear system Ax = b
    def solve(
        self,
        solver: str = "spsolve",
        nprocs: int = 1
    ) -> None:

        nx, ny, n = self.domain.nx, self.domain.ny, self.domain.n

        A, b = self.build()

        # Sparse direct solver of SciPy
        if solver == "spsolve":

            from scipy.sparse.linalg import spsolve
            x = spsolve(A, b)

        # MUMPS: Multifrontal Massively Parallel Sparse direct solver
        elif solver == "mumps":
            x = mumps_solver(A, b, nprocs)

        # Nested dissection
        elif solver == "nested_dissection":
            x = nested_dissection_solver(A, b)            

        # GMRES iterative solver of SciPy
        # Remark: iterative solver is much slower than the above direct solvers
        elif solver == "gmres": 
            x = gmres_solver(A, b, self.source.freq, self.domain.v_2d)

        else:
            raise Exception(f"Solver '{solver}' is not supported. Supported solvers are 'spsolve', 'mumps', 'nested_dissection' and 'gmres'.")

        self.x = x # wavefield solution in 1d-array
        self.x_2d = x.reshape([ny, nx]) # wavefield solution in 2d-array
        self.u = (x.reshape([ny, nx]))[n:-n, n:-n] # wavefield solution in 2d-array (exclude solution in absorbing domain)  


    # This method is a quick tool for visualizing various fields and medium property.
    def viz(
            self,
            data = "solution",  # available data: "solution", "velocity", "incident", "total"
            mode = "real",      # availabel mode: "real", "imag", "abs"
            unit = "m",         # available unit: "km", "m", "cm", "mm", "um", "nm"
            cmap = "jet",       # colormap
            vlim = None,        # value limit [vmin, vmax]
            title = "",
            xlabel = True,
            ylabel = True,
        ) -> None:
        
        import matplotlib.pyplot as plt

        if isinstance(data, str):

            n = self.abm.n

            if data == "velocity":

                viz2D = self.domain.vp_2d
                if title == "": title = "Velocity"

            elif data == "solution":

                viz2D = self.u

                if self.source.source_type == "plane_wave":
                    if title == "": title = "Scattered field"

                else:
                    if title == "": title = "Wavefield"

            elif data == "incident":

                if self.source.source_type == "plane_wave":
                    viz2D = self.source.uincp_2d

                else:
                    viz2D = np.zeros_like(self.u)

                if title == "": title = "Incident field"
                
            elif data == "total":

                if self.source.source_type == "plane_wave":
                    viz2D = self.u + self.source.uincp_2d

                else:
                    viz2D = self.u

                if title == "": title = "Total field"

            else:
                title = ""

        elif isinstance(data, np.ndarray):

            if data.shape == self.u.shape:
                viz2D = data

        if unit == "km": scale = 1e+3
        elif unit == "m" : scale = 1.0
        elif unit == "cm": scale = 1e-2
        elif unit == "mm": scale = 1e-3
        elif unit == "um": scale = 1e-6 # micrometer
        elif unit == "nm": scale = 1e-9 # nanometer

        extent = np.array([self.domain.xpmin, self.domain.xpmax, self.domain.ypmin, self.domain.ypmax]) / scale

        if mode == "real":
            viz2D = np.real(viz2D)

        elif mode == "imag":
            viz2D = np.imag(viz2D)
            
        elif mode == "abs":
            viz2D = np.abs(viz2D)
        
        try:
            n = len(vlim)

        except Exception as e:
            n = 0

        if n == 0:
            plt.imshow(np.flipud(np.real(viz2D)), extent=extent, cmap=cmap)

        else:
            plt.imshow(np.flipud(np.real(viz2D)), extent=extent, cmap=cmap, vmin=vlim[0], vmax=vlim[1])

        if xlabel: plt.xlabel(r'$x$' + ' [' + unit + ']')
        if ylabel: plt.ylabel(r'$y$' + ' [' + unit + ']')

        plt.colorbar()
        plt.title(title)

        if self.domain.positive_downward:
            plt.gca().invert_yaxis()

    # This method computes the analytic solution of the 2D Helmholtz equation in 
    # a homogeneous medium with wavenumber k = omega/v
    def analytic_solution_point_source(self):

        import scipy.special as sp

        if self.source.source_type == "point_source" and self.source.n_pointsource == 1 and self.domain.is_homogeneous():

            k0 = 2*np.pi*self.source.freq/self.domain.vp_2d[0,0]
            h_avg = (self.domain.hx + self.domain.hy)/2

            x_2d, y_2d = np.meshgrid(self.domain.xp, self.domain.yp)
            dist_from_source = np.sqrt((x_2d - self.source.xs)**2 + (y_2d - self.source.ys)**2)
            
            u = (1j/4)*sp.hankel1(0, k0*dist_from_source)

            # replace the singularity by a FD approximation
            isx = round((self.source.xs - self.domain.xpmin)/self.domain.hx)
            isy = round((self.source.ys - self.domain.ypmin)/self.domain.hy)

            ub = u[isy-1, isx] 
            ut = u[isy+1, isx]
            ul = u[isy, isx-1]
            ur = u[isy, isx+1]
            u_average = (ub + ut + ul + ur) / 4

            u[isy, isx] = u_average
            # u[isy, isx] = -(1e4*h_avg**2 + 4*u_average) / ((h_avg*k0)**2-4)

            return u
        
        else:
            return None

    def error_norm_point_source(self):

        isrcx = round((self.source.xs - self.domain.xpmin) / self.domain.hx)
        isrcy = round((self.source.ys - self.domain.ypmin) / self.domain.hy)

        wlen = self.domain.vp_2d[0, 0]/self.source.freq
        hwx = max(1, round(0.5*wlen/self.domain.hx)) # half width in x-direction
        hwy = max(1, round(0.5*wlen/self.domain.hy)) # half width in y-direction

        # compute analytical solution
        u_exact_2d = self.analytic_solution_point_source()

        # create a mask of size length 1λ
        mask_2d = np.ones_like(u_exact_2d)
        mask_2d[isrcy-hwy:isrcy+hwy+1, isrcx-hwx:isrcx+hwx+1] = 0

        # compute relative error
        error_percent = 100*np.linalg.norm(mask_2d*(self.u - u_exact_2d)) / np.linalg.norm(mask_2d*u_exact_2d)

        return error_percent
