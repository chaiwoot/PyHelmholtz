# Authors: Sirawit Inpuak and Chaiwoot Boonyasiriwat

import numpy as np
from .abm import *
from .domain import *
from .fd import *
from .source import *
from .util import *

LIGHT_SPEED = 299792458.0   # light speed in vacuum

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

    # This method forms the system matrix A and the source vector b.
    def build(self):
        A = self.abm.build_A(self.domain, self.source, self.fd)
        b = self.source.build_b(self.domain, self.abm.n)
        self.A = A
        return (A, b)

    # This method solve the linear system Ax = b
    # Modified by Chaiwoot on Dec 3, 2025 to add the solver from pymumps
    # Modified by Chaiwoot on Feb 25, 2026 to add the nested dissection and gmres
    # Modified by Chaiwoot on Mar 15, 2026 to avoid an error when MUMPS/PyMUMPS/scikit-sparse is not available
    # Note that SciPy is mandatory for this method to work properly.
    def solve(self, solver="spsolve") -> None:
        nx_pad, ny_pad = self.domain.nx_pad, self.domain.ny_pad
        n = self.abm.n
        A, b = self.build()
        if solver == "spsolve":    # sparse direct solver of SciPy
            from scipy.sparse.linalg import spsolve
            x = spsolve(A, b)      # use COLAMD ordering by default
        elif solver == "mumps":    # MUMPS: MUltifrontal Massively Parallel Sparse direct solver
            try:
                x = self.pymumps(A, b)
            except Exception:
                print("Since MUMPS or PyMUMPS is not available, SciPy's spsolve() is used instead!")
                from scipy.sparse.linalg import spsolve
                x = spsolve(A, b)
        elif solver == "nested_dissection":
            import scipy.sparse as sp
            from scipy.sparse.linalg import splu

            try:
                from sksparse.cholmod import analyze

                # Analyze the matrix A and compute the permutation vector p
                factor_structure = analyze(A.tocsc())
                p = factor_structure.P()
                
                # Apply permutation to A where P is the permutation matrix
                N = A.shape[0]
                P = sp.csr_matrix((np.ones(N), (np.arange(N), p)), shape=(N, N))
                A_perm = P @ A @ P.T
                b_perm = P @ b
                
                # Solve the linear system with SciPy's splu using 'NATURAL' ordering
                # because we've already reordered the matrix
                solver = splu(A_perm.tocsc(), permc_spec='NATURAL')
                x_perm = solver.solve(b_perm)

                # Reverse Permutation
                x = P.T @ x_perm
            except Exception:
                print("Since scikit-sparse is not available, SciPy's spsolve() is used instead!")
                from scipy.sparse.linalg import spsolve
                x = spsolve(A, b)
        # Remark: iterative solver is much slower than the above direct solvers
        elif solver == "gmres":    # GMRES iterative solver of SciPy
            import scipy.sparse as sp
            from scipy.sparse.linalg import gmres, spilu, LinearOperator
            beta = 0.5
            k = 2*np.pi*self.source.freq/self.domain.v_pad # wavenumber
            k_flat = k.flatten()
            shift_diag = 1j * beta * (k_flat**2)
            M = A + sp.diags(shift_diag, format='csr')
            try:
                ilu = spilu(M.tocsc(), drop_tol=1e-3, fill_factor=10, 
                            permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.1)
            except RuntimeError as e:
                return None, f"Preconditioner failed: {e}"
            M_x = lambda x: ilu.solve(x)
            preconditioner = LinearOperator(A.shape, M_x)  # create a preconditioner
            x, info = gmres(A, b, M=preconditioner, rtol=1e-6, atol=1e-8, restart=50)
            if info != 0:
                print(f"GMRES failed to converge ({info})")
        else:
            raise Exception(f"Solver '{solver}' is not supported. Supported solvers are 'spsolve' and 'pymumps'.")

        x = np.reshape(x, (ny_pad, nx_pad))
        self.u = x[n:-n, n:-n]

    # Interface to the complex-arithmetic MUMPS solver
    # Added by Chaiwoot on Dec 3, 2025
    # Last modified on Dec 5, 2025
    def pymumps(self, A, b):
        from mumps import ZMumpsContext
        from scipy.sparse import csr_matrix
        ctx = ZMumpsContext(sym=0, par=1)
        if ctx.myid == 0:
            row, col = Util.get_row_col_indices_of_csr_matrix(A)
            n = b.shape[0]
            irn = np.array(row.astype(np.int32)+1, dtype='i')
            jcn = np.array(col.astype(np.int32)+1, dtype='i')
            a   = np.array(A.data, dtype='D')
            bb = np.array(b, dtype='D')
            ctx.set_shape(n)
            ctx.set_centralized_assembled(irn, jcn, a)
            x = bb.copy()
            ctx.set_rhs(x)
        ctx.set_silent()
        ctx.run(job=6)
        if ctx.myid != 0:
            x = np.zeros(b.shape[0])
        ctx.destroy()
        return x

    # This method is a quick tool for visualizing various fields and medium property.
    # Modified by Chaiwoot on Dec 9, 2025
    # - combine <unit> and <scale>
    # - add <vlim>
    # - rename <plotmode> as <mode>, <colormap> as <cmap>
    # - allow <data> to be an array of the same size as the solution <self.u>
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
                viz2D = self.domain.v
                if title == "": title = "Velocity"

            elif data == "solution":
                viz2D = self.u
                if self.source.source_type == "plane_wave":
                    if title == "": title = "Scattered field"
                else:
                    if title == "": title = "Wave field"

            elif data == "incident":    # incident field
                if self.source.source_type == "plane_wave":
                    # viz2D = self.source.ui[n:-n,n:-n]
                    viz2D = self.source.ui
                else:
                    viz2D = np.zeros_like(self.u)
                if title == "": title = "Incident field"
                
            elif data == "total":       # total field
                if self.source.source_type == "plane_wave":
                    # viz2D = self.u + self.source.ui[n:-n,n:-n]
                    viz2D = self.u + self.source.ui
                else:
                    viz2D = self.u
                if title == "": title = "Total field"
            else:
                title = ""
        elif isinstance(data, np.ndarray):
            if data.shape == self.u.shape:
                viz2D = data

        if   unit == "km": scale = 1e+3
        elif unit == "m" : scale = 1.0
        elif unit == "cm": scale = 1e-2
        elif unit == "mm": scale = 1e-3
        elif unit == "um": scale = 1e-6     # micrometer
        elif unit == "nm": scale = 1e-9     # nanometer

        extent = np.array([self.domain.xmin, self.domain.xmax, self.domain.ymin, self.domain.ymax]) / scale

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

        if self.source.source_type == "point_source" and self.domain.is_homogeneous():
            x2d, y2d = np.meshgrid(self.domain.x, self.domain.y)
            r = np.sqrt((x2d-self.source.xs)**2 + (y2d-self.source.ys)**2)
            k = 2*np.pi*self.source.freq/self.domain.v[0,0]
            u = (1j/4)*sp.hankel1(0, k*r)

            # replace the singularity by a FD approximation
            h = self.domain.h
            isx = int((self.source.xs-self.domain.xmin)/h)
            isy = int((self.source.ys-self.domain.ymin)/h)
            uu = u[isy-1,isx]
            u[isy,isx] = -(1e4*h*h+4*uu)/((h*k)**2-4)
            return u
        else:
            return None

    def error_norm_point_source(self):
        h = self.domain.h
        isx = int((self.source.xs-self.domain.xmin)/h)
        isy = int((self.source.ys-self.domain.ymin)/h)
        wavelength = self.domain.v[0,0]/self.source.freq
        half = max(1,int(0.15*wavelength/h))
        G = self.analytic_solution_point_source()
        mask = np.ones(G.shape)
        mask[isy-half:isy+half+1,isx-half:isx+half+1] = 0
        return np.linalg.norm((self.u-G)*mask)*100/np.linalg.norm(G*mask)

    # This method computes an analytic solution, the total field, for a scattering of TM-mode EM planewave due to a circular cylinder.
    # The analytic solution was given in Appendix D of van der Sijs et al. (2020).
    # <object_info> is the information of the cylinder.
    # <M> is the maximum order of the finite series used to compute the total field.
    def analytic_solution_plane_wave(self, object_info, M=70):
        import scipy.special as sp

        domain = self.domain
        source = self.source
        xmin, xmax, ymin, ymax, nx, ny = domain.xmin, domain.xmax, domain.ymin, domain.ymax, domain.nx, domain.ny
        xc, yc, rc, epsr_cylinder, epsr_bg = object_info
        freq, theta = source.freq, source.theta

        x1d, y1d = np.linspace(xmin, xmax, nx, True), np.linspace(ymin, ymax, ny, True)
        X, Y = np.meshgrid(x1d, y1d)

        k0 = 2*np.pi*freq/LIGHT_SPEED
        kbg = np.sqrt(epsr_bg)*k0

        # Compute the total field
        ny, nx = X.shape
        k1 = k0*np.sqrt(epsr_cylinder)
        Dist2D = np.sqrt((X-xc)**2 + (Y-yc)**2)
        Dist1D = Dist2D.flatten()
        Phi2D = np.arctan2(Y, X)
        Ez_total = np.zeros_like(X, dtype=np.complex128)
        for m in range(-M, M+1):
            k0R, k1R = k0*rc, k1*rc
            k1k0 = k1/k0
            
            Am = 1j**m
            denom = sp.jv(m, k1R)*sp.hankel1(m+1, k0R) - k1k0*sp.jv(m+1, k1R)*sp.hankel1(m, k0R)
            Bm = sp.jv(m, k1R)*sp.jv(m+1, k0R) - k1k0*sp.jv(m+1, k1R)*sp.jv(m, k0R)
            Bm *= -Am/denom
            Cm = (-2j/(np.pi*k0R))*Am/denom
            
            coeff = np.zeros_like(Dist1D, dtype=np.complex128)
            cond_outside = (Dist1D > rc)
            cond_inside = (Dist1D <= rc)
            
            # compute the total field outside the cylinder
            k0r = k0*Dist1D[cond_outside]
            coeff[cond_outside] = Am*sp.jv(m, k0r) + Bm*sp.hankel1(m, k0r)
            
            # compute the total field inside the cylinder
            k1r = k1*Dist1D[cond_inside]
            coeff[cond_inside] = Cm*sp.jv(m, k1r)
            Ez_total += coeff.reshape([ny, nx])*np.exp(1j*m*Phi2D)

        # Compute the scattered field
        Ez_sc = Ez_total - self.source.ui
        
        return Ez_total, Ez_sc
