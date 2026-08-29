import numpy as np
from scipy.sparse import csr_matrix, diags

class Util:

    @staticmethod
    def pad_array2d(w: np.ndarray, npad: int) -> np.ndarray: # use in domain.py

        ny_old, nx_old = w.shape
        nx_new, ny_new = nx_old + 2*npad, ny_old + 2*npad

        w_pad = np.zeros([ny_new, nx_new])
        w_pad[npad:-npad, npad:-npad] = w

        ix1, ix2 = npad, (nx_new-1) - npad
        iy1, iy2 = npad, (ny_new-1) - npad

        # corner case
        w_pad[:iy1+1, :ix1+1] = w[0, 0]                 # left bottom corner
        w_pad[:iy1+1, ix2:] = w[0, -1]                  # right bottom corner
        w_pad[iy2:, :ix1+1] = w[-1, 0]                  # left top corner
        w_pad[iy2:, ix2:] = w[-1, -1]                   # right top corner

        # side case
        for j in range(npad):
            w_pad[iy1+1:iy2, j] = w[1:-1, 0]            # left side
            w_pad[iy1+1:iy2, ix2+1+j] = w[1:-1, -1]     # right side
            w_pad[j, ix1+1:ix2] = w[0, 1:-1]            # bottom side
            w_pad[iy2+1+j:, ix1+1:ix2] = w[-1, 1:-1]    # top side

        return w_pad

    @staticmethod
    def diag_matrix(val: np.ndarray, idx: np.ndarray, p: int) -> csr_matrix:
    
        # len(idx) < n    
    
        n = len(val) # len(val) = n = nx*ny 
        val = (val[idx])**p

        return csr_matrix((val, (idx, idx)), shape=(n, n), dtype=np.complex128)

    @staticmethod
    def scale_rows_by_diagonal(A: csr_matrix, idx: np.ndarray, weight: np.ndarray) -> csr_matrix:
        
        # A.shape = (ny, nx) and weight.shape = (nx*ny, ) 
        diagA = A.diagonal()

        factor = np.ones_like(diagA, dtype=np.complex128)
        factor[idx] = weight[idx]/diagA[idx]    
        factor_matrix = diags(factor, offsets=0, format='csr')
        
        scaled_A = factor_matrix @ A
    
        return scaled_A

    @staticmethod
    def dirichlet_bc_matrix(nx: int, ny: int) -> csr_matrix:

        n = nx*ny

        ixc1, ixc2 = 0, nx-1 
        iyc1, iyc2 = 0, ny-1

        l1 = np.arange(ixc1, ixc2+1) + nx*iyc1 # bottom side
        l2 = np.arange(ixc1, ixc2+1) + nx*iyc2 # top side
        l3 = ixc1 + nx*np.arange(iyc1+1, iyc2) # left side
        l4 = ixc2 + nx*np.arange(iyc1+1, iyc2) # right side

        idx_boundary = np.concatenate((l1, l2, l3, l4))

        ir = np.copy(idx_boundary)
        ic = np.copy(idx_boundary)
        val = np.ones_like(idx_boundary)
        
        return csr_matrix((val, (ir, ic)), shape=(n, n), dtype=np.complex128)
    
    @staticmethod
    def corner_treatment(A, val, nx, ny, nc):

        idx_corners = np.zeros(4*nc, dtype=np.int32)
        for k in range(nc): # k = [0, ..., nc-1] => klayer = [nc, ..., 1]

            ixc1, ixc2 = k, (nx-1)-k 
            iyc1, iyc2 = k, (ny-1)-k
            idx_ = np.array([ixc1+nx*iyc1, ixc1+nx*iyc2, ixc2+nx*iyc1, ixc2+nx*iyc2])
            idx_corners[4*k:4*k+4] = idx_

        At = A.copy()
        At[idx_corners, :] = val * A[idx_corners, :]

        return At
    
    @staticmethod
    def build_matrix(idx, nx, ny, dv_axis, swap, stencil):

        n = nx*ny
        translation = swap

        if dv_axis == "x":
            translation *= 1

        elif dv_axis == "y":
            translation *= nx

        nnode, dv_order, idx_stencil, coeff = stencil.unpack()
        
        ir = np.repeat(idx, nnode)
        ic = ir + np.tile(translation*idx_stencil, len(idx))
        val = np.tile(coeff, len(idx))

        iy = (idx // nx).astype(np.int32)
        ix = (idx % nx).astype(np.int32)
        
        return csr_matrix((val, (ir, ic)), shape=(n, n), dtype=np.complex128)
    
    @staticmethod
    def calculate_coeff_in_pml(ix_or_iy, nx_or_ny, nc, pmlstencil, q_profile):
    
        idx_qstencil = pmlstencil.idx_qstencil
        weight_q = pmlstencil.weight_q

        # calculate depth_q
        # In this work, the absorbing domain is implemented with uniform thickness on all four boundaries.
        # This calculation is based on left- or bottom-zones grids. 
        if ix_or_iy < round(nx_or_ny/2):
            iq = ix_or_iy + idx_qstencil
            depth_q = nc - iq
            depth_q0 = nc - ix_or_iy

        if depth_q0 < 0:
            depth_q0 = 0

        depth_q[depth_q<0] = 0

        # calculate q
        q = q_profile(depth_q, nc)

        # calculate coeff
        coeff = weight_q @ q

        q0 = q_profile(depth_q0, nc)
        coeff = q0*coeff

        return coeff
    
    @staticmethod
    def depth_from_physical_boundary(nx, ny, nc):

        distance_from_ends = lambda m: np.minimum(np.arange(m), np.arange(m)[::-1])

        depth_x = distance_from_ends(nx)
        depth_y = distance_from_ends(ny)
        depth2D_x = np.broadcast_to(depth_x, (ny, nx))
        depth2D_y = np.broadcast_to(depth_y.reshape(-1, 1), (ny, nx))

        depth2D = np.minimum(depth2D_x, depth2D_y)
        depth2D[nc:-nc, nc:-nc] = nc
        depth2D = nc - depth2D

        depth1D = depth2D.flatten() # depth1D.shape = (nx*ny, )

        return depth1D
    
    @staticmethod
    def build_radiusmap_and_flipmap(nx, ny, nc, noverlap, rmax, dv_axis,):

        if dv_axis == "x":
            radius_map = np.minimum(np.arange(nx), np.arange(nx)[::-1])
            radius_map[radius_map>rmax] = rmax
            radius_map = np.broadcast_to(radius_map, (ny, nx))

            flip_map = -1*np.ones(nx, dtype=np.int8)

            if noverlap == 0:
                if rmax != 0:
                    flip_map[:-rmax] = 1
                elif rmax == 0: # one-way wave equation (one-sided stencil r = 0)
                    flip_map[:-nc] = 1
            elif noverlap > 0:
                flip_map[:-nc-noverlap] = 1

            flip_map = np.broadcast_to(flip_map, (ny, nx))

        elif dv_axis == "y":
            radius_map = np.minimum(np.arange(ny), np.arange(ny)[::-1])
            radius_map[radius_map>rmax] = rmax
            radius_map = np.broadcast_to(radius_map.reshape(-1, 1), (ny, nx))

            flip_map = -1*np.ones(ny, dtype=np.int8)

            if noverlap == 0:
                if rmax != 0:
                    flip_map[:-rmax] = 1
                elif rmax == 0: # one-way wave equation (one-sided stencil r = 0)
                    flip_map[:-nc] = 1
            elif noverlap > 0:
                flip_map[:-nc-noverlap] = 1

            flip_map = np.broadcast_to(flip_map.reshape(-1, 1), (ny, nx))

        return radius_map, flip_map # flip อยากเปลี่ยนเป็นชื่ออื่น
    
    @staticmethod
    def get_idx_at_layer_k(nx, ny, k):

        ixc1, ixc2 = k, (nx-1)-k 
        iyc1, iyc2 = k, (ny-1)-k

        eb = np.arange(ixc1+1, ixc2) + nx*iyc1 # bottom edge
        et = np.arange(ixc1+1, ixc2) + nx*iyc2 # top edge
        el = ixc1 + nx*np.arange(iyc1+1, iyc2) # left edge
        er = ixc2 + nx*np.arange(iyc1+1, iyc2) # right edge

        cbl = np.array([ixc1 + nx*iyc1])
        ctl = np.array([ixc1 + nx*iyc2])
        cbr = np.array([ixc2 + nx*iyc1])
        ctr = np.array([ixc2 + nx*iyc2])
        
        edges = [eb, et, el, er]
        corners = [cbl, ctl, cbr, ctr]
        
        return edges, corners
    
    @staticmethod
    def get_idx(nx, ny, nc, noverlap, zone):

        """
        1) single-layer boundary method (D0/EM1/EM2/H2)
        - interior
        - boundary: Union of boundary_lr + boundary_bt (some points are overlap)

        2) Ren-Liu method (RLEM1/RLEM2/RLH2)
        - interior
        - transition: Union of transition_lr + transition_bt (some points are overlap)
        - boundary: Union of boundary_lr + boundary_bt

        3) PML method
        - interior
        - pml zone = (pml_lr_xs + pml_bt_x) + (pml_bt_ys + pml_lr_y)
        - boundary = Union of boundary_lr + boundary_bt (some points are overlap)
        """

        idx_2d = np.arange(nx*ny).reshape(ny, nx)
        
        ##### 1) Set-up
        Indices = {}

        ##### 2) boundary (apply D0/EM1/EM2)
        if zone == "boundary":

            idx_boundary = []
            idx_boundary_lr, idx_boundary_bt = [], []
            
            edges, corners = Util.get_idx_at_layer_k(nx, ny, 0)

            idx_boundary.extend(edges + corners)
            idx_boundary_lr.extend([edges[2], edges[3]] + corners)
            idx_boundary_bt.extend([edges[0], edges[1]] + corners)       

            Indices["boundary_lr"] = np.concatenate(idx_boundary_lr)
            Indices["boundary_bt"] = np.concatenate(idx_boundary_bt)
            Indices["boundary"] = np.concatenate(idx_boundary)

            return Indices

        ##### 2) interior (apply HH)
        elif zone == "interior":
            nco = nc + noverlap
            Indices["interior"] = idx_2d[nco:-nco, nco:-nco].flatten()

            return Indices

        ##### 3) transition (apply HH+OW) / pml (apply HH with PML)
        elif zone == "transition":

            if noverlap == 0 and nc >= 2: # non-PML method (RLEM1/RLEM2)

                idx_transition = []
                idx_transition_lr, idx_transition_bt = [], []

                for k in range(1, nc):
                                       
                    edges, corners = Util.get_idx_at_layer_k(nx, ny, k)

                    idx_transition.extend(edges + corners)
                    idx_transition_lr.extend([edges[2], edges[3]] + corners)
                    idx_transition_bt.extend([edges[0], edges[1]] + corners)
                
                Indices["transition_lr"] = np.concatenate(idx_transition_lr)
                Indices["transition_bt"] = np.concatenate(idx_transition_bt)
                Indices["transition"] = np.concatenate(idx_transition) 

                return Indices
            
            else:
                raise Exception("The condition does not match.")

        elif zone == "pml":

            if noverlap >= 1 and nc >= 2: # PML method 
                
                nco = nc + noverlap

                idx_pml_l_xs = idx_2d[1:-1, 1:nco].flatten()
                idx_pml_r_xs = idx_2d[1:-1, nx-nco:nx-1].flatten()
                Indices["pml_lr_xs"] = np.concatenate((idx_pml_l_xs, idx_pml_r_xs))

                idx_pml_b_x = idx_2d[1:nco, nco:-nco].flatten()
                idx_pml_t_x = idx_2d[ny-nco:ny-1, nco:-nco].flatten() 
                Indices["pml_bt_x"] = np.concatenate((idx_pml_b_x, idx_pml_t_x))

                idx_pml_b_ys = idx_2d[1:nco, 1:-1].flatten()
                idx_pml_t_ys = idx_2d[ny-nco:ny-1, 1:-1].flatten() 
                Indices["pml_bt_ys"] = np.concatenate((idx_pml_b_ys, idx_pml_t_ys))

                idx_pml_l_y = idx_2d[nco:-nco, 1:nco].flatten()
                idx_pml_r_y = idx_2d[nco:-nco, nx-nco:nx-1].flatten()
                Indices["pml_lr_y"] = np.concatenate((idx_pml_l_y, idx_pml_r_y))

                Indices["pml"] = np.concatenate((idx_pml_l_xs, idx_pml_r_xs, idx_pml_b_x, idx_pml_t_x))

                return Indices

            else:
                raise Exception("The condition does not match.")

        return Indices
    
    @staticmethod
    def group_stencils(idx, radius_map, direction_map):

        radius_data = (radius_map.flatten())[idx]
        direction_data = (direction_map.flatten())[idx]

        # pack keys: (radius * 100) + (direction + 1)
        # since radius < 100 and direction is -1 or 1, this creates unique IDs:
        # radius = 1, direction =-1 -> 100 + 0 = 100
        # radius = 1, direction = 1 -> 100 + 2 = 102
        packed_keys = (radius_data * 100) + (direction_data + 1)

        # sort by the packed ke ys
        sort_idx = np.argsort(packed_keys)
        sorted_keys = packed_keys[sort_idx]
        sorted_idx = idx[sort_idx]

        # find group boundaries
        diff_idx = np.flatnonzero(np.diff(sorted_keys)) + 1
        
        # split indices and keys into groups
        split_indices = np.split(sorted_idx, diff_idx)
        split_keys = np.split(sorted_keys, diff_idx)

        stencil_groups = {}
        for i, group in enumerate(split_indices):
            # Unpack the key from the first element of the group
            key_val = split_keys[i][0]
            r = int(key_val // 100)
            d = int((key_val % 100) - 1)
            stencil_groups[(r, d)] = group
            
        return stencil_groups
    
    @staticmethod
    def build_system_matrix(domain_grids, idx_zone, stc_catalog, dv_axis, pml_profile=None):
        
        nx, ny, nc, noverlap = domain_grids

        radius_max = max(stc_catalog)

        radius_map, direction_map = Util.build_radiusmap_and_flipmap(nx, ny, nc, noverlap, radius_max, dv_axis)
        stencil_groups = Util.group_stencils(idx_zone, radius_map, direction_map)

        A = None
        for (radius, flip), idx_group in stencil_groups.items():

            if pml_profile is None:
                
                # idx_group = the points that share the same stencil type
                selected_stencil = stc_catalog.get(radius)
                term = Util.build_matrix(idx_group, nx, ny, dv_axis, flip, selected_stencil)
                                        
                if A is None:
                    A = term
                else:
                    A += term    
 
            elif pml_profile is not None:

                selected_stencil = stc_catalog.get(radius)

                ix_group = idx_group % nx
                iy_group = idx_group // nx
                
                ix_unique = np.unique(ix_group)
                iy_unique = np.unique(iy_group)

                if dv_axis == "x":
                    
                    for ix in ix_unique:

                        idx = ix + nx*iy_unique

                        if flip == -1:
                            ix_cal = (nx-1) - ix
                        else:
                            ix_cal = ix

                        selected_stencil.coeff = Util.calculate_coeff_in_pml(ix_cal, nx, nc, selected_stencil, pml_profile)
                        term = Util.build_matrix(idx, nx, ny, "x", flip, selected_stencil)                                            
                        if A is None:
                            A = term
                        else:
                            A += term

                elif dv_axis == "y":           
                    
                    for iy in iy_unique:

                        idx = ix_unique + nx*iy

                        if flip == -1:
                            iy_cal = (ny-1) - iy
                        else:
                            iy_cal = iy

                        selected_stencil.coeff = Util.calculate_coeff_in_pml(iy_cal, ny, nc, selected_stencil, pml_profile)
                        term = Util.build_matrix(idx, nx, ny, "y", flip, selected_stencil)                                                
                        if A is None:
                            A = term
                        else:
                            A += term
        
        return A
    
    # This method returns the row and column indices of a sparse matrix of type csr_matrix
    @staticmethod
    def get_row_col_indices_of_csr_matrix(A):

        from scipy.sparse import csr_matrix

        if isinstance(A, csr_matrix):
            column_indices = A.indices
            indptr = A.indptr
            num_non_zero = A.nnz
            row_indices = np.empty(num_non_zero, dtype=np.intp)

            for i in range(A.shape[0]):
                start = indptr[i]
                end = indptr[i+1]
                row_indices[start:end] = i

            return (row_indices, column_indices)
        
        else:
            raise Exception("The input is not an instance of class scipy.sparse.csr_matrix!")
