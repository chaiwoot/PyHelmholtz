import numpy as np
from scipy.sparse import csr_matrix, diags

class Util:

    @staticmethod
    def pad_array2d(x: np.ndarray, n: int) -> np.ndarray:

        ny, nx = x.shape
        nx_pad, ny_pad = nx + 2*n, ny + 2*n

        x_pad = np.zeros([ny_pad, nx_pad])
        x_pad[n:-n, n:-n] = np.copy(x)

        ix1, ix2 = n, (nx_pad-1) - n
        iy1, iy2 = n, (ny_pad-1) - n

        # corner case
        x_pad[:iy1+1, :ix1+1] = x[0, 0]                 # left bottom corner
        x_pad[:iy1+1, ix2:] = x[0, -1]                  # right bottom corner
        x_pad[iy2:, :ix1+1] = x[-1, 0]                  # left top corner
        x_pad[iy2:, ix2:] = x[-1, -1]                   # right top corner

        # side case
        for j in range(n):
            x_pad[iy1+1:iy2, j] = x[1:-1, 0]            # left side
            x_pad[iy1+1:iy2, ix2+1+j] = x[1:-1, -1]     # right side
            x_pad[j, ix1+1:ix2] = x[0, 1:-1]            # bottom side
            x_pad[iy2+1+j:, ix1+1:ix2] = x[-1, 1:-1]    # top side

        return x_pad

    @staticmethod
    def get_idx(nx, ny, nly):

        # interior domain's indices
        ix = np.arange(nly, nx-nly)
        iy = np.arange(nly, ny-nly)
        ix, iy = np.meshgrid(ix, iy)
        idx_int = (ix + nx*iy).flatten()

        # absorbing domain's indices
        idx_abs, idx_abs_inner, idx_bd = [], [], []

        ixc1, ixc2 = 0, nx-1
        iyc1, iyc2 = 0, ny-1

        l1 = np.arange(ixc1, ixc2+1) + nx*iyc1 # bottom layer
        l2 = np.arange(ixc1, ixc2+1) + nx*iyc2 # top layer
        l3 = ixc1 + nx*np.arange(iyc1+1, iyc2) # left layer
        l4 = ixc2 + nx*np.arange(iyc1+1, iyc2) # right layer

        idx_abs = np.concatenate((l1, l2, l3, l4))
        idx_bd = np.concatenate((l1, l2, l3, l4))

        if nly == 1:
            idx_abs_inner = []

        elif nly > 1:
            for k in range(1, nly): # klayer = [nly,..., 1]

                ixc1, ixc2 = k, (nx-1)-k 
                iyc1, iyc2 = k, (ny-1)-k

                l1 = np.arange(ixc1, ixc2+1) + nx*iyc1 # bottom layer
                l2 = np.arange(ixc1, ixc2+1) + nx*iyc2 # top layer
                l3 = ixc1 + nx*np.arange(iyc1+1, iyc2) # left layer
                l4 = ixc2 + nx*np.arange(iyc1+1, iyc2) # right layer

                idx_ = np.concatenate((l1, l2, l3, l4))            
                idx_abs = np.concatenate((idx_abs, idx_))
                idx_abs_inner = np.concatenate((idx_abs_inner, idx_))
                idx_abs_inner = idx_abs_inner.astype(int)

        return idx_int, idx_abs, idx_abs_inner, idx_bd
    
    @staticmethod
    def diag_matrix(val: np.ndarray, idx: np.ndarray, p: int) -> csr_matrix:
    
        # len(idx) < n    
    
        n = len(val) # len(val) = n = nx*ny 
        val = (val[idx])**p

        return csr_matrix((val, (idx, idx)), shape=(n, n), dtype=np.complex128)

    @staticmethod
    def scale_rows_by_diagonal(A: csr_matrix, idx_abs: np.ndarray, weight: np.ndarray) -> csr_matrix:
        
        # A.shape = (ny, nx) and weight.shape = nx*ny 
        diagA = A.diagonal()

        factor = np.ones_like(diagA, dtype=np.complex128)
        factor[idx_abs] = weight[idx_abs]/diagA[idx_abs]    
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
    def compute_RL_weights(nx, ny, n, damping_profile):

        distance_from_ends = lambda m: np.minimum(np.arange(m), np.arange(m)[::-1])

        depth_x_1d = distance_from_ends(nx)
        depth_y_1d = distance_from_ends(ny)
        depth_x_2d, depth_y_2d = np.meshgrid(depth_x_1d, depth_y_1d)

        labeled_abslayer_2d = np.minimum(depth_x_2d, depth_y_2d)
        labeled_abslayer_2d[n:-n, n:-n] = n
        labeled_abslayer_2d = n - labeled_abslayer_2d

        labeled_abslayer_1d = labeled_abslayer_2d.flatten()
        weight_ow = damping_profile(labeled_abslayer_1d, n)

        weight_hh = 1 - weight_ow
        
        return weight_ow, weight_hh
    
    @staticmethod
    def corner_treatment(A, coeff, nx, ny, nly):

        idx_corners = np.zeros(4*nly, dtype=np.int32)
        for k in range(nly): # k = [0, ..., nly-1] => klayer = [nly, ..., 1]

            ixc1, ixc2 = k, (nx-1)-k 
            iyc1, iyc2 = k, (ny-1)-k
            idx_ = np.array([ixc1+nx*iyc1, ixc1+nx*iyc2, ixc2+nx*iyc1, ixc2+nx*iyc2])
            idx_corners[4*k:4*k+4] = idx_

        A[idx_corners, :] = coeff * A[idx_corners, :]

        return A
    
    @staticmethod
    def build_matrix(idx, nx, ny, dv_axis, swap, stencil):

        n = nx*ny
        translation = swap

        if dv_axis == "x":
            translation *= 1

        elif dv_axis == "y":
            translation *= nx

        # elif dev_axis == "z":
        #     translation *= nx*ny    

        nnode, dv_order, idx_stencil, coeff = stencil.unpack()
        
        ir = np.repeat(idx, nnode)
        ic = ir + np.tile(translation*idx_stencil, len(idx))
        val = np.tile(coeff, len(idx))
        
        return csr_matrix((val, (ir, ic)), shape=(n, n), dtype=np.complex128)
    
    @staticmethod
    def convert_to_idx(ix, iy, nx):

        ix, iy = np.meshgrid(ix, iy)
        idx = (ix + nx*iy).flatten()

        return idx
    
    @staticmethod
    def build_At(nx, ny, nly, roi, stc_list):

        if roi == "absorbing_domain":
            kmax = nly

        elif roi == "boundary":
            kmax = 1

        else:
            raise Exception("roi is either \"absorbing_domain\" or \"boundary\".")

        almax = max(stc_list)
        
        Axt, Ayt = [], []

        ### 1: inner zone
        ### 1.1: asymmetric stencil
        for k in range(min(kmax, almax), almax):
                                                                                        
            al = k
            stc = stc_list.get(al)
                                                            
            # bottom (fw)																		
            ix = k																		
            iy = np.arange(kmax)
            idx = Util.convert_to_idx(ix, iy, nx)		
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))

            # bottom (bw)																		
            ix = (nx-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))																	                                                                                                                                                                                                                       																		
                                                                                    
            # top (fw)																		
            ix = k																		
            iy = np.arange(ny-kmax, ny)
            idx = Util.convert_to_idx(ix, iy, nx)
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))

            # top (bw)																		
            ix = (nx-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))																																				
                                                                                    
            # left (fw)																		
            iy = k																		
            ix = np.arange(kmax)
            idx = Util.convert_to_idx(ix, iy, nx)
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))

            # left (bw)																		
            iy = (ny-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))																	
                                                                                    
            # right (fw)																		
            iy = k																		
            ix = np.arange(nx-kmax, nx)
            idx = Util.convert_to_idx(ix, iy, nx)
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))

            # right (bw)																		
            iy = (ny-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))														

        ### 1.2: symmetric stencil
        stc = stc_list.get(almax)

        # bottom																			
        ix = np.arange(max(kmax, almax), nx-max(kmax, almax))																			
        iy = np.arange(kmax)
        idx = Util.convert_to_idx(ix, iy, nx)
        Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))

        # top																			
        iy = np.arange(ny-kmax, ny)
        idx = Util.convert_to_idx(ix, iy, nx)																			
        Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))																																					
                                                                                    
        # left																			
        ix = np.arange(kmax)																			
        iy = np.arange(max(kmax, almax), ny-max(kmax, almax))
        idx = Util.convert_to_idx(ix, iy, nx)																			
        Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))																			

        # right																			
        ix = np.arange(nx-kmax, nx)
        idx = Util.convert_to_idx(ix, iy, nx)																			
        Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))

        ### 2: outer zone
        for k in range(1, kmax):

            al = min(k, almax)
            stc = stc_list.get(al)

            # bottom (fw)																		
            ix = k																		
            iy = np.arange(k)
            idx = Util.convert_to_idx(ix, iy, nx)																																					
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))																		
            
            # bottom (bw)																		
            ix = (nx-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)																																					
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))																		
                                                                                    
            # top (fw)																		
            ix = k																		
            iy = (ny - 1) - np.arange(k)
            idx = Util.convert_to_idx(ix, iy, nx)																																					
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))																		
            
            # top (bw)																		
            ix = (nx-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)																																				
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))																		
                                                                                    
            # left (fw)																		
            iy = k																		
            ix = np.arange(k)
            idx = Util.convert_to_idx(ix, iy, nx)																																						
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))																	
            
            # left (bw)																		
            iy = (ny-1) - k
            idx = Util.convert_to_idx(ix, iy,nx)																																						
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))																		
                                                                                    
            # right (fw)																		
            iy = k																		
            ix = (nx - 1) - np.arange(k)
            idx = Util.convert_to_idx(ix, iy,nx)																																						
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))																	
            
            # right (bw)																		
            iy = (ny-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)																																						
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))

        ### 3: corner
        for klayer in range(1, kmax+1):

            k = kmax - klayer
            al = min(k, almax)
            stc = stc_list.get(al)
            
            # left-bottom (+,+)
            ix = k
            iy = k
            idx = Util.convert_to_idx(ix, iy, nx)
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))
        
            # right-bottom (-,+)
            ix = (nx-1)-k
            iy = k
            idx = Util.convert_to_idx(ix, iy, nx)																			
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))
        
            # left-top (+,-)
            ix = k
            iy = (ny-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)																			
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))
            
            # right-top (-,-)
            ix = (nx-1)-k
            iy = (ny-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)																			
            Axt.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))
            Ayt.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))

        Axt, Ayt = sum(Axt), sum(Ayt)	

        return Axt, Ayt
    
    @staticmethod
    def build_An(nx, ny, nly, roi, stc_list):

        if roi == "absorbing_domain":
            kmax = nly

        elif roi == "boundary":
            kmax = 1

        else:
            raise Exception("roi is either \"absorbing_domain\" or \"boundary\".")

        almax = max(stc_list)
        
        Axn, Ayn = [], []        
        for k in range(kmax):

            ixc1, ixc2 = k, (nx-1) - k
            iyc1, iyc2 = k, (ny-1) - k
        
            al = min(k, almax)
            stc = stc_list.get(al)
        
            # left side (exclude corner)
            ix = ixc1
            iy = np.arange(iyc1+1, iyc2)
            idx = Util.convert_to_idx(ix, iy, nx)
            Axn.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))
            
            # right side (exclude corner)
            ix = ixc2
            iy = np.arange(iyc1+1, iyc2)
            idx = Util.convert_to_idx(ix, iy, nx)																			
            Axn.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))

            # bottom side (exclude corner)
            ix = np.arange(ixc1+1, ixc2)
            iy = iyc1
            idx = Util.convert_to_idx(ix, iy, nx)																			
            Ayn.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))
            
            # top side (exclude corner)
            ix = np.arange(ixc1+1, ixc2)
            iy = iyc2
            idx = Util.convert_to_idx(ix, iy, nx)																			
            Ayn.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))
            
            # left-bottom corner
            ix = ixc1
            iy = iyc1
            idx = Util.convert_to_idx(ix, iy, nx)
            Axn.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))
            Ayn.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))
            
            # right-bottom corner
            ix = ixc2
            iy = iyc1
            idx = Util.convert_to_idx(ix, iy, nx)
            Axn.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))
            Ayn.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))
            
            # left-top corner
            ix = ixc1
            iy = iyc2
            idx = Util.convert_to_idx(ix, iy, nx)
            Axn.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))
            Ayn.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))
            
            # right-top corner
            ix = ixc2
            iy = iyc2
            idx = Util.convert_to_idx(ix, iy, nx)
            Axn.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))
            Ayn.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))

        Axn, Ayn = sum(Axn), sum(Ayn)

        return Axn, Ayn
    
    @staticmethod
    def build_Axx(nx, ny, nly, nphy, absmethod, stcdv2_list):

        almax = max(stcdv2_list)

        if absmethod == "PML":
            iy = np.arange(1, ny-1)
        
        elif absmethod == "RenLiu":
            iy = np.arange(nly, ny-nly)

        else:
            raise Exception("absmethod is either \"PML\" or \"RenLiu\".")

        kmin = min(nly+nphy, almax)
        kmax = max(nly+nphy, almax)
        
        Axx = []
        for k in range(kmin, almax):
            
            al = k
            stc = stcdv2_list.get(al)

            # forward one-sided stencil
            ix = k
            idx = Util.convert_to_idx(ix, iy, nx)
            Axx.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))
                                                                                                                    
            # backward one-sided stencil
            ix = (nx-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)
            Axx.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))                                                                                                                                                                                                                       

        # symmetric stencil
        stc = stcdv2_list.get(almax)
        
        ix = np.arange(kmax, nx-kmax)	
        idx = Util.convert_to_idx(ix, iy, nx)
        Axx.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))

        Axx = sum(Axx)

        return Axx
    
    @staticmethod
    def build_Axx_abs_inner(nx, ny, nly, stcdv2_list):
    
        almax = max(stcdv2_list)

        # absorbing zone 1
        iy_bottom = np.arange(1, nly)
        iy_top = (ny-1) - iy_bottom
        iy = np.concatenate((iy_bottom, iy_top))
        
        Axx = []
        for k in range(1, almax):
            
            al = k
            stc = stcdv2_list.get(al)

            # forward one-sided stencil
            ix = k
            idx = Util.convert_to_idx(ix, iy, nx)
            Axx.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))                                                                                                                                                                                                                     

            # backward one-sided stencil
            ix = (nx-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)
            Axx.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))                                                                                                                                                                                                                       

        # symmetric stencil
        stc = stcdv2_list.get(almax)
        
        ix = np.arange(almax, nx-almax)	
        idx = Util.convert_to_idx(ix, iy, nx)
        Axx.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))

        # absorbing zone 2
        iy = np.arange(nly, ny-nly)
        for k in range(1, nly):
            
            al = min(k, almax) # when al = almax, swap paramether is not affect.
            stc = stcdv2_list.get(al)

            # forward one-sided stencil
            ix = k
            idx = Util.convert_to_idx(ix, iy, nx)
            Axx.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=stc))                                                                                                                                                                                                                     

            # backward one-sided stencil
            ix = (nx-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)
            Axx.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=stc))

        Axx = sum(Axx)

        return Axx

    @staticmethod
    def build_Ayy(nx, ny, nly, nphy, absmethod, stcdv2_list):
        
        almax = max(stcdv2_list)

        if absmethod == "PML":
            ix = np.arange(1, nx-1)

        elif absmethod == "RenLiu":
            ix = np.arange(nly, nx-nly)

        else:
            raise Exception("absmethod is either \"PML\" or \"RenLiu\".")

        kmin = min(nly+nphy, almax)
        kmax = max(nly+nphy, almax)

        Ayy = []
        for k in range(kmin, almax):
            
            al = k
            stc = stcdv2_list.get(al)

            # forward one-sided stencil
            iy = k
            idx = Util.convert_to_idx(ix, iy, nx)
            Ayy.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))                                                                                                                                                                                                                     

            # backward one-sided stencil
            iy = (ny-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)
            Ayy.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))                                                                                                                                                                                                                       

        # symmetric stencil
        stc = stcdv2_list.get(almax)

        iy = np.arange(kmax, ny-kmax)	
        idx = Util.convert_to_idx(ix, iy, nx)
        Ayy.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))

        Ayy = sum(Ayy)

        return Ayy
    
    @staticmethod
    def build_Ayy_abs_inner(nx, ny, nly, stcdv2_list):
    
        almax = max(stcdv2_list)

        # absorbing zone 1
        ix_left = np.arange(1, nly)
        ix_right = (nx-1) - ix_left
        ix = np.concatenate((ix_left, ix_right))

        Ayy = []
        for k in range(1, almax):
            
            al = k
            stc = stcdv2_list.get(al)

            # forward one-sided stencil
            iy = k
            idx = Util.convert_to_idx(ix, iy, nx)
            Ayy.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))                                                                                                                                                                                                                     

            # backward one-sided stencil
            iy = (ny-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)
            Ayy.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))                                                                                                                                                                                                                       

        # symmetric stencil
        stc = stcdv2_list.get(almax)
        
        iy = np.arange(almax, ny-almax)	
        idx = Util.convert_to_idx(ix, iy, nx)
        Ayy.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))

        # absorbing zone 2
        ix = np.arange(nly, nx-nly)
        for k in range(1, nly):
            
            al = min(k, almax) # when al = almax, swap paramether is not affect.
            stc = stcdv2_list.get(al)

            # forward one-sided stencil
            iy = k
            idx = Util.convert_to_idx(ix, iy, nx)
            Ayy.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=stc))                                                                                                                                                                                                                     

            # backward one-sided stencil
            iy = (ny-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)
            Ayy.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=stc))

        Ayy = sum(Ayy)

        return Ayy

    @staticmethod
    def build_Axxpml(nx, ny, nly, nphy, pmlstcdv2_list, p_profile):
    
        iy = np.arange(1, ny-1)
        almax = max(pmlstcdv2_list)

        Axxpml = []
        for k in range(1, nly+nphy):

            al = min(k, almax)
            
            # left zone
            ix = k
            pmlstc = pmlstcdv2_list.get(al)
            pmlstc.coeff = Util.calculate_coeff(ix, nx, nly, pmlstc, p_profile)

            idx = Util.convert_to_idx(ix, iy, nx)
            Axxpml.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=1, stencil=pmlstc))

            # right zone
            ix = (nx-1) - k
            idx = Util.convert_to_idx(ix, iy, nx)
            Axxpml.append(Util.build_matrix(idx, nx, ny, dv_axis="x", swap=-1, stencil=pmlstc))

        Axxpml = sum(Axxpml)

        return Axxpml
    
    @staticmethod
    def build_Ayypml(nx, ny, nly, nphy, pmlstcdv2_list, p_profile):

        ix = np.arange(1, nx-1)
        almax = max(pmlstcdv2_list)

        Ayypml = []
        for k in range(1, nly+nphy):

            al = min(k, almax)

            # bottom zone
            iy = k
            pmlstc = pmlstcdv2_list.get(al)
            pmlstc.coeff = Util.calculate_coeff(iy, ny, nly, pmlstc, p_profile)

            idx = Util.convert_to_idx(ix, iy, nx) 
            Ayypml.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=1, stencil=pmlstc))

            # top zone
            iy = (ny-1) - k
            idx = Util.convert_to_idx(ix, iy, nx) 
            Ayypml.append(Util.build_matrix(idx, nx, ny, dv_axis="y", swap=-1, stencil=pmlstc))

        Ayypml = sum(Ayypml)

        return Ayypml
    
    @staticmethod
    def calculate_coeff(ix_or_iy, nx_or_ny, nly, pmlstc, q_profile):
    
        idx_qstencil = pmlstc.idx_qstencil
        weight_q = pmlstc.weight_q

        # calculate depth_q
        # In this work, the absorbing domain is implemented with uniform thickness on all four boundaries.
        if ix_or_iy < round(nx_or_ny/2):
            iq = ix_or_iy + idx_qstencil
            depth_q = nly - iq
            depth_q0 = nly - ix_or_iy

        if depth_q0 < 0:
            depth_q0 = 0

        depth_q[depth_q<0] = 0

        # calculate q
        q = q_profile(depth_q, nly)

        # calculate coeff
        coeff = weight_q @ q

        q0 = q_profile(depth_q0, nly)
        coeff = q0*coeff

        return coeff