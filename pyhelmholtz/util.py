import numpy as np

class Util:

    @staticmethod
    def pad_array2d(x: np.ndarray, n: int) -> np.ndarray:

        """
        pad the velocity array v with n cells in all directions
        """

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
    def mask1d_sign(m: int) -> np.ndarray:
        
        """
        Assign a value of 1 to the first half of the 1D array of size m, and −1 to the remaining entries.
        """

        mid = int(m/2)
        mask1d = -1*np.ones(m, dtype=np.int32)
        mask1d[:mid] = 1

        return mask1d

    @staticmethod
    def mask2d_sign(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:

        """
        Construct two 2D sign masks of size (ny × nx):
        - mask2d_x: a 2D array marking left–right positions with +1 on the left and −1 on the right.
        - mask2d_y: a 2D array marking bottom–top positions with +1 at the bottom and −1 at the top.
        """

        mask1d_x = Util.mask1d_sign(nx)
        mask1d_y = Util.mask1d_sign(ny)

        mask2d_x, mask2d_y = np.meshgrid(mask1d_x, mask1d_y)

        return mask2d_x, mask2d_y

    @staticmethod
    def distance_from_edge(m: int) -> np.ndarray:
        
        """
        generate a symmetric distance-from-edge index array of length m
        """

        # m = 8 --> [0 1 2 3 3 2 1 0]
        # m = 9 --> [0 1 2 3 4 3 2 1 0]
        depth = np.minimum(np.arange(m), np.arange(m)[::-1]) 
        
        return depth
    
    @staticmethod
    def depth2d(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:

        """
        Construct two 2D distance-from-edge arrays of size (ny × nx):
        - depth2d_x: a 2D array storing the horizontal distance from the left or right edge.
        - depth2d_y: a 2D array storing the vertical distance from the bottom or top edge.
        """

        depth1d_x = Util.distance_from_edge(nx)
        depth1d_y = Util.distance_from_edge(ny)
        depth2d_x, depth2d_y = np.meshgrid(depth1d_x, depth1d_y)

        return depth2d_x, depth2d_y

    @staticmethod
    def mask2d_zone(nx: int, ny: int, nly: int) -> tuple[np.ndarray, np.ndarray]:

        """
        Construct two 2D sign masks of size (nx*ny):
        - mask2d_phy: a 2D array marking the physical/interior region with +1 and absorbing region with +0
        - mask2d_abs: a 2D array marking absorbing region with +1 and the physical/interior region with +0
        """

        mask2d_phy = np.zeros([ny, nx], dtype=np.int32)
        mask2d_phy[nly:-nly, nly:-nly] = 1
        mask2d_abs = np.abs(mask2d_phy-1)

        return mask2d_phy, mask2d_abs

    @staticmethod
    def get_idx_zone(
            nx: int,
            ny: int,
            nly: int,
            nphyouter: int = 0
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """
        Return a tuple of array containing 1D indices associated with each zone,
        namely the physical domain, inner physical domain, absorbing domain, and boundary.
        """

        n = nx*ny
        nexterior = nly + nphyouter

        idx2d_domain = np.arange(n, dtype=np.int32).reshape([ny, nx])
        idx1d_domain = idx2d_domain.flatten()
        
        idx1d_phy = idx2d_domain[nly:-nly, nly:-nly].flatten()
        idx1d_phy_inner = idx2d_domain[nexterior:-nexterior, nexterior:-nexterior].flatten()
        idx1d_abs = idx1d_domain[~np.isin(idx1d_domain, idx1d_phy)]

        idx1d_not_boundary = idx2d_domain[1:-1, 1:-1].flatten()
        idx1d_boundary = idx1d_domain[~np.isin(idx1d_domain, idx1d_not_boundary)]

        return idx1d_phy, idx1d_phy_inner, idx1d_abs, idx1d_boundary

    @staticmethod
    def get_sign(idx: int, nx:int, ny:int, axis: str) -> int:
        
        """
        Return a value (+1 or −1) indicating the use of a forward or backward stencil, respectively,
        based on the index position, domain size, and derivative axis.
        """

        mask2d_x, mask2d_y = Util.mask2d_sign(nx, ny)

        if axis == 'x':
            return (mask2d_x.flatten())[idx]
        
        elif axis == 'y':
            return (mask2d_y.flatten())[idx] 

    # absorbing layers
    @staticmethod
    def get_abslayer2d(nx: int, ny:int, nly: int) -> np.ndarray:

        """
        Return a 2D array that represents the k-th absorbing layer,
        with the physical domain assigned a value of 0.
        """

        depth2d_x, depth2d_y = Util.depth2d(nx, ny)
        abslayer2d = np.minimum(depth2d_x, depth2d_y)
        abslayer2d[nly:-nly, nly:-nly] = nly
        abslayer2d = nly - abslayer2d

        return abslayer2d

    @staticmethod
    def get_abszone(nx: int, ny:int, nly:int) -> np.ndarray:

        """
        Return a 2D array that represents the zone classification, where
        - corner regions in the absorbing layer are assigned a value of 0,
        - bottom/top sides in the absorbing layer are assigned a value of 1,
        - left/right sides in the absorbing layer are assigned a value of −1, and
        - the physical domain is assigned a value of 2.
        """

        n = nx*ny
        depth2d_x, depth2d_y = Util.depth2d(nx, ny)
        depth1d_x, depth1d_y = depth2d_x.flatten(), depth2d_y.flatten()

        abszone1d = np.empty(n, dtype=np.int32)
        abszone1d[depth1d_x == depth1d_y] = 0 # corner zone in absorbing layer
        abszone1d[depth1d_x > depth1d_y] = 1 # down/top sides in absorbing layer
        abszone1d[depth1d_x < depth1d_y] = -1 # left/right sides in absorbing layer
        abszone2d = abszone1d.reshape([ny, nx])
        abszone2d[nly:-nly, nly:-nly] = 2 # physical domain

        return abszone2d

    @staticmethod
    def get_idx_abslayer(k: int, abslayer2d:np.ndarray) -> np.ndarray:
        
        """
        Return the indices of the given k-th absorbing layer.
        """

        abslayer1d = abslayer2d.flatten()
        idx = np.where(abslayer1d == k)[0]

        return idx

    @staticmethod
    def get_nal2d_abs(nx: int, ny: int, nly: int, axis: str, nalstencil_max: int) -> np.ndarray:
        
        """
        Return a 2D array that specifies the armlength of stencil for each grid point in a 2D domain,
        based on the domain size (nx and ny), number of absorbing layer (nly),
        derivative axis (x- or y-direction), and maximum armlength.
        """
        
        const = nalstencil_max + 3
        mask2d_phy, mask2d_abs = Util.mask2d_zone(nx, ny, nly)

        # absorbing domain (axis = x or y)
        if axis == 'x':
            depth2d_axis, _ = Util.depth2d(nx, ny)
            mask2d_sign_axis, _ = Util.mask2d_sign(nx, ny)
            
        elif axis == 'y':
            _, depth2d_axis = Util.depth2d(nx, ny)
            _, mask2d_sign_axis = Util.mask2d_sign(nx, ny)

        nal1d_abs = np.copy((mask2d_sign_axis*depth2d_axis).flatten())
        nal1d_abs[np.abs(nal1d_abs)>=nalstencil_max] = nalstencil_max
        nal2d_abs = nal1d_abs.reshape([ny, nx])
        nal2d_abs = mask2d_abs*nal2d_abs + const*mask2d_phy

        return nal2d_abs

    @staticmethod
    def get_narmlength_arr(nal2d: np.ndarray, nly: int, nphyouter: int, zone: str) -> np.ndarray:

        """
        Return a 1D array of stencil lengths (from minimum to maximum)
        based on the given domain size and zone.
        """
        
        ny, nx = nal2d.shape
        nal1d = nal2d.flatten()
        idx1d_phy, idx1d_phy_inner, idx1d_abs, _ = Util.get_idx_zone(nx, ny, nly, nphyouter)

        if zone == 'phy':
            nalarr = np.unique(nal1d[idx1d_phy])
        
        # nphyouter = 1,2,... (for PML case)
        elif zone == 'phy_inner':
            nalarr = np.unique(nal1d[idx1d_phy_inner])

        elif zone == 'abs':
            nalarr = np.unique(nal1d[idx1d_abs])
        
        return nalarr[np.argsort(abs(nalarr))]

    @staticmethod
    def get_idx_abs_corners(
            klayer: int,
            nx: int,
            ny: int,
            nly: int
        ) -> tuple[int, int, int, int]:

        """
        Return the corner indices of the given k-layer absorbing domain.
        """

        if klayer > nly:
            raise Exception('klayer should be less than or equal to nly.')

        depth = nly - klayer

        ixc1 = depth
        ixc2 = nx - 1 - depth
        iyc1 = depth
        iyc2 = ny - 1 - depth
        
        return ixc1, ixc2, iyc1, iyc2

    @staticmethod
    def get_idx_abslayer_corners_and_sides(
            klayer: int,
            nx: int,
            ny: int,
            nly: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Return the index positions of the specified k-layer absorbing domain,
        ordered as left, right, bottom, top, and the four corner regions.
        """

        ixc1, ixc2, iyc1, iyc2 = Util.get_idx_abs_corners(klayer, nx, ny, nly)

        clb = np.array([ixc1 + nx*iyc1])
        crb = np.array([ixc2 + nx*iyc1])
        clt = np.array([ixc1 + nx*iyc2])
        crt = np.array([ixc2 + nx*iyc2])

        sl = ixc1 + nx*np.arange(iyc1+1, iyc2, dtype=np.int32)
        sr = ixc2 + nx*np.arange(iyc1+1, iyc2, dtype=np.int32)
        sb = np.arange(ixc1+1, ixc2, dtype=np.int32) + nx*iyc1
        st = np.arange(ixc1+1, ixc2, dtype=np.int32) + nx*iyc2

        return sl, sr, sb, st, clb, crb, clt, crt

    # This method returns the row and column indices of a sparse matrix of type csr_matrix
    # Added by Chaiwoot on Dec 5, 2025    
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