import numpy as np
from scipy.signal import windows

def compute_abc_reflection(angle_deg, method, R0=1e-5):

    angle_rad = np.deg2rad(angle_deg)

    if method == "EM1":
        R = (1 - np.cos(angle_rad)) / (1 + np.cos(angle_rad))

    elif method == "EM2":
        R = ((1 - np.cos(angle_rad)) / (1 + np.cos(angle_rad)))**2

    elif method == "H2":

        cos0 = 1
        cos45 = 1/np.sqrt(2)

        R1 = (cos0 - np.cos(angle_rad)) / (cos0 + np.cos(angle_rad))
        R2 = (cos45 - np.cos(angle_rad)) / (cos45 + np.cos(angle_rad))
        R = R1*R2

    elif method == "PML0":
        R = R0**np.cos(angle_rad)

    return R

def compute_fresnel_reflection(n1_over_n2, mode):

    angle_deg = np.arange(0, 90, 0.1)
    angle_inc = np.deg2rad(angle_deg)

    sin_inc = np.sin(angle_inc)
    cos_inc = np.cos(angle_inc)

    arg = 1.0 - (n1_over_n2 * sin_inc)**2
    cos_trans = np.sqrt(arg.astype(complex))

    if mode == "perpendicular":
        # perpendicular polarization
        R = (n1_over_n2*cos_inc - cos_trans) / (n1_over_n2*cos_inc + cos_trans)

    elif mode == "parallel":
        # parallel polarization
        R = (cos_inc - n1_over_n2*cos_trans) / (n1_over_n2*cos_trans + cos_inc)

    else:
       raise ValueError("mode must be 'perpendicular' or 'parallel'")
    
    return R, angle_deg

def compute_tamura_reflection(ut_line1, ut_line2, experiment_setting, algorithm_setting=("non_padding", 0.2)):

    """
    Compute the angle-dependent reflection coefficient using the Tamura method.

    Consider the reflection at interface y = y0, using wavefields recorded at 
    y1 and y2, with the condition y0 < y1 < y2.
    """

    ##### check the condition
    if len(ut_line1) != len(ut_line2):
        raise ValueError("The lengths of ut_line1 and ut_line2 must be the same.")

    ndata = len(ut_line1)

    x_step, y0, y1, y2, k_medium = experiment_setting
    preprocessing, alpha_tukey = algorithm_setting

    ##### 1) pre-processing: padding
    if preprocessing == "non_padding":
        pass

    elif preprocessing == "zero_padding":
        # zero-padding
        npad = ndata
        ut_line1 = np.pad(ut_line1, pad_width=npad, mode='constant')
        ut_line2 = np.pad(ut_line2, pad_width=npad, mode='constant')

    ##### 2) pre-processing: filtering
    window = windows.tukey(len(ut_line1), alpha=alpha_tukey)
    ut_line1_filtered = ut_line1 * window
    ut_line2_filtered = ut_line2 * window
    
    ##### 3) compute FFT of data
    U1 = np.fft.fftshift(np.fft.fft(ut_line1_filtered))
    U2 = np.fft.fftshift(np.fft.fft(ut_line2_filtered))

    ##### 4) compute propagating wavenumbers kx and ky
    kx_vectors = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(len(ut_line1_filtered), d=x_step))
    kx_step = np.abs(kx_vectors[1]-kx_vectors[0])

    idx_valid = np.abs(kx_vectors) < k_medium

    U1, U2 = U1[idx_valid], U2[idx_valid]
    kx_prop = kx_vectors[idx_valid]
    ky_prop = np.sqrt(k_medium**2 - kx_prop**2)            

    ##### 5) extract only positive angles
    angle_rad = np.arcsin(kx_prop/k_medium)
    angle_deg = np.rad2deg(angle_rad)
    idx_positive = angle_rad >=0

    angle_rad = angle_rad[idx_positive]
    angle_deg = angle_deg[idx_positive]
    U1, U2 = U1[idx_positive], U2[idx_positive]
    kx_prop, ky_prop = kx_prop[idx_positive], ky_prop[idx_positive]

    nvalid = len(idx_valid)

    ##### 6) compute the reflection using Tamura method
    Ur_ = ( U2*np.exp(-1j*ky_prop*(y1-y0)) - U1*np.exp(-1j*ky_prop*(y2-y0)) )
    Ui_ = ( U1*np.exp(+1j*ky_prop*(y2-y0)) - U2*np.exp(+1j*ky_prop*(y1-y0)) )

    R_TMR = Ur_/Ui_

    return R_TMR, angle_deg
