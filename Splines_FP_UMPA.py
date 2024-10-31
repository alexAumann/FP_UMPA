import numpy as np
from scipy import signal as sig
from scipy import interpolate as interpolate
from multiprocessing import Process, Queue
from numpy import fft as fft





def FP_UMPA_Tensor_Splines(I_sample, I_ref, kx = 3, ky = 3, Nw = 0, N_blur = 0):
    """
    Compare speckle images with sample (Isample) and w/o sample
    (Iref) using a given window.
    max_shift can be set to the number of pixels for an "acceptable"
    speckle displacement.

    :param Isample: A list  of measurements, with the sample aligned but speckles shifted
    :param Iref: A list of empty speckle measurements with the same displacement as Isample.
    :param Nw: 2*Nw + 1 is the width of the window.
    :param step: perform the analysis on every other _step_ pixels in both directions (default 1)
    :param max_shift: Do not allow shifts larger than this number of pixels (default 4)
    :param df: Compute dark field (default True)

    Return T, dx, dy, df, f
    """

    kx = kx
    ky = ky
    Ish = I_sample[0].shape

    # Create the window
    w = np.multiply.outer(np.hamming(2*Nw+1), np.hamming(2*Nw+1))
    w /= w.sum()

    NR = len(I_sample)
    
    # Get the derivative of the reference images

    Isample = [0]*NR
    Iref = [0]*NR
    DxIref = [0]*NR
    DyIref = [0]*NR
    Dx2Iref = [0]*NR
    DxDyIref = [0]*NR
    Dy2Iref = [0]*NR


    x = np.arange(0, Ish[0])
    y = np.arange(0, Ish[1])
    
    i = np.arange(1, Ish[0] - 1, 1/(2*Nw + 1))
    j = np.arange(1, Ish[1] - 1, 1/(2*Nw + 1))



    for index in range(NR):
        temp = interpolate.RectBivariateSpline(x, y, I_ref[index], kx = kx, ky = ky, s = 0)

        Isample[index] = (interpolate.RectBivariateSpline(x, y, I_sample[index], kx = kx, ky = ky, s = 0))(i, j)
        Iref[index] = temp(i, j)
        DxIref[index]   = -(temp.partial_derivative(0, 1))(i, j)
        DyIref[index]   = (temp.partial_derivative(1, 0))(i, j)
        Dx2Iref[index]  = (temp.partial_derivative(0, 2))(i, j)
        
        DxDyIref[index] = -(temp.partial_derivative(1, 1))(i, j)
        
        Dy2Iref[index]  = (temp.partial_derivative(2, 0))(i, j)



    # Define entries of our coefficient matrix    
    L1  = cc(sum(I**2 for I in Iref), w)[::2*Nw+1, ::2*Nw+1]
    L2  = cc(sum(Iref[i]*DxIref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L3  = cc(sum(Iref[i]*DyIref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L4  = cc(sum(Iref[i]*Dx2Iref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L5  = cc(sum(Iref[i]*DxDyIref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L6  = cc(sum(Iref[i]*Dy2Iref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L7  = cc(sum(I**2 for I in DxIref), w)[::2*Nw+1, ::2*Nw+1]
    L8  = cc(sum(DxIref[i]*DyIref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L9  = cc(sum(DxIref[i]*Dx2Iref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L10 = cc(sum(DxIref[i]*DxDyIref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L11 = cc(sum(DxIref[i]*Dy2Iref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L12 = cc(sum(I**2 for I in DyIref), w)[::2*Nw+1, ::2*Nw+1]
    L13 = cc(sum(DyIref[i]*Dx2Iref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L14 = cc(sum(DyIref[i]*DxDyIref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L15 = cc(sum(DyIref[i]*Dy2Iref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L16 = cc(sum(I**2 for I in Dx2Iref), w)[::2*Nw+1, ::2*Nw+1]
    L17 = cc(sum(Dx2Iref[i]*DxDyIref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L18 = cc(sum(Dx2Iref[i]*Dy2Iref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L19 = cc(sum(I**2 for I in DxDyIref), w)[::2*Nw+1, ::2*Nw+1]
    L20 = cc(sum(DxDyIref[i]*Dy2Iref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L21 = cc(sum(I**2 for I in Dy2Iref), w)[::2*Nw+1, ::2*Nw+1]





    V1 = cc(sum(Iref[i]*Isample[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    V2 = cc(sum(DxIref[i]*Isample[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    V3 = cc(sum(DyIref[i]*Isample[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    V4 = cc(sum(Dx2Iref[i]*Isample[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    V5 = cc(sum(DxDyIref[i]*Isample[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    V6 = cc(sum(Dy2Iref[i]*Isample[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    

    # Create parameter images
    A00 = np.zeros((Ish[0] - 2, Ish[1] - 2))
    A10 = np.zeros((Ish[0] - 2, Ish[1] - 2))
    A01 = np.zeros((Ish[0] - 2, Ish[1] - 2))
    A20 = np.zeros((Ish[0] - 2, Ish[1] - 2))
    A11 = np.zeros((Ish[0] - 2, Ish[1] - 2))
    A02 = np.zeros((Ish[0] - 2, Ish[1] - 2))


    
    
    # Loop through all positions
    for i in range(Ish[0] - 2):
        for j in range(Ish[1] - 2):
            M = np.matrix([[L1[i, j], L2[i, j],  L3[i, j],  L4[i, j],  L5[i, j],  L6[i, j]],
                           [L2[i, j], L7[i, j],  L8[i, j],  L9[i, j],  L10[i, j], L11[i, j]],
                           [L3[i, j], L8[i, j],  L12[i, j], L13[i, j], L14[i, j], L15[i, j]],
                           [L4[i, j], L9[i, j],  L13[i, j], L16[i, j], L17[i, j], L18[i, j]],
                           [L5[i, j], L10[i, j], L14[i, j], L17[i, j], L19[i, j], L20[i, j]],
                           [L6[i, j], L11[i, j], L15[i, j], L18[i, j], L20[i, j], L21[i, j]]
                          ])
            V = np.matrix([
                          [V1[i, j]],
                          [V2[i, j]],
                          [V3[i, j]],
                          [V4[i, j]],
                          [V5[i, j]],
                          [V6[i, j]]])

            A_param = (M.I)*V

            # store everything
            A00[i, j] = A_param[0, 0]
            A10[i, j] = A_param[1, 0]
            A01[i, j] = A_param[2, 0]
            A20[i, j] = A_param[3, 0]
            A11[i, j] = A_param[4, 0]
            A02[i, j] = A_param[5, 0]


    Dx = (1/2) * np.array([[-1, 0, 1]]) 

    Dy = (1/2) * np.array([[ 1],
                       [ 0],
                       [-1]]) 

    Dx2 = sig.convolve2d(Dx, Dx)

    DxDy = sig.convolve2d(Dx, Dy)

    Dy2 = sig.convolve2d(Dy, Dy)



    w_blur = np.multiply.outer(np.hamming(2*N_blur+1), np.hamming(2*N_blur+1))
    w_blur /= w_blur.sum()
    
    A00 = sig.convolve2d(A00, w_blur, mode='same')
    A10 = sig.convolve2d(A10, w_blur, mode='same')
    A01 = sig.convolve2d(A01, w_blur, mode='same')
    A20 = sig.convolve2d(A20, w_blur, mode='same')
    A11 = sig.convolve2d(A11, w_blur, mode='same')
    A02 = sig.convolve2d(A02, w_blur, mode='same')

    
    S00 = A00 - sig.convolve2d(A10, Dx,  mode='same') - sig.convolve2d(A01, Dy,  mode='same') + sig.convolve2d(A20, Dx2,  mode='same') + sig.convolve2d(A11, DxDy,  mode='same') + sig.convolve2d(A02, Dy2,  mode='same')

    S10 = sig.convolve2d(A20, Dx,  mode='same') + (1/2)*sig.convolve2d(A11, Dy,  mode='same') + A20/S00 * sig.convolve2d(S00, Dx,  mode='same') + A11/(2*S00)*sig.convolve2d(S00, Dy,  mode='same') - A10
    S10 = S10 / S00
    
    S01 = sig.convolve2d(A02, Dy,  mode='same') + (1/2)*sig.convolve2d(A11, Dx,  mode='same') + A02/S00 * sig.convolve2d(S00, Dy,  mode='same') + A11/(2*S00)*sig.convolve2d(S00, Dx,  mode='same') - A01
    S01 = S01 / S00
    
    S20 = A20/S00
    S11 = A11/(2*S00)
    S02 = A02/S00
    
    return {'S00': S00, 'S10': S10, 'S01': S01, 'S20': S20, 'S11': S11, 'S02': S02,
           "A00": A00, "A10": A10, "A01": A01, "A20": A20, "A11": A11, "A02": A02}



def FP_UMPA_Scalar_Splines(I_sample, I_ref, kx = 3, ky = 3, Nw = 0, N_blur = 0):
    """
    Compare speckle images with sample (Isample) and w/o sample
    (Iref) using a given window.
    max_shift can be set to the number of pixels for an "acceptable"
    speckle displacement.

    :param Isample: A list  of measurements, with the sample aligned but speckles shifted
    :param Iref: A list of empty speckle measurements with the same displacement as Isample.
    :param Nw: 2*Nw + 1 is the width of the window.
    :param step: perform the analysis on every other _step_ pixels in both directions (default 1)
    :param max_shift: Do not allow shifts larger than this number of pixels (default 4)
    :param df: Compute dark field (default True)

    Return T, dx, dy, df, f
    """

    kx = kx
    ky = ky
    Ish = I_sample[0].shape

    # Create the window
    w = np.multiply.outer(np.hamming(2*Nw+1), np.hamming(2*Nw+1))
    w /= w.sum()

    NR = len(I_sample)
    
    # Get the derivative of the reference images

    Isample = [0]*NR
    Iref = [0]*NR
    DxIref = [0]*NR
    DyIref = [0]*NR
    Dx2Dy2Iref = [0]*NR


    x = np.arange(0, Ish[0])
    y = np.arange(0, Ish[1])
    
    i = np.arange(1, Ish[0]-1, 1/(2*Nw + 1))
    j = np.arange(1, Ish[1]-1, 1/(2*Nw + 1))



    for index in range(NR):
        temp = interpolate.RectBivariateSpline(x, y, I_ref[index], kx = kx, ky = ky, s = 0)

        Isample[index] = (interpolate.RectBivariateSpline(x, y, I_sample[index], kx = kx, ky = ky, s = 0))(i, j)
        Iref[index] = temp(i, j)
        DxIref[index]   = -(temp.partial_derivative(0, 1))(i, j)
        DyIref[index]   = (temp.partial_derivative(1, 0))(i, j)
        Dx2Dy2Iref[index]  = (temp.partial_derivative(0, 2))(i, j)
        Dx2Dy2Iref[index]  += (temp.partial_derivative(2, 0))(i, j)



    # Define entries of our coefficient matrix and bin our data
    L1  = cc(sum(I**2 for I in Iref), w)[::2*Nw+1, ::2*Nw+1]
    L2  = cc(sum(Iref[i]*DxIref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L3  = cc(sum(Iref[i]*DyIref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L4  = cc(sum(Iref[i]*Dx2Dy2Iref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L5  = cc(sum(I**2 for I in DxIref), w)[::2*Nw+1, ::2*Nw+1]
    L6  = cc(sum(DxIref[i]*DyIref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L7  = cc(sum(DxIref[i]*Dx2Dy2Iref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L8  = cc(sum(I**2 for I in DyIref), w)[::2*Nw+1, ::2*Nw+1]
    L9  = cc(sum(DyIref[i]*Dx2Dy2Iref[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    L10 = cc(sum(I**2 for I in Dx2Dy2Iref), w)[::2*Nw+1, ::2*Nw+1]


    V1 = cc(sum(Iref[i]*Isample[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    V2 = cc(sum(DxIref[i]*Isample[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    V3 = cc(sum(DyIref[i]*Isample[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]
    V4 = cc(sum(Dx2Dy2Iref[i]*Isample[i] for i in range(NR)), w)[::2*Nw+1, ::2*Nw+1]





    # Create parameter images
    A00 = np.zeros((Ish[0] - 2, Ish[1] - 2))
    A10 = np.zeros((Ish[0] - 2, Ish[1] - 2))
    A01 = np.zeros((Ish[0] - 2, Ish[1] - 2))
    A20 = np.zeros((Ish[0] - 2, Ish[1] - 2))

    
    
    # Loop through all positions
    for i in range(Ish[0] - 2):
        for j in range(Ish[1] - 2):
            # Define local values of L1, L2, ...
            M = np.matrix([[L1[i, j], L2[i, j],  L3[i, j],  L4[i, j]],
                           [L2[i, j], L5[i, j], L6[i, j], L7[i, j]],
                           [L3[i, j], L6[i, j], L8[i, j], L9[i, j]],
                           [L4[i, j], L7[i, j], L9[i, j], L10[i, j]]
                          ])

            V = np.matrix([
                          [V1[i, j]],
                          [V2[i, j]],
                          [V3[i, j]],
                          [V4[i, j]]
                            ])

            A_param = (M.I)*V

            # store everything
            A00[i, j] = A_param[0, 0]
            A10[i, j] = A_param[1, 0]
            A01[i, j] = A_param[2, 0]
            A20[i, j] = A_param[3, 0]

    Dx = (1 / 2) * np.array([[-1, 0, 1]])

    Dy = (1 / 2) * np.array([[1],
                             [0],
                             [-1]])

    Dx2 = sig.convolve2d(Dx, Dx)

    DxDy = sig.convolve2d(Dx, Dy)

    Dy2 = sig.convolve2d(Dy, Dy)

    # THIS IS FOR BLURRING
    w_blur = np.multiply.outer(np.hamming(2*N_blur+1), np.hamming(2*N_blur+1))
    w_blur /= w_blur.sum()

    A00 = sig.convolve2d(A00, w_blur,  mode='same')
    A10 = sig.convolve2d(A10, w_blur,  mode='same')
    A01 = sig.convolve2d(A01, w_blur,  mode='same')
    A20 = sig.convolve2d(A20, w_blur,  mode='same')

    S00 = A00 - sig.convolve2d(A10, Dx,  mode='same') - sig.convolve2d(A01, Dy,  mode='same') + sig.convolve2d(A20, Dx2,  mode='same') + sig.convolve2d(A20, Dy2,  mode='same')

    S10 = (-A10 + 2 * sig.convolve2d(A20, Dx,  mode='same')) / S00

    S01 = (-A01 + 2 * sig.convolve2d(A20, Dy,  mode='same')) / S00

    S20 = A20 / S00


    return {'S00': S00, 'S10': S10, 'S01': S01, 'S20': S20,
           'A00': A00, 'A10': A10, 'A01': A01, 'A20': A20}
    


def cc(A, B, mode='same'):
    """
    A fast cross-correlation based on scipy.signal.fftconvolve.

    :param A: The reference image
    :param B: The template image to match
    :param mode: one of 'same' (default), 'full' or 'valid' (see help for fftconvolve for more info)
    :return: The cross-correlation of A and B.
    """
    return sig.fftconvolve(A, B[::-1, ::-1], mode=mode)


def quad_fit(a):
    """\
    (c, x0, H) = quad_fit(A)
    Fits a parabola (or paraboloid) to A and returns the
    parameters (c, x0, H) such that

    a ~ c + (x-x0)' * H * (x-x0)

    where x is in pixel units. c is the value at the fitted optimum, x0 is
    the position of the optimum, and H is the hessian matrix (curvature in 1D).
    """

    sh = a.shape

    i0, i1 = np.indices(sh)
    i0f = i0.flatten()
    i1f = i1.flatten()
    af = a.flatten()

    # Model = p(1) + p(2) x + p(3) y + p(4) x^2 + p(5) y^2 + p(6) xy
    #       = c + (x-x0)' h (x-x0)
    A = np.vstack([np.ones_like(i0f), i0f, i1f, i0f**2, i1f**2, i0f*i1f]).T
    r = np.linalg.lstsq(A, af)
    p = r[0]
    x0 = - (np.matrix([[2*p[3], p[5]], [p[5], 2*p[4]]]).I * np.matrix([p[1], p[2]]).T).A1
    c = p[0] + .5*(p[1]*x0[0] + p[2]*x0[1])
    h = np.matrix([[p[3], .5*p[5]], [.5*p[5], p[4]]])
    return c, x0, h


    

