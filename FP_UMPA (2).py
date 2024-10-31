import numpy as np
from scipy import signal as sig
from scipy import interpolate as interpolate
from multiprocessing import Process, Queue

# Note, savgol filters along axis = 1 have negative sign to make it consistent with finite difference approach.

def deriv_for_scalar_rep(NR, Iref, savgol, polyorder, window_length):
    """
    :param NR: Number of reference/sample images
    :param Iref: A 3D numpy array where axis = 0 has a reference scan for each diffuser position
    :param savgol: A boolean where True indicate savgol filters should be used, otherwise central finite differences
    are used
    :param polyorder: The polynomial order that should be used for savgol filters
    :param window_length: The window size to be used for savgol filters
    :return: Calculates the Dx, Dy and Dx^2 + Dy^2 of Iref
    """
    DxIref = np.array([0] * NR, dtype=object)
    DyIref = np.array([0] * NR, dtype=object)
    Dx2_p_Dy2Iref = np.array([0] * NR, dtype=object)
    if savgol:
        for i in range(NR):
            DxIref[i] = -sig.savgol_filter(Iref[i], window_length, polyorder, deriv=1, axis=1)
            DyIref[i] = sig.savgol_filter(Iref[i], window_length, polyorder, deriv=1, axis=0)
            Dx2_p_Dy2Iref[i] = sig.savgol_filter(Iref[i], window_length, polyorder, deriv=2, axis=1)
            Dx2_p_Dy2Iref[i] += sig.savgol_filter(Iref[i], window_length, polyorder, deriv=2, axis=0)
    else:
        Dx = (1/2) * np.array([[-1, 0, 1]])
        Dy = (1/2) * np.array([[1], [0], [-1]])
        Dx2 = sig.convolve2d(Dx, Dx)
        Dy2 = sig.convolve2d(Dy, Dy)
        for i in range(NR):
            DxIref[i] = sig.convolve2d(Iref[i], Dx, mode='same')
            DyIref[i] = sig.convolve2d(Iref[i], Dy, mode='same')
            Dx2_p_Dy2Iref[i] = sig.convolve2d(Iref[i], Dx2, mode='same') + sig.convolve2d(Iref[i], Dy2, mode='same')

    # Ensuring tha these are 3D numpy arrays
    DxIref = np.stack(DxIref)
    DyIref = np.stack(DyIref)
    Dx2_p_Dy2Iref = np.stack(Dx2_p_Dy2Iref)
    return DxIref, DyIref, Dx2_p_Dy2Iref


def deriv_for_tensor_rep(NR, Iref, savgol, polyorder, window_length):
    """
    :param NR: Number of reference/sample images
    :param Iref: A 3D numpy array where axis = 0 has a reference scan for each diffuser position
    :param savgol: A boolean where True indicate savgol filters should be used, otherwise central finite differences
    are used
    :param polyorder: The polynomial order that should be used for savgol filters
    :param window_length: The window size to be used for savgol filters
    :return: Calculates the Dx, Dy,  Dx^2, DxDy and Dy^2 of Iref
    """
    DxIref = np.array([0] * NR, dtype=object)
    DyIref = np.array([0] * NR, dtype=object)
    Dx2Iref = np.array([0] * NR, dtype=object)
    DxDyIref = np.array([0] * NR, dtype=object)
    Dy2Iref = np.array([0] * NR, dtype=object)
    if savgol:
        for i in range(NR):
            DxIref[i] = -sig.savgol_filter(Iref[i], window_length, polyorder, deriv=1, axis=1)
            DyIref[i] = sig.savgol_filter(Iref[i], window_length, polyorder, deriv=1, axis=0)
            Dx2Iref[i] = sig.savgol_filter(Iref[i], window_length, polyorder, deriv=2, axis=1)
            DxDyIref[i] = -sig.savgol_filter(Iref[i], window_length, polyorder, deriv=1, axis=1)
            DxDyIref[i] = sig.savgol_filter(DxDyIref[i], window_length, polyorder, deriv=1, axis=0)
            Dy2Iref[i] = sig.savgol_filter(Iref[i], window_length, polyorder, deriv=2, axis=0)
            print("hi")
    else:
        Dx = (1/2) * np.array([[-1, 0, 1]])
        Dy = (1/2) * np.array([[1], [0], [-1]])
        Dx2 = sig.convolve2d(Dx, Dx)
        DxDy = sig.convolve2d(Dx, Dy)
        Dy2 = sig.convolve2d(Dy, Dy)
        for i in range(NR):
            DxIref[i] = sig.convolve2d(Iref[i], Dx, mode='same')
            DyIref[i] = sig.convolve2d(Iref[i], Dy, mode='same')
            Dx2Iref[i] = sig.convolve2d(Iref[i], Dx2, mode='same')
            DxDyIref[i] = sig.convolve2d(Iref[i], DxDy, mode='same')
            Dy2Iref[i] = sig.convolve2d(Iref[i], Dy2, mode='same')

    # Ensuring that these are 3D numpy arrays
    DxIref = np.stack(DxIref)
    DyIref = np.stack(DyIref)
    Dx2Iref = np.stack(Dx2Iref)
    DxDyIref = np.stack(DxDyIref)
    Dy2Iref = np.stack(Dy2Iref)
    return DxIref, DyIref, Dx2Iref, DxDyIref, Dy2Iref


def matrix_entries_for_scalar_rep(NR, Iref, Isam, DxIref, DyIref, Dx2_p_Dy2Iref):
    """
    :param NR: Number of reference and sample images
    :param Iref: A 3D numpy array where axis = 0 has a reference scan for each diffuser position
    :param Isam: A 3D numpy array where axis = 0 has a sample scan for each diffuser position
    :param DxIref: A 3D numpy array which contains the first x-derivative of each reference scan
    :param DyIref: A 3D numpy array which contains the first y-derivative of each reference scan
    :param Dx2_p_Dy2Iref: A 3D numpy array which contains the Laplacian of each reference scan
    :return: The matrix entries for the minimisation equation
    """
    # Define entries of our coefficient matrix
    L1 = sum(I ** 2 for I in Iref)
    L2 = sum(Iref[i] * DxIref[i] for i in range(NR))
    L3 = sum(Iref[i] * DyIref[i] for i in range(NR))
    L4 = sum(Iref[i] * Dx2_p_Dy2Iref[i] for i in range(NR))
    L5 = sum(I ** 2 for I in DxIref)
    L6 = sum(DxIref[i] * DyIref[i] for i in range(NR))
    L7 = sum(DxIref[i] * Dx2_p_Dy2Iref[i] for i in range(NR))
    L8 = sum(I ** 2 for I in DyIref)
    L9 = sum(DyIref[i] * Dx2_p_Dy2Iref[i] for i in range(NR))
    L10 = sum(I ** 2 for I in Dx2_p_Dy2Iref)

    # Define entries of our column vector
    V1 = sum(Iref[i] * Isam[i] for i in range(NR))
    V2 = sum(DxIref[i] * Isam[i] for i in range(NR))
    V3 = sum(DyIref[i] * Isam[i] for i in range(NR))
    V4 = sum(Dx2_p_Dy2Iref[i] * Isam[i] for i in range(NR))

    return (L1, L2, L3, L4, L5, L6, L7, L8, L9, L10), (V1, V2, V3, V4)


def matrix_entries_for_tensor_rep(NR, Iref, Isam, DxIref, DyIref, Dx2Iref, DxDyIref, Dy2Iref):
    """
    :param NR: Number of reference and sample images
    :param Iref: A 3D numpy array where axis = 0 has a reference scan for each diffuser position
    :param Isam: A 3D numpy array where axis = 0 has a sample scan for each diffuser position
    :param DxIref: A 3D numpy array which contains the first x-derivative of each reference scan
    :param DyIref: A 3D numpy array which contains the first y-derivative of each reference scan
    :param Dx2Iref: A 3D numpy array which contains the second x-derivative of each reference scan
    :param DxDyIref: A 3D numpy array which contains the  mixed x-y-derivative of each reference scan
    :param Dy2Iref: A 3D numpy array which contains the second y-derivative of each reference scan
    :return: The matrix entries for the minimisation equation
    """
    # Define entries of our coefficient matrix
    L1  = sum(I**2 for I in Iref)
    L2  = sum(Iref[i]*DxIref[i] for i in range(NR))
    L3  = sum(Iref[i]*DyIref[i] for i in range(NR))
    L4  = sum(Iref[i]*Dx2Iref[i] for i in range(NR))
    L5  = sum(Iref[i]*DxDyIref[i] for i in range(NR))
    L6  = sum(Iref[i]*Dy2Iref[i] for i in range(NR))
    L7  = sum(I**2 for I in DxIref)
    L8  = sum(DxIref[i]*DyIref[i] for i in range(NR))
    L9  = sum(DxIref[i]*Dx2Iref[i] for i in range(NR))
    L10 = sum(DxIref[i]*DxDyIref[i] for i in range(NR))
    L11 = sum(DxIref[i]*Dy2Iref[i] for i in range(NR))
    L12 = sum(I**2 for I in DyIref)
    L13 = sum(DyIref[i]*Dx2Iref[i] for i in range(NR))
    L14 = sum(DyIref[i]*DxDyIref[i] for i in range(NR))
    L15 = sum(DyIref[i]*Dy2Iref[i] for i in range(NR))
    L16 = sum(I**2 for I in Dx2Iref)
    L17 = sum(Dx2Iref[i]*DxDyIref[i] for i in range(NR))
    L18 = sum(Dx2Iref[i]*Dy2Iref[i] for i in range(NR))
    L19 = sum(I**2 for I in DxDyIref)
    L20 = sum(DxDyIref[i]*Dy2Iref[i] for i in range(NR))
    L21 = sum(I**2 for I in Dy2Iref)

    # Define entries of our column vector
    V1 = sum(Iref[i]*Isam[i] for i in range(NR))
    V2 = sum(DxIref[i]*Isam[i] for i in range(NR))
    V3 = sum(DyIref[i]*Isam[i] for i in range(NR))
    V4 = sum(Dx2Iref[i]*Isam[i] for i in range(NR))
    V5 = sum(DxDyIref[i]*Isam[i] for i in range(NR))
    V6 = sum(Dy2Iref[i]*Isam[i] for i in range(NR))

    return ((L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15, L16, L17, L18, L19, L20, L21),
            (V1, V2, V3, V4, V5, V6))


def calc_param_for_scalar_rep(L, V):
    """
    :param L: A list of entries for the coefficient matrix for the scalar representation
    :param V: A list of entries for the constant vector for the scalar representation
    :return: The entries which solves L.A =  V for the scalar representation
    """
    # Store the shape of the recorded images
    Ish = np.shape(L[0])

    # Define variables which will hold the resultant parameter images
    A00 = np.zeros(Ish)
    A10 = np.zeros(Ish)
    A01 = np.zeros(Ish)
    A20 = np.zeros(Ish)

    # Loop through all pixels
    for i in range(Ish[0]):
        for j in range(Ish[1]):
            # Get the local values of L1, L2 ... and V1, V2...
            # to construct a linear system at pixel (i, j) where the solution vector
            # minimises the cost function and corresponds to the parameters, the ones defined in the thesis, at (i, j)
            M = np.matrix([[L[0][i, j], L[1][i, j], L[2][i, j], L[3][i, j]],
                           [L[1][i, j], L[4][i, j], L[5][i, j], L[6][i, j]],
                           [L[2][i, j], L[5][i, j], L[7][i, j], L[8][i, j]],
                           [L[3][i, j], L[6][i, j], L[8][i, j], L[9][i, j]],
                           ], dtype=np.float64)

            B = np.matrix([
                [V[0][i, j]],
                [V[1][i, j]],
                [V[2][i, j]],
                [V[3][i, j]]
            ], dtype=np.float64)

            # There may exist local regions within the image which aren't physical which may result in a coefficient
            # matrix which is singular. For example, a region where there are no photons recorded in the reference
            # images for the forward model of speckle modulation.
            try:
                A_vec = np.linalg.solve(M, B)
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    A_vec = np.array([[0], [0], [0], [0]])
                else:
                    raise

            # store everything
            A00[i, j] = A_vec[0, 0]
            A10[i, j] = A_vec[1, 0]
            A01[i, j] = A_vec[2, 0]
            A20[i, j] = A_vec[3, 0]

    return A00, A10, A01, A20


def calc_param_for_tensor_rep(L, V):
    """
    :param L: A list of entries for the coefficient matrix for the tensor representation
    :param V: A list of entries for the constant vector for the tensor representation
    :return: The entries which solves L.A =  V for the tensor representation
    """
    # Store the shape of the recorded images
    Ish = np.shape(L[0])

    # Define variables which will hold the resultant parameter images
    A00 = np.zeros(Ish)
    A10 = np.zeros(Ish)
    A01 = np.zeros(Ish)
    A20 = np.zeros(Ish)
    A11 = np.zeros(Ish)
    A02 = np.zeros(Ish)

    # Loop through all pixels
    for i in range(Ish[0]):
        for j in range(Ish[1]):
            # Get the local values of L1, L2 ... and V1, V2...
            # to construct a linear system at pixel (i, j) where the solution vector
            # minimises the cost function and corresponds to the parameters, the ones defined in the thesis, at (i, j)
            M = np.matrix([[L[0][i, j], L[1][i, j],  L[2][i, j],   L[3][i, j],   L[4][i, j],   L[5][i, j] ],
                           [L[1][i, j], L[6][i, j],  L[7][i, j],   L[8][i, j],   L[9][i, j],   L[10][i, j]],
                           [L[2][i, j], L[7][i, j],  L[11][i, j],  L[12][i, j],  L[13][i, j],  L[14][i, j]],
                           [L[3][i, j], L[8][i, j],  L[12][i, j],  L[15][i, j],  L[16][i,j],   L[17][i, j]],
                           [L[4][i, j], L[9][i, j],  L[13][i, j],  L[16][i, j],  L[18][i, j],  L[19][i, j]],
                           [L[5][i, j], L[10][i, j], L[14][i, j], L[17][i, j],   L[19][i, j],  L[20][i, j]]
                           ], dtype=np.float64)

            B = np.matrix([
                [V[0][i, j]],
                [V[1][i, j]],
                [V[2][i, j]],
                [V[3][i, j]],
                [V[4][i, j]],
                [V[5][i, j]]
            ], dtype=np.float64)

            # There may exist local regions within the image which aren't physical which may result in a coefficient
            # matrix which is singular. For example, a region where there are no photons recorded in the reference
            # images for the forward model of speckle modulation.
            try:
                A_vec = np.linalg.solve(M, B)
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    A_vec = np.array([[0], [0], [0], [0], [0], [0]])
                else:
                    raise

            # store everything
            A00[i, j] = A_vec[0, 0]
            A10[i, j] = A_vec[1, 0]
            A01[i, j] = A_vec[2, 0]
            A20[i, j] = A_vec[3, 0]
            A11[i, j] = A_vec[4, 0]
            A02[i, j] = A_vec[5, 0]

    return A00, A10, A01, A20, A11, A02



def UMPA_FP_minimise_helper(out_queue, i_proc, i_sam, i_ref, rep, savgol, polyorder = 3, window_length = 5):
    """
    :param out_queue: A queue to be used for parallel computation
    :param i_proc: The process to be considered for computation
    :param i_sam: A 3D numpy array of sample images where axis = 0 correspond to different mask positions
    :param i_ref: A 3D numpy array of reference images where axis = 0 corresponds to different mask positions
    :param rep: A string which takes 'scalar' or 'tensor' which dictates what representation of the diffusion field to be
    used.
    :param savgol: A boolean where True indicates the savgol are to be employed, otherwise central finite difference are
    to be used.
    :param polyorder: The polynomial order to be used for savgol filters
    :param window_length: The window length to be used for savgol filters
    :return:
    """
    # Ensuring that the data is presented as a 3D numpy array
    Isam = np.stack(i_sam)
    Iref = np.stack(i_ref)

    # Storing the number of mask positions in the variable NR
    NR = len(Isam)

    if rep.lower() == 'scalar':
        DxIref, DyIref, Dx2_p_Dy2Iref = deriv_for_scalar_rep(NR, Iref, savgol, polyorder, window_length)
        L, V = matrix_entries_for_scalar_rep(NR, Iref, Isam, DxIref, DyIref, Dx2_p_Dy2Iref)
        del DxIref, DyIref, Dx2_p_Dy2Iref
        A00, A10, A01, A20 = calc_param_for_scalar_rep(L, V)

        out_queue.put([i_proc, [A00, A10, A01, A20]])

    elif rep.lower() == 'tensor':
        DxIref, DyIref, Dx2Iref, DxDyIref, Dy2Iref = deriv_for_tensor_rep(NR, Iref, savgol, polyorder, window_length)
        L, V = matrix_entries_for_tensor_rep(NR, Iref, Isam, DxIref, DyIref, Dx2Iref, DxDyIref, Dy2Iref)
        del DxIref, DyIref, Dx2Iref, DxDyIref, Dy2Iref
        A00, A10, A01, A20, A11, A02 = calc_param_for_tensor_rep(L, V)

        out_queue.put([i_proc, [A00, A10, A01, A20, A11, A02]])

    else:
        return "Please select 'scalar' or 'tensor' representation for the diffusion field."


def collect_param_for_tensor_rep(results):
    """
    :param results: A list of results where each along axis = 0 corresponds to different sub-problem which has been solved
    in the tensor representation
    :return: A list of parameter images for the tensor representation
    """
    A00 = []
    A10 = []
    A01 = []
    A20 = []
    A11 = []
    A02 = []

    for data in results:
        A00.append(data[0])
        A10.append(data[1])
        A01.append(data[2])
        A20.append(data[3])
        A11.append(data[4])
        A02.append(data[5])

    # Use vstack to recreate the images
    A00 = np.vstack(A00)
    A10 = np.vstack(A10)
    A01 = np.vstack(A01)
    A20 = np.vstack(A20)
    A11 = np.vstack(A11)
    A02 = np.vstack(A02)

    return [A00, A10, A01, A20, A11, A02]


def collect_param_for_scalar_rep(results):
    """
    :param results: A list of results where each along axis = 0 corresponds to different sub-problem which has been solved
    in the scalar representation
    :return: A list of parameter images for the scalar representation
    """
    A00 = []
    A10 = []
    A01 = []
    A20 = []

    for data in results:
        A00.append(data[0])
        A10.append(data[1])
        A01.append(data[2])
        A20.append(data[3])

    # Use vstack to recreate the images
    A00 = np.vstack(A00)
    A10 = np.vstack(A10)
    A01 = np.vstack(A01)
    A20 = np.vstack(A20)

    return [A00, A10, A01, A20]


def subdivide_parallel_helper(n_proc, i_sam, i_ref, rep, savgol, polyorder, window_length):
    """
    :param n_proc: The number of processes to be used for parallel computation
    :param i_sam: A 3D numpy array of sample images where axis = 0 corresponds to different scan positions
    :param i_ref: A 3D numpy array of reference images where axis = 0 corresponds to different scan positions
    :param rep: The representation of the diffusion field to be used. Takes "tensor" or "scalar".
    :param savgol: A boolean where True indicates the savgol are to be employed, otherwise central finite difference are
    to be used.
    :param polyorder: The polynomial order to be used for savgol filters
    :param window_length: The window length to be used for savgol filters
    :return: Retrieved parameter images for the whole region over which parallel computation was performed
    """
    # Ensuring that the data is presented as a 3D numpy array
    Isam = np.stack(i_sam)
    Iref = np.stack(i_ref)

    # Store the shape of the images
    Ish = Isam[0].shape

    # create a list of processes and a queue
    all_p = []
    out = Queue()

    if savgol:
        window_extent = window_length - 1
    else:
        window_extent = 4
    half_win_ext = int(window_extent/2)
    # The bin size is the vertical height of a full subtask
    bin_size_parallel = int(np.ceil((Ish[0] - window_extent) / float(n_proc)))
    if window_extent >= bin_size_parallel - window_extent:
        raise Exception("Please decrease either n_proc or N_div")

    for i_proc in range(n_proc):
        print("create subtask " + str(1 + i_proc))
        # Create individual subtasks to be computed in parallel
        # The subtasks are defined over horizontal regions in the sample and reference images

        # This is where the region should start for each subtask
        line_start_p = half_win_ext + i_proc * bin_size_parallel
        # This is where the region should end for each subtask
        line_end_p = line_start_p + bin_size_parallel
        if line_end_p > Ish[0] - half_win_ext:
            line_end_p = Ish[0] - half_win_ext
            bin_size_parallel = line_end_p - line_start_p

        p = Process(target=UMPA_FP_minimise_helper, args=(out, i_proc, Isam[:, line_start_p - half_win_ext:line_end_p +
                                                          half_win_ext], Iref[:, line_start_p - half_win_ext:line_end_p +
                                                          half_win_ext], rep, savgol, polyorder, window_length))
        p.deamon = True
        p.start()
        all_p.append(p)
    # Get results from parallel computation
    results = []
    for _ in range(n_proc):
        results.append(out.get())

    for p in all_p:
        p.join()

    # Use the meaningful region to reconstruct parameter images
    param_res = []
    for row in sorted(results):
        param_res.append(np.stack(row[1])[:, half_win_ext:-half_win_ext])

    if rep.lower() == 'scalar':
        return collect_param_for_scalar_rep(param_res)
    elif rep.lower() == 'tensor':
        return collect_param_for_tensor_rep(param_res)
    else:
        raise Exception("Please select 'scalar' or 'tensor' representation for the diffusion field.")


def subdivide_helper(N_div, n_proc, i_sam, i_ref, rep, savgol, polyorder, window_length):
    """
    :param N_div: An integer which corresponds to how subregions the image is to be divided into.
    :param n_proc: The number of processes to be used for parallel computation
    :param i_sam: A 3D numpy array of sample images where axis = 0 corresponds to different scan positions
    :param i_ref: A 3D numpy array of reference images where axis = 0 corresponds to different scan positions
    :param rep: The representation of the diffusion field to be used. Takes "tensor" or "scalar".
    :param savgol: A boolean where True indicates the savgol are to be employed, otherwise central finite difference are
    to be used.
    :param polyorder: The polynomial order to be used for savgol filters
    :param window_length: The window length to be used for savgol filters
    :return: Retrieved parameter images over the entire domain of the image
    """
    # Ensuring that the data is presented as a 3D numpy array
    Isam = np.stack(i_sam)
    Iref = np.stack(i_ref)

    # Store the shape of the images
    Ish = Isam[0].shape

    # the number of pixels adjacent to the centre pixel
    if savgol:
        window_extent = window_length - 1
    else:
        window_extent = 4
    half_win_ext = int(window_extent/2)
    # subdivide problem so that memory requirements aren't as demanding (mostly applicable for interpolation approach)
    # The extent of the window needs to be considered as the window for differentiation along y
    # as this also dictates the size of the subregions which need to be considered.
    bin_size = int(np.ceil((Ish[0] - window_extent) / float(N_div)))
    results = []
    for i in range(N_div):
        print("subregion " + str(i + 1))
        # This is where the region should start for each subregion
        line_start = half_win_ext + i * bin_size
        # This is where the region should end for each subregion
        line_end = line_start + bin_size
        if line_end > Ish[0] - half_win_ext:
            line_end = Ish[0] - half_win_ext
            bin_size = line_end - line_start
        temp = subdivide_parallel_helper(n_proc, Isam[:, line_start - half_win_ext:line_end + half_win_ext],
                                         Iref[:, line_start - half_win_ext:line_end + half_win_ext], rep, savgol,
                                         polyorder, window_length)
        # Extract the meaningful region to reconstruct parameter images
        results.append(np.stack(temp))
    del temp

    if rep.lower() == 'scalar':
        return collect_param_for_scalar_rep(results)
    elif rep.lower() == 'tensor':
        return collect_param_for_tensor_rep(results)
    else:
        return Exception("Please select 'scalar' or 'tensor' representation for the diffusion field.")


def calculate_signal_scalar_rep(A00, A10, A01, A20, N_blur, savgol, polyorder, window_length):
    """
    :param A00: 2D numpy array which corresponds to the A00 parameter in the scalar representation
    :param A10: 2D numpy array which corresponds to the A10 parameter in the scalar representation
    :param A01: 2D numpy array which corresponds to the A01 parameter in the scalar representation
    :param A20: 2D numpy array which corresponds to the A20 parameter in the scalar representation
    :param N_blur: An integer which species the size of the blurring kernel to be applied to the parameters. The size of
     the kernel is (2 * N_blur + 1) * (2 * N_blur + 1)
    :param savgol: A boolean where True indicates the savgol are to be employed, otherwise central finite difference are
    to be used.
    :param polyorder: The polynomial order to be used for savgol filters
    :param window_length: The window length to be used for savgol filters
    :return: A dictionary of the calculated signals and the blurred parameters in the scalar representation of diffusion
    """
    # Create blurring kernel
    w = np.multiply.outer(np.hamming(2 * N_blur + 1), np.hamming(2 * N_blur + 1))
    w /= w.sum()

    # Blur parameters
    A00 = sig.correlate2d(A00, w, mode='same')
    A10 = sig.correlate2d(A10, w, mode='same')
    A01 = sig.correlate2d(A01, w, mode='same')
    A20 = sig.correlate2d(A20, w, mode='same')

    if savgol:
        DxA10 = -sig.savgol_filter(A10, window_length, polyorder, deriv=1, axis=1)
        DyA01 = sig.savgol_filter(A01, window_length, polyorder, deriv=1, axis=0)
        DxA20 = -sig.savgol_filter(A20, window_length, polyorder, deriv=1, axis=1)
        DyA20 = sig.savgol_filter(A20, window_length, polyorder, deriv=1, axis=0)
        Dx2A20 = sig.savgol_filter(A20, window_length, polyorder, deriv=2, axis=1)
        Dy2A20 = sig.savgol_filter(A20, window_length, polyorder, deriv=2, axis=0)
    else:
        Dx = (1/2) * np.array([[-1, 0, 1]])
        Dy = (1/2) * np.array([[1], [0], [-1]])
        Dx2 = sig.convolve2d(Dx, Dx)
        Dy2 = sig.convolve2d(Dy, Dy)

        DxA10 = sig.convolve2d(A10, Dx, mode='same')
        DyA01 = sig.convolve2d(A01, Dy, mode='same')
        DxA20 = sig.convolve2d(A20, Dx, mode='same')
        DyA20 = sig.convolve2d(A20, Dy, mode='same')
        Dx2A20 = sig.convolve2d(A20, Dx2, mode='same')
        Dy2A20 = sig.convolve2d(A20, Dy2, mode='same')


    # This error is ignored when the signals are calculated. This error
    # should only occur when the coefficient matrix for minimisation is singular
    with np.errstate(divide='ignore'):
        # Transmission
        S00 = A00 - DxA10 - DyA01 + Dx2A20 + Dy2A20

        # Differential phase in x up to a multiplicative constant, where the constant is delta/k
        S10 = (-A10 + 2 * DxA20) / S00
        # Differential phase in x up to a multiplicative constant, where the constant is delta/k
        S01 = (-A01 + 2 * DyA20) / S00

        # Isotropic diffusion coefficient up to a multiplicative constant, where the constant is delta**2
        S20 = A20 / S00

        # Remove any potential numerical artefacts with calculated signals
        S00 = np.nan_to_num(S00, nan=0.0, posinf=0.0, neginf=0.0)
        S10 = np.nan_to_num(S10, nan=0.0, posinf=0.0, neginf=0.0)
        S01 = np.nan_to_num(S01, nan=0.0, posinf=0.0, neginf=0.0)
        S20 = np.nan_to_num(S20, nan=0.0, posinf=0.0, neginf=0.0)

    return {'S00': S00, 'S10': S10, 'S01': S01, 'S20': S20, "A00": A00, "A10": A10, "A01": A01, "A20": A20}


def calculate_signal_tensor_rep(A00, A10, A01, A20, A11, A02, N_blur, savgol, polyorder, window_length):
    """
    :param A00: 2D numpy array which corresponds to the A00 parameter in the tensor representation
    :param A10: 2D numpy array which corresponds to the A10 parameter in the tensor representation
    :param A01: 2D numpy array which corresponds to the A01 parameter in the tensor representation
    :param A20: 2D numpy array which corresponds to the A20 parameter in the tensor representation
    :param A11: 2D numpy array which corresponds to the A11 parameter in the tensor representation
    :param A02: 2D numpy array which corresponds to the A02 parameter in the tensor representation
    :param N_blur: An integer which species the size of the blurring kernel to be applied to the parameters. The size of
     the kernel is (2 * N_blur + 1) * (2 * N_blur + 1)
    :param savgol: A boolean where True indicates the savgol are to be employed, otherwise central finite difference are
    to be used.
    :param polyorder: The polynomial order to be used for savgol filters
    :param window_length: The window length to be used for savgol filters
    :return: A dictionary of the calculated signals and the blurred parameters in the tensor representation of diffusion
    """
    # Create blurring kernel
    w = np.multiply.outer(np.hamming(2 * N_blur + 1), np.hamming(2 * N_blur + 1))
    w /= w.sum()

    # Blur parameters
    A00 = sig.correlate2d(A00, w, mode='same')
    A10 = sig.correlate2d(A10, w, mode='same')
    A01 = sig.correlate2d(A01, w, mode='same')
    A20 = sig.correlate2d(A20, w, mode='same')
    A11 = sig.correlate2d(A11, w, mode='same')
    A02 = sig.correlate2d(A02, w, mode='same')

    # Calculate the derivatives of the parameters
    if savgol:
        DxA10 = -sig.savgol_filter(A10, window_length, polyorder, deriv=1, axis=1)
        DyA01 = sig.savgol_filter(A01, window_length, polyorder, deriv=1, axis=0)

        DxA20 = -sig.savgol_filter(A20, window_length, polyorder, deriv=1, axis=1)
        DxA11 = -sig.savgol_filter(A11, window_length, polyorder, deriv=1, axis=1)
        DyA11 = sig.savgol_filter(A11, window_length, polyorder, deriv=1, axis=0)
        DyA02 = sig.savgol_filter(A02, window_length, polyorder, deriv=1, axis=0)

        Dx2A20 = sig.savgol_filter(A20, window_length, polyorder, deriv=2, axis=1)
        DxDyA11 = sig.savgol_filter(DyA11, window_length, polyorder, deriv=1, axis=1)
        Dy2A02 = sig.savgol_filter(A02, window_length, polyorder, deriv=2, axis=0)
    else:
        Dx = (1/2) * np.array([[-1, 0, 1]])
        Dy = (1/2) * np.array([[1], [0], [-1]])
        Dx2 = sig.convolve2d(Dx, Dx)
        DxDy = sig.convolve2d(Dx, Dy)
        Dy2 = sig.convolve2d(Dy, Dy)

        DxA10 = sig.convolve2d(A10, Dx, mode='same')
        DyA01 = sig.convolve2d(A01, Dy, mode='same')

        DxA20 = sig.convolve2d(A20, Dx,  mode='same')
        DxA11 = sig.convolve2d(A11, Dx, mode='same')
        DyA11 = sig.convolve2d(A11, Dy, mode='same')
        DyA02 = sig.convolve2d(A02, Dy, mode='same')

        Dx2A20 = sig.convolve2d(A20, Dx2, mode='same')
        DxDyA11 = sig.convolve2d(A11, DxDy, mode='same')
        Dy2A02 = sig.convolve2d(A02, Dy2, mode='same')


    # This error is ignored when the signals are calculated. This error
    # should only occur when the coefficient matrix for minimisation is singular
    with np.errstate(divide='ignore'):
        # Transmission
        S00 = A00 - DxA10 - DyA01 + Dx2A20 + DxDyA11 + Dy2A02

        # Either have separate param for polynomial order when take derivatives of the retrieved parameters,
        # or hard-code a value. For now, its hard-coded as 3.
        if savgol:
            DxS00 = sig.savgol_filter(S00, window_length, 3, deriv=1, axis=1)
            DyS00 = sig.savgol_filter(S00, window_length, 3, deriv=1, axis=0)
        else:
            Dx = (1 / 2) * np.array([[-1, 0, 1]])
            Dy = (1 / 2) * np.array([[1], [0], [-1]])
            DxS00 = sig.convolve2d(S00, Dx, mode = 'same')
            DyS00 = sig.convolve2d(S00, Dy, mode = 'same')

        # Differential phase in x up to a multiplicative constant, where the constant is delta/k
        S10 = DxA20 + (1 / 2) * DyA11 + A20 / S00 * DxS00 + A11 / (2 * S00) * DyS00 - A10
        # Differential phase in x up to a multiplicative constant, where the constant is delta/k
        S01 = DyA02 + (1 / 2) * DxA11 + A02 / S00 * DyS00 + A11 / (2 * S00) * DxS00 - A01

        # D20 component of diffusion tensor up to a multiplicative constant, where the constant is delta**2
        S20 = A20 / S00
        # D11 component of diffusion tensor up to a multiplicative constant, where the constant is delta**2
        S11 = A11 / (2 * S00)
        # D02 component of diffusion tensor up to a multiplicative constant, where the constant is delta**2
        S02 = A02 / S00

        # Remove any potential numerical artefacts with calculated signals
        S00 = np.nan_to_num(S00, nan=0.0, posinf=0.0, neginf=0.0)
        S10 = np.nan_to_num(S10, nan=0.0, posinf=0.0, neginf=0.0)
        S01 = np.nan_to_num(S01, nan=0.0, posinf=0.0, neginf=0.0)
        S20 = np.nan_to_num(S20, nan=0.0, posinf=0.0, neginf=0.0)
        S11 = np.nan_to_num(S11, nan=0.0, posinf=0.0, neginf=0.0)
        S02 = np.nan_to_num(S02, nan=0.0, posinf=0.0, neginf=0.0)

    return {'S00': S00, 'S10': S10, 'S01': S01, 'S20': S20, 'S11': S11, 'S02': S02,
            'A00': A00, 'A10': A10, 'A01': A01, 'A20': A20, 'A11': A11, 'A02': A02}


def do_FP_UMPA(N_div, n_proc, i_sam, i_ref, rep, N_blur=0, savgol=True, polyorder=3, window_length=5):
    """
    :param N_div: An integer which corresponds to how subregions the image is to be divided into.
    :param n_proc: The number of processes to be used for parallel computation
    :param i_sam: A 3D numpy array of sample images where axis = 0 corresponds to different scan positions
    :param i_ref: A 3D numpy array of reference images where axis = 0 corresponds to different scan positions
    :param rep: The representation of the diffusion field to be used. Takes "tensor" or "scalar"
    :param N_blur: An integer which species the size of the blurring kernel to be applied to the parameters. The size of
     the kernel is (2 * N_blur + 1) * (2 * N_blur + 1)
    :param savgol: A boolean where True indicates the savgol are to be employed, otherwise central finite difference are
    to be used.
    :param polyorder: The polynomial order to be used for savgol filters
    :param window_length: The window length to be used for savgol filters
    :return: Retrieved a dictionary of retrieved signals and blurred parameters in the specified representation for diffusion
    """

    # Ensuring that the data is presented as a 3D numpy array
    Isam = np.stack(i_sam)
    Iref = np.stack(i_ref)
    
    # Storing the dimensions of the image in a variable
    Ish = Isam[0].shape
    
    if n_proc > Ish[0]:
        raise ValueError("n_proc should not be larger than no. of rows in the data")
    
    if rep.lower() == 'scalar':
        A00, A10, A01, A20 = subdivide_helper(N_div, n_proc, Isam, Iref, rep, savgol, polyorder, window_length)
        return calculate_signal_scalar_rep(A00, A10, A01, A20, N_blur, savgol, polyorder, window_length)
    elif rep.lower() == 'tensor':
        A00, A10, A01, A20, A11, A02 = subdivide_helper(N_div, n_proc, Isam, Iref, rep, savgol, polyorder, window_length)
        return calculate_signal_tensor_rep(A00, A10, A01, A20, A11, A02, N_blur, savgol, polyorder, window_length)
    else:
        return "Please select 'scalar' or 'tensor' representation for the diffusion field."


