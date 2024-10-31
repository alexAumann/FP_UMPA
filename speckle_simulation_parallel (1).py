"""
Wave optics speckle simulation

Author: Alex Aumann
Date: July 2024
"""

import numpy as np
import random as rand
import multiprocessing as mp


# TO DO: create simple unit tests
# document my code
# type hinting? 

def superimpose_matrix(small_mat, large_mat, pos_i, pos_j):
    """
    A function which superimposes a small matrix, smallMat, onto the large matrix, largeMat
    at position (row, col) = (posI, posJ). Only the parts of the small matrix get superimposed
    onto the large matrix.

    :param small_mat: A small matrix which to be superimposed on the large matrix. The
    small need an odd number of rows and columns
    :param large_mat: A large matrix which a small matrix will be superimposed on.
    :param pos_i: A non-negative integer which specifies which row in the large matrix
    the centre of the small matrix get superimposed on.
    :param pos_j: A non-negative integer which specifies which column in the large matrix
    the centre of the small matrix get superimposed on.
    :return:
    """
    res = np.copy(large_mat)
    smallDim = np.shape(small_mat)
    largeDim = np.shape(large_mat)
    sIndices = [0] * 4  # [i_min, i_max, j_min, j_max], these are the relevant indices for the small matrix
    lIndices = [0] * 4  # these are the relevant indices for the large matrix
    smallMidIndex = [np.floor(smallDim[0] / 2).astype(int), np.floor(smallDim[1] / 2).astype(int)]

    if pos_i < 0 or pos_i >= largeDim[0]:
        raise Exception("bad pos_i in superimpose_matrix")
    if pos_j < 0 or pos_j >= largeDim[1]:
        raise Exception("bad pos_j in superimpose_matrix")
    if smallDim[0] % 2 == 0 or smallDim[1] % 2 == 0:
        raise Exception("small matrix should have an odd shape")

    # Checking if the top of the small column is contained within the large column
    if pos_i - smallMidIndex[0] >= 0:
        sIndices[0] = 0
        lIndices[0] = pos_i - smallMidIndex[0]
    else:
        sIndices[0] = smallMidIndex[0] - pos_i
        lIndices[0] = 0

    # Checking if the bottom of the small column is contained within the large column
    if pos_i + smallMidIndex[0] <= largeDim[0] - 1:
        sIndices[1] = smallDim[0]
        lIndices[1] = pos_i + smallMidIndex[0] + 1
    else:
        sIndices[1] = largeDim[0] - pos_i + smallMidIndex[0]
        lIndices[1] = largeDim[0]

    # Checking if the left of the small row is contained within the large row
    if pos_j - smallMidIndex[1] >= 0:
        sIndices[2] = 0
        lIndices[2] = pos_j - smallMidIndex[1]
    else:
        sIndices[2] = smallMidIndex[1] - pos_j
        lIndices[2] = 0

    # Checking if the right of the small row is contained within the large row
    if pos_j + smallMidIndex[1] <= largeDim[1] - 1:
        sIndices[3] = smallDim[1]
        lIndices[3] = pos_j + smallMidIndex[1] + 1
    else:
        sIndices[3] = largeDim[1] - pos_j + smallMidIndex[1]
        lIndices[3] = largeDim[1]

    large_mat[lIndices[0]:lIndices[1], lIndices[2]:lIndices[3]] = large_mat[lIndices[0]:lIndices[1], lIndices[2]:lIndices[3]] \
                                                            + small_mat[sIndices[0]:sIndices[1], sIndices[2]:sIndices[3]]





def proj_thickness_of_sphere(r: int):
    """
    A function which finds the projected thickness of a half-sphere.

    :paramr r: A non-negative integer which specifies the radius of the sphere. This
    radius needs to be in units of pixels.
    :return: A 2D numpy array where each entry has the projected thickness of the half-sphere expressed
    in pixels. The centre/middle of the array corresponds to the coordinates (x, y) = (0, 0).
    """
    ran = range(-r, r + 1, 1)
    return np.array(
        [[np.sqrt(r ** 2 - i ** 2 - j ** 2) if r ** 2 - i ** 2 - j ** 2 > 0 else 0 for j in ran] for i in ran])




def place_spheres(diff_sh, min_r, max_r, stp_i, stp_j, pr):
    """
    Original Implementation place_spheres. Not designed to be run in parallel
    """
    dim = diff_sh
    res = np.zeros(diff_sh)
    temp = np.copy(res)
    for i in range(0, dim[0], stp_i):
        for j in range(0, dim[1], stp_j):
            if rand.random() > pr:
                res += superimpose_matrix(proj_thickness_of_sphere(rand.randint(min_r, max_r)), temp, i, j)
    return res


def place_spheres_2(diff_sh, min_r, max_r, stp_i, stp_j, start, out_q):
    """
    A function to be run in parallel which places the projected thickness of half-spheres at random on an empty 2D numpy array
    of shape diff_sh. The radius of the spheres also randomly vary where the maximum radius is max_r and the minimum radius is min_r.

    :param diff_sh: A tuple which corresponds to the shape of the 2D numpy array with
    the projected thickness of the half-spheres to be returned.
    :min_r: An non-negative integer which specifies the minimum radius of the half-spheres.
    :max_r: An non-negative integer which specifies the maximum radius of the half-spheres.
    :stp_i: An integer which specifies how frequently to place half spheres vertically (i.e
    At every pixel, at every second pixel...). 
    :stp_j: An integer which specifies how frequently to place the half spheres horizontally (i.e
    At every pixel, at every second pixel...).
    :out_q: A queue from the multiprocessing module in python which will store the results.
    :return: 

    TO DO: DELETE START, I CAN APPLY PSHIFT FROM UMPA INSTEAD.
    """
    res = np.zeros(diff_sh)
    for i in range(start, diff_sh[0], stp_i):
        for j in range(start, diff_sh[1], stp_j):
            superimpose_matrix(proj_thickness_of_sphere(rand.randint(min_r, max_r)), res, i, j)
    out_q.put(res)




# This a function used for constructing a diffuser in parallel by subdividing the diffuser into strips
def place_spheres_worker(diff_sh, min_r, max_r, stp_i, stp_j, pr, out_q):
    """
    Worker function to compute diffuser constituents in strips
    """
    dim = diff_sh
    res = np.zeros(diff_sh)
    temp = np.copy(res)
    for i in range(0, dim[0], stp_i):
        for j in range(0, dim[1], stp_j):
            if rand.random() > pr:
                res += superimpose_matrix(proj_thickness_of_sphere(rand.randint(min_r, max_r)), temp, i, j)
    print("done")
    out_q.put(res)


# This a function used for constructing a diffuser in parallel by subdividing the diffuser into strips
def mp_place_spheres(diff_sh, min_r, max_r, stp_i, stp_j, pr):
    """
    A function which computes a diffuser in parallel. This is acheived by computing stips of the
    diffuser in parallel. However, vertical streaks do appear in the result.
    """
    out_q = mp.Queue()
    results = []
    nproc = mp.cpu_count()
    proc = []
    new_diff_sh = [diff_sh[0], int(np.ceil(diff_sh[1]/nproc))]

    for _ in range(nproc):
        p = mp.Process(target=place_spheres_worker, args=(new_diff_sh, min_r, max_r, stp_i, stp_j, pr, out_q))
        proc.append(p)
        p.start()

    for _ in range(nproc):
        results.append(out_q.get())

    for p in proc:
        p.join()

    res = np.concatenate(tuple(results), axis=1)
    return res[:, :diff_sh[1]]



def create_diffuser(diff_sh, num_of_diff, min_r = 1, max_r = 3, stp_i = 1, stp_j = 1, pr = 0.90):
    """
    A function which computers a diffuser
    """
    res = np.zeros(diff_sh)
    for _ in range(num_of_diff):
        res +=  mp_place_spheres(diff_sh, min_r, max_r, stp_i, stp_j, pr)
    return res



def mp_create_diffusers(diff_sh, min_r = 1, max_r = 3, stp_i = 1, stp_j = 1, pr = 0.90):
    """
    A function which tries to compute the constituent diffusers in parallel.
    """
    out_q = mp.Queue()
    results = []
    nproc = 10
    proc = []


    for _ in range(nproc):
        p = mp.Process(target=place_spheres_worker, args=(diff_sh, min_r, max_r, stp_i, stp_j, pr, out_q))
        proc.append(p)
        p.start()

    for _ in range(nproc):
        results.append(out_q.get())

    for p in proc:
        p.join()

    return results



def create_kp2(image, dx):
    """
    a function which finds the the magnitude squared of the reciprocal space value of each pixel.
    The array isn't centered about |k|^2 = 0.

    :param image: A 2-dimensional numpy array which corresponds to the image.
    :param dx: The length of each pixel in real space.
    :return: A 2D array where each entry has a reciprocal space value.
    """
    imageSize = image.shape
    kp2 = 0
    for i in range(len(imageSize)):
        ki = np.square(2 * np.pi * np.fft.fftfreq(imageSize[i], d=dx))
        kp2 = np.add.outer(kp2, ki)
    return kp2


def fresnel_propagate(arr, k, z, pixel_size):
    """
    A function which performs Fresnel propagation on the arr.

    :param arr: A 2D numpy array which has the amplitude of a complex scalar wavefield.
    :param k: A number which corresponds to the wavenumber of the complex scalar wavefield.
    :param z: A number which corresponds to the distance in which the complex scalar wavefield
    will propagate over.
    :param pizel_size: The physical dimensions of each pixel in the 2D numpy array arr.
    :return: A 2D numpy array which has been propagated over a distance z with the Fresnel propagator
    """
    return np.exp(1j * k * z) * np.fft.ifft2(np.exp(-1j * z / (2 * k) * create_kp2(arr, pixel_size)) * np.fft.fft2(arr))


def freespace_propagate(arr, k, z, pixel_size):
    """
    A function which performs the angular spectrum method for freespace propagation.

    :param arr: A 2D numpy array which has the amplitude of a complex scalar wavefield.
    :param k: A number which corresponds to the wavenumber of the complex scalar wavefield.
    :param z: A number which corresponds to the distance in which the complex scalar wavefield
    will propagate over.
    :param pizel_size: The physical dimensions of each pixel in the 2D numpy array arr.
    :return: A 2D numpy array which has been propagated over a distance z within the angular spectrum formalism.
    """
    return np.fft.ifft2(np.exp(1j * z * np.sqrt(np.full(arr.shape, k ** 2) - create_kp2(arr, pixel_size))) * np.fft.fft2(arr))




def pshift(a, ctr):
    """
    TAKEN FROM MARIE'S ORIGINAL UMPA CODE
    
    Shift an array so that ctr becomes the origin.
    """
    sh  = np.array(a.shape)
    out = np.zeros_like(a)

    ctri = np.floor(ctr).astype(int)
    ctrx = np.empty((2, a.ndim))
    ctrx[1,:] = ctr - ctri     # second weight factor
    ctrx[0,:] = 1 - ctrx[1,:]  # first  weight factor

    # walk through all combinations of 0 and 1 on a length of a.ndim:
    #   0 is the shift with shift index floor(ctr[d]) for a dimension d
    #   1 the one for floor(ctr[d]) + 1
    comb_num = 2**a.ndim
    for comb_i in range(comb_num):
        comb = np.asarray(tuple(("{0:0" + str(a.ndim) + "b}").format(comb_i)), dtype=int)

        # add the weighted contribution for the shift corresponding to this combination
        cc = ctri + comb
        out += np.roll( np.roll(a, -cc[1], axis=1), -cc[0], axis=0) * ctrx[comb,range(a.ndim)].prod()

    return out

