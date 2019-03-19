# CSC320 Winter 2018
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure


def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                ):
    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    # find the dimension of the source image
    x, y= source_patches.shape[1], source_patches.shape[0]

    # try to find the max index that reduce the search radius below 1 pixel.
    max_index = 0
    while (w * (alpha ** max_index) >= 1):
        max_index += 1

    coefficient = np.zeros((max_index + 2, 2)).astype(int)

    for i in range(1, max_index+2):
        coefficient[i, :] = np.array([w * (alpha ** i), w * (alpha ** i)])
    # do the iteration
    # if the iteration number is odd
    if odd_iteration:
        for y_pos in range(0, y):
            for x_pos in range(0, x):
                if not propagation_enabled:
                    index_list = None
                    offset_list = None
                    if y_pos == 0:
                        if x_pos == 0:
                            break
                        elif x_pos > 0:
                            index_list = np.array([[y_pos, x_pos], [y_pos, x_pos - 1]])
                            offset_list = np.array([new_f[y_pos, x_pos], new_f[y_pos, x_pos - 1]])
                    else:
                        if x_pos == 0:
                            index_list = np.array([[y_pos, x_pos], [y_pos -1 , x_pos]])
                            offset_list = np.array([new_f[y_pos, x_pos], new_f[y_pos - 1, x_pos]])
                        elif x_pos > 0:
                            index_list = np.array([[y_pos, x_pos], [y_pos - 1, x_pos], [y_pos, x_pos - 1]])
                            offset_list = np.array([new_f[y_pos, x_pos], new_f[y_pos - 1, x_pos], new_f[y_pos, x_pos - 1]])
                    if index_list is not None:
                        targetImageIndex = index_list + offset_list
                        target_idxy = np.clip(targetImageIndex[:, 0], 0, y - 1)
                        target_idxx = np.clip(targetImageIndex[:, 1], 0, x - 1)
                        # a list of candidate patches of the source image
                        source_list = np.nan_to_num(source_patches[index_list[:, 0], index_list[:, 1]])
                        # a list of candidate patches of the target image
                        target_list = np.nan_to_num(target_patches[target_idxy, target_idxx])
                        distance_sqaure = np.sum(np.abs(target_list - source_list)*np.abs(target_list - source_list), axis=(1,2))
                        distance_list = np.sqrt(distance_sqaure)
                        # find the index of min arg
                        min_index = np.argmin(distance_list)
                        new_f[y_pos, x_pos] = offset_list[min_index]


                if not random_enabled:
                    index_list = np.array([[y_pos, x_pos], ] * (max_index+2))
                    Ri = np.random.uniform(-1, 1, (max_index+2, 2))
                    offset_list = np.array([new_f[y_pos, x_pos], ] * (max_index+2))
                    Ui = offset_list + coefficient * Ri
                    targetImageIndex = index_list + Ui
                    #print('yyyyyy', y)
                    #print("before cliping ", targetImageIndex[:, 0])
                    target_idxy = np.clip(targetImageIndex[:, 0], 0, y - 1).astype(int)
                    #print("after cliping ", target_idxy)
                    target_idxx = np.clip(targetImageIndex[:, 1], 0, x - 1).astype(int)
                    # a list of candidate patches of the target image
                    target_list = np.nan_to_num(target_patches[target_idxy, target_idxx])
                    # a list of candidate patches of the source image
                    source_list = np.nan_to_num(source_patches[index_list[:, 0], index_list[:, 1]])

                    #print('the size of the patch is ', target_patches.shape)
                    #print('the size of the source is ', source_patches.shape)
                    distance_sqaure = np.sum(np.abs(target_list - source_list) * np.abs(target_list - source_list),
                                         axis=(1,2))
                    distance_list = np.sqrt(distance_sqaure)

                    min_index = np.argmin(distance_list)
                    new_off_y = target_idxy - index_list[:, 0]
                    new_off_x = target_idxx - index_list[:, 1]
                    new_f[y_pos, x_pos] = np.array([new_off_y[min_index], new_off_x[min_index]])

    else:
        for y_pos in range(y - 1, -1, -1):
            for x_pos in range(x - 1, -1, -1):
                if not propagation_enabled:
                    index_list = None
                    offset_list = None
                    if y_pos == y - 1:
                        if x_pos == x - 1:
                            pass
                        elif x_pos < x - 1:
                            index_list = np.array([[y_pos, x_pos],[y_pos, x_pos + 1]])
                            offset_list = np.array([new_f[y_pos, x_pos], new_f[y_pos, x_pos + 1]])
                    elif y_pos < y -1:
                        if x_pos == x - 1:
                            index_list = np.array([[y_pos, x_pos], [y_pos + 1, x_pos]])
                            offset_list = np.array([new_f[y_pos, x_pos], new_f[y_pos + 1, x_pos]])
                        elif x_pos < x - 1:
                            index_list = np.array([[y_pos, x_pos], [y_pos + 1, x_pos], [y_pos, x_pos + 1]])
                            offset_list = np.array(
                                [new_f[y_pos, x_pos], new_f[y_pos + 1, x_pos], new_f[y_pos, x_pos + 1]])
                    if index_list is not None:
                        targetImageIndex = index_list + offset_list
                        target_idxy = np.clip(targetImageIndex[:, 0], 0, y - 1)
                        target_idxx = np.clip(targetImageIndex[:, 1], 0, x - 1)
                        # a list of candidate patches of the source image
                        source_list = np.nan_to_num(source_patches[index_list[:, 0], index_list[:, 1]])
                        # a list of candidate patches of the target image
                        target_list = np.nan_to_num(target_patches[target_idxy, target_idxx])
                        # distance_list = np.sum(np.linalg.norm(target_list - source_list, axis=2), axis=1)
                        distance_sqaure = np.sum(np.abs(target_list - source_list) * np.abs(target_list - source_list),
                                                 axis=(1,2))
                        distance_list = np.sqrt(distance_sqaure)

                        # find the index of min arg
                        min_index = np.argmin(distance_list)
                        new_f[y_pos, x_pos] = offset_list[min_index]

                if not random_enabled:
                    index_list = np.array([[y_pos, x_pos], ] * (max_index + 2))
                    Ri = np.random.uniform(-1, 1, (max_index + 2, 2))
                    offset_list = np.array([new_f[y_pos, x_pos], ] * (max_index + 2))
                    Ui = offset_list + coefficient * Ri
                    targetImageIndex = index_list + Ui
                    target_idxy = np.clip(targetImageIndex[:, 0], 0, y - 1).astype(int)
                    target_idxx = np.clip(targetImageIndex[:, 1], 0, x - 1).astype(int)
                    # a list of candidate patches of the source image
                    source_list = np.nan_to_num(source_patches[index_list[:, 0], index_list[:, 1]])
                    # a list of candidate patches of the target image
                    target_list = np.nan_to_num(target_patches[target_idxy, target_idxx])
                    distance_sqaure = np.sum(np.abs(target_list - source_list) * np.abs(target_list - source_list),
                                             axis=(1, 2))
                    distance_list = np.sqrt(distance_sqaure)

                    min_index = np.argmin(distance_list)
                    new_off_y = target_idxy - index_list[:, 0]
                    new_off_x = target_idxx - index_list[:, 1]
                    new_f[y_pos, x_pos] = np.array([new_off_y[min_index], new_off_x[min_index]])
    #############################################

    return new_f, best_D, global_vars


# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    target_i = make_coordinates_matrix(target.shape) + f
    x = np.clip(target_i[:, :, 0], 0, target.shape[0] - 1)
    y = np.clip(target_i[:, :, 1], 0, target.shape[1] - 1)
    rec_source = target[x, y]
    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))