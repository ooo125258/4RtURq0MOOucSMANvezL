# CSC320 Winter 2019
# Assignment 4
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
# import the heapq package
from heapq import heappush, heappushpop, nlargest, nsmallest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()
import os
from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')

def getFromNPArray(i, j, x):
    # Get [i,j] if existed.
    # Return [] if out of bound by 1
    if i == -1 or j == -1 or i == x.shape[0] or j == x.shape[1]:
        return []
    else:
        return [i, j]
# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
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
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
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
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function


def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):
    print("start propagation_and_random_search_k")
    new_f, best_D = NNF_heap_to_NNF_matrix(f_heap)
    #################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES   ###
    ###  THEN START MODIFYING IT AFTER YOU'VE     ###
    ###  IMPLEMENTED THE 2 HELPER FUNCTIONS BELOW ###
    #################################################
    iNumRows, iNumCols = source_patches.shape[0:2]

    # alphai matrix, used for random process:
    wais = np.array([])
    i_max = 0
    if random_enabled:
        # as w * alpha^i >= 1, record u, then i <= -logw/logalpha, record u
        i_max = int(np.ceil(- np.log(w) / np.log(alpha)))
        alphai_s = np.logspace(0, i_max, num=i_max, base=alpha, endpoint=False)
        wais = w * alphai_s
        wais = np.repeat(wais, 2).reshape(-1, 2)

    # Loop through whole patches
    i = 0
    j = 0
    K = len(f_heap[0][0])
    for iteri in range(iNumRows):
        print(iteri)
        for iterj in range(iNumCols):
        
            if not odd_iteration:
                i = iNumRows - iteri - 1
                j = iNumCols - iterj - 1
            else:
                i = iteri
                j = iterj
        
            curr_pos = [i, j]
            source_patch = source_patches[i][j]
        
            skip_propagation = (i == 0 and j == 0) or (i == iNumRows - 1 and j == iNumCols - 1)
            # skip_propagation = False
            # Do propagation:
            
            if not skip_propagation and propagation_enabled:
                
                #curr_fxy = best_D[k][i, j]
                # The position of 3 source points(or 2 or 1)
                source_pts = np.array([], dtype=int)
                if odd_iteration:
                    # When odd, we propagate information up and left
                    # coherent bot and right
                    source_pts = np.append(source_pts, getFromNPArray(i - 1, j, new_f[0]))
                    source_pts = np.append(source_pts, getFromNPArray(i, j - 1, new_f[0]))
                else:
                    # When even, we propagate information down and right
                    # coherent up and left
                    source_pts = np.append(source_pts, getFromNPArray(i + 1, j, new_f[0]))
                    source_pts = np.append(source_pts, getFromNPArray(i, j + 1, new_f[0]))
                source_pts = source_pts.reshape(source_pts.size / 2, 2).astype(int)
        
                f_pts = new_f[:,source_pts[:, 0], source_pts[:, 1]] #TODO: it must be k shape
                # f_pts is 2 * 2 ... might be 1*2
        
                # As f(a) = b-a, b = f(a) + a
                target_pts = f_pts + curr_pos

                #Check if targets in bound, implement later
                # TODO: Is this correct? Should we keep them? or abandon them?
                target_pts[..., 0] = np.clip(target_pts[..., 0], 0, iNumRows - 1)
                target_pts[..., 1] = np.clip(target_pts[..., 1], 0, iNumCols - 1)
                target_pts = target_pts.astype(int)

                
                dists = np.zeros((0))
            
                target_patch = target_patches[target_pts[..., 0], target_pts[..., 1]]
            
                # Compute D now between the source patch and the target patch
                square_diff = np.square(source_patch - target_patch)
                #Reweighted by nan number
                #nan_pos = np.isnan(square_diff)
                #nan_count = np.sum(np.sum(nan_pos, -1), -1)
                #count_for_each_patch = square_diff.shape[-1] * square_diff.shape[-2]
                #ratio = nan_count * 1.0 / count_for_each_patch
                
                square_diff[np.isnan((square_diff))] = 65025  # 255^2
                #TODO: weight by existed pixel
                dists = np.sum(np.nansum(square_diff, axis=-1),
                               axis=-1) #/ ratio  #TODO: zero or max or avg?
                actual_offset = target_pts - curr_pos
                for k in range(k):
                    for tar_idx in range(target_pts.shape[1]):# TODO: confirm all tar_idx items have the same shape!
                        #tupled_src_coord = tuple(f_pts[k][tar_idx])
                        tupled_src_coord = tuple(actual_offset[k][tar_idx])
                        if tupled_src_coord in f_coord_dictionary[i][j]:
                            continue
                        #if -nlargest(1, f_heap[i][j])[0][0] < dists[tar_idx]: #if truly the new result is smaller
                        tup = (-dists[k][tar_idx], _tiebreaker.next(), tupled_src_coord) #It's actually with number of target pts
                        
                        #if tup[0] == f_heap[i][j][0][0]:
                        #    continue
                        #popped = heappushpop(f_heap[i][j], tup)
                        if tup == heappushpop(f_heap[i][j], tup):
                            continue
                            
                        #else:
                        #heappushpop(f_heap[i][j], tup)#heappushpop?
                        f_coord_dictionary[i][j][tupled_src_coord] = dists[k][tar_idx]
                        best_D[k][i][j] = dists[k][tar_idx]
                        new_f[k][i][j] = tupled_src_coord#f_pts[k][tar_idx]
                        #curr_fxy = dists[tar_idx]
                            
                    # 2 * target_patch_shape(2d)
                    #dists = np.append(dists, curr_fxy)
                    #min_dist_idx = np.argmin(dists)
            
                    # If min_dist_idx doesn't point the the original one, then move best_D
                    #if min_dist_idx != (dists.shape[0] - 1):
                        

            # Random Process!
            if random_enabled:
                
                # print("i:{} j:{}".format(i,j))
                #curr_fxy = best_D[:][i, j]
                # We examine patches for i = 0, 1, 2, ... until the current search radius wa i is below 1 pixel.
                uis = np.array([], dtype=int)  # As this is used as INDEX!
                R = np.random.uniform(-1, 1, (K, i_max, 2))
                # R = 2*(np.random.rand(2 * i_max) - 0.5)
                #R = R.reshape(-1, 2)
                wr = np.multiply(wais, R) #4 * 10 * 2
                uis_unshaped = new_f[:,i, j,:] + np.moveaxis(wr, 1, 0) #TODO: check dim
                uis = np.moveaxis(uis_unshaped, 0, 1).astype(int)
                
                target_pts = uis + curr_pos
                # TODO: Is this correct? Should we keep them? or abandon them?
                target_pts[..., 0] = np.clip(target_pts[..., 0], 0, iNumRows - 1)
                target_pts[..., 1] = np.clip(target_pts[..., 1], 0, iNumCols - 1)
                target_pts = target_pts.astype(int)
            
                # similar with propogate:
                dists = np.zeros((0))
                # for target_pt_idx in target_pts:
                target_patch = target_patches[target_pts[..., 0], target_pts[..., 1]]
            
                # Compute D now between the source patch and the target patch
                square_diff = np.square(source_patch - target_patch)
                # Reweighted by nan number
                #nan_pos = np.isnan(square_diff)
                #nan_count = np.sum(np.sum(nan_pos, -1), -1)
                #count_for_each_patch = square_diff.shape[-1] * square_diff.shape[-2]
                #ratio = nan_count * 1.0 / count_for_each_patch

                square_diff[np.isnan((square_diff))] = 65025  # 255^2
                # TODO: weight by existed pixel
                dists = np.sum(np.nansum(square_diff, axis=-1),
                               axis=-1)# / (1 - ratio)
                actual_offset = target_pts - curr_pos
                for k in range(K):
                    for tar_idx in range(target_pts.shape[1]):# TODO: confirm all tar_idx items have the same shape!
                        # TODO: for here, I just put if it's not exists. Should I also replace with the smallest one?
                        #tupled_src_coord = tuple(uis[k][tar_idx]) # TODO: it should represent the offset. Should I use the clipped offset? or the origin offset?
                        tupled_src_coord = tuple(actual_offset[k][tar_idx])
                        if tupled_src_coord in f_coord_dictionary[i][j]: #TODO: check if it's this in f_coord
                            continue
                        #else:
                        tup = (-dists[k][tar_idx], _tiebreaker.next(),
                               tupled_src_coord)  # It's actually with number of target pts
                        #if tup[0] == f_heap[i][j][0][0]:#: #if truly the new result is smaller
                        #    continue
                        #popped = heappushpop(f_heap[i][j], tup)
                        if tup == heappushpop(f_heap[i][j], tup):
                            continue
                        f_coord_dictionary[i][j][tupled_src_coord] = dists[k][tar_idx]
                        best_D[k][i][j] = dists[k][tar_idx]
                        new_f[k][i][j] = tupled_src_coord#target_pts[k][tar_idx] - curr_pos#uis[tar_idx]
                        #curr_fxy = dists[k][tar_idx]

                # 2 * target_patch_shape(2d)
                #dists = np.append(dists, curr_fxy)
                #min_dist_idx = np.argmin(dists)
            
                #if min_dist_idx != len(dists) - 1:
            #############################################

    return global_vars



# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders

def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    f_heap = None
    f_coord_dictionary = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    #holds k NNFs
    k = f_k.shape[0]
    f_coord_dictionary = []
    f_heap = []
    #2d array of dicts...
    
    for i in range(source_patches.shape[0]):
        f_heap.append([])
        f_coord_dictionary.append([])
        for j in range(source_patches.shape[1]):
            dict = {}
            f_heap[i].append([])
            f_coord_dictionary[i].append({})
            target_pts = f_k[:,i,j,:]
            source_patch = source_patches[i][j]
            target_patch = target_patches[target_pts[..., 0], target_pts[..., 1]]

            # Compute D now between the source patch and the target patch
            #square_diff = np.square(source_patch - target_patch)
            # Reweighted by nan number
            #nan_pos = np.isnan(square_diff)
            #nan_count = np.sum(np.sum(nan_pos, -1), -1)
            #count_for_each_patch = square_diff.shape[-1] * square_diff.shape[-2]
            #ratio = nan_count * 1.0 / count_for_each_patch

            # square_diff[np.isnan((square_diff))] = 65025  # 255^2
            # TODO: weight by existed pixel
            #dists = np.sum(np.nansum(square_diff, axis=-1),
            #               axis=-1) / (1 - ratio)
            square_diff = np.square(source_patch - target_patch)
            square_diff[np.isnan((square_diff))] = 65025  # 255^2
            dists = np.sum(np.sum(square_diff, axis=-1),
                           axis=-1)  # np.append(dists, np.sum(square_diff)) # Notice: negative here!

            for each_idx in range(k):
                # (priority, counter, displacement)
                displacement = f_k[each_idx][i][j]
                #Push to Heap!

                priority = -dists[each_idx]
                counter = _tiebreaker.next()
                heappush(f_heap[i][j], (priority, counter, displacement))
                f_coord_dictionary[i][j][tuple(displacement)] = dists[each_idx]

            
    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):
    #Impossible to make it vectorized
    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    k = len(f_heap[0][0])
    f_k = np.zeros((len(f_heap[0][0]),len(f_heap), len(f_heap[0]),  2), dtype=int)
    D_k = np.zeros((len(f_heap[0][0]), len(f_heap), len(f_heap[0])))

    for i in range(len(f_heap)):
        for j in range(len(f_heap[0])):
            largest = nlargest(k, f_heap[i][j])
            for k_idx in range(k):
                f_k[k_idx,i,j,:] = largest[k_idx][2][:]
                D_k[k_idx,i,j] = -largest[k_idx][0]

    f_k = f_k.astype(int)
    #############################################
    
    return f_k, D_k


def nlm(target, f_heap, h):


    # this is a dummy statement to return the image given as input
    #denoised = target

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    f_k, D_k = NNF_heap_to_NNF_matrix(f_heap)
    e_item = np.exp(- np.sqrt(D_k) / h ** 2) #Assume D is a distance square here, or square root
    z_item = np.sum(e_item, axis=0) #sigma j
    w_item = np.multiply(e_item, np.reciprocal(z_item)) #TODO: check the dimension
    denoised_k = np.einsum("ijc,kij->kijc", target, w_item)
    #denoised_k = np.multiply(target, w_item)
    denoised = np.sum(denoised_k, axis=0)

    #############################################

    return denoised




#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################



#############################################



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
    print "shape target: " + str(target.shape)
    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################
    iNumRows, iNumCols = target.shape[0:2]
    coord = make_coordinates_matrix([iNumRows, iNumCols])

    # (x, y) + f(x, y)
    target_idxs = coord + f
    # TODO: check indexes here.
    rec_source = target[target_idxs[:, :, 0], target_idxs[:, :, 1]]

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
