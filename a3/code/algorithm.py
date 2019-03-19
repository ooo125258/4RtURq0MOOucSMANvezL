# CSC320 Winter 2019
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
    print("lets' print f!")
    print(f[0, 0])
    #odd_iteration = False
    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    iNumRows, iNumCols = source_patches.shape[0:2]
    if best_D is None:
        # Create best_D, if it is None.
        coord = make_coordinates_matrix([iNumRows, iNumCols])
        
        target_idxs = new_f + coord
        #TODO: check indexes here.
        target_patches_reordered = target_patches[target_idxs[...,0], target_idxs[...,1]]
        square_diff = np.square(source_patches - target_patches_reordered)
        square_diff[np.isnan(square_diff)] = 65025
        #Best d is still 2 dimension, so I cannot add them together.
        best_D = np.sum(np.sum(square_diff, axis=3), axis=2)
    
    #alphai matrix, used for random process:
    wais = np.array([])
    i_max = 0
    if random_enabled:
        #as w * alpha^i >= 1, record u, then i <= -logw/logalpha, record u
        i_max = int(np.ceil(- np.log(w) / np.log(alpha)))
        alphai_s = np.logspace(0, i_max, num=i_max, base=alpha, endpoint=False)
        wais = w * alphai_s
        wais = np.repeat(wais, 2).reshape(-1, 2)
    
    # Loop through whole patches
    i = 0
    j = 0
    for iteri in range(iNumRows):
        for iterj in range(iNumCols):
            
            if not odd_iteration:
                i = iNumRows - iteri - 1
                j = iNumCols - iterj - 1
            else:
                i = iteri
                j = iterj
            
            curr_pos = [i,j]
            source_patch = source_patches[i][j]
            
            #skip_propagation = (i == 0 and j == 0) or (i == iNumRows - 1 and  j == iNumCols - 1)
            skip_propagation = False
            # Do propagation:
            if not skip_propagation and propagation_enabled:
                curr_fxy = best_D[i, j]
                # The position of 3 source points(or 2 or 1)
                source_pts = np.array([], dtype=int)
                if odd_iteration:
                    # When odd, we propagate information down and left
                    # coherent bot and right
                    source_pts = np.append(source_pts, getFromNPArray(i - 1, j, best_D))
                    source_pts = np.append(source_pts, getFromNPArray(i, j - 1, best_D))
                else:
                    # When even, we propagate information up and right
                    # coherent up and left
                    source_pts = np.append(source_pts, getFromNPArray(i + 1, j, best_D))
                    source_pts = np.append(source_pts, getFromNPArray(i, j + 1, best_D))
                source_pts = source_pts.reshape(source_pts.size / 2, 2).astype(int)
                
                f_pts = new_f[source_pts[:, 0], source_pts[:, 1]]
                # f_pts is 2 * 2 ... might be 1*2
                
                # As f(a) = b-a, b = f(a) + a
                target_pts = f_pts + curr_pos
                
                dists = np.zeros((0))
                #for target_pt_idx in target_pts:#TODO: Change to matrix form later.
                
                target_patch = target_patches[target_pts[..., 0], target_pts[..., 1]]
                
                # Compute D now between the source patch and the target patch
                square_diff = np.square(source_patch - target_patch)
                square_diff[np.isnan((square_diff))] = 65025  # 255^2
                dists = np.sum(np.sum(square_diff, axis=-1), axis=-1)#np.append(dists, np.sum(square_diff)) #TODO:Should I treat nan as zero? or max?
                
                # 2 * target_patch_shape(2d)
                dists = np.append(dists, curr_fxy)
                min_dist_idx = np.argmin(dists)
                
                # If min_dist_idx doesn't point the the original one, then move best_D
                if min_dist_idx != (dists.shape[0] - 1):
                    best_D[i][j] = dists[min_dist_idx]
                    new_f[i][j] = f_pts[min_dist_idx]
                    curr_fxy = dists[min_dist_idx]
            
            #Random Process!
            if random_enabled:
                if i == 0 and j == 0:
                    print(1)
                #print("i:{} j:{}".format(i,j))
                curr_fxy = best_D[i, j]
                # We examine patches for i = 0, 1, 2, ... until the current search radius wa i is below 1 pixel. 
                alphai = alpha
                uis = np.array([], dtype=int)#As this is used as INDEX!
                #R = np.random.uniform(-1,1,2 * i_max).reshape(-1,2)
                R = 2*(np.random.rand(2 * i_max) - 0.5)
                R = R.reshape(-1, 2)
                uis = new_f[i,j] + np.multiply(wais, R)
                '''
                while w * alphai >= 1:
                    #Ri is a uniform random in [ 1, 1] * [ 1, 1]
                    Ri = np.random.uniform(-1,1,2)
                    #ui = v0 + w*alpha^i Ri
                    alphai *= alpha
                    ui = new_f[i,j] + w * alphai * Ri
                    uis = np.append(uis, ui)
                uis = uis.reshape(len(uis) / 2, 2)
                '''
                target_pts = uis + curr_pos
                # TODO: Is this correct? Should we keep them? or abandon them?
                target_pts[:, 0] = np.clip(target_pts[:, 0], 0, iNumRows - 1)
                target_pts[:, 1] = np.clip(target_pts[:, 1], 0, iNumCols - 1)
                target_pts = target_pts.astype(int)
                

                #similar with propogate:
                dists = np.zeros((0))
                #for target_pt_idx in target_pts:
                target_patch = target_patches[target_pts[..., 0], target_pts[..., 1]]

                # Compute D now between the source patch and the target patch
                square_diff = np.square(source_patch - target_patch)
                square_diff[np.isnan((square_diff))] = 65025  # 255^2
                dists = np.sum(np.sum(square_diff, axis=-1), axis=-1)
                # 2 * target_patch_shape(2d)
                dists = np.append(dists, curr_fxy)
                min_dist_idx = np.argmin(dists)
                
                if min_dist_idx != len(dists) - 1:
                    target_pt = target_pts[min_dist_idx] #This is b
                    #f(a) = b - a
                    new_f[i][j] = target_pt - curr_pos #
                    best_D[i][j] = dists[min_dist_idx]
    #############################################
                
                
                
    return new_f, best_D, global_vars


def getFromNPArray(i, j, x):
    # Get [i,j] if existed.
    # Return [] if out of bound by 1
    if i == -1 or j == -1 or i == x.shape[0] or j == x.shape[1]:
        return []
    else:
        return [i, j]


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
    
    iNumRows, iNumCols = target.shape[0:2]
    coord = make_coordinates_matrix([iNumRows, iNumCols])
    
    #(x, y) + f(x, y)
    target_idxs = coord + f
    # TODO: check indexes here.
    rec_source = target[target_idxs[:,:, 0], target_idxs[:,:, 1]]
    
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
