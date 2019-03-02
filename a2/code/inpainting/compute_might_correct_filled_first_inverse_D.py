## CSC320 Winter 2019 
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    
    # Replace this dummy value with your own code
    C = 1    
    #########################################
    conf, valid = copyutils.getWindow(confidenceImage, psiHatP._coords, psiHatP._w)
    C = np.sum(conf) / np.sum(valid)
    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).
#
    
def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    
    # Replace these dummy values with your own code
    Dy = 1
    Dx = 0
    valid = None

    #Image_gray = cv.cvtColor(psiHatP.value(), cv.COLOR_BGR2GRAY)
    #Gradient from the whole gray image, and cut from it.

    #get coord of p
    center = (psiHatP.row(), psiHatP.col())
    w = psiHatP.radius()
    if center[0] == 145 and center[1] ==  77:
        print 1
    if center[0] == 141 and center[1] ==  120:
        print 1
    selected_block, valid = copyutils.getWindow(inpaintedImage, center, w + 2)
    block_gray = cv.cvtColor(selected_block, cv.COLOR_BGR2GRAY)
    
    #OK, this time I try to use filled first
    filled_patch, valid = copyutils.getWindow(filledImage, center, w + 2)
    block_gray = filled_patch * 1.0 / 255 * block_gray
    
    block_scharrx = cv.Sobel(block_gray, cv.CV_64F, 1, 0, ksize=5)
    block_scharry = cv.Sobel(block_gray, cv.CV_64F, 0, 1, ksize=5)
    #For here, I added one outer range to ensure the accuracy for the outtest gradient.
    #cut it back to correct size
    cut_x_available, _ = copyutils.getWindow(block_scharrx, (w + 2, w + 2), w)
    cut_y_available, _ = copyutils.getWindow(block_scharry, (w + 2, w + 2), w)

    #cut!
    #cut_x_patch_gray, valid = copyutils.getWindow(Image_scharrx, center, w)
    #cut_y_patch_gray, valid = copyutils.getWindow(Image_scharry, center, w)
    #Filled?
    #filledImage = filledImage
    #filled_patch, valid = copyutils.getWindow(filledImage, center, w)
    #cut_x_available = cut_x_patch_gray * filled_patch / 255
    #cut_y_available = cut_y_patch_gray * filled_patch / 255

    squared_sum = np.multiply(cut_x_available, cut_x_available) \
                  + np.multiply(cut_y_available, cut_y_available)
    idx = np.unravel_index(squared_sum.argmax(), squared_sum.shape)
    Dx = cut_y_available[idx[0], idx[1]]
    Dy = cut_x_available[idx[0], idx[1]]#TODO:Should I keep here? Or should I switch?
    #########################################
    '''

    # get coords and w
    coords = (psiHatP.row(), psiHatP.col())
    w = psiHatP.radius()

    # get the patch in grascale with only filled pixels
    imgGray = cv.cvtColor(inpaintedImage, cv.COLOR_BGR2GRAY)
    validGray = imgGray * (filledImage / 255)
    patchGray = copyutils.getWindow(validGray, coords, w)[0]

    # compute gradients use sobel (since we use 3 by 3 kernel, thus set ksize=-1
    # in order to use 3 by 3 scharr kernel)
    gradientsX = cv.Sobel(patchGray, cv.CV_64F, 1, 0, ksize=-1)
    gradientsY = cv.Sobel(patchGray, cv.CV_64F, 0, 1, ksize=-1)

    # compute the magnitude for each pair of gradients for each pixel
    magnitudes = np.add(gradientsX ** 2, gradientsY ** 2)

    # find the maximum gradients and set the values to Dy and Dx
    ind = np.unravel_index(np.argmax(magnitudes, axis=None), magnitudes.shape)
    Dx = gradientsX[ind]
    Dy = gradientsY[ind]
    '''
    return Dy, Dx
    
#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#

def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    '''
    Nx, Ny = None, None
    pcord = (psiHatP.row(), psiHatP.col())
    w = psiHatP.radius()
    imagei, imageb = copyutils.getWindow(filledImage, pcord, w)
    sx = cv.Sobel(imagei, cv.CV_64F, 1, 0)
    sy = cv.Sobel(imagei, cv.CV_64F, 0, 1)
    x, y = sx[w, w], sy[w, w]
    i = np.sqrt(x ** 2 + y ** 2)
    if i != 0:
        Nx = x / i
        Ny = -y / i

    #########################################


    return Ny, Nx
    '''
    # Replace these dummy values with your own code
    Ny = None
    Nx = None
    #########################################
    #The fill front only shows a line. It may points to a different direction.
    
    center = (psiHatP.row(), psiHatP.col())
    if center[0] == 145 and center[1] ==  77:
        print 1
    if center[0] == 141 and center[1] ==  120:
        print 1
    w = psiHatP.radius()
    
    cut_fillFront, valid = copyutils.getWindow(fillFront, center, w)
    #Remove the patch that normal has no neighbour!
    #for i in range(-1, 2):
    #    for j in range(-1, 2):
            #if cut_fillFront[w + i, w + j] != 0:
                #OK, neighbour exists.
                #Treat out of bound values as filled. So there might be a normal points from non-exists to exists
    filledImage, valid1 = copyutils.getWindow(filledImage, center, w + 2)#TODO: test if there exists something here: when 0 or 255
    filledFrontImage, valid2 = copyutils.getWindow(fillFront, center, w + 2)#TODO: test if there exists something here: when 0 or 255
    Unfilled = np.logical_not(filledImage * 1.0 / 255) / 1.0
    filledImage_scharrx = cv.Sobel(Unfilled, cv.CV_64F, 1, 0, ksize=5)
    filledImage_scharry = cv.Sobel(Unfilled, cv.CV_64F, 0, 1, ksize=5)
    cut_x_patch, valid = copyutils.getWindow(filledImage_scharrx, (w+2, w+2), w)
    cut_y_patch, valid = copyutils.getWindow(filledImage_scharry, (w+2, w+2), w)

    if (np.count_nonzero(filledFrontImage[2:-2,2:-2]) <=1):
        return None, None
    magnitude = np.sqrt(cut_x_patch[w, w] ** 2 + cut_y_patch[w, w] ** 2)

    if magnitude != 0:#TODO: if this order and posneg correct?
        Nx = cut_y_patch[w,w] / magnitude
        Ny = -cut_x_patch[w,w] / magnitude
    return Ny, Nx
    
    '''

    # get coords and w and patch in source image
    coords = (psiHatP.row(), psiHatP.col())
    w = psiHatP.radius()

    # get masks in patch size
    # patchFilled = copyutils.getWindow(filledImage, coords, w)[0] / float(255)
    patchFront = copyutils.getWindow(fillFront, coords, w)[0]
    # patchUnfilled = np.logical_not(patchFilled) / float(1)

    # if the fill front consists of exactly one pixel, the fill front is degenerate
    # and has no well-defined normal
    if np.count_nonzero(patchFront) == 1:
        Nx = None
        Ny = None
        return Ny, Nx

    # compute gradients at patch center
    centerGX = cv.Sobel(patchFront, cv.CV_64F, 1, 0, ksize=5)[w, w]
    centerGY = cv.Sobel(patchFront, cv.CV_64F, 0, 1, ksize=5)[w, w]

    # compute magnitude for tangent at patch center
    magnitude = np.sqrt(np.add(centerGX ** 2, centerGY ** 2))

    # set Ny and Nx
    Nx = - centerGY
    Ny = centerGX
    return Ny, Nx
    '''
