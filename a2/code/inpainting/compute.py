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
    #Gradient from the whole gray image, and cut from it.
    Image_gray = cv.cvtColor(inpaintedImage, cv.COLOR_BGR2GRAY)
    Image_scharrx = cv.Scharr(Image_gray, cv.CV_32F, 1, 0)
    Image_scharry = cv.Scharr(Image_gray, cv.CV_32F, 0, 1)

    #get coord of p
    center = [psiHatP.row(), psiHatP.col()]
    w = psiHatP.radius()

    #cut!
    cut_x_patch_gray, valid = copyutils.getWindow(Image_scharrx, center, w)
    cut_y_patch_gray, valid = copyutils.getWindow(Image_scharry, center, w)
    #Filled?
    filled_patch, valid = copyutils.getWindow(filledImage, center, w) / 255
    cut_x_available = cut_x_patch_gray * filled_patch
    cut_y_available = cut_y_patch_gray * filled_patch

    squared_sum = np.multiply(cut_x_available, cut_x_available) \
                  + np.multiply(cut_y_available, cut_y_available)
    idx = np.unravel_index(squared_sum.argmax(), squared_sum.shape)
    Dx = cut_x_patch_gray[idx]
    Dy = cut_y_patch_gray[idx]
    #########################################
    
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
    
    # Replace these dummy values with your own code
    Ny = 0
    Nx = 1    
    #########################################
    #The fill front only shows a line. It may points to a different direction.
    center = [psiHatP.row(), psiHatP.col()]
    w = psiHatP.radius()
    
    filledImage_scharrx = cv.Scharr(filledImage, cv.CV_32F, 1, 0)
    filledImage_scharry = cv.Scharr(filledImage, cv.CV_32F, 0, 1)
    cut_x_patch, valid = copyutils.getWindow(filledImage_scharrx, center, w)
    cut_y_patch, valid = copyutils.getWindow(filledImage_scharry, center, w)

    magnitude = np.sqrt(np.multiply(cut_x_patch, cut_x_patch)
                        + np.multiply(cut_y_patch, cut_y_patch))
    
    if (np.count_mpmzero(psiHatP.filled() <= 1)):
        Ny, Nx = None, None
    else:
        Ny = filledImage_scharrx[center] / magnitude
        Nx = -filledImage_scharry[center] / magnitude
    
    return Ny, Nx
