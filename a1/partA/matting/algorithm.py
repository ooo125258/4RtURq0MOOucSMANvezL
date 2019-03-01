## CSC320 Winter 2019 
## Assignment 1
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

# import basic packages
import numpy as np
import scipy.linalg as sp
import cv2 as cv

# If you wish to import any additional modules
# or define other utility functions, 
# include them here

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################
import os
import numpy as np


#########################################

#
# The Matting Class
#
# This class contains all methods required for implementing 
# triangulation matting and image compositing. Description of
# the individual methods is given below.
#
# To run triangulation matting you must create an instance
# of this class. See function run() in file run.py for an
# example of how it is called
#
class Matting:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts as a container
    # for all input and all output images of the triangulation matting and compositing 
    # algorithms. These images are initialized to None and populated/accessed by 
    # calling the the readImage(), writeImage(), useTriangulationResults() methods.
    # See function run() in run.py for examples of their usage.
    #
    def __init__(self):
        self._images = {
            'backA': None,
            'backB': None,
            'compA': None,
            'compB': None,
            'colOut': None,
            'alphaOut': None,
            'backIn': None,
            'colIn': None,
            'alphaIn': None,
            'compOut': None,
        }

    # Return a dictionary containing the input arguments of the
    # triangulation matting algorithm, along with a brief explanation
    # and a default filename (or None)
    # This dictionary is used to create the command-line arguments
    # required by the algorithm. See the parseArguments() function
    # run.py for examples of its usage
    def mattingInput(self):
        return {
            'backA': {'msg': 'Image filename for Background A Color', 'default': None},
            'backB': {'msg': 'Image filename for Background B Color', 'default': None},
            'compA': {'msg': 'Image filename for Composite A Color', 'default': None},
            'compB': {'msg': 'Image filename for Composite B Color', 'default': None},
        }

    # Same as above, but for the output arguments
    def mattingOutput(self):
        return {
            'colOut': {'msg': 'Image filename for Object Color', 'default': ['color.tif']},
            'alphaOut': {'msg': 'Image filename for Object Alpha', 'default': ['alpha.tif']}
        }

    def compositingInput(self):
        return {
            'colIn': {'msg': 'Image filename for Object Color', 'default': None},
            'alphaIn': {'msg': 'Image filename for Object Alpha', 'default': None},
            'backIn': {'msg': 'Image filename for Background Color', 'default': None},
        }

    def compositingOutput(self):
        return {
            'compOut': {'msg': 'Image filename for Composite Color', 'default': ['comp.tif']},
        }

    # Copy the output of the triangulation matting algorithm (i.e., the 
    # object Color and object Alpha images) to the images holding the input
    # to the compositing algorithm. This way we can do compositing right after
    # triangulation matting without having to save the object Color and object
    # Alpha images to disk. This routine is NOT used for partA of the assignment.
    def useTriangulationResults(self):
        if (self._images['colOut'] is not None) and (self._images['alphaOut'] is not None):
            self._images['colIn'] = self._images['colOut'].copy()
            self._images['alphaIn'] = self._images['alphaOut'].copy()

    # If you wish to create additional methods for the 
    # Matting class, include them here

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    #########################################

    # Use OpenCV to read an image from a file and copy its contents to the 
    # matting instance's private dictionary object. The key 
    # specifies the image variable and should be one of the
    # strings in lines 54-63. See run() in run.py for examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def readImage(self, fileName, key):  ##TODO: Assume it's a colorful image, all pictures same size
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        loader = cv.imread(fileName)
        if loader is None:
            msg = 'Error: File ' + fileName + ' at key ' + key + ' reading Failed!'
        else:
            loader = loader.astype(np.float) / 255
            # loader = cv.cvtColor(loader, cv.COLOR_BGR2RGB).astype(np.float) / 255
            # loader = cv.normalize(loader, loader, alpha=0, beta=1, norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)
            # loader = cv.normalize(loader, None, 1, 0, cv.NORM_MINMAX)
            self._images[key] = loader
            success = True
            msg = "Img" + fileName + ' at key ' + key + " loaded."

        #########################################
        return success, msg

    # Use OpenCV to write to a file an image that is contained in the 
    # instance's private dictionary. The key specifies the which image
    # should be written and should be one of the strings in lines 54-63. 
    # See run() in run.py for usage examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # return False, along with an error message
    def writeImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################Input: 0-1 picture
        loader = self._images[key] * 255
        if len(self._images[key].shape) >= 3:
            loader = loader.astype(np.uint8)
            # loader = cv.cvtColor(loader, cv.COLOR_RGB2BGR)
        else:
            loader = loader.astype(np.uint8)
        cv.imwrite(fileName, loader)
        if os.path.exists(fileName):
            success = True
            msg = "Img" + fileName + ' at key ' + key + " written!"
        else:
            msg = 'Error: File ' + fileName + ' at key ' + key + ' writing Failed!'

        #########################################
        return success, msg

    # Method implementing the triangulation matting algorithm. The
    # method takes its inputs/outputs from the method's private dictionary 
    # ojbect. 
    def triangulationMatting(self):
        """
success, errorMessage = triangulationMatting(self)
        
        Perform triangulation matting. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        np.seterr(all='raise')
        img_photos = ['backA', 'backB', 'compA', 'compB']
        for key in img_photos:
            if self._images[key] is None:
                success = False
                msg = 'triangulationMatting Error:  key ' + key + ' image not found!'
                return success, msg
        # C-B, m * n* 6
        shape = self._images['backA'].shape
        B = np.concatenate((self._images['backA'], self._images['backB']), axis=2)
        deltaC = np.concatenate(
            (self._images['compA'] - self._images['backA'], self._images['compB'] - self._images['backB']), axis=2)
        I = np.identity(3, dtype=int)
        matrix = np.concatenate((I, I), axis=0)
        try:
            colOut = np.zeros(shape)
            alphaOut = np.zeros(shape[:2])
            '''
            This piece of code require python3, numpy 1.14+. Which means it can only run  on python3.7 on cdf computer
            wholemx = np.tile(matrix, (shape[:2])).reshape(shape[0],shape[1], 6,3)
            neg_B = B * -1
            matrix_111 = np.concatenate((wholemx, neg_B[..., np.newaxis]), axis=-1)
            try_pinv = np.linalg.pinv(matrix_111)
            reshape_deltaC = deltaC.reshape(shape[0],shape[1], 6, 1)
            rst = np.matmul(try_pinv, reshape_deltaC).reshape(shape[0],shape[1], 4)
            sample_colOut = rst[:,:,3]
            sample_alpha = rst[:,:,1]
            '''
            # Using the formula on essay SIGGRAPH 1996
            # alpha = 1- ((R_f1 - R_f2)(R_k1-R_k2) + (G_f1 - G_f2)(G_k1-G_k2) + (B_f1 - B_f2)(B_k1-B_k2)) / (R_k1 - R_k2)^2 + (G_k1 - G_k2)^2 + (B_k1 - B_k2)^2
            step1 = np.multiply(self._images['compA'] - self._images['compB'],
                                self._images['backA'] - self._images['backB'])
            nomi = step1.sum(axis=-1)
            step2 = np.multiply(self._images['backA'] - self._images['backB'],
                                self._images['backA'] - self._images['backB'])
            deno = step2.sum(axis=-1)
            deno[deno == 0] = 0.00000001
            one_minus_alpha = np.divide(nomi, deno)  # type: np.array()
            one_minus_alpha[one_minus_alpha > 1] = 1
            alpha = 1 - one_minus_alpha
            one_minus_alpha = one_minus_alpha
            C0 = self._images['compA'] - np.multiply(np.expand_dims(one_minus_alpha, axis=-1), self._images['backA'])
            #C0 = C0 * alpha[:,:,None].repeat(3,2) #Does alpha

            # unnormalized_colOut = np.clip(colOut, 0, 1)
            colOut = np.clip(C0, 0, 1)
            alphaOut = np.clip(alpha, 0, 1)
        except Exception as e:
            msg = "triangulationMatting Error! Reason: {}".format(e)
            return success, msg

        if colOut is None:
            success = False
            msg = 'triangulationMatting Error:  colOut error! None!'
            return success, msg
        elif alphaOut is None:
            success = False
            msg = 'triangulationMatting Error:  alphaOut error! None!'
            return success, msg
        success = True
        self._images['colOut'] = colOut  # unnormalized_colOut
        self._images['alphaOut'] = alphaOut
        msg = 'triangulationMatting completed!'
        return success, msg

    '''
                    RGB bitwise. But it's so slow
                        for i in range(colOut.shape[0]):
                        for j in range(colOut.shape[1]):
                        ver_neg_B = B[i,j].reshape(6,1) * -1
                        coeff_Fprime = np.hstack((matrix, ver_neg_B))
                        deltaC_ij = deltaC[i,j]
                        #deltaC_ij =coeff_Fprime * f_prime
                        #(C.T * C)^-1 * C^T * d = F
                        #F = pinv(coeff_Fprime) * deltaC_ij
                        something = np.linalg.pinv(coeff_Fprime)
                        unclip_Fprime = np.dot(np.linalg.inv(np.dot(coeff_Fprime.T, coeff_Fprime)), np.dot(coeff_Fprime.T, deltaC_ij))
                        Fprime = np.clip(unclip_Fprime, 0, 1)
                        colOut[i, j] = Fprime[:3]#Fprime[:3] / (Fprime[-1] + 0.00000001)#divided by zero?
                        alphaOut[i,j] = Fprime[-1]
                '''

    def createComposite(self):
        """
success, errorMessage = createComposite(self)
        
        Perform compositing. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        alphaIn, colIn, backIn, Return: compOut
"""

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        # First, if imgs are loaded.

        img_photos = ['alphaIn', 'colIn', 'backIn']
        for key in img_photos:
            if self._images[key] is None:
                success = False
                msg = 'triangulationMatting Error:  key ' + key + ' image not found!'
                return success, msg

        # C0 = self._images['compA'] - np.multiply(np.expand_dims(one_minus_alpha, axis=-1), self._images['backA'])
        alphaIn = self._images['alphaIn'][:, :, 0]
        try:
            compOut = self._images['colIn'] + np.multiply(np.expand_dims(1 - alphaIn, axis=-1), self._images['backIn'])
        except Exception as e:
            msg = "createComposite Error! Reason: {}".format(e)
            return success, msg
        self._images['compOut'] = compOut
        success = True
        msg = 'createComposite completed!'
        #########################################

        return success, msg
