#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:56:11 2018

@author: eo
"""

import os
import cv2
import numpy as np


#%% OpenCV macros

# OpenCV videocapture property macros
vc_pos_ms = 0
vc_pos_frames = 1
vc_width = 3
vc_height = 4
vc_fps = 5
vc_fourcc = 6
vc_framecount = 7
readAsGrayscale = 0


#%% Class definitions


# Class for creating a stack of grayscale frames
class FrameStack:
    
    def __init__(self, frameWidthHeight, stackSize, bgImage=None, maskImage=None, stackName=""):
        self.width = frameWidthHeight[0]
        self.height = frameWidthHeight[1]
        self.stackSize = stackSize
        self.bg = bgImage
        self.mask = maskImage
        self.name = stackName
        self.stack = np.zeros((self.height, self.width, stackSize), np.uint8)
        self.pointer = 0
    
    # -----------------------------------------------------------------------------------------------------------------
    
    # Function for adding elements to the stack (using a ring/FIFO pattern)
    def append(self, newFrame, useMask=None):
        self.pointer = self.wrapIndex(1)
        if useMask is not None:
             self.stack[:, :, self.pointer] = cv2.bitwise_and(newFrame, newFrame, mask=useMask)
             return None
        self.stack[:, :, self.pointer] = newFrame.copy()
        
    # -----------------------------------------------------------------------------------------------------------------
        
    # Function for loading in a background image
    def loadBackground(self, image=None, imagePath=None, asGrayscale=True):
        
        # If an image is directly supplied
        if image is not None:
            
            # Make sure the background is resized to match the frame stack dimensions
            resizeBG = cv2.resize(image, dsize=(self.width, self.height))
            
            # Convert to grayscale if needed
            if asGrayscale:
                self.bg = cv2.cvtColor(resizeBG, cv2.COLOR_BGR2GRAY)    
            else:
                self.bg = resizeBG.copy()
            
            return None
        
        # If a path to a background image is supplied
        if imagePath is not None:
            
            # Check if the file exists
            if not os.path.exists(imagePath):
                print("No background file found at:")
                print(imagePath)
                print("")
                raise FileNotFoundError
                
            # Load in the image
            if asGrayscale:
                pathedImage = cv2.imread(imagePath, readAsGrayscale)
            else:
                pathedImage = cv2.imread(imagePath)
            
            # Resize the image before storing it
            self.bg = cv2.resize(pathedImage, dsize=(self.width, self.height))
            
            return None
        
        # Image/path options failed, so no background image will be used
        print("No image/path supplied! No background image was loaded")
        return None
    
    # -----------------------------------------------------------------------------------------------------------------
    
    # Function for blurring the most recent frame in the stack
    def blur(self, kernelSize=(3,3), kernelSigma=(0,0), inplace=False):
        
        blurFrame = cv2.GaussianBlur(self.getFrame(), kernelSize, 
                                     sigmaX=kernelSigma[0], sigmaY=kernelSigma[1])
        
        # Write the blurred frame back into the stack if desired
        if inplace:
            self.replaceNew(blurFrame)
            return None
        
        return blurFrame
    
    # -----------------------------------------------------------------------------------------------------------------
        
    # Function for performing a frame-to-frame difference
    def absDiff(self, backwardIndex=1):
        
        # Get difference between current frame and a previous frame
        return cv2.absdiff(self.getFrame(), self.getFrame(backwardIndex))   
    
    # -----------------------------------------------------------------------------------------------------------------
    
    # Function for performing background subtraction
    def bgDiff(self):
        
        # Get difference between the current frame and the background image (assuming one is loaded!)
        return cv2.absdiff(self.getFrame(), self.bg)
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def andSelf(self, backwardIndex=1):
        
        # Get boolean AND with a previous frame (acts similar to a frame difference)
        return cv2.bitwise_and(self.getFrame(), self.getFrame(backwardIndex))    
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def orSelf(self, backwardIndex=1):
        
        # Get boolean OR with a previosu frame (acts similar to summing frames)
        return cv2.bitwise_or(self.getFrame(), self.getFrame(backwardIndex))
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def applyMask(self, maskImage=None, inplace=False):
        
        # Use a built-in mask if an image isn't supplied
        if maskImage is not None:
            useMask = maskImage
        else:
            useMask = self.mask
        
        # Apply mask to the most recent frame
        masked = cv2.bitwise_and(self.getFrame(), useMask)
        
        # Write the frame back into the stack if desired
        if inplace:
            self.replaceNew(masked)
            return None
        
        return masked
    
    # -----------------------------------------------------------------------------------------------------------------
    
    # Function for performing morphological processing on the most recent frame
    def morpho(self, morphKernel, kernelType=cv2.MORPH_CLOSE, inplace=False):
        
        # Perform morphology (can be grayscale or binary, depending on frame type)
        morphFrame = cv2.morphologyEx(self.getFrame(), kernelType, morphKernel)
            
        # Write the frame back into the stack if desired
        if inplace:
            self.replaceNew(morphFrame)
            return None
        
        return morphFrame
    
    # -----------------------------------------------------------------------------------------------------------------
    
    # Function for backwards-summing frames in the frame stack
    def sumStack(self, numToSum=1):
        
        # Initialize an empty summation frame, then add up (backwards) as many frames as needed
        summationFrame = np.zeros((self.height, self.width), dtype=np.uint16)
        
        # Get multiple frames and sum them up    
        for backIndex in range(numToSum):
            frameToSum = np.uint16(self.getFrame(backIndex))
            summationFrame = np.sum((summationFrame, frameToSum), 0, dtype=np.uint16)

        # Clip the summation frame back to an 8-bit image
        clippedSumFrame = np.uint8(np.clip(summationFrame, 0, 255))
        
        return clippedSumFrame
    
    # -----------------------------------------------------------------------------------------------------------------
    
    # Function for getting a thresholded image from the frame stack
    def threshold(self, thresholdValue=127, inplace=False):
        
        # Get only the threshold frame (the [1] index)
        thresholdFrame = cv2.threshold(self.getFrame(), thresholdValue, 255, cv2.THRESH_BINARY)[1]
        
        # Write the frame back into the stack if desired
        if inplace:
            self.replaceNew(thresholdFrame)
            return None
            
        return thresholdFrame
    
    # -----------------------------------------------------------------------------------------------------------------
    
    # Function for getting Canney edge-detection on the most recent frame in the stack
    def edges(self, lowThresh=100, highThresh=200, inplace=False):
        
        # Get edge-detected frame
        edgeFrame = cv2.Canny(self.getFrame(), lowThresh, highThresh)
        
        # Write the frame back into the stack if desired
        if inplace:
            self.replaceNew(edgeFrame)
            return None
        
        return edgeFrame

    # -----------------------------------------------------------------------------------------------------------------
    
    
    # ****************************************************************************************************************
    # ********************************************* Convenience functions ********************************************
    
    # Convenience function for calculating wrap-around indices in the frame stack
    def wrapIndex(self, indexShift):        
        return (self.pointer + indexShift) % self.stackSize
    
    # -----------------------------------------------------------------------------------------------------------------
    
    # Convenience function for returning a single frame from the stack
    def getFrame(self, frameIndex=0):
        wrappedIndex = self.wrapIndex(-frameIndex)
        return self.stack[:, :, wrappedIndex].copy()
    
    # -----------------------------------------------------------------------------------------------------------------
    
    # Convenience function for grabbing a reference to the most recent frame in the stack
    def getFrameRef(self):
        return self.stack[:, :, self.pointer]
    
    # -----------------------------------------------------------------------------------------------------------------
    
    # Convenience function for replacing the most recent frame in the stack
    def replaceNew(self, newFrame):
        self.stack[:, :, self.pointer] = newFrame.copy()
    
    # -----------------------------------------------------------------------------------------------------------------
    
    # Convenience function for displaying a frame from the stack (for debugging)
    def display(self, frameIndex=0, windowName=None):
        
        wrappedIndex = self.wrapIndex(-frameIndex)        
        if windowName is None:
            windowName = " ".join([self.name, "- Frame:", str(frameIndex)])
        cv2.imshow(windowName, self.stack[:, :, wrappedIndex])        
        cv2.waitKey(500)
        
    # -----------------------------------------------------------------------------------------------------------------
        
    
#                                       **********
#                                       **********
#                                       **********
#                                       **********
#                                       **********
#                                       **********
