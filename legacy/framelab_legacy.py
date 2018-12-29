#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:38:27 2018

@author: eo
"""

import cv2
import numpy as np
from functools import partial


class FrameLab:
    
    def __init__(self, name="Unnamed Frame Processor"):
        
        # Name this object to use for plotting/feedback
        self._name = name
        
        # Variables used for keeping track of image dimensions/changes
        self._historyWHC = []
        self._crop_points = ()
   
        # Storage for masking/background differences
        self._bgImage = None
        self._resources = {}
        
        # Allocate variables for implementing a framestack
        self._pointerList = []
        self._stackList = []
        self._stackSizeList = []
        
        # Variables used to manage function sequencing
        self._setSequence = False
        self._functionSequence = []
        self._funcIndex = 0
        
        # Variables used to handle temporal subsampling
        self._countUpdates = np.int64(0)
        self._requestBreak = False
        self._requestHoldFrame = False
        self._holdFrame = None
        self._changed = False
        self._timeResourcesList = []
        
        # Variables used to store processed frames
        self._intermediates = []
        self._seqDictionary = {}
        
    # .................................................................................................................
    
    def __str__(self):
        return self._name

    # .................................................................................................................

    def __repr__(self):
        
        print("")
        outStringList = []
        outStringList.append(" ".join(["FrameLab sequence:", self._name]))
        for eachFunc in self._functionSequence:
            
            if type(eachFunc) is partial:                
                funcString = "".join(["  ", eachFunc.func.__name__, "()"])
            else:
                funcString = "".join(["  ", eachFunc.__name__, "()"])
                
            outStringList.append(funcString)
                
        return "\n".join(outStringList)
    
    # -----------------------------------------------------------------------------------------------------------------
    #%% Sequence Utilities  
    
    # .................................................................................................................
    
    def startSequence(self, setInput=None):
        
        # Quick sanity check to avoid weirdness from calling start multiple times
        if self._setSequence:
            print("")
            print(self._name)
            print("Error, a sequence has already been started!")
            print("")
            raise SyntaxError
        
        # If a sequence was already created, give a warning about clearing the function list
        if len(self._functionSequence) > 0:
            print("")
            print(self._name)
            print("WARNING:")
            print("  A function sequence already exists and will be cleared!")
            
        # Setup expected input dimensions
        self._setInput(setInput)
        
        # Clear any existing functions and set flag
        self._funcIndex = 0
        self._functionSequence = []
        self._intermediates = []
        self._setSequence = True
    
    # .................................................................................................................
    
    def endSequence(self, storeIntermediates=False):
        
        # Quick sanity check! Don't allow sequencing without a function sequence
        if len(self._functionSequence) == 0:
            print("")
            print(self._name)
            print("Can't end sequence because there is no function sequence!")
            print("")
            FrameLab._quit()
        
        # Make sure a start command was given before calling this function
        if not self._setSequence:
            print("")
            print(self._name)
            print("No sequence to end!")
            print("Must call .startSequence() and processing functions before calling .endSequence()")
            print("")
            FrameLab._quit()
            
        # End sequencing
        self._setSequence = False    
        
        # Enable storage of intermediate frames
        if storeIntermediates:
            
            # Allocate space for intermediate frame storage
            self._intermediates = [None] * (1 + len(self._functionSequence))    # Add 1 to store output
            
            self.update = self._updateWithIntermediates
            
            return
        
        # If no special settings, replace update function with no-storage version
        self.update = self._updateWithNoStorage
    
    # .................................................................................................................
    
    def update(self, inFrame):
        
        # The endSequence() function will select an appropriate update() function
        # In case a user doesn't set endSequence, display the warning below
        print("")
        print(self._name)
        print("Need use .endSequence() function before using update!")
        print("")
        raise SyntaxError
    
    # .................................................................................................................
    
    def _updateWithNoStorage(self, inFrame):
        
        # Assume that the output will change on this update (this may be altered by a break later)
        self._changed = True
        
        # Repeatedly apply each function in the func. sequence on the result from the previous function
        prevFrame = inFrame.copy()
        for idx, eachFunction in enumerate(self._functionSequence):
            
            self._funcIndex = idx    # Allows for debugging
            prevFrame = eachFunction(prevFrame)
            
            # Allow functions in the function list to break the update loop (dangerous!)
            if self._requestBreak:
                self._changed = False
                break
        
        # Store a copy of the output if requested by one of the sequence functions
        if self._requestHoldFrame:
            self._holdFrame = prevFrame.copy()
            self._requestHoldFrame = False       
        
        # Update processing count
        self._countUpdates += 1
        
        return prevFrame
    
    
    # .................................................................................................................
    
    def _updateWithIntermediates(self, inFrame):
        
        # Assume that the output will change on this update (this may be altered by a break later)
        self._changed = True
        
        # Store input frame in the first list index (which has been allocated by endSequence function)
        self._intermediates[0] = inFrame.copy()
        for idx, eachFunction in enumerate(self._functionSequence):
            
            nextIndex = 1 + idx
            
            self._funcIndex = idx
            self._intermediates[nextIndex] = eachFunction(self._intermediates[idx]).copy()
            
            if self._requestBreak:
                self._changed = False
                break
            
        if self._requestHoldFrame:
            self._holdFrame = self._intermediates[nextIndex].copy()
            self._requestHoldFrame = False     
            
        self._countUpdates += 1
        
        return self._intermediates[nextIndex].copy()
    
    # -----------------------------------------------------------------------------------------------------------------
    #%% Input utilities
    
    # .................................................................................................................
    
    def _setInput(self, setInput):
        
        if type(setInput) is cv2.VideoCapture:
            self._setInputFromVideoCapture(setInput)
            return
        
        if type(setInput) is FrameLab:
            self._setInputFromOtherFrameLab(setInput)
            return
        
        if type(setInput) is np.ndarray:
            self._setInputFromImage(setInput)
            return
        
        if type(setInput) in [tuple, list]:
            inWidth = setInput[0]
            inHeight = setInput[1]
            inChannels = setInput[2] if len(setInput) > 2 else 3    # Assume a 3-channel image if not specified   
            self._recordDimensions(inWidth, inHeight, inChannels)        
            return
        
        if setInput is None:
            # Not gonna set input size... hope for the best!
            self._recordDimensions(None, None, None)
            return
        
        print("")
        print(self._name)
        print("Unrecognized input!")
        print("")
        FrameLab._quit()
        
    # .................................................................................................................
    
    def _setInputFromOtherFrameLab(self, framelabObj):
        
        otherWHC = framelabObj.getDimensions()
        
        if None in otherWHC or len(otherWHC) == 0:
            self._recordDimensions(None, None, None)
            print("")
            print(self._name)
            print("Error setting input based on other FrameLab object!")
            print(framelabObj._name, "does not have valid dimensions set (most likely it's processing does not require storage)")
            print("Explicit dimensions may need to be provided if another error occurs.")
            print("")
            
        else:
            self._recordDimensions(otherWHC)
        
    # .................................................................................................................
    
    def _setInputFromImage(self, inFrame):
        
        inDimensions = inFrame.shape
        inWidth = inDimensions[1]
        inHeight = inDimensions[0]
        inChannels = inDimensions[2] if len(inDimensions) > 2 else 1
        self._recordDimensions(inWidth, inHeight, inChannels)
    
    # .................................................................................................................
    
    def _setInputFromVideoCapture(self, videoCaptureObject):
        
        # OpenCV enumerator values
        vc_width = 3
        vc_height = 4

        # Set dimension tracker variables based on video frames
        vidWidth = int(videoCaptureObject.get(vc_width))
        vidHeight = int(videoCaptureObject.get(vc_height))        
        self._recordDimensions(vidWidth, vidHeight, 3)
    
    # .................................................................................................................
    
    def appendFrameLab(self, framelabObj):
        
        raise NotImplementedError
    
    
    # -----------------------------------------------------------------------------------------------------------------
    #%% Sequence functions    
    
    
    # .................................................................................................................
    
    def customFunction(self, inputFunction, **kwargs):
        
        # Record the change from (presumably) 3 channel BGR to to a single grayscale channel
        inWidth, inHeight, inChannels = self.getDimensions()        
        if None in [inWidth, inHeight, inChannels]:
            print("")
            print(self._name)
            print("Error running custom function:", inputFunction.__name__)
            print("Must supply input frame dimensions prior to using custom functions")
            print("Use startSequence(setInput) to specific input dimensions!")
            raise AttributeError
        
        # Call input function will all keyword arguments
        customFunc = partial(inputFunction, **kwargs)
                
        # Figure out what this custom function does to image dimensions by passing a dummy frame through it
        dummyFrame = np.zeros((inHeight, inWidth, inChannels), dtype=np.uint8)
        try:
            outFrame = customFunc(dummyFrame)
        except Exception as e:
            print("")
            print(self._name)
            print("Error running custom function:", customFunc.func.__name__)
            print("Tried inputting frame of dimensions:")
            print("WHC:", " x ".join([str(inWidth), str(inHeight), str(inChannels)]))
            print("")
            raise e
        
        # Update dimension record to account for any resizing this function performs
        outDimensions = outFrame.shape
        outWidth = outDimensions[1]
        outHeight = outDimensions[0]
        outChannels = outDimensions[2] if len(outDimensions) > 2 else 1
        self._recordDimensions(outWidth, outHeight, outChannels)
        
        # Some feedback
        print("")
        print("Custom function:", customFunc.func.__name__)
        print("  Input size  (WHC):", " x ".join([str(inWidth), str(inHeight), str(inChannels)]))
        print("  Output size (WHC):", " x ".join([str(outWidth), str(outHeight), str(outChannels)]))
        
        return self._seqReturn(customFunc)
    
    # .................................................................................................................
    
    def grayscale(self):
        
        # Record the change from (presumably) 3 channel BGR to to a single grayscale channel
        width, height, channels = self.getDimensions()
        self._recordDimensions(width, height, 1)
        
        # OpenCV: cv2.cvtColor(src, code)
        grayFunc = partial(cv2.cvtColor, code=cv2.COLOR_BGR2GRAY)
        
        return self._seqReturn(grayFunc)
    
    
    # .................................................................................................................
    
    def resize(self, dimensionsWH=None, scale_factorXY=None):
        
        # Sanity check
        if (dimensionsWH is None) and (scale_factorXY is None):
            print("")
            print(self._name)
            print("Must set dimensions or scaling parameter when resizing!")
            print("")
            raise AttributeError
            
        # Set width/height using scaling values (if provided)
        if scale_factorXY is not None:
            lastWidth, lastHeight, lastChannels = self.getDimensions()            
            scaledWidth = np.int(np.round(lastWidth*scale_factorXY[0]))
            scaledHeight = np.int(np.round(lastHeight*scale_factorXY[1]))
            
            # OpenCV: cv2.resize(src, fx, fy)
            resizeFunc = partial(cv2.resize, dsize=None, fx=scale_factorXY[0], fy=scale_factorXY[1]) 
            
        # Set width/height directly (if provided, overrides scaling if present)
        if dimensionsWH is not None:
            scaledWidth = dimensionsWH[0]
            scaledHeight = dimensionsWH[1]
            
            # Check if the resize is not needed
            last_width, last_height, last_channels = self.getDimensions()
            if last_width == scaledWidth and last_height == scaledHeight:
                print("")
                print(self._name)
                print("No resizing performed, since input already matches target dimensions!")
                
                # Define a pass-through function and return it
                def no_resize(inFrame): return inFrame
                resizeFunc = no_resize
            else:
                
                # OpenCV: cv2.resize(src, dsize)
                resizeFunc = partial(cv2.resize, dsize=(scaledWidth, scaledHeight)) 
        
        # Record the change in image dimensions
        self._recordDimensions(scaledWidth, scaledHeight)
        
        return self._seqReturn(resizeFunc)
        
    # .................................................................................................................
    
    def mask(self, maskImage):
        
        # Warn if no mask is supplied
        if maskImage is None:
            self._error_out(AttributeError, "No mask supplied!")
            
        # Make sure the mask image is correctly sized for bitwise_and operation
        maskImage = self._matchToSelfDimensions(maskImage, "mask", "masking")
        
        # Resizing can mess up mask images, so re-threshold the image
        maskImage = cv2.threshold(maskImage, 200, 255, cv2.THRESH_BINARY)[1]
        
        # OpenCV: cv2.bitwise_and(src1, src2, mask=optional)
        # Using RGB mask as src2 seems much faster than using the optional mask input!
        maskFunc = partial(cv2.bitwise_and, src2=maskImage)
        
        return self._seqReturn(maskFunc)
    
    # .................................................................................................................
    
    def crop(self, cropImage):
        
        # Warn if the input image is bad
        if type(cropImage) is not (np.ndarray):
            self._error_out(AttributeError, "No crop mask supplied!")
            
        # Make sure the crop mask is correctly sized for bitwise_anding
        cropImage = self._matchToSelfDimensions(cropImage, "crop image", "cropping")  
        
        # Resizing can mess up binary images, so re-threshold the image
        cropImage = cv2.threshold(cropImage, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Get cropping point indices
        _, crop_points = self.cropped_mask(cropImage)
        self._crop_points = crop_points
        
        # Create function for cropping the input image
        def crop_(inFrame, cropping_points):            
            return inFrame[cropping_points[0]:cropping_points[1], cropping_points[2]:cropping_points[3]]
        
        # Store cropping image with crop points
        crop_func = partial(crop_, cropping_points=crop_points)
        
        # Record the change in image dimensions
        self._recordDimensions((crop_points[3] - crop_points[2]), (crop_points[1] - crop_points[0]))
        
        return self._seqReturn(crop_func)
    
    # .................................................................................................................
    
    def cropAndMask(self, cropMaskImage):
        
        # Warn if the input image is bad
        if type(cropMaskImage) is not (np.ndarray):
            self._error_out(AttributeError, "No crop mask supplied!")
            
        # Make sure the cropmask is correctly sized for bitwise_anding
        cropMaskImage = self._matchToSelfDimensions(cropMaskImage, "cropmask image", "cropping")  
        
        # Resizing can mess up binary images, so re-threshold the image
        cropMaskImage = cv2.threshold(cropMaskImage, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Get cropping point indices and mask
        cropped_input_mask, crop_points = self.cropped_mask(cropMaskImage)
        self._crop_points = crop_points
        
        
        # Create function for cropping and applying the input cropmask
        def cropAndMask_(inFrame, cropping_points, mask_image):        
            # Crop the incoming frame
            cropped_image = inFrame[cropping_points[0]:cropping_points[1], cropping_points[2]:cropping_points[3]]
            
            # OpenCV: cv2.bitwise_and(src1, src2)
            return cv2.bitwise_and(src1 = cropped_image, src2 = cropped_image, mask = mask_image)
        
        # Store cropping image with crop points
        cropmask_func = partial(cropAndMask_, cropping_points=crop_points, mask_image=cropped_input_mask)
        
        # Record the change in image dimensions
        self._recordDimensions((crop_points[3] - crop_points[2]), (crop_points[1] - crop_points[0]))
        
        return self._seqReturn(cropmask_func)
    
    # .................................................................................................................
    
    def blur(self, kernelSize=(3,3), kernelSigma=(0,0)):
        
        # OpenCV: cv2.GaussianBlur(src, ksize, sigmaX, sigmaY)
        blurFunc = partial(cv2.GaussianBlur, ksize=kernelSize, sigmaX=kernelSigma[0], sigmaY=kernelSigma[1])
        return self._seqReturn(blurFunc)        
        
    # .................................................................................................................
    
    def diffWithBG(self, bgImage):
        
        # Warn if no background image is supplied
        if bgImage is None:
            print("")
            print(self._name)
            print("No background image supplied!")
            print("")
            raise AttributeError
        
        # Make sure the background image is correctly sized for background subtraction operation
        bgImage = self._matchToSelfDimensions(bgImage, "background", "background subtraction")
        
        # Copy background to internal storage
        self._bgImage = bgImage.copy()
        
        # Use internal function so that internal self._bgImage can be used in difference (and updated dynamically)
        def diffWithBG_(inFrame):
            # OpenCV: cv2.absdiff(src1, src2)
            return cv2.absdiff(inFrame, self._bgImage)
        
        return self._seqReturn(diffWithBG_)
    
    # .................................................................................................................
    
    def diffWithSelf(self, backwardStep=1):
        
        # Update minimum stack sizing requirements, since self difference requires at least 2 frames
        minSize = 1 + abs(backwardStep)        
        stackIndex, stackSize = self._buildNewStack(minSize)
        
        # Function for getting an absolute difference 
        def diffWithSelf_(inFrame, stackIndex=0, backStep=1):
            
            # Add inFrame to the stack before performing difference
            self._addToStack(inFrame, stackIndex)
            
            # OpenCV: cv2.absdiff(src1, src2)
            return cv2.absdiff(inFrame, self._returnStackFrame(stackIndex=stackIndex, relIndex=backStep))
        
        selfDiffFunc = partial(diffWithSelf_, stackIndex=stackIndex, backStep=backwardStep)
        
        return self._seqReturn(selfDiffFunc)
        
    # .................................................................................................................
    
    def diffWith(self, storage_name, initial_frame=None):
        
        # Check that we aren't already using the storage name
        if storage_name in self._resources:
            print("")
            print("Error: {}".format(self._name))
            print("  Storage name", storage_name, "already in use!")
            print("")
            raise AttributeError
            
        # Store initial frame if provided
        self._resources[storage_name] = None if initial_frame is None else initial_frame.copy()
        
        def diffWith_(inFrame, resource_key):
            
            # Get the resource in case it was changed
            diff_frame = self._resources[resource_key]
            
            # If no resource is present, just pass back the incoming frame (i.e. no difference)
            if diff_frame is None:
                return inFrame
            
            # OpenCV: cv2.absdiff(src1, src2)
            return cv2.absdiff(inFrame, diff_frame)
            
        diffFunc = partial(diffWith_, resource_key=storage_name)
        
        return self._seqReturn(diffFunc)
    
    # .................................................................................................................
    
    def morphology(self, kernelSize=(3,3), kernelShape=cv2.MORPH_RECT, morphKernel=None, operation=cv2.MORPH_CLOSE):
        
        # Generate the morphological kernel if it isn't supplied
        if morphKernel is None:
            morphKernel = cv2.getStructuringElement(kernelShape, kernelSize)
        
        # OpenCV: cv2.morphologyEx(src, op, kernel)
        morphFunc = partial(cv2.morphologyEx, op=operation, kernel=morphKernel)

        return self._seqReturn(morphFunc)
    
    # .................................................................................................................
    
    def threshold(self, thresholdLevel=127):
        
        # Function for getting the 1-index return argument (frame data) from the OpenCV function
        def threshold_(inFrame, threshVal):
            # OpenCV: cv2.threshold(src, thresh, maxval, type) 
            return cv2.threshold(inFrame, thresh=threshVal, maxval=255, type=cv2.THRESH_BINARY)[1]
        
        threshFunc = partial(threshold_, threshVal=thresholdLevel)

        return self._seqReturn(threshFunc)
        
    # .................................................................................................................
    
    def andWithSelf(self, backwardStep=1):
        
        # Set minimum stack size requirement to perform ANDing across previous frames
        minSize = 1 + abs(backwardStep)        
        stackIndex, _ = self._buildNewStack(minSize)
        
        # Function for ANDing together two frames in a framestack
        def andWithSelf_(inFrame, stackIndex=0, backStep=1):
            
            # Add inFrame to the stack before ANDing frames
            self._addToStack(inFrame, stackIndex)
            
            # OpenCV: cv2.bitwise_and(src1, src2)
            return cv2.bitwise_and(inFrame, self._returnStackFrame(stackIndex=stackIndex, relIndex=backStep))
        
        selfAndFunc = partial(andWithSelf_, stackIndex=stackIndex, backStep=backwardStep)
        
        return self._seqReturn(selfAndFunc)
    
    # .................................................................................................................
    
    def andWith(self, storage_name, initial_frame=None):
        
        # Check that we aren't already using the storage name
        if storage_name in self._resources:
            print("")
            print("Error: {}".format(self._name))
            print("  Storage name", storage_name, "already in use!")
            print("")
            raise AttributeError
            
        # Store initial frame if provided
        self._resources[storage_name] = None if initial_frame is None else initial_frame.copy()
            
        def andWith_(self, inFrame, resource_key):
            
            # Get the resource in case it was changed
            and_frame = self._resources[resource_key]
            
            # If no resource is present, just pass back the incoming frame (i.e. no ANDing)
            if and_frame is None:
                return inFrame
            
            # OpenCV: cv2.bitwise_and(src1, src2)
            return cv2.bitwise_and(inFrame, and_frame)
        
        # OpenCV: cv2.bitwise_and(src1, src2)
        andFunc = partial(andWith_, resource_key=storage_name)
        
        return self._seqReturn(andFunc)
    
    # .................................................................................................................
    
    def orWithSelf(self, backwardStep=1):
        
        # Set minimum stack size requirement to perform ORing across previous frames
        minSize = 1 + abs(backwardStep)        
        stackIndex, _ = self._buildNewStack(minSize)
        
        # Function for ORing together two frames in a framestack
        def orWithSelf_(inFrame, stackIndex=0, backStep=1):
            
            # Add inFrame to the stack before ORing frames
            self._addToStack(inFrame, stackIndex)
            
            # OpenCV: cv2.bitwise_or(src1, src2)
            return cv2.bitwise_or(inFrame, self._returnStackFrame(stackIndex=stackIndex, relIndex=backStep))
        
        selfAndFunc = partial(orWithSelf_, stackIndex=stackIndex, backStep=backwardStep)
        
        return self._seqReturn(selfAndFunc)
        
    # .................................................................................................................
    
    def orWith(self, storage_name, initial_frame=None):
        
        # Check that we aren't already using the storage name
        if storage_name in self._resources:
            print("")
            print("Error: {}".format(self._name))
            print("  Storage name", storage_name, "already in use!")
            print("")
            raise AttributeError
            
        # Store initial frame if provided
        self._resources[storage_name] = None if initial_frame is None else initial_frame.copy()
            
        def orWith_(self, inFrame, resource_key):
            
            # Get the resource in case it was changed
            or_frame = self._resources[resource_key]
            
            # If no resource is present, just pass back the incoming frame (i.e. no ORing)
            if or_frame is None:
                return inFrame
            
            # OpenCV: cv2.bitwise_or(src1, src2)
            return cv2.bitwise_or(inFrame, or_frame)
        
        # OpenCV: cv2.bitwise_or(src1, src2)
        orFunc = partial(orWith_, resource_key=storage_name)
        
        return self._seqReturn(orFunc)
     
    # .................................................................................................................

    def invert(self):        
        return self._seqReturn(cv2.bitwise_not)
    
    # .................................................................................................................
    
    def backSum(self, numToSum):
        
        # Set minimum stack size requirement to sum enough frames
        minSize = abs(numToSum)        
        stackIndex, _ = self._buildNewStack(minSize)
                
        def backSum_(inFrame, listIndex, framesToSum):
            
            # Add inFrame to the stack before performing summation
            self._addToStack(inFrame, listIndex)
            
            # Get convenient variables
            stackSize = self._stackSizeList[listIndex]
            startPoint = self._pointerList[listIndex]
            endPoint = self._wrapPointer(startPoint, 1 - framesToSum, stackSize)
            
            # Figure out which stack indices to include in the summation
            indexingVector = np.arange(stackSize)
            if endPoint > startPoint:
                selectionVector = np.logical_or(indexingVector <= startPoint, indexingVector >= endPoint)
            else:
                # endPoint < startPoint
                selectionVector = np.logical_and(indexingVector >= endPoint, indexingVector <= startPoint)
                
            sumFrame = np.sum(self._stackList[listIndex][selectionVector], axis=0, dtype=np.uint16)            
            return np.uint8(np.clip(sumFrame, 0, 255))
        
        sumFunc = partial(backSum_, listIndex=stackIndex, framesToSum=numToSum)
        
        return self._seqReturn(sumFunc)
    
    # .................................................................................................................
    
    def temporalSubsampleByFrames(self, sampleEveryNFrames):
        
        # Quick sanity check
        if sampleEveryNFrames < 1:
            print("")
            print(self._name)
            print("Error, temporal subsampling needs a frame jump of at least 1!")
            print("")
            raise AttributeError
            
        # Warning about requiring integer sampling indices (non-integer sampling could be implemented however...)
        if type(sampleEveryNFrames) is not int:
            print("")
            print(self._name)
            print("Warning! Temporal subsampling requires integer inputs!")
            print("Got:", sampleEveryNFrames)
            print("Converted to:", int(sampleEveryNFrames))
            sampleEveryNFrames = int(sampleEveryNFrames)
        
        def storeSubsamples(inFrame, sampleRate):#, listIndex):
            
            # Only record/update frames on subsample indices. Otherwise, pass previous frame through
            sampleCycleIndex = (self._countUpdates % sampleRate)
            subsampleUpdate = (sampleCycleIndex == 0)
            
            if subsampleUpdate:
                #self._addToStack(inFrame, listIndex)
                self._requestHoldFrame = True
                self._requestBreak = False
                return inFrame.copy()
            
            # No subsample update, so just pass the previously stored frame and request a break to the update loop!
            self._requestBreak = True
            return self._holdFrame
        
        subsampleFunc = partial(storeSubsamples, sampleRate=sampleEveryNFrames)#, listIndex=stackIndex)
        
        return self._seqReturn(subsampleFunc)
    
    # .................................................................................................................
    
    def temporalSubsample(self, timedelta=None, hours=None, minutes=None, seconds=None):
        
        import datetime as dt
        
        
        # Quick sanity check, don't let all keyword arguments be empty
        if (timedelta, hours, minutes, seconds) == (None, None, None, None):
            print("")
            print("Must supply at least 1 keyword argument to temporalSubsample() function!")
            print("")
            raise TypeError
        
        # If a timedelta value isn't supplied, build a timedelta out of individual time components
        if timedelta is None:
            hours = 0 if hours is None else hours
            minutes = 0 if minutes is None else minutes
            seconds = 0 if seconds is None else seconds            
            timedelta = dt.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        
        # Make sure timedelta is valid
        if type(timedelta) is not dt.timedelta:
            print("")
            print("Input must be a datetime.timedelta object!")
            print("")
            raise TypeError
            
        # Initialize next update time as now, to force immediate update
        timeIndex = len(self._timeResourcesList)
        self._timeResourcesList.append(dt.datetime.now())
            
        def downsample(inFrame, timestep, timeResourceIndex):
            
            # Check if we need to perform an update
            currentTime = dt.datetime.now()
            nextUpdateTime = self._timeResourcesList[timeResourceIndex]
            if currentTime >= nextUpdateTime:
                # Store result on the processing after this iteration (in holdFrame)
                self._requestHoldFrame = True
                self._requestBreak = False
                
                # Update the record of when to process next update
                nextUpdateTime = currentTime + timestep
                self._timeResourcesList[timeResourceIndex] = nextUpdateTime
                
                return inFrame.copy()
            
            # No subsampling update, so just pass previously stored frame and request a break to update loop!
            self._requestBreak = True
            return self._holdFrame
        
        subsampleFunc = partial(downsample, timestep=timedelta, timeResourceIndex=timeIndex)
        
        return self._seqReturn(subsampleFunc)
    
    # .................................................................................................................
    
    def norm(self, order=np.inf):
        
        # Update channel size, since norm-ing will reduce to 1 channel
        lastWidth, lastHeight, lastChannels = self.getDimensions()        
        self._recordDimensions(lastWidth, lastHeight, 1)
        
        # Handle incorrect input (single channel) by passing the incoming frame through
        if lastChannels == 1:
            print("")
            print(self._name)
            print("Warning:")
            print("norm() operation cannot be applied to an input with 1 channel!")
            print("Skipping norm operation...")
            
            def noNorm(inFrame):
                return inFrame
        
            return self._seqReturn(noNorm)
        
        def norm_(inFrame, inOrder):
            # Numpy: np.linalg.norm(x, ord, axis)
            return np.uint8(np.linalg.norm(inFrame, ord=inOrder, axis=2))
        
        normFunc = partial(norm_, inOrder=order)
        
        return self._seqReturn(normFunc)
    
    # .................................................................................................................
    
    def outputAs(self, outputKeyName):
        
        def outputAs_(inFrame, keyName):            
            self._seqDictionary[keyName] = inFrame.copy()            
            return inFrame
        
        saveFunc = partial(outputAs_, keyName=outputKeyName)
        
        return self._seqReturn(saveFunc)
    
    # .................................................................................................................
    
    def replaceResource(self, storage_name, updated_value):
        
        # Make sure we have an existing storage spot
        if storage_name not in self._resources:
            print("")
            print("Error: {}".format(self._name))
            print("  Storage name", storage_name, "does not exist!")
            print("")
            raise AttributeError
         
        # Assuming everything went well, so store the new value. May run into 'copying' issues though...
        self._resources[storage_name] = updated_value
    
    # .................................................................................................................
    
    def _seqReturn(self, funcRef):
        
        if self._setSequence:
            self._functionSequence.append(funcRef)
            return len(self._functionSequence)
        else:
            return funcRef
        
    # -----------------------------------------------------------------------------------------------------------------
    #%% Stack functions
    
    # .................................................................................................................
    
    def _buildNewStack(self, stackSize):
        
        # Set frame and stack dimensions (using lastWidth/lastHeight/lastChannel buffers)
        frameDimensions = list(self._efficientDimensions())
        stackDimensions = [stackSize] + frameDimensions

        try:
            # Allocate space for the new (empty) stack
            emptyFrame = np.zeros(frameDimensions, dtype=np.uint8)
            newStack = np.full(stackDimensions, emptyFrame, dtype=np.uint8)
            
        except ValueError:
            print("")
            print(self._name)
            print("Error building storage!")
            print("Input size is likely set incorrectly.")
            print("An example input can be set using .startSequence() function.")
            print("For example:")
            print("  .startSequence(exampleImage)")
            print("  .startSequence(VideoCaptureObject)")
            print("  .startSequence(OtherFrameLabObject)")
            FrameLab._quit()
        
        # Add stack, sizing info and initial pointer to the appropriate lists
        stackIndex = len(self._stackList)
        self._stackSizeList.append(stackSize)
        self._stackList.append(newStack)
        self._pointerList.append(0)        
        
        return stackIndex, stackSize
    
    # .................................................................................................................
    
    # Function for adding frames to a framestack
    def _addToStack(self, inFrame, stackIndex, copyFrame=True):
        
        # Update the stack pointer
        pointer = self._advancePointer(stackIndex)
        
        try:
            # Store the new frame at the updated pointer value
            self._stackList[stackIndex][pointer] = inFrame.copy() if copyFrame else inFrame
            
        except ValueError:
            # Errors can occur when frame shapes don't match up to expectations
            print("")
            print(self._name)
            print("Error storing frame data!")
            print("  Trying to copy image of shape:", inFrame.shape)
            print("  into frame stack with shape:", self._stackList[stackIndex].shape[1:])
            print("  Probably need to use/adjust setInput() FrameLab function to use proper image dimensions!")
            FrameLab._quit()
        
    # .................................................................................................................
    
    def _advancePointer(self, listIndex):
        
        # Get some convenience variables
        stackSize = self._stackSizeList[listIndex]
        currPointer = self._pointerList[listIndex]
        
        # Calculate new pointer and replace old value
        newPointer = (currPointer + 1) % stackSize
        self._pointerList[listIndex] = newPointer
        
        return newPointer
    
    # .................................................................................................................
    
    def _returnStackFrame(self, stackIndex, relIndex=0):
        
        # Get convenience variables
        stackSize = self._stackSizeList[stackIndex]
        currPointer = self._pointerList[stackIndex]
        
        frameIndex = FrameLab._wrapPointer(currPointer, -relIndex, stackSize)
        return self._stackList[stackIndex][frameIndex]
    
    # -----------------------------------------------------------------------------------------------------------------
    #%% Helper functions
    
    # .................................................................................................................
    
    def getDimensions(self):        
        return self._historyWHC[-1]
    
    # .................................................................................................................
    
    def _recordDimensions(self, *args):
        
        # Get length for convenience
        argLength = len(args)
        
        # Initialize outputs
        prevDimensions = self._historyWHC[-1] if len(self._historyWHC) > 0 else (None, None, None)
        newWidth, newHeight, newChannels = prevDimensions
        
        # Give an error if we get the wrong number of inputs
        if argLength not in (1,2,3):
            print("")
            print("Error recording dimensions!")
            print("Input must be either (channel inputs are optional):")
            print(" - width, height, channels")
            print(" - (width, height, channels)")
            print(" - [width, height, channels]")
            print(" - dimensions from another FrameLab object")
            print("")
            raise TypeError
        
        # For separated inputs
        if argLength > 1:
            newWidth = args[0]
            newHeight = args[1]
            newChannels = args[2] if argLength > 2 else newChannels
            
        # For the case of single-entry inputs
        if argLength == 1:
            
            # For the case of tuples or lists
            argVal = args[0]
            if type(argVal) in [tuple, list]:
                newWidth = argVal[0]
                newHeight = argVal[1]
                newChannels = argVal[2] if len(argVal) > 2 else newChannels
            
            # For the case of inputing another FrameLab object
            if type(argVal) is FrameLab:
                newWidth, newHeight, newChannels = argVal.getDimensions()
            
        # If all went well, we'll record the new dimensions in the internal history variable
        self._historyWHC.append((newWidth, newHeight, newChannels))
    
    # .................................................................................................................
    
    def _efficientDimensions(self):
        lastWidth, lastHeight, lastChannels = self.getDimensions()
        dimensions = (lastHeight, lastWidth, lastChannels) if lastChannels > 1 else (lastHeight, lastWidth)
        return dimensions
        
    # .................................................................................................................
    
    def _matchToSelfDimensions(self, inImage, imageName="image", operationName="processing"):
        
        # Get current image size to compare to incoming image
        lastWidth, lastHeight, lastChannels = self.getDimensions()
        
        # Figure out if input/self are color images
        selfIsColor = (lastChannels > 1)
        selfIsGrayscale = not selfIsColor
        inputIsColor = FrameLab.isColor(inImage)
        inputIsGrayscale = not inputIsColor
        
        # Convert input image to grayscale if the previous 'self' frames are grayscale
        if inputIsColor and selfIsGrayscale:
            inImage = FrameLab.toGray(inImage)
            
        # Convert grayscale image to BGR if the incoming frames are BGR
        if selfIsColor and inputIsGrayscale:
            print("")
            print("WARNING:")
            print("  Grayscale ", imageName, " converted to BGR image to perform ", operationName, "!", sep="")
            inImage = FrameLab.toBGR(inImage)
            
        # Match mask size to the image size
        inputDimensions = inImage.shape
        selfDimensions = self._efficientDimensions()        
        if inputDimensions != selfDimensions:            
            print("")
            print("WARNING:")
            print(" ", imageName.capitalize(), "has been resized to match frame processing!")
            dsizeFormat = (selfDimensions[1], selfDimensions[0])
            inImage = cv2.resize(inImage, dsize=dsizeFormat)
        
        return inImage
    
    # .................................................................................................................

    def retrieveOutput(self, keyName):
        return self._seqDictionary[keyName]
    
    # .................................................................................................................
    
    def _error_out(self, exception_type, exception_message=""):
        out_msg = self._name if exception_message == "" else (exception_message + "\n({})".format(self._name))
        raise exception_type(out_msg)
    
    # .................................................................................................................

    @staticmethod
    def _wrapPointer(pointer, indexShift, stackSize):
        return (pointer + indexShift) % stackSize

    # .................................................................................................................
    
    @staticmethod
    def _quit():
        import os
        from inspect import currentframe, getframeinfo
        
        frameinfo = getframeinfo(currentframe())
        print("")
        print("File:", frameinfo.filename)
        print("Line:", frameinfo.lineno)
        print("")        
        if any('SPYDER' in name for name in os.environ): raise SystemExit()  # Crash to stop that menacing spyder IDE
        quit()  # Works nicely everywhere else! 
        
    
    # .................................................................................................................
    
    @staticmethod
    def toGray(inFrame):
        return cv2.cvtColor(inFrame, cv2.COLOR_BGR2GRAY)
    
    # .................................................................................................................
    
    @staticmethod
    def toBGR(inFrame):
        return cv2.cvtColor(inFrame, cv2.COLOR_GRAY2BGR)
    
    # .................................................................................................................    

    @staticmethod
    def isColor(npImage):        
        return (len(npImage.shape) > 2)

    # .................................................................................................................
    
    @staticmethod
    def cropped_mask(mask_image):
        
        # Convert to a single channel if needed
        if len(mask_image.shape) > 2:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        
        # Find extent of mask so we can use co-ordinates for cropping
        _, cropContour, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cropX, cropY, cropWidth, cropHeight = cv2.boundingRect(cropContour[0])
        
        # Create some convenience variables for the crop indexing
        cropX1, cropX2 = cropX, cropX + cropWidth
        cropY1, cropY2 = cropY, cropY + cropHeight
        crop_points = (cropY1, cropY2, cropX1, cropX2)
        
        # First crop the input mask
        return mask_image[cropY1:cropY2, cropX1:cropX2], crop_points

    # .................................................................................................................
    
    def crop_offsets(self):
        return self._crop_points[2], self._crop_points[0]

    # .................................................................................................................
    
    def changed(self):
        return self._changed

    # .................................................................................................................
    
    def replaceBG(self, newBg):
        
        # Warning if new background doesn't match old background
        if newBg.shape != self._bgImage.shape:
            print("")
            print("New background dimensions do not match original background!")
            print("Old:", self._bgImage.shape)
            print("New:", newBg.shape)
        
        self._bgImage = newBg.copy()
    
    # -----------------------------------------------------------------------------------------------------------------
    #%% Debugging functions
    
    # .................................................................................................................
    
    def state(self):
        
        numUpdates = self._countUpdates
        lastFuncIndex = self._funcIndex
        
        lastFunction = self._functionSequence[lastFuncIndex]
        
        if type(lastFunction) is partial:
            lastFunctionName = lastFunction.func.__name__
        else:
            lastFunctionName = lastFunction.__name__
            
        print("")
        print("Number of full update cycles:", numUpdates)
        print("Stopped at function:", lastFunctionName, "()")
        print("(Function index ", lastFuncIndex, ")", sep="")
    
    # .................................................................................................................
    
    def displaySequence(self, numToDisplay=None):
        
        # Warning and quit if storage isn't enabled
        if len(self._intermediates) == 0:
            print("")
            print("No intermediate frames were stored!")
            print("Need to enable storage to display intermediate frames using:")
            print("")
            print("  .endSequence(storeIntermediates=True)")
            return
        
        # Set maximum indices
        seqStart = 0
        seqEnd = len(self._functionSequence)
        
        # Initialize display to show all images
        startVal = seqStart
        stopVal = seqEnd
        
        # Allow for more specific frame display selections
        if numToDisplay is not None:
            
            # CHECK FOR LIST INPUTS TO SELECT SPECIFIC FRAMES!
            
            # Allow positive/negative indexing to display frames
            if numToDisplay > 0:
                startVal = seqStart
                stopVal = min(numToDisplay, seqEnd)
            else:
                stopVal = seqEnd
                startVal = max(seqStart, seqEnd + numToDisplay)
        
        # Show input frame
        cv2.imshow("Input", self._intermediates[0])
        cv2.waitKey(200)
        
        # Display intermediate frames
        for idx in range(startVal, stopVal):
            eachFunc = self._functionSequence[idx]
            funcName = eachFunc.func.__name__ if type(eachFunc) is partial else eachFunc.__name__
            intermIndex = 1 + idx
            cv2.imshow(str(intermIndex) + " - " + funcName, self._intermediates[intermIndex])
            cv2.waitKey(200)
        
        # Can't q/esc quit out of these windows, so better give the user some help...
        print("Use closeall() or cv2.destroyAllWindows() to close images.")
        
    # .................................................................................................................
    
    def collage(self, dimensionsWH=(1280, 720), maxCols=4):
            
        # Warning and quit if storage isn't enabled
        numIntermediates = len(self._intermediates)
        if numIntermediates <= 0:
            print("")
            print("No intermediate frames were stored!")
            print("Need to enable storage to display intermediate frames using:")
            print("")
            print("  .endSequence(storeIntermediates=True)")
            return np.zeros((50,50,3), dtype=np.uint8)
        
        # Figure out how many rows and columns to have in the collage
        numCols = min(maxCols, numIntermediates)
        numRows = int(1 + np.floor((numIntermediates-1)/numCols))
        
        # Set the frame size for each piece of the collage
        maxWidth = int(np.floor(dimensionsWH[0] / numCols))
        maxHeight = int(np.floor(dimensionsWH[1] / numRows))
        
        # Set the fill color for the border
        borderFill = (50, 50, 50)
        
        # Get scaled copies of each intermediate frame to use in the collage
        collageImages = []
        for idx, eachInterFrame in enumerate(self._intermediates):
            
            # Get the frame data and it's dimensions        
            interHeight, interWidth = eachInterFrame.shape[0:2]
            
            # Resize the frame to fit the collage
            scaleVal = min(maxWidth/interWidth, maxHeight/interHeight, 1)
            scaledFrame = cv2.resize(eachInterFrame, dsize=None, fx=scaleVal, fy=scaleVal)
            
            # Figure out the size of the borders on the left/right of the image (if any)
            widthBorder = maxWidth - scaledFrame.shape[1]
            leftBorder = int(widthBorder/2)
            rightBorder = widthBorder - leftBorder
            
            # Figure out the size of the borders on the top/bottom of the image (if any)
            heightBorder = maxHeight - scaledFrame.shape[0]
            topBorder = int(heightBorder/2)
            botBorder = heightBorder - topBorder
            
            # Convert to BGR if needed
            numChannels = scaledFrame.shape[2] if len(scaledFrame.shape) > 2 else 1
            if numChannels < 3:
                scaledFrame = FrameLab.toBGR(scaledFrame)
    
            # Place borders around each image so that all scaled frames have the same height/width
            collageFrame = cv2.copyMakeBorder(scaledFrame, 
                                              topBorder, 
                                              botBorder, 
                                              leftBorder, 
                                              rightBorder, 
                                              borderType=cv2.BORDER_CONSTANT,
                                              value=borderFill)
            
            # Print function name in top left corner of the images
            if idx == 0:
                funcName = "0: Input"
            else:                
                eachFunc = self._functionSequence[max(0, idx-1)]
                funcName = str(idx) + ": " + eachFunc.func.__name__ if type(eachFunc) is partial else eachFunc.__name__
            cv2.putText(collageFrame, funcName, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)
            #cv2.putText(img, text, pos, font, scale, col, thic)
            
            # Finally, draw a rectangle around image border to visually separate collage images
            cv2.rectangle(collageFrame, (-10,-10), (maxWidth-1, maxHeight-1), (0,0,0), 1)
            
            # Add frame to the collage set
            collageImages.append(collageFrame)
        
        # Add blank frames if images don't completely fill out collage
        blankFrame = np.full((maxHeight, maxWidth, 3), borderFill, dtype=np.uint8)
        numBlanks = numCols*numRows - numIntermediates
        for idx in range(numBlanks):
            collageImages.append(blankFrame.copy())
        
        # Stack frames, row-by-row
        rowImages = [np.hstack(collageImages[(rowIdx*numCols):((1+rowIdx)*numCols)]) for rowIdx in range(numRows)]
        
        # Create final collage image by vertically stacking row images
        collageOut = np.vstack(rowImages)
        
        return collageOut
    
    # .................................................................................................................
    
    def blame(self, iterations=100, exampleFrame=None):
        
        # Set up functions for getting the process timing
        tickFreq = cv2.getTickFrequency()
        calculate_time_ms = lambda t1, t2: 1000*(t2 - t1)/tickFreq
        
        print("")
        print("-------------------------------------------------------------------")
        print("Blame report:", self._name)
        print("-------------------------------------------------------------------")
        
        # Generate an example frame if one isn't given
        if exampleFrame is None:
            print("")
            print("Example frame not supplied. Using an image of random colors.")
            print("This may affect processing times!")
            inWidth, inHeight, inChannels = self._historyWHC[0]     # Get input dimensions
            exampleFrame = np.random.randint(0, 256, (inHeight, inWidth, inChannels), dtype=np.uint8)
            
        # Allocate storage for timing
        procTimes = np.zeros((len(self._functionSequence), iterations), dtype=np.float)
        startTime = 0
        endTime = 0
        
        # Some feedback
        print("")
        print("Beginning frame processor timing ({:.0f} iterations)...".format(iterations))
        
        for k in range(iterations):
        
            # Repeatedly apply each function in the func. sequence on the result from the previous function
            prevFrame = exampleFrame.copy()
            for idx, eachFunction in enumerate(self._functionSequence):
                
                # Start process timer
                startTime = cv2.getTickCount()
                
                prevFrame = eachFunction(prevFrame)
                
                # Stop process timer and add to accumulator
                endTime = cv2.getTickCount()
                procTimes[idx, k] += calculate_time_ms(startTime, endTime)
                
                # Allow functions in the function list to break the update loop
                if self._requestBreak: break # Will leave zero times for all following functions!
            
        
        # Get processing time stats
        avgProcTimes = np.mean(procTimes, axis=1)
        stDevs = np.std(procTimes, axis=1)
        timesAsPercents = 100*avgProcTimes/np.sum(avgProcTimes)
        
        # Get totals stats
        totalTimes = np.sum(procTimes, axis=0)
        totalTimeAvg = np.mean(totalTimes)      # Should be identical to sum of proc. averages
        totalTimeStDev = np.std(totalTimes)     # Should be very different from sum of proc. stDevs (n-th root?)
        totalAsPercent = 100*totalTimeAvg/np.sum(avgProcTimes)      # Sanity check, should be 100%
        
        # Some useful info before printing...
        funcNames = [ef.func.__name__ if type(ef) is partial else ef.__name__ for ef in self._functionSequence]
        longestName = max([len(eachName) for eachName in funcNames])
        
        # Print out average processing times
        print("")
        print("Total run time (ms):", "{:.3f}".format(np.sum(totalTimes)))
        print("Average processing times based on", iterations, "iterations (ms)")
        for idx, eachFunc in enumerate(self._functionSequence):
            funcName = funcNames[idx]
            timeString = "{:.3f} +/- {:.3f} ({:.1f}%)".format(avgProcTimes[idx], stDevs[idx], timesAsPercents[idx])
            print("  ", funcName.ljust(longestName), ": ", timeString, sep="")
        totalString = "{:.3f} +/- {:.3f} ({:.1f}%)".format(totalTimeAvg, totalTimeStDev, totalAsPercent)
        print("  ", "TOTAL".ljust(longestName), ": ", totalString, sep="")
        
        # Bit of a warning
        print("")
        print("Note:")
        print("These timing numbers do not account for the time needed")
        print("to copy frames through the processing sequence or the")
        print("python for loop. Actual process timing will be longer!")
        
        print("")
        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")
        
        return totalTimeAvg, totalTimeStDev

    # .................................................................................................................


# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================


class Variable_FrameLab(FrameLab):
    
    def __init__(self, name="Unnamed Variable Frame Processor"):
        super().__init__(name)
        
    # .................................................................................................................
    
    def resize(self, dimensionsWH_control=None, scaleXY_control=None):
        
        # WARNING:
        # This function can mess up any proceeding functions expecting specific input frame dimensions!
        # This can be fixed in the future, but for now be careful!
        
        # Convert inputs to Control objects if needed
        dimensionsWH_control = self._convert_to_Control(dimensionsWH_control)
        scaleXY_control = self._convert_to_Control(scaleXY_control)
        
        def resize_(inFrame, dimWH_control, facXY_control):
            
            # Get control settings for convenience
            dimWH = dimWH_control.value
            facXY = facXY_control.value
            
            # If both controls are none, just return the input frame un-modified
            if (dimWH is None) and (facXY is None):
                return inFrame
            
            # Return dimension based resizing if provided
            if dimWH is not None:
                dimWH = (dimWH, dimWH) if type(dimWH) is int else dimWH                
                return cv2.resize(inFrame, dsize=dimWH)
            
            # Return scale factor based resizing if provided
            if facXY is not None:
                facXY = (facXY, facXY) if type(facXY) is float else facXY
                return cv2.resize(inFrame, dsize=None, fx=facXY[0], fy=facXY[1])
            
            # If we didn't already return, then both inputs are none and we should just return the input unmodified
            return inFrame
        
        resizeFunc = partial(resize_, dimWH_control=dimensionsWH_control, facXY_control=scaleXY_control)
        
        return self._seqReturn(resizeFunc)
    
    # .................................................................................................................
    
    def blur(self, 
             kernelSize_control=(3,3), 
             kernelSigma_control=(0,0)):
        
        # Convert inputs to Control objects if needed
        kernelSize_control = self._convert_to_Control(kernelSize_control)
        kernelSigma_control = self._convert_to_Control(kernelSigma_control)
        
        # Create a function that can access controls before applying blur
        def variable_blur(inFrame, ksize_control, kSigma_control):            
            # Get control values
            blurSize = ksize_control.value
            sigmaSize = kSigma_control.value
            
            # Disable blurring if the kernel is too small
            if max(blurSize) < 2:
                return inFrame
            
            # OpenCV: cv2.GaussianBlur(src, ksize, sigmaX, sigmaY)
            return cv2.GaussianBlur(inFrame, ksize=blurSize, sigmaX=sigmaSize[0], sigmaY=sigmaSize[1])
            
        # Build output function
        blurFunc = partial(variable_blur, 
                           ksize_control=kernelSize_control, 
                           kSigma_control=kernelSigma_control)
        
        return self._seqReturn(blurFunc) 

    # .................................................................................................................

    def morphology(self, 
                   kernelSize_control=(3,3), 
                   kernelShape_control=cv2.MORPH_RECT,
                   operation_control=cv2.MORPH_CLOSE):
        
        # Convert inputs to Control objects if needed
        kernelSize_control = self._convert_to_Control(kernelSize_control)
        kernelShape_control = self._convert_to_Control(kernelShape_control)
        operation_control = self._convert_to_Control(operation_control)
        
        def variable_morphology(inFrame, kernelSize_control, kernelShape_control, operation_control):
            
            # Get the parameters needed to create the kernel
            kernelShape = kernelShape_control.value
            kernelSize = kernelSize_control.value  
            
            # Disable morphology if the kernel is too small
            if max(kernelSize) < 2:
                return inFrame
            
            # Generate the morphological kernel based on control values
            new_kernel = cv2.getStructuringElement(shape = kernelShape, 
                                                   ksize = kernelSize)
            
            # OpenCV: cv2.morphologyEx(src, op, kernel)
            return cv2.morphologyEx(inFrame, op = operation_control.value, kernel = new_kernel)
        
        # Build output function including controls
        morphFunc = partial(variable_morphology, 
                            kernelSize_control = kernelSize_control, 
                            kernelShape_control = kernelShape_control,
                            operation_control = operation_control)

        return self._seqReturn(morphFunc)
    
    # .................................................................................................................
    
    def threshold(self, thresholdLevel_control=127):
        
        # Convert inputs to Control objects if needed
        thresholdLevel_control = self._convert_to_Control(thresholdLevel_control)
        
        # Function for getting the 1-index return argument (frame data) from the OpenCV function
        def variable_threshold(inFrame, threshCtrl):
            
            # Get thresholding value
            new_thresh = thresholdLevel_control.value
            
            # Disable thresholding if the threshold is too small
            if new_thresh < 1:
                return inFrame
            
            # OpenCV: cv2.threshold(src, thresh, maxval, type)
            return cv2.threshold(inFrame, thresh=new_thresh, maxval=255, type=cv2.THRESH_BINARY)[1]
        
        threshFunc = partial(variable_threshold, threshCtrl=thresholdLevel_control)

        return self._seqReturn(threshFunc)

    # .................................................................................................................
    
    def diffWithSelf(self, backwardStep_control=1):
        
        # Convert inputs to Control objects if needed
        backwardStep_control = self._convert_to_Control(backwardStep_control)
        
        # Figure out how deep the stack needs to be
        maxStep = backwardStep_control.max_value
        maxStep = backwardStep_control.value if maxStep is None else maxStep
        
        # Update minimum stack sizing requirements, since self difference requires at least 2 frames
        minSize = 1 + maxStep    
        stackIndex, _ = self._buildNewStack(minSize)
        
        # Function for getting an absolute difference 
        def variable_diffWithSelf(inFrame, stackIndex, backStep_control):
        #def getDiff(inFrame, stackIndex=0, backStep=1):
            
            # Add inFrame to the stack before performing difference
            self._addToStack(inFrame, stackIndex)
            
            # Get the backward step control value
            back_step = backStep_control.value
            
            # Disable backward step if the step is too low
            if back_step < 1:
                return inFrame
            
            # OpenCV: cv2.absdiff(src1, src2)
            return cv2.absdiff(inFrame, self._returnStackFrame(stackIndex=stackIndex, 
                                                               relIndex=back_step))
        
        selfDiffFunc = partial(variable_diffWithSelf, stackIndex=stackIndex, backStep_control=backwardStep_control)
        
        return self._seqReturn(selfDiffFunc)
    
    # .................................................................................................................
    
    def backSum(self, numToSum_control=1):
        
        # Convert inputs to Control objects if needed
        numToSum_control = self._convert_to_Control(numToSum_control)
        
        # Figure out how deep the stack needs to be
        maxSize = numToSum_control.max_value
        maxSize = numToSum_control.value if maxSize is None else maxSize
        
        # Set minimum stack size requirement to sum enough frames
        minSize = 1 + maxSize
        stackIndex, _ = self._buildNewStack(minSize)
                
        def variable_backSum(inFrame, listIndex, framesToSum_control):
            
            # Add inFrame to the stack before performing summation
            self._addToStack(inFrame, listIndex)
            
            # Get the backward sum index from the controls
            backward_index = framesToSum_control.value
            
            # Disable backward sum if the index is too low
            if backward_index < 1:
                return inFrame
            
            # Get convenient variables
            stackSize = self._stackSizeList[listIndex]
            startPoint = self._pointerList[listIndex]
            endPoint = self._wrapPointer(startPoint, 0 - backward_index, stackSize)
            
            # Figure out which stack indices to include in the summation
            indexingVector = np.arange(stackSize)
            if endPoint > startPoint:
                selectionVector = np.logical_or(indexingVector <= startPoint, indexingVector >= endPoint)
            else:
                # endPoint < startPoint
                selectionVector = np.logical_and(indexingVector >= endPoint, indexingVector <= startPoint)
                
            sumFrame = np.sum(self._stackList[listIndex][selectionVector], axis=0, dtype=np.uint16)            
            return np.uint8(np.clip(sumFrame, 0, 255))
        
        sumFunc = partial(variable_backSum, listIndex=stackIndex, framesToSum_control=numToSum_control)
        
        return self._seqReturn(sumFunc)

    # .................................................................................................................
    
    def mask(self, maskImage_control):
        
        # Convert inputs to Control objects if needed
        maskImage_control = self._convert_to_Control(maskImage_control)
        
        def variable_mask(inFrame, mask_control):
            
            # First check if the mask needs to be resized
            mask_image = mask_control.value
            maskWH = mask_image.shape[0:2][::-1]
            imageWH = inFrame.shape[0:2][::-1]
            
            # Resize if the mask doesn't match the incoming image dimensions
            if np.any(maskWH != imageWH):
                print("")
                print("Warning: {}".format(self._name))
                print("  Mask size is mismatched with incoming image!")
                print("  Incoming size: {} x {}".format(*imageWH))
                print("  Mask size:     {} x {}".format(*maskWH))
                print("  Mask will be resized!")
                
                # Resizing can mess up mask images, so make sure to re-threshold the image
                mask_image = cv2.resize(mask_image, dsize=imageWH)
                mask_image = cv2.threshold(mask_image, 200, 255, cv2.THRESH_BINARY)[1]
                mask_control.update(mask_image)
                
            # OpenCV: cv2.bitwise_and(src1, src2)
            return cv2.bitwise_and(src1=inFrame, src2=mask_image)

        maskFunc = partial(variable_mask, mask_control=maskImage_control)
        
        return self._seqReturn(maskFunc)

    # .................................................................................................................

    @staticmethod
    def _convert_to_Control(input_data, feedback=False):
        
        # Convert incoming data to a control variable
        if type(input_data) != Variable_FrameLab.Control:       
            
            # Provide feedback, if enabled
            if feedback:
                print("")
                print("Input data is type ({}), converting to Control object!".format(type(input_data)))
            
            return Variable_FrameLab.Control(name = "unnamed",
                                             value = input_data)
        return input_data

    # .................................................................................................................
    # .................................................................................................................

    class Control:
        
        def __init__(self, name="", *, 
                     value=None, min_value=None, max_value=None, 
                     update_func=None, 
                     report_func = None,
                     initial_input=None):
            
            # Store handy values
            self._name = name
            self._value = value
            self._min_value = min_value
            self._max_value = max_value
            self._initial_value = initial_input
            
            # Store custom update function (if provided)
            self._update_func = lambda x: x          
            if update_func is not None:
                self.change_update_function(update_func)
                
            # Store custom reporting function (if provided)
            self._report_func = lambda x: x
            if report_func is not None:
                self.change_report_function(report_func)
            
            # If an input is provided, pass it through the update function and store it as the initial value
            if initial_input is not None:
                self.update(initial_input)
            
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        @property
        def name(self):
            return self._name
            
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        @property
        def value(self):
            return self._value
        
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        def report(self):
            return self._report_func(self._value)
        
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        def update(self, new_value):
            # Handle input bounding, if needed
            new_value = new_value if self._max_value is None else min(new_value, self._max_value)
            new_value = new_value if self._min_value is None else max(new_value, self._min_value)
            
            # Update internal value
            self._value = self._update_func(new_value)
            
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        def update_direct(self, new_value):
            
            # Update internal value directly (no update function)
            self._value = new_value
        
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        def update_from_window_trackbar(self, window_ref):
            
            # Assume the trackbar was set up with trackbar config
            val_changed, new_val = window_ref.readTrackbar(self._name)
            if val_changed:
                self.update(new_val)
                
            return val_changed
        
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            
        def change_update_function(self, new_function):
            self._update_func = new_function
            
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        def change_report_function(self, new_function):
            self._report_func = new_function
            
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        def trackbar_config(self):
            
            trackbar_config = {"bar_name": self._name,
                               "start_value": self._initial_value,
                               "max_value": self._max_value}
            
            return trackbar_config
            
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        @property
        def min_value(self):
            return self._min_value
        
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        @property
        def max_value(self):
            return self._max_value
            
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        @property
        def min_max_values(self):
            return self._min_value, self._max_value
        
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
        
# ---------------------------------------------------------------------------------------------------------------------
#%% Testing
    
if __name__ == "__main__":
    
    closeall = cv2.destroyAllWindows
    
    
    videoObj = cv2.VideoCapture(0)#"/home/eo/Desktop/PythonData/Shared/videos/dtb2.avi")
    _, bgImg = videoObj.read()
    

    # Not a practical processor. Just for example
    gg = FrameLab("Glow Tar")    
    gg.startSequence(videoObj)

    gg.resize((500,500))
    gg.blur((5,5))
    gg.blur((11,11))
    gg.blur((21,21))
    gg.outputAs("Dummy1")
    gg.grayscale()
    #gg.diffWithBG(bgImg)
    gg.diffWithSelf(2)
    gg.outputAs("Dummy2")
    #gg.norm()  # Much slower than grayscale
    gg.backSum(1)    
    gg.backSum(2)
    gg.backSum(3)
    gg.backSum(4)
    gg.backSum(5)
    #gg.backSum(6)
    #gg.outputAs("Dummy3")
    #gg.backSum(7)
    gg.morphology((5,5))
    gg.andWithSelf(3)
    gg.andWithSelf(2)
    gg.andWithSelf(1)
    
    gg.endSequence(storeIntermediates=True)
    
    
    # Blame example
    gg.blame(150)#, exampleFrame=bgImg)

    
    
    # Create window with control bars
    def blankFunc(*args, **kwargs): return
    dispWin = cv2.namedWindow("Display")
    cv2.createTrackbar("blurSize", "Display", 5, 25, blankFunc)
    
    frameCount = 0
    while videoObj.isOpened():
        
        recFrame, inFrame = videoObj.read()
        
        if not recFrame:
            print("No more frames!")
            break
        frameCount += 1
        
        procFrame = gg.update(inFrame)
        
        cv2.imshow("Dummy 1", gg.retrieveOutput("Dummy1"))
        cv2.imshow("Dummy 2", gg.retrieveOutput("Dummy2"))
        cv2.imshow("Collage", gg.collage(dimensionsWH=(1280,720), maxCols=7))
        cv2.imshow("Output Frame", procFrame)

        
        # Get key press values
        keyPress = cv2.waitKey(1) & 0xFF
        if (keyPress == ord('q')) | (keyPress == 27):  # q or Esc key to close window
            print("")
            print("Key pressed to stop!")
            break
    
    # Clean up this window
    videoObj.release()
    cv2.destroyAllWindows()
    
    
# ---------------------------------------------------------------------------------------------------------------------
#%% Scrap



