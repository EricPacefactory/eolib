#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 18:04:23 2018

@author: eo
"""

import cv2
from collections import namedtuple



# ---------------------------------------------------------------------------------------------------------------------
#%% Define classes

class BoundingBox:
    
    # Create a container for storing bounding box data
    _StandardBox = namedtuple("StandardBox", ["xcen", "ycen", "w", "h", "xbase", "ybase"])
    
    # Class-wide calibration function used to convert from raw-box data to calibrated boxes
    _calibrationFunction = None
    
    # .................................................................................................................
    
    def __init__(self, inContour, contourArea=None):
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(inContour)      # TopLeftX, TopLeftY, width, height
        
        # Get centers
        xcen = int(x + (w/2))
        ycen = int(y + (h/2))
        
        # Get base co-ordinates
        xbase = xcen
        ybase = y + h
        
        # Store the object co-ordinates
        self._raw = BoundingBox._StandardBox(xcen, ycen, w, h, xbase, ybase)
        self._cal = BoundingBox._calibrate(self._raw)
        self._tracked = False

        #self._contour = inContour
        #self._convHull = cv2.convexHull(inContour)        
        self._area = cv2.contourArea(inContour) if contourArea is None else contourArea
    
    # .................................................................................................................
    
    def __repr__(self):
        xyval = "({:>4.0f}, {:>4.0f})".format(self._raw.xcen, self._raw.ycen)
        whval = "({:>4.0f}, {:>4.0f})".format(self._raw.w, self._raw.h)
        return "".join(["xycen", xyval, "  ", "wh", whval])
    
    # .................................................................................................................
    
    def getXYCenter(self, useCalibrated=False):
        if useCalibrated: return self._cal.xcen, self._cal.ycen
        return self._raw.xcen, self._raw.ycen
    
    # .................................................................................................................
    
    def getXYBase(self, useCalibrated=False):
        if useCalibrated: return self._cal.xbase, self._cal.ybase
        return self._raw.xbase, self._raw.ybase
    
    # .................................................................................................................
    
    def getWH(self, useCalibrated=False):
        if useCalibrated: return self._cal.w, self._cal.h
        return self._raw.w, self._raw.h
    
    # .................................................................................................................
    
    def getXYWH(self, useCalibrated=False):
        if useCalibrated: return self._cal.xcen, self._cal.ycen, self._cal.w, self._cal.h
        return self._raw.xcen, self._raw.ycen, self._raw.w, self._raw.h
    
    # .................................................................................................................
    
    def getArea(self):
        return self._area
    
    # .................................................................................................................
    
    def setToTracked(self):
        self._tracked = True
    
    # .................................................................................................................
        
    def isTracked(self):
        return self._tracked
    
    # .................................................................................................................
    
    @classmethod
    def setCalibration(cls, caliFunc):
        cls._calibrationFunction = caliFunc
        
    # .................................................................................................................

    # Function which takes an input (binary) image, looks for object contours and creates associated bounding boxes
    @classmethod
    def imageToBBList(cls, searchFrame, minArea=0, maxArea=1E9):
        
        # Get (messy) contours
        _, contourList, _ = cv2.findContours(searchFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Only extract contours with the correct sizing
        bbList = []
        for eachContour in contourList:
            contourArea = cv2.contourArea(eachContour)
            if minArea < contourArea < maxArea:
                bbList.append(cls(eachContour, contourArea))
                
        return bbList
       
    # .................................................................................................................
    
    @classmethod
    def _calibrate(cls, raw):
        
        # In case no calibration function exists, define a simple pass-through function
        if cls._calibrationFunction is None:
            
            # Define a template calibration function which just copies each entry through to the output
            def defaultCaliFunc(rawBox):
                xccal = rawBox.xcen
                yccal = rawBox.ycen
                wcal = rawBox.w
                hcal = rawBox.h
                xbcal = rawBox.xbase
                ybcal = rawBox.ybase                
                return cls._StandardBox(xccal, yccal, wcal, hcal, xbcal, ybcal)
            
            # Store the template function so we don't have to keep re-defining it
            cls._calibrationFunction = defaultCaliFunc
        
        return cls._calibrationFunction(raw)
    
    
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================



# ---------------------------------------------------------------------------------------------------------------------
#%% Define functions


def drawSingleBoundingBox(displayFrame, bBox, color=(0,255,0), thickness=2, padding=0):
    
    # Get bounding box info
    xcen, ycen, w, h = bBox.getXYWH()
    
    # Pre-calculate the width and height shifting
    wOffset = w/2 + padding
    hOffset = h/2 + padding
    
    # Get OpenCV drawing co-ordinates
    topLeft = (int(xcen - wOffset), int(ycen - hOffset))
    botRight = (int(xcen + wOffset), int(ycen + hOffset))
    
    # Draw to screen
    cv2.rectangle(displayFrame, topLeft, botRight, color, thickness)
    
# .....................................................................................................................

def drawBoundingBoxList(displayFrame, bbList, color=(0,255,0), thickness=2, padding=0):
    for eachBox in bbList:
        drawSingleBoundingBox(displayFrame, eachBox, color, thickness, padding)

# .....................................................................................................................



