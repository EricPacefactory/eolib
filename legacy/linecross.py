
from collections import namedtuple
import numpy as np
import cv2






# Class for handling line counter objects
class LineHandler:
    
    
    # Function for getting the text location for a LineCounter
    @staticmethod
    def locateText(line, textDistance=20, exampleText="00"):
        
        # Get position of text by rotating line vector by 90 degrees and placing it next to line start point
        rotVec = np.array((line.vec[1], -line.vec[0]))      # Swap minus/plus sign for +/- 90 degree rotation
        rotVec = rotVec * (1 / np.linalg.norm(rotVec))      # Normalize vector
        textShift = line.startPoint + textDistance*rotVec
        
        # Figure out textbox sizing so that the placement is correctly 'centered'
        textSize = cv2.getTextSize(exampleText, LineCounter.textFont, LineCounter.textScale, LineCounter.textThickness)
        textSize = np.array(textSize[0])
        textSize[1] = -textSize[1]
        
        # Include shifting (since text is registered relative to top corner of textbox)
        textPos = textShift - (textSize/2)
        textPos = tuple(textPos.astype(int))
        
        return textPos
    
    #                       ******************************************************
    
    # Function for getting a bounding rectangle for a given LineCounter
    @staticmethod
    def getBoundingRect(line):
        
        # Create the bounding box co-ordinates
        xcenter = (line.startPoint[0] + line.endPoint[0]) * (0.5)
        ycenter = (line.startPoint[1] + line.endPoint[1]) * (0.5)
        width = 2*line.hpad
        height = 2*line.vpad
        
        # Create the transformation matrix
        angle = np.arccos(line.vec[0] / np.linalg.norm(line.vec))
        transformMatrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        
        return LineCounter.BoundingRectangle(xcenter, ycenter, width, height, transformMatrix)
    
    
    #                       ******************************************************    
    
    # Function for drawing lines onto an image
    @staticmethod
    def drawLine(inImage, line, circleRadius=5, overwriteText=None, showText=True):
        cv2.line(inImage, tuple(line.startPoint), tuple(line.endPoint), line.lineColor, line.lineThickness)
        cv2.circle(inImage, tuple(line.startPoint), circleRadius, line.lineColor, -1)
        cv2.circle(inImage, tuple(line.endPoint), circleRadius, line.lineColor, -1)
        
        # Get counter text
        countText = str(line.count).zfill(2)
        if overwriteText is not None:
            countText = str(overwriteText).zfill(2)
        
        if showText:
            cv2.putText(inImage, countText, line.textPosition,
                        LineCounter.textFont, LineCounter.textScale, line.lineColor, LineCounter.textThickness)
        
    #                       ******************************************************

    # Function for drawing LineCounter intersection points onto an image
    @staticmethod
    def drawLiveIntersections(inImage, line):
        # Draw each intersection point back onto an image
        for eachIntersection in line.liveIntersections:            
            cv2.circle(inImage, eachIntersection.point, 6, (0,0,255), -1)
            
    #                       ******************************************************
    
    # Function for drawing recorded intersection events
    @staticmethod
    def drawIntersectionEvent(inImage, line, eventIndex=-1):
        cv2.circle(inImage, line.events[eventIndex].point, 6, (0,0,255), -1)
  
    #                       ******************************************************
    
    # Function for outputting a description of the line segment to use for recording
    @staticmethod
    def toDictionary(line):
        
        myDictionary = {"startPoint": list(line.startPoint),
                        "endPoint": list(line.endPoint),
                        "lineColor": line.lineColor,
                        "vpad": line.vpad,
                        "hpad": line.hpad}
        
        return myDictionary
    
    
    #                       ******************************************************
    
    # Function for finding intersections between the line counter and other line segments
    # The other line segments must have .startpoint and .vec properties 
    @staticmethod
    def getIntersections(line, otherLineSegments, currentTime=None):
        
        # Clear storage variable variable
        intersections = []
        
        # Define some convenience terms
        pa = line.startPoint
        va = line.vec
        
        # Loop through all other lines to look for intersections
        for eachOtherSegment in otherLineSegments:
            
            # Convenience terms
            pb = eachOtherSegment.startPoint
            vb = eachOtherSegment.vec
            objID = eachOtherSegment.objID
            
            # Find cross-product terms
            aXb = np.cross(va, vb)
            
            # Check if the lines are parallel/co-linear
            if aXb == 0:
                # Don't add line to intersection list, so just skip...
                continue
            
            # Pre-calculate the vector 'coefficient' used for both t calculations
            coeffVector = (pb - pa) * (1/aXb)
            
            # Calculate ta and tb to see if the intersection is within the line segments
            ta = np.cross(coeffVector, vb)
            tb = np.cross(coeffVector, va)
            
            # The line segments only intersect if the t values are both between [0, 1]
            if (ta >= 0 and ta <= 1) and (tb >= 0 and tb <= 1):
                
                intersectPt = pa + va*ta
                intersectPt = tuple(intersectPt.astype(int))
                intersectDir = np.sign(aXb) 
                
                if (line.dir == 0) or (intersectDir == line.dir):                
                    intersections.append(line.Intersection(time=currentTime, point=intersectPt, 
                                                           dir=intersectDir, objID=objID))
                
        # Count up the intersections and return the count
        numIntersections =  len(intersections)
        
        return intersections, numIntersections
    
    
    @staticmethod
    def getPathIntersections(line, pathX, pathY):
        

        # First transform the path points using the line-transform mapping
        tpX, tpY = LineHandler.transformPath(line, pathX, pathY)
        
        # Find transformed path boundaries
        pxMin = min(tpX)
        pxMax = max(tpX)
        pyMin = min(tpY)
        pyMax = max(tpY)
        
        # Get line boundary (after transform, the line is horizontal, so it doesn't have a y-extent)
        lxMin = line.tfXs[0]
        lxMax = line.tfXs[1]
        ly = line.tfYs[0]       # Could take either index, both are equal
        
        # Check if the two boundaries are left-right separated
        if (pxMin > lxMax) or (lxMin > pxMax):
            # Not overlaping
            return None
        if not (pyMin < ly < pyMax):
            # Not overlaping
            return None
        
        # If we get here, the boxes do have overlap, so we need to check for intersections
        numPoints = len(tpY)
        
        # Get indices where the transformed path crosses the line
        crossX = [idx for idx in range(numPoints-1) if tpY[idx] < ly < tpY[idx+1]]
        
        # Now check that the x-coordinates contain the line when 
        
        # Finally, find if there is an intersection and where it is located
        
        raise NotImplementedError
        
        return crossX
        
    
    
    # Function for applying a line rotation transform on a set of points (maps the line to be parallel to the x-axis)
    @staticmethod
    def transformPath(line, inPathX, inPathY):
        
        # Create a matrix out of input data, then perform rotation matrix multiply to transform points
        #   - pxy is formed as two ROW vectors, since this simplifies the matrix multplication
        #   - The transformed output is also in ROW vector format
        pxy = np.row_stack((inPathX, inPathY))
        transformedXY = np.dot(line.transform, pxy)
        
        # Extract transformed rows
        #   - After extracting, numpy doesn't seem to store row/column shape info, so tpx and tpy are just 'arrays'
        tpx = transformedXY[0, :]
        tpy = transformedXY[1, :]
        
        return tpx, tpy
    
    
    
    # Function which returns a rotation matrix that rotates a line parallel to the x-axis (i.e. horizontal line)
    @staticmethod
    def getLineTransform(line, sortOutputs=True, tol=1E-5):
        
        # Get a 'simplified' vector from the line (i.e. vector is always mapped to have a positive x value)
        if line.vec[0] < 0:
            svec = -line.vec
        else:
            svec = line.vec
        
        # Get rotation angle of simple vector with sign perserved (so that we can get line crossing directions)
        transformAngle = -1*np.arctan2(svec[1], svec[0])
        
        #print("T-Angle:", transformAngle*180*(1/np.pi))
        
        # Create rotation matrix
        cosVal = np.cos(transformAngle)
        sinVal = np.sin(transformAngle)
        rotMatrix = np.array(((cosVal,-sinVal), (sinVal, cosVal)))        
        
        transformedStartPoint = np.dot(rotMatrix, line.startPoint)
        transformedEndPoint = np.dot(rotMatrix, line.endPoint)
        
        transformedXs = (transformedStartPoint[0], transformedEndPoint[0])
        transformedYs = (transformedStartPoint[1], transformedEndPoint[1])
        
        # Create new (transformed) start and end points to make no funny business has occurred!
        nsp = (transformedXs[0], transformedYs[0])
        nep = (transformedXs[1], transformedYs[1])
        nvec = np.array(nep) - np.array(nsp)
        olen = np.linalg.norm(line.vec)
        nlen = np.linalg.norm(nvec)
        
        # Check that the transformed vector has the same length as the input vector
        if abs(olen - nlen) > tol:
            print("")
            print("ERROR: Something wrong with tranformed vector length!")
            print("")
            raise ArithmeticError
        
        # Check that y-values are inline
        if abs(transformedYs[0] - transformedYs[1]) > tol:
            print("")
            print("ERROR: Something wrong with transformed vector y-values!")
            print("")
            raise ArithmeticError
            
            
        # It may be more useful to have output values sorted
        if sortOutputs:            
            transformedXs = np.sort(transformedXs)
            
            meanY = np.mean(transformedYs)
            transformedYs = (meanY, meanY)
        
        
        return rotMatrix, transformedXs, transformedYs
    
    @staticmethod
    def checkBoundaryOverlap():
        
        
        pass
    
    pass



#                                       **********
#                                       **********
#                                       **********
#                                       **********
#                                       **********
#                                       **********
    
    
class LineCounter:
    
    textFont = cv2.FONT_HERSHEY_SIMPLEX
    textScale = 0.6
    textThickness = 2
    lineThickness = 2
    Intersection = namedtuple("Intersection", ["time", "point", "dir", "objID"])
    BoundingRectangle = namedtuple("BoundingRectangle", ["xcenter", "ycenter", "width", "height", "matrix"])
    handler = LineHandler()
    
    def __init__(self, inStartPoint, inEndPoint, inColor=(0, 255, 0), inVertPad = 5, inHorzPad = 5, inDir=0):
        self.startPoint = np.array(inStartPoint)
        self.endPoint = np.array(inEndPoint)
        self.vec = self.endPoint - self.startPoint  
        self.dir = inDir
        self.lineColor = inColor
        self.count = 0
        self.vpad = inVertPad
        self.hpad = inHorzPad
        self.boundingRect = self.handler.getBoundingRect(self)
        self.textPosition = self.handler.locateText(self)
        self.liveIntersections = []
        self.events = []
        
        
        self.transform, self.tfXs, self.tfYs = self.handler.getLineTransform(self)
        
    #                       ******************************************************
    
    
    def getIntersections(self, otherLineSegments, time=None):
        
        intersections, numIntersections = self.handler.getIntersections(self, otherLineSegments, time)
        
        self.liveIntersections = intersections
        
        # Only update if an intersection occurred
        if numIntersections > 0:
            
            # Add intersection events to the list
            self.events += intersections # Append list
            self.count += numIntersections
        
        return numIntersections
        
    
    '''
    def locateText(self, textDistance=20, exampleText="00"):
        
        # Get position of text by rotating line vector by 90 degrees and placing it next to line start point
        rotVec = np.array((self.vec[1], -self.vec[0]))      # Swap minus/plus sign for +/- 90 degree rotation
        rotVec = rotVec * (1 / np.linalg.norm(rotVec))      # Normalize vector
        textShift = self.startPoint + textDistance*rotVec
        
        # Figure out textbox sizing so that the placement is correctly 'centered'
        textSize = cv2.getTextSize(exampleText, self.textFont, self.textScale, self.textThickness)
        textSize = np.array(textSize[0])
        textSize[1] = -textSize[1]
        
        # Include shifting (since text is registered relative to top corner of textbox)
        textPos = textShift - (textSize/2)
        textPos = tuple(textPos.astype(int))
        
        return textPos
    
    #                       ******************************************************
    '''
    
    '''
    def getBoundingRect(self):
        
        # Create the bounding box co-ordinates
        xcenter = (self.startPoint[0] + self.endPoint[0]) * (0.5)
        ycenter = (self.startPoint[1] + self.endPoint[1]) * (0.5)
        width = 2*self.hpad
        height = 2*self.vpad
        
        # Create the transformation matrix
        angle = np.arccos(self.vec[0] / np.linalg.norm(self.vec))
        transformMatrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        
        return self.BoundingRectangle(xcenter, ycenter, width, height, transformMatrix)
    '''
    
    #                       ******************************************************    
    
    '''
    def drawLine(self, inImage):
        cv2.line(inImage, tuple(self.startPoint), tuple(self.endPoint), self.lineColor, self.lineThickness)
        cv2.circle(inImage, tuple(self.startPoint), 5, self.lineColor, -1)
        cv2.circle(inImage, tuple(self.endPoint), 5, self.lineColor, -1)
        
        cv2.putText(inImage, str(self.count).zfill(2), self.textPosition, self.textFont, self.textScale, self.lineColor, self.textThickness)
    '''
    
    #                       ******************************************************
    
    '''
    # Function for finding intersections between the line counter and other line segments
    # The other line segments must have .startpoint and .vec properties 
    def getIntersections(self, otherLineSegments):
        
        # Clear storage variable variable
        self.intersections = []
        
        # Define some convenience terms
        pa = self.startPoint
        va = self.vec
        
        # Loop through all other lines to look for intersections
        for eachOtherSegment in otherLineSegments:
            
            # Convenience terms
            pb = eachOtherSegment.startPoint
            vb = eachOtherSegment.vec
            
            # Find cross-product terms
            aXb = np.cross(va, vb)
            
            # Check if the lines are parallel/co-linear
            if aXb == 0:
                # Don't add line to intersection list, so just skip...
                continue
            
            # Pre-calculate the vector 'coefficient' used for both t calculations
            coeffVector = (pb - pa) * (1/aXb)
            
            # Calculate ta and tb to see if the intersection is within the line segments
            ta = np.cross(coeffVector, vb)
            tb = np.cross(coeffVector, va)
            
            # The line segments only intersect if the t values are both between [0, 1]
            if (ta >= 0 and ta <= 1) and (tb >= 0 and tb <= 1):
                
                intersectPt = pa + va*ta
                intersectPt = tuple(intersectPt.astype(int))
                intersectDir = np.sign(aXb) 
                
                if (self.dir == 0) or (intersectDir == self.dir):                
                    self.intersections.append(self.Intersection(point=intersectPt, dir=intersectDir))
                
        # Count up the intersections and return the count
        numIntersections =  len(self.intersections)
        self.count += numIntersections
        
        return numIntersections
    '''
    #                       ******************************************************

    '''
    # Function for drawing intersection points onto an image
    def drawIntersections(self, inImage):
        # Draw each intersection point back onto an image
        for eachIntersection in self.intersections:            
            cv2.circle(inImage, eachIntersection.point, 6, (0,0,255), -1)
            #print(eachIntersection.dir)
    '''

    #                       ******************************************************
    
    '''
    # Function for outputting a description of the line segment to use for recording
    def toDictionary(self):
        
        myDictionary = {"startPoint": list(self.startPoint),
                        "endPoint": list(self.endPoint),
                        "lineColor": self.lineColor,
                        "vpad": self.vpad,
                        "hpad": self.hpad}
        
        return myDictionary
    '''

#%% Line transform demo
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    sp = np.random.rand(2)*10
    ep = np.random.rand(2)*10
    
    maxVal = max(sp + ep)
    minVal = min(sp + ep)
    largestVal = 1.5*max(abs(maxVal), abs(minVal))
    
    ox = (sp[0], ep[0])
    oy = (sp[1], ep[1])
        
    ll = LineCounter(sp, ep)
    
    
    tm, tx, ty = LineHandler.getLineTransform(ll)
    
    
    nsp = (tx[0], ty[0])
    nep = (tx[1], ty[1])
    
    
    plt.figure(1)
    plt.plot(ox, oy)
    plt.plot(tx, ty)
    plt.scatter(ox[0], oy[0])
    plt.scatter(tx[0], ty[0])
    
    
    plt.ylim(ymin=-largestVal)
    plt.ylim(ymax=largestVal)
    plt.xlim(xmin=-largestVal)
    plt.xlim(xmax=largestVal)
    
    
    
#%% Path intersection demo
    
    pathx = (np.arange(11) + np.random.rand(11)*3)
    pathy = (np.arange(11)*3 + np.random.rand(11)*5)
    
    plt.plot(pathx, pathy)
    
    pxy = np.row_stack((pathx,pathy))
    tpxy = np.dot(tm, pxy)
    
    plt.plot(tpxy[0,:],tpxy[1,:])
    
    
    res = LineHandler.getPathIntersections(ll, pathx, pathy)
    
    if res is not None:
        print("Overlap!")
        print(res)
    else:
        print("No overlap!")