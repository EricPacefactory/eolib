
import cv2
import numpy as np
from matplotlib import pyplot as plt

class SumTracker:
    
    def __init__(self, inNumFrames, inMaskImage, inTriggerLevels, targetEdge="rising", initialCycleState=0, initialTriggerState=0, triggersPerCycle=1, timeFirstCycle=True, debounceFrames=15, displayInverted=False, name="Tracker"):
        
        # Frame parameters
        self.frameIndex = 0
        self.maskImage = inMaskImage
        self.cropMask, self.cropPts = SumTracker.getCropMask(inMaskImage)
        self.numFrames = inNumFrames
        self.timing = timeFirstCycle
        self.name = name
        self.dispInv = displayInverted
        
        # Debounce parameters
        self.debouceFrames = debounceFrames
        self.debounceWindow = self.frameIndex
        
        # Trigger parameters
        self.triggerCount = 0
        self.triggersPerCycle = triggersPerCycle
        self.triggerLowLevel = min(inTriggerLevels)
        self.triggerHighLevel = max(inTriggerLevels)
        self.startedHigh = initialTriggerState
        
        # Cycle timing storage
        self.currentCycleState = initialCycleState
        self.cycleStartTiming = []
        self.cycleIntervals = []
        
        # Trigger timing storage
        self.currentEdgeState = 0
        self.currentTriggerState = initialTriggerState
        self.triggerLowTiming = []
        self.triggerHighTiming = []
        self.triggerEvents = []
        
        # Create storage space for mask sums and state records
        self.maskSum = np.zeros((self.numFrames), dtype=np.float)
        self.cycleState = np.full(self.numFrames, initialCycleState, dtype=np.bool)
        self.triggerState = np.full(self.numFrames, initialTriggerState, dtype=np.bool)
        
        # Set initial trigger timing values depending on the initial state
        if self.startedHigh:
            self.triggerHighTiming.append(self.frameIndex)
        else:
            self.triggerLowTiming.append(self.frameIndex)            
        self.triggerEvents.append(self.frameIndex)
            
        # Set timing the first cycle then the first cycle start is the initial frame
        if self.timing:
            self.cycleStartTiming.append(self.frameIndex)
        
        # Convert input trigger edge selection into a numerical value for ease of use            
        if (targetEdge == "rising") or (targetEdge == 1):
            self.targetEdge = 1
        elif (targetEdge == "falling") or (targetEdge == -1):
            self.targetEdge = -1
        else:
            print("")
            print("Non-valid trigger edge:", self.targetEdge)
            print("Must be either 'rising' or 'falling'")
            print("")
            raise NotImplementedError
    
    # .................................................................................................................  
    
    # 
    def regionSum(self, inFrame):
        
        # Get crop indices for convenience
        y1 = self.cropPts[0]
        y2 = self.cropPts[1]
        x1 = self.cropPts[2]
        x2 = self.cropPts[3]
        
        # Extract crop area from the input image and mask it
        cropFrame = inFrame[y1:y2, x1:x2]
        maskedImage = cv2.bitwise_and(cropFrame, cropFrame, mask=self.cropMask)
        
        # Store the sum of pixels in the masked area to use as a signal for tracking events
        self.maskSum[self.frameIndex] = np.sum(maskedImage)
        
    # .................................................................................................................  
    
    # 
    def checkCurrentTriggerState(self):
        
        # Get previous trigger state for convenience
        prevState = self.getPrevState()
        
        # Debouce the trigger
        if self.frameIndex > self.debounceWindow:
            
            # Check current level (using hysteresis)
            if prevState:
                currentState = (self.maskSum[self.frameIndex] > self.triggerLowLevel)
            else:
                currentState = (self.maskSum[self.frameIndex] > self.triggerHighLevel)
        
        else:
            # Debouncing, carry the previous level value through
            currentState = prevState
        
        # Record trigger level after debouncing + hysteresis
        self.currentTriggerState = currentState
    
    # .................................................................................................................  
    
    # Function for determining if we're seeing a rising or falling edge (also handles debouncing)
    def checkForTriggerStateChange(self):
        
        # Get state booleans
        currentState = self.currentTriggerState
        prevState = self.getPrevState()
        
        # Figure out if we have a rising or falling edge (or neither if both return false)
        risingEdge = currentState and not prevState     # Currently high, previously low
        fallingEdge = prevState and not currentState    # Currently low, previously high
        
        # Get edge type (Rising: +1, Falling: -1, Steady: 0)
        self.currentEdgeState = int(risingEdge) - int(fallingEdge)
        
        # If edge type matches the target edge type, then record the timing (for deboucing)
        if self.currentEdgeState == self.targetEdge: 
            self.debounceWindow = self.frameIndex + self.debouceFrames
    
    # .................................................................................................................  
    
    # Function for keeping track of trigger events (i.e. rising/falling edges in the tracker signal)
    def manageTriggerEdgeEvents(self):
        
        # Only react to non-steady-state conditions (i.e. currentEdge != 0)
        if self.currentEdgeState != 0:
            
            # Define convenience terms for readability
            currentFrame = self.frameIndex
            self.triggerEvents.append(currentFrame)
            
            # Handle rising edges (+1)
            if self.currentEdgeState > 0:
                # Record 'trigger-high' event timing
                self.triggerHighTiming.append(currentFrame)
                
            # Handle falling edges (-1)
            elif self.currentEdgeState < 0:
                # Record 'trigger-low' event timing
                self.triggerLowTiming.append(currentFrame)
                
            # Some additional book-keeping if the edge matches the target edge type 
            # (i.e. this edge is possiblly the start of a new cycle)
            self.countCycles()
                                        
    # .................................................................................................................  

    # Function that counts cycle from trigger events (useful if 2 or more triggers equals one cycle)
    def countCycles(self):
        
        if self.currentEdgeState == self.targetEdge:
                self.triggerCount += 1
                
                # Count cycles if enough triggers have occurred
                if self.triggerCount >= self.triggersPerCycle:
                    
                    # Record interval if there is more than one start timing
                    if len(self.cycleStartTiming) > 0:
                        self.cycleIntervals.append(self.frameIndex - self.cycleStartTiming[-1])
                    
                    # Record 'cycle-start' event
                    self.cycleStartTiming.append(self.frameIndex)
                    
                    # Toggle the cycle state
                    self.currentCycleState = not self.currentCycleState
                    
                    # Reset trigger count
                    self.triggerCount = 0
    
    # .................................................................................................................  
    
    # Function used to update state data for a cycle tracker
    def update(self, inFrame):
        
        # Get mask sum
        self.regionSum(inFrame)
        
        # Check level
        self.checkCurrentTriggerState()
        
        # Check for state changes
        self.checkForTriggerStateChange()
        
        # Check for trigger edge events and count cycles if needed
        self.manageTriggerEdgeEvents()
        
        # Keep track of the cycle and trigger states as well as the frame index
        self.triggerState[self.frameIndex] = self.currentTriggerState
        self.cycleState[self.frameIndex] = self.currentCycleState
        self.frameIndex += 1
    
    # .................................................................................................................   
    
    # Function for re-running a cycle tracker after-the-fact, using new trigger levels
    def re_eval(self, newTriggerLow, newTriggerHigh):
        
        # Reset the frame index so we can re-evaluate the maskSum data using new trigger levels
        self.frameIndex = 0
        self.debounceWindow = 0
        self.triggerLowLevel = min(newTriggerLow, newTriggerHigh)
        self.triggerHighLevel = max(newTriggerHigh, newTriggerLow)
        
        # Cycle timing storage
        self.currentCycleState = self.cycleState[0]
        self.cycleStartTiming = []
        self.cycleIntervals = []
        
        # Trigger timing storage
        self.currentEdgeState = 0
        self.currentTriggerState = self.triggerState[0]
        self.triggerLowTiming = []
        self.triggerHighTiming = []
        self.triggerEvents = []
        
        # Set initial trigger timing values depending on the initial state
        if self.startedHigh:
            self.triggerHighTiming.append(self.frameIndex)
        else:
            self.triggerLowTiming.append(self.frameIndex)
        self.triggerEvents.append(self.frameIndex)
            
        # Set timing the first cycle then the first cycle start is the initial frame
        if self.timing:
            self.cycleStartTiming.append(self.frameIndex)
        
        # Loop through maskSum data and re-check triggers/cycles
        for k in range(self.numFrames):
            self.checkCurrentTriggerState()
            self.checkForTriggerStateChange()
            self.manageTriggerEdgeEvents()
            self.triggerState[self.frameIndex] = self.currentTriggerState
            self.cycleState[self.frameIndex] = self.currentCycleState
            self.frameIndex += 1
            
        self.name = self.name + "-re_eval"
        self.summary()
    
    # .................................................................................................................  
    
    # Function for generating useful plots after running a cycle tracker
    def summary(self, enableMarkers=True):
    
        # Create a figure for this object
        plt.figure(self.name)
        
        # Plot mask sum in upper plot
        plt.subplot(211)
        plt.plot(self.maskSum)
        plt.plot([0, self.numFrames-1], [self.triggerLowLevel, self.triggerLowLevel])
        plt.plot([0, self.numFrames-1], [self.triggerHighLevel, self.triggerHighLevel])
        
        # Only plot markers if it's enabled, since it can be messy
        if enableMarkers:
            lowMarkersX = self.triggerLowTiming
            lowMarkersY = [self.maskSum[eachFrame] for eachFrame in lowMarkersX]
            plt.plot(lowMarkersX, lowMarkersY, 
                     linestyle='None', 
                     marker="v", 
                     markerfacecolor='k',
                     markeredgecolor='k')
            
            highMarkersX = self.triggerHighTiming
            highMarkersY = [self.maskSum[eachFrame] for eachFrame in highMarkersX]
            plt.plot(highMarkersX, highMarkersY, 
                     linestyle='None', 
                     marker="^", 
                     markerfacecolor='k',
                     markeredgecolor='k')
        
        # Set maskSum plot text
        plt.xlabel("Frame Index")
        plt.ylabel("Sum")
        plt.title("Region Pixel Sum", fontweight="bold")
        
        # Plot trigger state (i.e. the result of processing the maskSum) on the lower plot
        plt.subplot(212)
        plt.plot(self.triggerState)
        
        # Set trigger state plot text
        plt.xlabel("Frame Index")
        plt.ylabel("State")
        plt.title("Trigger State", fontweight="bold")

    # .................................................................................................................  

    # Convenience function for grabbing latest trigger state
    def getTrigState(self, indexOffset=-1):        
        return self.triggerState[self.frameIndex + indexOffset]
    
    
    # Convenience function for grabbing previous trigger state
    def getPrevState(self):
        prevIndex = max(0, self.frameIndex - 1)
        return self.triggerState[prevIndex]

    # .................................................................................................................  
    
    # Function for cropping masks and generating indices used to crop incoming images to match cropped mask
    @staticmethod
    def getCropMask(inMask):
        
        # Find extent of mask so we can use co-ordinates for cropping
        _, cropContour, _ = cv2.findContours(inMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cropX, cropY, cropWidth, cropHeight = cv2.boundingRect(cropContour[0])
        
        # Create some convenience variables for the crop indexing
        cropX1 = cropX
        cropX2 = cropX + cropWidth
        cropY1 = cropY
        cropY2 = cropY + cropHeight
        
        # Build outputs
        cropMask = inMask[cropY1:cropY2, cropX1:cropX2]
        cropPts= (cropY1, cropY2, cropX1, cropX2)
        
        return cropMask, cropPts        
    
    # .................................................................................................................  
    