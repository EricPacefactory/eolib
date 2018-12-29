#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:14:35 2018

@author: eo
"""

#import os
#import datetime as dt


# ---------------------------------------------------------------------------------------------------------------------
#%% Define classes





# ---------------------------------------------------------------------------------------------------------------------
#%% Define functions

'''
def getRTSP(ipAddr, userName="", password="", port=554, command=""):
    
    videoSource = "".join(["rtsp://", userName, ":", password, "@", ipAddr, ":", str(port), "/", command])
    
    splitIP = ipAddr.split(".")
    padIP = [eachNumber.zfill(3) for eachNumber in splitIP]
    solidIP = "".join(padIP)
    
    videoName = "".join(["RTSP-", solidIP]) 
    videoFile = "".join([solidIP, ".rtsp"])
    
    return videoSource, videoFile, videoName

# .....................................................................................................................
    
def getRTSP(ipAddr, userName="", password="", port=554, command=""):
    
    rtspSource = "".join(["rtsp://", userName, ":", password, "@", ipAddr, ":", str(port), "/", command])
    
    splitIP = ipAddr.split(".")
    padIP = [eachNumber.zfill(3) for eachNumber in splitIP]
    blockIP = "".join(padIP)
    
    return rtspSource, blockIP
'''

# ---------------------------------------------------------------------------------------------------------------------
#%% File demo

'''
# Demo if running this file directly
if __name__ == "__main__":
    
    fakeIP = "192.5.77.123"
    fakeUser = "Jimi"
    fakePass = "Hendrix!"
    fakeCommand = "groovy"
    
    exampleSource, exampleFile, exampleName = getRTSP(fakeIP, fakeUser, fakePass, command=fakeCommand)
    
    print("")
    print("Example RTSP string:")
    print(" ", exampleSource)
    print("")
    print("Example file:")
    print(" ", exampleFile)
    print("")
    print("Example name:")
    print(" ", exampleName)
'''
    
# ---------------------------------------------------------------------------------------------------------------------
#%% Scrap