#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:50:06 2018
@author: johndoe

Classes:
    Sedan
    Compact
    Emergency
    Semi

Attributes:
    LENGTH_HALF:        
        Half the length of the vehicle (units and values are TBD)
    
    DISTANCE_STANDOFF:  
        The 'safe' stand-off distance, based on the type and size, which will 
        be used for trailing distance and to define 'cutting-off' someone when 
        lane changing. Semis and emergency vehicles will need more space than 
        sedans and compacts. This could be a base value that can be scaled or 
        modified to be dependent upon relative velocities.
    
    IMG:
        Path to image for animation/visualization

Methods:

"""

class Sedan:
    def __init__(self,driver_type):
        self.LENGTH_HALF = 5
        self.DIST_STANDOFF = 0
        self.IMG = ''
        self.PREFERRED_SPEED_RATIO = driver_type.PREFERRED_SPEED_RATIO

class Compact:
    def __init__(self,driver_type):
        self.LENGTH_HALF = 5
        self.DIST_STANDOFF = 0
        self.IMG = ''
        self.PREFERRED_SPEED_RATIO = driver_type.PREFERRED_SPEED_RATIO

class Emergency:
    def __init__(self,driver_type):
        self.LENGTH_HALF = 5
        self.DIST_STANDOFF = 0
        self.IMG = ''
        self.PREFERRED_SPEED_RATIO = driver_type.PREFERRED_SPEED_RATIO

class Semi:
    def __init__(self,driver_type):
        self.LENGTH_HALF = 5
        self.DIST_STANDOFF = 0
        self.IMG = ''
        self.PREFERRED_SPEED_RATIO = driver_type.PREFERRED_SPEED_RATIO