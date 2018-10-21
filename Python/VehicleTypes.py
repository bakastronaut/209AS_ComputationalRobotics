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
    LENGTH_HALF -   half the length of the vehicle (units and values are TBD)
    DIST_STANDOFF - the 'safe' stand-off distance, based on the type and size,
                    which will be used for trailing distance and to define 
                    'cutting-off' someone when lane changing. Semis and
                    emergency vehicles will need more space than sedans and 
                    compacts.

"""

class Sedan:
    def __init__(self):
        self.LENGTH_HALF = 5
        self.DIST_STANDOFF = 0

class Compact:
    def __init__(self):
        self.LENGTH_HALF = 5
        self.DIST_STANDOFF = 0

class Emergency:
    def __init__(self):
        self.LENGTH_HALF = 5
        self.DIST_STANDOFF = 0

class Semi:
    def __init__(self):
        self.LENGTH_HALF = 5
        self.DIST_STANDOFF = 0