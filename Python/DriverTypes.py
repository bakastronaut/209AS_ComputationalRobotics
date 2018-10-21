#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:02:45 2018

@author: johndoe

Classes:
    Passive
    Moderate
    Aggressive

Attributes:
    PREFERRED_SPEED_RATIO_MEAN:
        The driver's speed preference as a fraction of the speed limit.
    PREFERRED_SPEED_RATIO_STDDEV:
        Standard deviation in speed ratios for driver of that type.
    
Methods:
    
    
Functions:
    speed_ratio_calculate
    
"""

import numpy as np

def speed_ratio_calculate(mean,stddev):
        return stddev * (np.random.randn() + mean)

class Passive:
    def __init__(self):
        self.PREFERRED_SPEED_RATIO_MEAN = 0.70
        self.PREFERRED_SPEED_RATIO_STDDEV = 0.15
        
        mean = self.PREFERRED_SPEED_RATIO_MEAN
        stddev = self.PREFERRED_SPEED_RATIO_STDDEV
        self.PREFERRED_SPEED_RATIO = speed_ratio_calculate(mean,stddev)
    
class Moderate:
    def __init__(self):
        self.PREFERRED_SPEED_RATIO = 0.80
        self.PREFERRED_SPEED_RATIO_STDDEV = 0.1
        
        mean = self.PREFERRED_SPEED_RATIO_MEAN
        stddev = self.PREFERRED_SPEED_RATIO_STDDEV
        self.PREFERRED_SPEED_RATIO = speed_ratio_calculate(mean,stddev)
        
class Aggressive:
    def __init__(self):
        self.PREFERRED_SPEED_RATIO = 1
        self.PREFERRED_SPEED_RATIO_STDDEV = 0.1
        
        mean = self.PREFERRED_SPEED_RATIO_MEAN
        stddev = self.PREFERRED_SPEED_RATIO_STDDEV
        self.PREFERRED_SPEED_RATIO = speed_ratio_calculate(mean,stddev)