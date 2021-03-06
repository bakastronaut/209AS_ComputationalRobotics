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
    PROBABILITY_LANE_CHANGE:
        Since other cars will be 'unpredictable,' this probability will be used
        to determine when they'll randomly make a lane change assuming it is
        possible given their current location relative to other cars.
    
    PREFERRED_SPEED_RATIO_MEAN:
        The driver's speed preference as a fraction of the speed limit.
    
    PREFERRED_SPEED_RATIO_STDDEV:
        Standard deviation in speed ratios for driver of that type.
    
Methods:
    
    
Functions:
    speed_ratio_calculate:
        Returns a speed ratio by sampling from normal distribution with 
        specified mean and standard deviation
    
"""

import numpy as np

class Driver:
    
    def __init__(self,driver_type):
        if driver_type.lower().strip() == 'passive':
            self.PROBABILITY_LANE_CHANGE = 0.1
            self.PREFERRED_SPEED_RATIO_MEAN = 0.70
            self.PREFERRED_SPEED_RATIO_STDDEV = 0.15
        
        elif driver_type.lower().strip() == 'moderate':
            self.PROBABILITY_LANE_CHANGE = 0.2
            self.PREFERRED_SPEED_RATIO = 0.80
            self.PREFERRED_SPEED_RATIO_STDDEV = 0.1
        
        elif driver_type.lower().strip() == 'aggressive':
            self.PROBABILITY_LANE_CHANGE = 0.3
            self.PREFERRED_SPEED_RATIO = 1
            self.PREFERRED_SPEED_RATIO_STDDEV = 0.1
        else:
            raise ValueError('Unrecognized driver type in driver_type string')
    
        mean = self.PREFERRED_SPEED_RATIO_MEAN
        stddev = self.PREFERRED_SPEED_RATIO_STDDEV
        self.PREFERRED_SPEED_RATIO = self.speed_ratio_calculate(mean,stddev)

    def speed_ratio_calculate(mean,stddev):
        ratio = stddev * (np.random.randn() + mean)
        return ratio