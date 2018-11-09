#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 02:49:22 2018

@author: johndoe
"""

import numpy as np
import matplotlib as mpl
import scipy.interpolate as interpolate

def fit_spline(xvals,yvals,kind='quadratic',demo=False):
    '''
    Receives set of x values and y values, fits a spline and returns the spline
    
    Inputs:
        xvals   x coordinates of points to be fit
        yvals   y coordinates of points to be fit
        kind    spline polynomial degree
        demo    whether to demo its output by plotting the fit
    Ouputs:
        f       function that takes x values as inputs and outputs 
                corresponding y values of cubic spline
    '''
    
    # Interpolation to fit the spline
    f = interpolate.interp1d(xvals,yvals,kind=kind)
    
    # If demo is true, show a plot of the cubic function.
    if demo:
        mpl.pyplot.scatter(xvals,yvals)
        x = np.linspace(np.min(xvals),np.max(xvals),10000)
        mpl.pyplot.plot(x,f(x))
        mpl.pyplot.show()
    
    return f
