#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 02:49:22 2018

@author: johndoe
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate as interpolate

def fit_spline(xvals,yvals,kind='cubic',demo=False,returncubics=False):
    '''
    Receives set of x values and y values, fits a spline and returns the spline
    
    Inputs:
        xvals           x coordinates of points to be fit
        yvals           y coordinates of points to be fit
        kind            spline polynomial degree
        demo            whether to demo its output by plotting the fit
        returncubics    whether to return coefficients for cubic fits
    Ouputs:
        f       function that takes x values as inputs and outputs 
                corresponding y values of cubic spline
        cubics  list of arrays with coefficients inside
    '''
    
    assert len(xvals) > 4 and len(yvals) > 4, "Need at least 4 points for a cubic fit"
    
    # Interpolation to fit the spline
    f = interpolate.interp1d(xvals,yvals,kind=kind)
    
    # If demo is true, show a plot of the cubic function.
    if demo:
        plt.scatter(xvals,yvals)
        x = np.linspace(np.min(xvals),np.max(xvals),100)
        plt.plot(x,f(x))
        plt.show()
    
    if returncubics:
        def f_cubic(x,a,b,c,d):
            return a*x**3 + b*x**2 + c*x + d
        
        cubics = []
        N_curves = len(xvals) - 1
        for n in range(N_curves):
            x = np.linspace(xvals[n],xvals[n+1],100)
            y = f(x)
            params,_ = sp.optimize.curve_fit(f_cubic,x,y)
            cubics.append(params)
        
        return f,cubics
    
    else:
        
        return f

io0 = [1,2,3,4,5,6,7,8,9]
io1 = [7,10,2,7,6,1,1,9,0]
func_fit,params = fit_spline(io0,io1, kind='cubic', demo=False, returncubics=True)

x = np.linspace(-5,15,1000)
for n in range(len(params)):
    p = params[n]
    a,b,c,d = p[0],p[1],p[2],p[3]
    y = a * x**3 + b * x**2 + c * x + d
    plt.plot(x,y)
plt.scatter(io0,io1)