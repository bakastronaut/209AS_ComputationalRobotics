# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:27:38 2018

@author: Big Champ
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as cir

def ObsGenerator(h, k, radius, demo = False):
    '''
    h and k inputs are center of circle (h,k)
    r input are radius of circle
    '''
    angle = np.arange(0, 2*np.pi, .01)
    x = radius*np.sin(angle)
    y = radius*np.cos(angle)
    
    #General equation for a circle
    circle = ((x - h)*(x - h)) + ((y - k)*(y - k))
    
    
    #Demo
    if demo == True:
        plt.plot(x - h, y - k)
        plt.show()
    
    return circle