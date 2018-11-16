#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 02:49:22 2018

@author: johndoe
"""
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import matplotlib.patches as patches

class Road:    
    '''
    Checkpoints are a WIDE array of [x_i, y_i]
    '''
    def __init__(self, checkpoints, width):
        self.width = width
        self.checkpoints = checkpoints
        [self.f, self.cubics] = fit_spline(checkpoints[0,:], checkpoints[1,:], kind='cubic', demo=False,returncubics=True)
        self.cubic_lengths = [self.getArcLength(self.checkpoints[0,i], self.checkpoints[0,i+1]) for i in range(len(self.cubics)-1)]
        self.road_length = sum(self.cubic_lengths)
        
    def drawRoad(self):
        x = np.linspace(np.min(checkpoints[0,:]), np.max(checkpoints[0,:]),100)
        plt.plot(x,self.f(x))
        plt.show()
            
    def getS_RhoCoords(self, x,y):
        [projected_x, projected_y] = getProjectionOntoRoad(self,x,y)
        rho = dist(projected_x,projected_y, x,y)
        if(projected_x > x):
            rho = -rho
        s = self.getArcLength(self.checkpoints[0,0],projected_x)       
        return [s,rho]
    
    def getCartesianCoords(self,s,rho):
        x_start = self.checkpoints[0,0]
        dx = (self.checkpoints[0,-1] - self.checkpoints[0,0])/1000
        x = x_start
        s_accumulated = 0
        while(s_accumulated < s):
            s_accumulated += self.getArcLength(x, x+dx)
            x += dx
        road_x = x
        road_y = self.f(road_x)
        prev_x = road_x - dx
        prev_y = self.f(prev_x)
        next_x = road_x + dx
        next_y = self.f(next_x)
        
        tangent_slope = (next_y - prev_y)/(next_x - prev_x)
        normal_slope = -1/tangent_slope
        theta  = np.arctan(normal_slope)
        tempx = rho*np.cos(theta)
        tempy = rho*np.sin(theta)
        return [road_x + tempx, road_y + tempy]
    
    def getArcLength(self, x_start, x_end):
        x_vals = np.linspace(x_start, x_end, int((x_end-x_start) * 100) + 2)
        return sum([dist(x_vals[i+1], self.f(x_vals[i+1]), x_vals[i], self.f(x_vals[i])) for i in range(len(x_vals)-1)])
        
        
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
            params,_ = opt.curve_fit(f_cubic,x,y)
            cubics.append(params)
        
        return f,cubics
    
    else:
        
        return f    

#returns the x,y coordinates on the road, closest to the input point
def getProjectionOntoRoad(road, Px, Py):
    checkpoints = road.checkpoints
    cubics = road.cubics
    
    def Distance_Function(x,a,b,c,d, Px, Py):
        return (Px - x)**2 + (Py - a*x*x*x - b*x*x - c*x - d)**2
    def Cubic_Function(x,a,b,c,d):
        return a*(x*x*x)+b*(x*x)+c*x+d
    
    minSoFar = float("inf")
    min_x = 0
    min_y = 0
    for i in range(len(cubics)):
        a,b,c,d = cubics[i][0],cubics[i][1],cubics[i][2],cubics[i][3]
        x = np.linspace(checkpoints[0,i], checkpoints[0,i+1], 2000)
        
        y = Distance_Function(x,a,b,c,d,Px,Py)
        #plt.plot(x,y)
        tempMin = np.min(y)
        if(tempMin < minSoFar):
            minSoFar = tempMin
            index = np.argmin(y)
            min_x = x[index]
            min_y = Cubic_Function(min_x,a,b,c,d)
    return [min_x, min_y]

#cubic is an array of the a/b/c/d weights in that order
def evaluate_cubic(x, cubic):
    return cubic[0]*x*x*x + cubic[1]*x*x + cubic[2]*x + cubic[3]

def dist(x1,y1, x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def getArcLengthOnCubic(cubic, x_start, x_end):
    x_vals = np.linspace(x_start, x_end, int(x_end-x_start)*1000)
    return sum([dist(x_vals[i+1], evaluate_cubic(x_vals[i+1],cubic), x_vals[i], evaluate_cubic(x_vals[i],cubic)) for i in range(len(x_vals)-1)])

plt.close('all')

#make and draw the road checkpoints
x_vals = [1,2,2.2,4,5,7,8,10]
y_vals = [1,2,3,4,5,6,1,2]
checkpoints = np.vstack((x_vals,y_vals))
plt.scatter(checkpoints[0,:],checkpoints[1,:])

#make and draw the road
myRoad = Road(checkpoints, 10)
myRoad.drawRoad()

#current position of the car
Px = 2
Py = 10
#projected coordinates are the point on the road closest to Px, Py
[projected_x,projected_y] = getProjectionOntoRoad(myRoad, Px, Py)
#myS and myRho are the s/rho coordinates of Px, Py
[myS, myRho] = myRoad.getS_RhoCoords(Px, Py)
#cart_x and cart_y are the estimated position of the car, given the derived S and Rho coordinates
[cart_x, cart_y] = myRoad.getCartesianCoords(myS, myRho)


#plot the actual position of the car
plt.scatter(Px,Py)
#plot the projected position of the car on the center line
plt.scatter(projected_x,projected_y)
#plot the remapped position of the car
plt.scatter(cart_x, cart_y, marker='x')

#plot distance circle for verification
circle = plt.Circle((Px,Py),myRho, edgecolor = 'b', facecolor='none')
circle.fill
plt.gcf().gca().add_artist(circle)



#minx = min(x_vals)
#miny = min(y_vals)
#maxx = max(x_vals)
#maxy = max(y_vals)
#lims = (min([minx,miny])-1,max([maxx,maxy])+1)
#plt.xlim(lims)
#plt.ylim(lims)




    
    
