# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:26:49 2018

@author: Jay Jackman
"""
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

#fits a spline (default cubic) to a set of checkpoints.
def fit_spline(xvals, yvals, kind='cubic', demo=False, returncubics=False):
    '''
    Inputs:
        xvals         x coordinates of points to be fit
        yvals         y coordinates of points to be fit
        kind          spline polynomial degree
        demo          whether to demo its output by plotting the fit
        returncubics  whether to return coefficients for cubic fits
    Outputs:
        f             function that takes x values as inputs and outputs 
                      corresponding y values of cubic spline
        cubics        list of arrays with coefficients inside
    '''
    assert len(xvals) > 4 and len(yvals) > 4, "Need at least 4 points for a cubic fit"
    
    f = interpolate.interp1d(xvals, yvals, kind=kind)
    
    if(demo):
        plt.figure()
        plt.scatter(xvals,yvals)
        x = np.linspace(np.min(xvals), np.max(xvals), 100)
        plt.plot(x,f(x))
        plt.show()
    
    if(returncubics):
        def f_cubic(x,a,b,c,d):
            return a*x*x*x + b*x*x + c*x + d
        cubics = []
        N_curves = len(xvals) - 1
        for n in range(N_curves):
            x = np.linspace(xvals[n], xvals[n+1], 100)
            y = f(x)
            params,_ = opt.curve_fit(f_cubic,x,y)
            cubics.append(params)
        
        return f,cubics
    else:
        return f

'''
Helper functions for calculating distances
'''
def dist(x1,y1, x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)
def distSquared(x1,y1, x2,y2):
    return (x2-x1)**2 + (y2-y1)**2
'''
Cost between 0-1 denotes how bad it is to hit an obstacle
'''
class Obstacle:
    def __init__(self, x,y,r,cost):
        self.x = x
        self.y = y
        self.r = r
        self.cost = cost
        
class Road:
    
    def __init__(self, checkpoints, width):
        self.width = width
        self.checkpoints = checkpoints
        [self.f, self.cubics] = fit_spline(self.checkpoints[0,:], self.checkpoints[1,:], kind='cubic', demo=False,returncubics=True)
        #self.cubic_lengths = [self.getArcLength(self.checkpoints[0,i], self.checkpoints[0,i+1]) for i in range(len(self.cubics)-1)]
        #self.road_length = sum(self.cubic_lengths)
        self.obstacles = []
        
        
    def draw(self):
        #draw center line
        N = len(self.checkpoints[0])*3
        x = np.linspace(np.min(self.checkpoints[0,:]), np.max(self.checkpoints[0,:]),N)
        plt.plot(x,self.f(x), '--y', linewidth=3)
        
        #draw left and right boundary of road
        left_xs = []
        left_ys = []
        right_xs = []
        right_ys = []
        for i in range(len(x)):
            [pos_x,pos_y] = self.getCartesianCoords(x[i],self.width)
            [pos_s,pos_rho] = self.getS_RhoCoords(pos_x,pos_y)
            [neg_x,neg_y] = self.getCartesianCoords(x[i],-self.width)
            [neg_s,neg_rho] = self.getS_RhoCoords(neg_x,neg_y)
            if(pos_y > neg_y):
                if(np.abs(np.abs(pos_rho) - self.width) < 0.01):
                    left_xs.append(pos_x)
                    left_ys.append(pos_y)
                if(np.abs(np.abs(neg_rho) - self.width) < 0.01):
                    right_xs.append(neg_x)
                    right_ys.append(neg_y)
            else:
                if(np.abs(np.abs(neg_rho) - self.width) < 0.01):
                    left_xs.append(neg_x)
                    left_ys.append(neg_y)
                if(np.abs(np.abs(pos_rho) - self.width) < 0.01):
                    right_xs.append(pos_x)
                    right_ys.append(pos_y)
        plt.plot(left_xs,left_ys, 'black', linewidth=3)
        plt.plot(right_xs,right_ys, 'black', linewidth=3)
        
        #draw obstacles
        for i in range(len(self.obstacles)):
            x = self.obstacles[i].x
            y = self.obstacles[i].y
            r = self.obstacles[i].r
            cost = self.obstacles[i].cost
            circle = plt.Circle((x,y),r,facecolor=(cost,(1-cost)*0.5,(1-cost)*0.5),edgecolor='black')
            circle.fill
            plt.gcf().gca().add_artist(circle) 
    
    def addObstacle(self, s,rho,r,cost):
        [x,y] = self.getCartesianCoords(s,rho)
        self.obstacles.append(Obstacle(x,y,r,cost))
        
    #find the cubic function associated with a given x value
    def getCubic(self, x):
        index = 0
        while(x > self.checkpoints[0,index + 1]):
            index += 1
        return self.cubics[index]
    
    #return the cartesian coordinates of the point on the center line closest to the input point
    def getProjectionOntoRoad(self,Px,Py):
        def Distance_Function(x,a,b,c,d,Px,Py):
            return (Px-x)**2 + (Py - a*x*x*x - b*x*x - c*x - d)**2
        def Cubic_Function(x,a,b,c,d):
            return a*x*x*x + b*x*x + c*x + d
        
        minSoFar = float('inf')
        min_x = 0
        min_y = 0
        for i in range(len(self.cubics)):
            a,b,c,d = self.cubics[i][0], self.cubics[i][1], self.cubics[i][2], self.cubics[i][3]
            x = np.linspace(self.checkpoints[0,i], self.checkpoints[0,i+1],200)
            y = Distance_Function(x,a,b,c,d,Px,Py)
            tempMin = np.min(y)
            if(tempMin < minSoFar):
                minSoFar = tempMin
                index = np.argmin(y)
                min_x = x[index]
                min_y = Cubic_Function(min_x,a,b,c,d)
        return [min_x, min_y]
    
    #return the S-Rho coordinates of a given x-y point
    def getS_RhoCoords(self, x,y):
        [projected_x, projected_y] = self.getProjectionOntoRoad(x,y)
        rho = dist(projected_x, projected_y, x,y)
        if(projected_y > y):
            rho = -rho
        s = projected_x
        return [s,rho]
    
    #returns the x-y coordinates of a given S-Rho point
    def getCartesianCoords(self,s,rho):
        def getSlope(x,cubic):
            return 3*cubic[0]*x*x + 2*cubic[1]*x + cubic[2]
        center_x = s
        center_y = self.f(center_x)
        cubic = self.getCubic(center_x)
        slope = getSlope(center_x, cubic)    
        normal_slope = -1/slope
        
        theta = np.arctan2(normal_slope,1)
        tempx = rho*np.cos(theta)
        tempy = rho*np.sin(theta)
        
        if(normal_slope > 0 and rho > 0):
            tempx = abs(tempx)
            tempy = abs(tempy)
        elif(normal_slope > 0 and rho < 0):
            tempx = -1*abs(tempx)
            tempy = -1*abs(tempy)
        elif(normal_slope < 0 and rho > 0):
            tempx = -1*abs(tempx)
            tempy = abs(tempy)
        else:
            tempx = abs(tempx)
            tempy = -1*abs(tempy)
        return [center_x + tempx, center_y + tempy]
    
    '''
    inputs:
        rho_start:      starting distance from the center line
        rho_end:        ending distance from the center line
        s_start:        starting s-position
        s_end:          ending s-position
        d_rho_start:    difference in angle of cars position and tangent line to the projected point
    outputs:
        path:           a 2x100 array of (x,y) coordinates consisting of the path
    '''
    def getPath(self, rho_start, rho_end, s_start, s_end, d_rho_start):
        def calcParams(s_start, s_end, rho_start, rho_end, d_rho_start):
            d = rho_start
            c = np.tan(d_rho_start)
            d_s = s_end - s_start
            b = (2*c*d_s - 3*rho_end +3*d)/(-(d_s**2))
            a = (rho_end - b*(d_s**2) - c*d_s - d)/(d_s**3)
            return [a,b,c,d]
        def getRho(s,a,b,c,rho_start,s_start):
            return a*(s-s_start)**3 + b*(s-s_start)**2 + c*(s-s_start) + rho_start
        
        path_xs = []
        path_ys = []
        s = np.linspace(s_start, s_end,100)
        [a,b,c,d] = calcParams(s_start, s_end, rho_start, rho_end, d_rho_start)
        rhos = getRho(s,a,b,c,d,s_start)
        for i in range(len(s)):
            [x,y] = self.getCartesianCoords(s[i],rhos[i])
            path_xs.append(x)
            path_ys.append(y)
        path = np.vstack((path_xs, path_ys))
        return path
    
    def getPathCollisionValue(self, path):
        collisions = [0]
        
        #check collision with obstacles
        for i in range(len(self.obstacles)):
            c_x = self.obstacles[i].x
            c_y = self.obstacles[i].y
            r = self.obstacles[i].r
            cost = self.obstacles[i].cost
            for k in range(len(path[0,:])):
                x = path[0,k]
                y = path[1,k]
                distance = dist(x,y, c_x,c_y)
                if(distance <= r):
                    collisions.append(cost)
                    break
        
        #check if the car leaves the road
        for i in range(len(path[0,:])):
            x = path[0,i]
            y = path[1,i]
            [s,rho] = self.getS_RhoCoords(x,y)
            if(rho > self.width or rho < -self.width):
                collisions.append(1)
                break
        return max(collisions)
    
    def generatePaths(self, s_start, s_end, rho_start, d_rho_start, num_paths):
        '''
        outputs:
            paths                       list of num_paths paths. paths are defined as in getPath(...)
            collision_values_raw        list of the actual collision number of a path
            collision_values_blurred    list of numbers, denoting the collision value of each path after gaussian blur
        '''
        paths = []
        collision_values_raw = []
        
        rho_ends = np.linspace(-width,width,num_paths)
        for i in range(num_paths):
            path = self.getPath(rho_start, rho_ends[i], s_start, s_end, d_rho_start)
            paths.append(path)
            collision_values_raw.append(self.getPathCollisionValue(path))
            
        def gauss(i,k,sig):
            return 1/(np.sqrt(np.pi*2*sig))*np.exp(-((k)**2)/(2*(sig**2)))
        def GaussianBlur(R,i,N,sig):
            toReturn = 0
            for k in range(-N,N+1):
                g_result = gauss(i,k,sig)
                if(k+i < 0 or k+i >= len(R)):
                    toReturn += g_result
                else:
                    toReturn += g_result*R[k+i]
            return toReturn
         
        collision_values_blurred = []
        for i in range(len(collision_values_raw)):
            collision_values_blurred.append(GaussianBlur(collision_values_raw,i,num_paths,1))

        return [paths,collision_values_raw,collision_values_blurred]
    
    def drawPaths(self, paths, collision_values_raw, collision_values_blurred):
        index = collision_values_blurred.index(min(collision_values_blurred))
        for i in range(len(paths)):
            if(i == index):
                color = 'g'
            elif(collision_values_raw[i] == 0):
                color = 'b'
            else:
                color = 'r'
            plt.plot(paths[i][0,:],paths[i][1,:],color)

'''
MAIN
'''
 
plt.close('all')   
#create the centerline
x_vals = np.linspace(-10,10,40)
y_vals = [np.sqrt(100-x**2) for x in np.linspace(-10,0,20)]
y_vals = y_vals + [-np.sqrt(100-x**2)+20 for x in np.linspace(0,10,20)]
myCheckpoints = np.vstack((x_vals,y_vals))

#create the road
width = 5
myRoad = Road(myCheckpoints,width)

#add an obstacle onto the road
myRoad.addObstacle(s=7,rho=1,r=3,cost=0.3)
myRoad.addObstacle(s=4,rho=0,r=1,cost=1)

myRoad.draw()

#generate path parameters 
s_start = -9.6
s_end = 8
rho_start = 0
d_rho_start = 0
num_paths = 10

[paths, collisions_raw, collisions_blurred] = myRoad.generatePaths(s_start, s_end, rho_start, d_rho_start,num_paths)
myRoad.drawPaths(paths, collisions_raw, collisions_blurred)

plt.figure()
plt.plot(collisions_raw)
plt.plot(collisions_blurred)

   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        