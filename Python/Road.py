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
    
    def __init__(self, checkpoints, lane_width, num_lanes):
        self.lane_width = lane_width
        self.num_lanes = num_lanes
        self.checkpoints = checkpoints
        [self.f, self.cubics] = fit_spline(self.checkpoints[0,:], self.checkpoints[1,:], kind='cubic', demo=False,returncubics=True)
        self.obstacles = []        
        
    def draw(self):
        #draw center line
        N = len(self.checkpoints[0])*3
        x = np.linspace(np.min(self.checkpoints[0,:]), np.max(self.checkpoints[0,:]),N)
        
        #draw the center line in thick, dashed black
        plt.plot(x,self.f(x), '--k', linewidth=3,label='Road')
        
        #draw left and right lane demarcations
        for k in range(self.num_lanes):
            left_xs = []
            left_ys = []
            right_xs = []
            right_ys = []
            curr_width = lane_width*(k+1)
            for i in range(len(x)):
                [pos_x,pos_y] = self.getCartesianCoords(x[i],curr_width)
                [pos_s,pos_rho] = self.getS_RhoCoords(pos_x,pos_y)
                [neg_x,neg_y] = self.getCartesianCoords(x[i],-curr_width)
                [neg_s,neg_rho] = self.getS_RhoCoords(neg_x,neg_y)
                if(pos_y > neg_y): #need to check this to prevent flipping when road slope changes from positive to negative
                    if(np.abs(np.abs(pos_rho) - curr_width) < 0.01):
                        left_xs.append(pos_x)
                        left_ys.append(pos_y)
                    if(np.abs(np.abs(neg_rho) - curr_width) < 0.01):
                        right_xs.append(neg_x)
                        right_ys.append(neg_y)
                else:
                    if(np.abs(np.abs(neg_rho) - curr_width) < 0.01):
                        left_xs.append(neg_x)
                        left_ys.append(neg_y)
                    if(np.abs(np.abs(pos_rho) - curr_width) < 0.01):
                        right_xs.append(pos_x)
                        right_ys.append(pos_y)

            #draw the terminal edges in thick black
            if(k == self.num_lanes-1):
                plt.plot(left_xs,left_ys, 'black', linewidth=3)
                plt.plot(right_xs,right_ys, 'black', linewidth=3)
            #draw the dashed lane separaters in thin yellow
            else:
                plt.plot(left_xs,left_ys, '--y', linewidth=1.5)
                plt.plot(right_xs,right_ys, '--y', linewidth=1.5)
        
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
            x = np.linspace(self.checkpoints[0,i], self.checkpoints[0,i+1],100)
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
        road_bound = self.lane_width*self.num_lanes
        lane_markers = [lane_width*i for i in np.arange(-self.num_lanes,self.num_lanes+1,1)]
        lane_markers.remove(0)
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
        #check for crossing lanes and leaving the road
        rho_prev = self.getS_RhoCoords(path[0,0],path[1,0])[1]
        lane_crossing_counter = 0
        for i in range(len(path[0,:])):
            x = path[0,i]
            y = path[1,i]
            rho_curr = self.getS_RhoCoords(x,y)[1]
            #if the car leaves the lane
            if(rho_curr > road_bound or rho_curr < -road_bound):
                collisions.append(1)
                break
            #if the car passes the center line
            if(np.sign(rho_prev) != np.sign(rho_curr)):
                lane_crossing_counter += 0.6
            #check to see if it changes lanes
            for j in lane_markers:
                if(np.sign(j-rho_prev) != np.sign(j-rho_curr)):
                    lane_crossing_counter += 0.2
            rho_prev = rho_curr
        collisions.append(lane_crossing_counter)        
        return max(collisions)
    
    def generatePaths(self, s_start, s_end, rho_start, d_rho_start, num_paths):
        '''
        outputs:
            paths                       list of num_paths paths. paths are defined as in getPath(...)
            collision_values_raw        list of the actual collision number of a path
            collision_values_blurred    list of numbers, denoting the collision value of each path after gaussian blur
            curvature_values            list of numbers, denoting the curvature value of each path, normalized.
        '''
        paths = []
        collision_values_raw = []
        collision_values_blurred = []
        curvature_values = []
        bound = self.lane_width*self.num_lanes
        rho_ends = np.linspace(-bound,bound,num_paths)
        for i in range(num_paths):
            path = self.getPath(rho_start, rho_ends[-(i+1)], s_start, s_end, d_rho_start)
            paths.append(path)
            collision_values_raw.append(self.getPathCollisionValue(path))
            curvature_values.append(self.getCurvature(path))
            
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
        
        #get the blurred collision values
        for i in range(len(collision_values_raw)):
            collision_values_blurred.append(GaussianBlur(collision_values_raw,i,num_paths,1))
        
        #normalize the curvature values
        curvature_values = curvature_values/max(curvature_values)
    
        return [paths,collision_values_raw,collision_values_blurred, curvature_values]
    
    def getCurvature(self, path):
        N = len(path[0,:])
        K = 0
        for i in range(N-2):
            x1 = path[0,i]
            y1 = path[1,i]
            x2 = path[0,i+1]
            y2 = path[1,i+1]
            x3 = path[0,i+2]
            y3 = path[1,i+2]
            x21 = x2-x1
            y21 = y2-y1
            x32 = x3-x2
            y32 = y3-y2
            
            mag21 = np.sqrt(x21**2 + y21**2)
            mag32 = np.sqrt(x32**2 + y32**2)
            
            x_diff = x32/mag32 - x21/mag21
            y_diff = y32/mag32 - y32/mag32
            
            K += np.sqrt(x_diff**2 + y_diff**2)
        return K/(N-2)           
            
    def drawPaths(self, paths, collision_values_raw, collision_values_blurred, curvature_values,final_costs):
        index_safety = collision_values_blurred.index(min(collision_values_blurred))
        index_curvature = list(curvature_values).index(min(curvature_values))
        index_final = list(final_costs).index(min(final_costs))
        for i in range(len(paths)):
            if(i == index_final):
                plt.plot(paths[i][0,:],paths[i][1,:], 'g', linewidth=4)
                continue
            elif(i == index_safety):
                plt.plot(paths[i][0,:],paths[i][1,:],'cyan', linewidth=4)
                continue
            elif(i == index_curvature):
                plt.plot(paths[i][0,:],paths[i][1,:],'magenta', linewidth=4)
                continue
            elif(collision_values_raw[i] >= 0.3 and collision_values_raw[i] < 0.7):
                color = 'y'
            elif(collision_values_raw[i] < 0.3):
                color = 'b'
            else:
                color = 'r'
            plt.plot(paths[i][0,:],paths[i][1,:],color)    
        

'''
MAIN
'''
 
plt.close('all') 

#define cost weights
w_s = 0.5 #safety cost
w_c = 1-w_s #comfort cost

lane_width = 5
num_lanes = 2
MAX_CURVATURE = 20
NUM_CHECKPOINTS = 40
#create the centerline
x_vals = np.linspace(-MAX_CURVATURE, MAX_CURVATURE,NUM_CHECKPOINTS)

'''
S_Bend road generation
'''
y_vals = [np.sqrt(MAX_CURVATURE**2-x**2) for x in np.linspace(-MAX_CURVATURE,0,NUM_CHECKPOINTS/2)]
y_vals = y_vals + [-np.sqrt(MAX_CURVATURE**2-x**2)+2*MAX_CURVATURE for x in np.linspace(0,MAX_CURVATURE,NUM_CHECKPOINTS/2)]

'''
Straight road generation
'''
#y_vals = [5 for i in range(len(x_vals))]
#y_vals = x_vals

myCheckpoints = np.vstack((x_vals,y_vals))
#create the road
myRoad = Road(myCheckpoints,lane_width, num_lanes)

#add an obstacle onto the road
#myRoad.addObstacle(s=7,rho=3,r=3,cost=0.5)
#myRoad.addObstacle(s=-5,rho=lane_width/2,r=lane_width/4,cost=1)

myRoad.draw()

#generate path parameters 
s_start = x_vals[0]
s_end = 0
rho_start = lane_width/2
d_rho_start = np.pi/4
num_paths = 30

#generate and draw prospective paths
[paths, collisions_raw, collisions_blurred, curvatures] = myRoad.generatePaths(s_start, s_end, rho_start, d_rho_start,num_paths)
final_cost = [w_s*collisions_blurred[i]+w_c*curvatures[i] for i in range(len(collisions_blurred))]
myRoad.drawPaths(paths, collisions_raw, collisions_blurred, curvatures,final_cost)

#P_x = -15
#P_y = 20
#[proj_x,proj_y] = myRoad.getProjectionOntoRoad(P_x,P_y)
#[s,rho] = myRoad.getS_RhoCoords(P_x,P_y)
#[final_x,final_y] = myRoad.getCartesianCoords(s,rho)

#plt.scatter(P_x,P_y,s=100,label='original (x,y) point')
#plt.scatter(proj_x,proj_y,s=100,label='projected (x,y) point')
#plt.scatter(final_x, final_y,marker='x',color='r',label='remapped (s,rho) point')
#plt.legend()

final_cost = [w_s*collisions_blurred[i]+w_c*curvatures[i] for i in range(len(collisions_blurred))]


plt.figure()
plt.plot(final_cost,'g')
plt.plot(collisions_raw,'r')
plt.plot(collisions_blurred,'cyan')
plt.plot(curvatures,'magenta')


plt.figure()
NUM_CHECKPOINTS_LONG = 20
S_START = 0
S_END = lane_width*20
long_x_vals = np.linspace(S_START, S_END, NUM_CHECKPOINTS_LONG)
long_y_vals = [0 for i in long_x_vals]
long_checkpoints = np.vstack((long_x_vals, long_y_vals))

long_road = Road(long_checkpoints, lane_width, num_lanes)

long_road.addObstacle(s=20, rho=-lane_width/2, r = lane_width/2, cost = 1)
long_road.addObstacle(s=40, rho=-lane_width*3/2, r = lane_width/2, cost =1)
long_road.addObstacle(s=70, rho=-lane_width/2, r = lane_width/2, cost = 1)

plt.subplot(211)
long_road.draw()

s_start = long_x_vals[0]
s_end = 10 
rho_start = -lane_width/2
d_rho_start = 0
num_paths = 30

[paths, collisions_raw, collisions_blurred, curvatures] = long_road.generatePaths(s_start, s_end, rho_start, d_rho_start,num_paths)
final_cost = [w_s*collisions_blurred[i]+w_c*curvatures[i] for i in range(len(collisions_blurred))]
long_road.drawPaths(paths, collisions_raw, collisions_blurred, curvatures,final_cost)

bestPathIndex = list(final_cost).index(min(final_cost))
bestPath = paths[bestPathIndex]
best_paths = [bestPath]

for k in range(9):
    [s_start, rho_start] = long_road.getS_RhoCoords(best_paths[k][0,-1],best_paths[k][1,-1])
    s_end = s_start + 10
    d_rho_start = 0
    [paths, collisions_raw, collisions_blurred, curvatures] = long_road.generatePaths(s_start, s_end, rho_start, d_rho_start, num_paths)
    final_cost = [w_s*collisions_blurred[i] + w_c*curvatures[i] for i in range(len(collisions_blurred))]
    long_road.drawPaths(paths, collisions_raw, collisions_blurred, curvatures,final_cost)
    
    
    
    bestPathIndex = list(final_cost).index(min(final_cost))
    bestPath = paths[bestPathIndex]
    best_paths.append(bestPath)

plt.subplot(212)

long_road.draw()
for i in range(len(best_paths)):
    plt.plot(best_paths[i][0,:],best_paths[i][1,:])
   
    
    
plt.figure()
plt.subplot(211)
long_road.draw()

s_start = long_x_vals[0]
s_end = 20
rho_start = -lane_width/2
d_rho_start = 0
num_paths = 30

[paths, collisions_raw, collisions_blurred, curvatures] = long_road.generatePaths(s_start, s_end, rho_start, d_rho_start,num_paths)
final_cost = [w_s*collisions_blurred[i]+w_c*curvatures[i] for i in range(len(collisions_blurred))]
long_road.drawPaths(paths, collisions_raw, collisions_blurred, curvatures,final_cost)

bestPathIndex = list(final_cost).index(min(final_cost))
bestPath = paths[bestPathIndex]
best_paths = [bestPath]

for k in range(4):
    [s_start, rho_start] = long_road.getS_RhoCoords(best_paths[k][0,-1],best_paths[k][1,-1])
    s_end = s_start + 20
    d_rho_start = 0
    [paths, collisions_raw, collisions_blurred, curvatures] = long_road.generatePaths(s_start, s_end, rho_start, d_rho_start, num_paths)
    final_cost = [w_s*collisions_blurred[i] + w_c*curvatures[i] for i in range(len(collisions_blurred))]
    long_road.drawPaths(paths, collisions_raw, collisions_blurred, curvatures,final_cost)
    
    
    
    bestPathIndex = list(final_cost).index(min(final_cost))
    bestPath = paths[bestPathIndex]
    best_paths.append(bestPath)

plt.subplot(212)

long_road.draw()
for i in range(len(best_paths)):
    plt.plot(best_paths[i][0,:],best_paths[i][1,:])

#%%
plt.figure()
R = 50
NUM_CHECKPOINTS = 2*R
curved_x_vals = np.linspace(-R,R, NUM_CHECKPOINTS)
curved_y_vals = [np.sqrt(R**2-x**2) for x in np.linspace(-R,0,NUM_CHECKPOINTS/2)]
curved_y_vals = curved_y_vals + [-np.sqrt(R**2-x**2)+2*R for x in np.linspace(0,R,NUM_CHECKPOINTS/2)]
curved_checkpoints = np.vstack((curved_x_vals,curved_y_vals))


curvedRoad = Road(curved_checkpoints, lane_width, num_lanes)
curvedRoad.addObstacle(s=-30, rho=-lane_width/2, r = lane_width/2, cost = 1)
curvedRoad.addObstacle(s=-10, rho=-lane_width*3/2, r = lane_width/2, cost =1)
curvedRoad.addObstacle(s=20, rho=-lane_width/2, r = lane_width/2, cost = 1)

plt.subplot(121)
curvedRoad.draw()

s_start = curved_x_vals[0]
s_end = s_start + 10
rho_start = -lane_width/2
d_rho_start = 0
num_paths = 30

[paths, collisions_raw, collisions_blurred, curvatures] = curvedRoad.generatePaths(s_start, s_end, rho_start, d_rho_start,num_paths)
final_cost = [w_s*collisions_blurred[i]+w_c*curvatures[i] for i in range(len(collisions_blurred))]
curvedRoad.drawPaths(paths, collisions_raw, collisions_blurred, curvatures,final_cost)

bestPathIndex = list(final_cost).index(min(final_cost))
bestPath = paths[bestPathIndex]
best_paths = [bestPath]

for k in range(9):
    [s_start, rho_start] = curvedRoad.getS_RhoCoords(best_paths[k][0,-1],best_paths[k][1,-1])
    s_end = s_start + 10
    d_rho_start = 0
    [paths, collisions_raw, collisions_blurred, curvatures] = curvedRoad.generatePaths(s_start, s_end, rho_start, d_rho_start, num_paths)
    final_cost = [w_s*collisions_blurred[i] + w_c*curvatures[i] for i in range(len(collisions_blurred))]
    curvedRoad.drawPaths(paths, collisions_raw, collisions_blurred, curvatures,final_cost)
    
    
    
    bestPathIndex = list(final_cost).index(min(final_cost))
    bestPath = paths[bestPathIndex]
    best_paths.append(bestPath)

plt.subplot(122)

curvedRoad.draw()
for i in range(len(best_paths)):
    plt.plot(best_paths[i][0,:],best_paths[i][1,:])
    
    
plt.figure()
plt.subplot(121)
curvedRoad.draw()

s_start = curved_x_vals[0]
s_end = s_start + 20
rho_start = -lane_width/2
d_rho_start = 0
num_paths = 30

[paths, collisions_raw, collisions_blurred, curvatures] = curvedRoad.generatePaths(s_start, s_end, rho_start, d_rho_start,num_paths)
final_cost = [w_s*collisions_blurred[i]+w_c*curvatures[i] for i in range(len(collisions_blurred))]
curvedRoad.drawPaths(paths, collisions_raw, collisions_blurred, curvatures,final_cost)

bestPathIndex = list(final_cost).index(min(final_cost))
bestPath = paths[bestPathIndex]
best_paths = [bestPath]

for k in range(4):
    [s_start, rho_start] = curvedRoad.getS_RhoCoords(best_paths[k][0,-1],best_paths[k][1,-1])
    s_end = s_start + 20
    d_rho_start = 0
    [paths, collisions_raw, collisions_blurred, curvatures] = curvedRoad.generatePaths(s_start, s_end, rho_start, d_rho_start, num_paths)
    final_cost = [w_s*collisions_blurred[i] + w_c*curvatures[i] for i in range(len(collisions_blurred))]
    curvedRoad.drawPaths(paths, collisions_raw, collisions_blurred, curvatures,final_cost)
    
    
    
    bestPathIndex = list(final_cost).index(min(final_cost))
    bestPath = paths[bestPathIndex]
    best_paths.append(bestPath)

plt.subplot(122)
curvedRoad = Road(curved_checkpoints, lane_width, num_lanes)
curvedRoad.addObstacle(s=-30, rho=-lane_width/2, r = lane_width/2, cost = 1)
curvedRoad.addObstacle(s=-10, rho=-lane_width*3/2, r = lane_width/2, cost =1)
curvedRoad.addObstacle(s=20, rho=-lane_width/2, r = lane_width/2, cost = 1) 
curvedRoad.draw()
for i in range(len(best_paths)):
    plt.plot(best_paths[i][0,:],best_paths[i][1,:])        
        
