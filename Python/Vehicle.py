#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:50:06 2018
@author: johndoe

Classes:
    Vehicle
    
Subclasses:
    StateGlobal:
        Instantiates a state object within the vehicle class that can be
        indexed by its x & y position & velocity

Vehicle Types:
    Sedan
    Compact
    Emergency
    Semi

Attributes:
    UNIQUE_ID:
        Unique integer identifying the vehicle
        
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
    
    STATE_GLOBAL:
        State vector containing object's state at the global scale - absolute
        position components (x & y), absolute velocity (x & y)

Methods:

"""
import numpy as np

class Vehicle(object):
    def __init__(self,vehicle_type,driver):
        
        self.UNIQUE_ID = np.random.randint(0,100000)
#        self.PREFERRED_SPEED_RATIO = driver.PREFERRED_SPEED_RATIO
        
        if vehicle_type.lower() == 'sedan':
            self.LENGTH_HALF = 5
            self.DIST_STANDOFF = 0
            self.IMG = ''
        
        elif vehicle_type.lower() == 'compact':
            self.LENGTH_HALF = 5
            self.DIST_STANDOFF = 0
            self.IMG = ''
        
        elif vehicle_type.lower() == 'emergency':
            self.LENGTH_HALF = 5
            self.DIST_STANDOFF = 0
            self.IMG = ''
        
        elif vehicle_type.lower() == 'semi':
            self.LENGTH_HALF = 5
            self.DIST_STANDOFF = 0
            self.IMG = ''
        
        else:
            raise ValueError('Unrecognized vehicle type in vehicle_type string')
        
        self.HORIZON_SPATIAL = 3 * (2*self.LENGTH_HALF)  # 3 vehicle lengths in all directions from vehicle center
        
        self.STATE_GLOBAL = StateGlobal()
        self.STATE_RELATIVE = Neighbors()
        
class StateGlobal():
    '''
    '''
    def __init__(self,x_pos=np.inf,y_pos=np.inf,x_vel=np.inf,y_vel=np.inf):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_vel = x_vel
        self.y_vel = y_vel

class StateRelative(Vehicle):
    '''
    Calculates the relative x & y positions, x & yvelocities, distance, 
    speed between current vehicle and a neighboring vehicle.
    '''
    def __init__(self,vehicle_neighbor):
        # Relative x and y positions
        self.x_pos_rel = vehicle_neighbor.STATE_GLOBAL.x_pos - self.STATE_GLOBAL.x_pos
        self.y_pos_rel = vehicle_neighbor.STATE_GLOBAL.y_pos - self.STATE_GLOBAL.y_pos
        
        # Relative x and y velocities
        self.x_vel_rel = vehicle_neighbor.STATE_GLOBAL.x_vel - self.STATE_GLOBAL.x_vel
        self.y_vel_rel = vehicle_neighbor.STATE_GLOBAL.x_vel - self.STATE_GLOBAL.x_vel
        
        # Relative absolute distance and speed
        self.dist_rel = np.linalg.norm([self.x_pos_rel,self.y_pos_rel],ord=2)
        self.speed_rel = np.linalg.norm([self.x_vel_rel,self.y_vel_rel],ord=2)
        
class Neighbors():
    '''
    Dictionary holding the relative state of all neighbors within the spatial horizon.
    
    Methods:
        update_relative_neighbor_state:
            Calculate the state of the indicated neighbor relative to the 
            current vehicle and stores it in the NEIGHBORS dictionary. Its key
            is that vehicle's unique ID.
    
    Attributes:
        NEIGHBORS - DICT that will hold a StateRelative object for each neighbor
    '''
    def __init__(self):
        self.NEIGHBORS = dict()
    
    # TO DO: Wrap this inside another function that will get all neighbors in horizon and run this for each.
    def update_relative_neighbor_state(self,neighbor_vehicle):
        
        def remove_if_beyond_spatial_horizon(self,neighbor_vehicle):
            '''
            Checks all neighbors, removes neighbors that are beyond the sptial 
            horizon of the current vehicle.
            '''
            key = neighbor_vehicle.UNIQUE_ID
            if self.NEIHBORS[key].dist_rel > self.HORIZON_SPATIAL:
                del self.NEIGHBORS[key]
        
        ############
        
        neighborID = neighbor_vehicle.UNIQUE_ID
        check = False
        if neighborID in self.NEIGHBORS.keys():
            check = True
            N_keys0 = len(self.NEIGHBORS.keys())
        
        # Update relative state between current vehicle and neighbor
        self.NEIGHBORS[neighborID] = StateRelative(neighbor_vehicle)

        if check:
            assert N_keys0 == len(self.NEIGHBORS.keys())
        
        # Remove neighbor if it is beyond current vehicle's spatial horizon
        self.remove_if_beyond_spatial_horizon(neighborID)
            
        