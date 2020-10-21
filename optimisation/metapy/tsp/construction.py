# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 07:48:50 2017

@author: tm3y13
"""
from metapy.tsp.objective import tour_cost



class NearestNeighbour(object):
    
    def __init__(self, cities, matrix):
        self.cities = cities
        self.matrix = matrix
        self.solution = []
        self.best_cost = 99999
        self.best_solution = self.solution
        
    def solve(self):
        """
        Constructs a tour based on nearest neighbour method.
        Assume that first city in tour is base city.
        """
        
        from_city = self.cities[0]
        
        self.solution.append(from_city)
                
        for i in range(len(self.cities) - 2):
            to_city = self.closest_city_not_in_tour(from_city)
            self.solution.append(self.cities[to_city])
            #print(self.solution)
            from_city = to_city
        
        self.solution.append(self.cities[0])
        self.best_cost = tour_cost(self.solution, self.matrix)
        self.best_solution = self.solution
                   
            
                    
    def closest_city_not_in_tour(self, from_city):
        min_cost = 999
        min_index = from_city
        
        for to_city in range(len(self.cities) - 1):
            
            if (min_cost > self.matrix[from_city][to_city]):
                              
                if (self.cities[to_city] not in self.solution):
                    min_index = to_city
                    min_cost = self.matrix[from_city][to_city]
        
        return min_index
        
        
class FurthestInsertion(object):
    """
    Furthest insertion contruction heuristic.  
    Finds city furthest away from any point on the subtour.
    Finds a place to insert into tour that minimises the increase in length
    
    Notes: code needs a tidy - TM.
    """
    def __init__(self, cities, matrix):
        self.cities = cities
        self.matrix = matrix
        self.solution = []
        self.best_cost = 0
        self.best_solution = self.solution
        
    
    def solve(self):
        from_city = self.cities[0]
        
        self.solution.append(from_city)
        print(self.solution)
        
        #step 2
        to_insert = self.select_city(self.solution)
        result = self.furthest_city_index(from_city)
        self.solution.append(result[1])
        self.best_cost = result[0]
        
        print(self.solution)
        
        
        #loop until full tour
        for i in range(len(self.cities) - 3):
            
            #find city to insert
            to_insert = self.select_city(self.solution)
            
            #find insertion point
            self.insert_link(to_insert)
        
        self.solution.append(self.solution[0])
        print(self.solution)
        
        self.best_cost = tour_cost(self.solution, self.matrix)
        
    def select_city(self, sub_solution):           
        max_cost = 0
        index = 0
        #[self.farthest_city_index(from_city) for from_city in sub_solution]
        
        for from_city in sub_solution:
            result =  self.furthest_city_index(from_city)
            
            if max_cost < result[0]:
                max_cost = result[0]
                index = result[1]
                
        return index
               
               
               
        
        
    def furthest_city_index(self, from_city):
        cost = 0
        index = 0
        for to_city in self.cities[1:len(self.cities)]:
            #print("furthest city from {0}".format(from_city))
            if(self.matrix[from_city][to_city] > cost):
                
                if(to_city not in self.solution):
                    
                    cost = self.matrix[from_city][to_city]
                    index = to_city
        
        return [cost, index]
                
        
    def insert_link(self, to_insert):
        """
        Insert edge in between cities to achieve
        minimum increase in length
        """
        
        #print("to insert {0}".format(to_insert))
        deltas = [[self.calculate_delta(pos, to_insert), pos] for pos in range(1, len(self.solution))]
        min_insertion = min(deltas)
        self.solution.insert(min_insertion[1], to_insert)   
        print(self.solution)
        self.best_cost += min_insertion[0]
        
        
    def calculate_delta(self, i, to_insert):
        """
        Calculate the difference between current tour length 
        and additional after potential insert of @to_insert
        @to_insert - the city to insert 
        """
        #print("C_ik = {0}".format(self.matrix[i][to_insert]))
        #print("C_ki = {0}".format(self.matrix[to_insert][i+1]))
        #print("C_ij = {0}".format(self.matrix[i][i+1]))
        
        return self.matrix[i][to_insert] + self.matrix[to_insert][i+1] \
            - self.matrix[i][i+1]                     
            
        
    