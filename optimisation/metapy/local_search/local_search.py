# -*- coding: utf-8 -*-
"""
Local search algorithms for TSP (a.k.a neighbourhood search)
Implementations:
1. Ordinary Search - find first improvement
"""

from metapy.tsp.objective import tour_cost

class OrdinaryDecent(object):
    
    def __init__(self, init_solution, matrix):
        """
        Constructor Method
        @init_solution = initial tour
        @matrix = matrix of travel costs
        """
        self.matrix = matrix
        self.set_init_solution(init_solution)
    
    def set_init_solution(self, solution):  
        self.solution = solution
        self.best_solutions = [solution]
        self.best_cost = tour_cost(self.solution, self.matrix)        
    
    def solve(self):
        
        improvement = True
        
        while(improvement):
            improvement = False
            
            for city1 in range(1, len(self.solution) - 1):
                
                for city2 in range(city1 + 1, len(self.solution) - 1):
                    
                    self.swap_cities(city1, city2)
                    
                    new_cost = tour_cost(self.solution, self.matrix)
                    
                    #if (new_cost == self.best_cost):
                        #self.best_solutions.append(self.solution)
                        #improvement = True
                    if (new_cost < self.best_cost):
                        self.best_cost = new_cost
                        self.best_solutions = [self.solution]
                        improvement = True
                    else:
                        self.swap_cities(city1, city2)
                        
                        

                    
    def swap_cities(self, city1, city2):
        self.solution[city1], self.solution[city2] = \
            self.solution[city2], self.solution[city1]
        
                    
class SteepestDecent(object):
    def __init__(self, init_solution, matrix):
        """
        Constructor Method
        @init_solution = initial tour
        @matrix = matrix of travel costs
        """
        self.matrix = matrix
        self.set_init_solution(init_solution)
        
    def set_init_solution(self, solution):  
        self.solution = solution
        self.best_solutions = [solution]
        self.best_cost = tour_cost(self.solution, self.matrix)     
        
    def solve(self):
        
        improvement = True
        best_swap_city1 = 0
        best_swap_city2 = 0
        
        while(improvement):
            improvement = False
            
            for city1 in range(1, len(self.solution) - 1):
                
                for city2 in range(city1 + 1, len(self.solution) - 1):
                    
                    self.swap_cities(city1, city2)
                    
                    new_cost = tour_cost(self.solution, self.matrix)
                    
                    if (new_cost < self.best_cost):
                        self.best_cost = new_cost
                        best_swap_city1 = city1
                        best_swap_city2 = city2
                        improvement = True
                    
                    self.swap_cities(city1, city2)
            
            self.swap_cities(best_swap_city1, best_swap_city2)
            self.best_solutions = [self.solution]
            best_swap_city1 = 0
            best_swap_city2 = 0
                        
                        

                    
    def swap_cities(self, city1, city2):
        self.solution[city1], self.solution[city2] = \
            self.solution[city2], self.solution[city1]
    
    
    