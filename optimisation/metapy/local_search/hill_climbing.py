# -*- coding: utf-8 -*-
"""
hill-climbing local search and associated classes/functions

@author: Tom Monks
"""

import numpy as np
import time


class HillClimberRandomRestarts():
    '''
    Implementation of Hill-Climbing with Random Restarts
    '''
    def __init__(self, objective, localsearch, init_solution,
                 maxiter=20, random_seed=None):
        self._objective = objective
        
        self.set_init_solution(init_solution)
        self._maxiter = maxiter
        self._rng = np.random.default_rng(random_seed)
        self.localsearch = localsearch
        
    def set_init_solution(self, solution):  
        self.solution = solution
        self.best_solutions = [solution]
        
    def solve(self):
        
        local_solutions = []
        local_costs = []
                
        for i in range(self._maxiter):
            
            #create the hill climber
            self.localsearch.set_init_solution(np.copy(self.solution))
        
            #complete local search and get new solution
            self.localsearch.solve()
            
            #store local solution
            local_solutions.append(np.copy(self.localsearch.best_solutions[0]))
            local_costs.append(self.localsearch.best_cost)
                            
            #random restart
            self._rng.shuffle(self.solution)
            
        # best solution
        best_index = np.argmax(np.array(local_costs))
        self.best_cost = local_costs[best_index]
        self.best_solutions = [local_solutions[best_index]]

        

class HillClimber(object):
    '''
    Simple first improvement hill climbing algorithm
    
    '''
    def __init__(self, objective, init_solution, tweaker, maximisation=True,
                 time_limit=None):
        '''
        Constructor
        
        Params:
        ------
        objective:object
            optimisation target
        
        init_solution: np.ndarray
            numpy representation of solution
            
        tweaker: object
            tweak operation for hill climber
            
        maximisation: bool, optional (default=True)
            Is this a max or min optimisation?
            
        time_limit: float, optional (default=None)
            If set to float hill climbing termates when time limit is reached.
        '''
        self._objective = objective
        
        if maximisation:
            self._negate = 1.0
        else:
            self._negate = -1.0
        
        self.set_init_solution(init_solution)
        self._tweaker = tweaker
        
        #hill climbing time limit
        if time_limit is None:
            self._time_limit = np.Inf
        else:
            self._time_limit = time_limit
 
    def set_init_solution(self, solution):  
        self.solution = solution
        self.best_solutions = [solution]
        self.best_cost = self._objective.evaluate(solution) * self._negate 
               
    def solve(self):
        '''
        Run first improvement hill climbing
        
        Returns:
        --------
        None
        '''
        improvement = True
        start = time.time()
        
        while improvement and ((time.time() - start) < self._time_limit):
            improvement = False
            
            for city1 in range(1, len(self.solution) - 1):
                #print("city1: {0}".format(city1))
                for city2 in range(city1 + 1, len(self.solution) - 1):
                    #print("city2: {0}".format(city2))
                    
                    self._tweaker.tweak(tour=self.solution, start_index=city1, 
                                        end_index=city2)

                    neighbour_cost = \
                        self._objective.evaluate(self.solution) * self._negate

                    if (neighbour_cost > self.best_cost):
                        self.best_cost = neighbour_cost
                        self.best_solutions = [self.solution]
                        improvement = True
                    else:
                        self._tweaker.tweak(self.solution, city1, city2)
        



class TweakTwoOpt(object):
    '''
    Perform a 2-Opt swap for a tour (reverse a section of the route)
    '''
    def __init__(self, random_seed=None):
        '''
        Constructor method
        
        Params:
        ------
        random_seed: int, optional (default=None)
            control sampling in the algorithm.  By default is set
            to None i.e. random init.
        '''
        self.rng = np.random.default_rng(random_seed)
    
    def tweak(self, tour, start_index, end_index):
        '''
        Perform a 2-Opt swap for a tour (reverse a section of the route)
        
        Params:
        ------
        tour - np.ndarray
            vector representing tour
            
        start_index: int
            Index of first city in section of tour to reverse
            
        end_index: int
            Index of second city in section of tour to reverse
            
        
        '''
        self.reverse_section(tour, start_index, end_index)

    def reverse_section(self, tour, start, end):
        """
        Reverse a slice of the @tour elements between
        @start and @end. Note Operation happens in place.

        Params:
        --------
        tour - np.array, 
            vector representing a solution
            
        start - int, 
            start index of sublist (inclusive)
            
        end - int, 
            end index of sublist (inclusive)

        """
        tour[start:end] = tour[start:end][::-1]
        
        

class SimpleTweak(object):
    '''
    Perform a simple swap of two cities in a tour
    '''
    def __init__(self, random_seed=None):
        '''
        Constructor method
        
        Params:
        ------
        random_seed: int, optional (default=None)
            control sampling in the algorithm.  By default is set
            to None i.e. random init.
        '''
        self.rng = np.random.default_rng(random_seed)
    
    def tweak(self, tour, start_index, end_index):
        '''
        Swap two cities in tour
        
        Params:
        ------
        tour - np.ndarray
            vector representing tour
            
        start_index: int
            Index of first city to swap
            
        end_index: int
            Index of second city to swap
        
        '''
        tour[start_index], tour[end_index] = tour[end_index], tour[start_index]
        
        
        


class RandomTweakTwoOpt(object):
    '''
    Perform a 2-Opt swap for a tour (reverse a section of the route)
    Randomly selects section
    '''
    def __init__(self, random_seed=None):
        '''
        Constructor method
        
        Params:
        ------
        random_seed: int, optional (default=None)
            control sampling in the algorithm.  By default is set
            to None i.e. random init.
        '''
        self.rng = np.random.default_rng(random_seed)
        self._tweaker = TweakTwoOpt()
    
    def tweak(self, tour):
        '''
        Perform a 2-Opt swap for a tour (reverse a section of the route)
        
        Params:
        ------
        tour - np.ndarray
            vector representing tour
            
        Returns:
        --------
        np.ndarray
            tweaked tour
        
        '''
        sample = np.sort(self.rng.integers(0, high=len(tour), size=2))
        self.reverse_section(tour, tour[sample[0]], tour[sample[1]])
        return tour

    def reverse_section(self, tour, start, end):
        """
        Reverse a slice of the @tour elements between
        @start and @end. Note Operation happens in place.

        Params:
        --------
        tour - np.array, 
            vector representing a solution
            
        start - int, 
            start index of sublist (inclusive)
            
        end - int, 
            end index of sublist (inclusive)

        """
        tour[start:end] = tour[start:end][::-1]







    


