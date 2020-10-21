# -*- coding: utf-8 -*-
"""
local search implemented with 2-opt swap

2-opt = Switch 2 edges

@author: Tom Monks
"""

from metapy.tsp.objective import tour_cost
import numpy as np
import time
from joblib import Parallel, delayed


class LocalSearchArgs(object):
    """
    Argument class for local search classes
    """
    def __init__(self):
        pass
    
    
class OrdinaryDecent2Opt(object):
    """
    
    Local (neighbourhood) search implemented as first improvement 
    with 2-opt swaps
    
    """   
    def __init__(self, args):
        """
        Constructor Method
        @init_solution = initial tour
        @matrix = matrix of travel costs
        """
        self.matrix = args.matrix
        self.set_init_solution(args.init_solution)
        #self.swapper = args.swapper
        
    
    def set_init_solution(self, solution):  
        self.solution = solution
        self.best_solutions = [solution]
        self.best_cost = tour_cost(self.solution, self.matrix)        
    
    def solve(self):
        """
        Run solution algoritm.
        Note: algorithm is the same as ordinary decent
        where 2 customers are swapped apare from call to swap 
        code.  Can I encapsulate the swap code so that it can be reused?
        """
        improvement = True
        
        while(improvement):
            improvement = False
            
            for city1 in range(1, len(self.solution) - 1):
                #print("city1: {0}".format(city1))
                for city2 in range(city1 + 1, len(self.solution) - 1):
                    #print("city2: {0}".format(city2))
                    
                    self.reverse_sublist(self.solution, city1, city2)
                    
                    new_cost = tour_cost(self.solution, self.matrix)
                    
                    #if (new_cost == self.best_cost):
                        #self.best_solutions.append(self.solution)
                        #improvement = True
                    if (new_cost < self.best_cost):
                        self.best_cost = new_cost
                        self.best_solutions = [self.solution]
                        improvement = True
                    else:
                        self.reverse_sublist(self.solution, city1, city2)
                        
                 
                    
    def reverse_sublist(self, lst, start, end):
        """
        Reverse a slice of the @lst elements between
        @start and @end
        """
        lst[start:end+1] = reversed(lst[start:end+1])
        return lst


class OrdinaryDecent2OptNew(object):
    """
    
    Local (neighbourhood) search implemented as first improvement 
    with 2-opt swaps
    
    """   
    def __init__(self, objective, init_solution):
        """
        Constructor Method
        
        Parameters:
        objective - objective function 
        init_solution = initial tour
        
        """
        self._objective = objective
        self.set_init_solution(init_solution)
        
    def set_init_solution(self, solution):  
        self.solution = solution
        self.best_solutions = [solution]
        self.best_cost = self._objective.evaluate(self.solution)        
    
    def solve(self):
        """
        Run solution algoritm.
        Note: algorithm is the same as ordinary decent
        where 2 customers are swapped apare from call to swap 
        code.  Can I encapsulate the swap code so that it can be reused?
        """
        improvement = True
        
        while(improvement):
            improvement = False
            
            for city1 in range(1, len(self.solution) - 1):
                #print("city1: {0}".format(city1))
                for city2 in range(city1 + 1, len(self.solution) - 1):
                    #print("city2: {0}".format(city2))
                    
                    self.reverse_sublist(self.solution, city1, city2)
                    
                    #new_cost = tour_cost(self.solution, self.matrix)
                    new_cost = self._objective.evaluate(self.solution)
                    #if (new_cost == self.best_cost):
                        #self.best_solutions.append(self.solution)
                        #improvement = True
                    if (new_cost < self.best_cost):
                        self.best_cost = new_cost
                        self.best_solutions = [self.solution]
                        improvement = True
                    else:
                        self.reverse_sublist(self.solution, city1, city2)
                        
                 
                    
    def reverse_sublist(self, lst, start, end):
        """
        Reverse a slice of the @lst elements between
        @start and @end

        Parameters:
        --------
        lst - np.array, vector representing a solution
        start - int, start index of sublist (inclusive)
        end - int, end index of sublist (inclusive)

        """
        lst[start:end] = lst[start:end][::-1]
        
        


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




class OrdinaryDecent2OptNew__(object):
    """
    
    Local (neighbourhood) search implemented as first improvement 
    
    """   
    def __init__(self, objective, init_solution):
        """
        Constructor Method

        Parameters:
        ----------

        @init_solution = initial tour
        @matrix = matrix of travel costs
        """

        self._objective = objective
        self.set_init_solution(init_solution)
        #self.swapper = args.swapper unused?
        
    
    def set_init_solution(self, solution):  
        self.solution = solution
        self.best_solutions = [solution]
        #self.best_cost = tour_cost(self.solution, self.matrix)   
        self.best_cost = self._objective.evaluate(solution)     
    
    def solve(self):
        
        improvement = True
        
        while(improvement):
            improvement = False
            
            for city1 in range(1, len(self.solution) - 1):
                
                for city2 in range(city1 + 1, len(self.solution) - 1):
                    
                    self.reverse_sub_list(self.solution, city1, city2)
                    
                    #new_cost = tour_cost(self.solution, self.matrix)
                    new_cost = self._objective.evaluate(self.solution)
                    
                    if (new_cost == self.best_cost):
                        self.best_solutions.append(self.solution)
                        improvement = True
                    elif (new_cost < self.best_cost):
                        self.best_cost = new_cost
                        self.best_solutions = [self.solution]
                        improvement = True
                    else:
                        self.swap_cities(city1, city2)
                 
                    
    def reverse_sub_list(self, lst, start, end):
        """
        Reverse a slice of the @lst elements between
        @start and @end
        """
        lst[start:end] = lst[start:end][::-1]
        return lst


    

class SteepestDecent2Opt(object):
    """
    
    Local (neighbourhood) search implemented as steepest decent
    with 2-opt swaps
    
    """   
    def __init__(self, args):
        """
        Constructor Method
        @init_solution = initial tour
        @matrix = matrix of travel costs
        """
        self.matrix = args.matrix
        self.set_init_solution(args.init_solution)
        
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
                    
                    self.reverse_sublist(self.solution, city1, city2)
                    
                    new_cost = tour_cost(self.solution, self.matrix)
                    
                    if (new_cost < self.best_cost):
                        self.best_cost = new_cost
                        best_swap_city1 = city1
                        best_swap_city2 = city2
                        improvement = True
                    
                    self.reverse_sublist(self.solution, city1, city2)
            
            self.reverse_sublist(self.solution, best_swap_city1, best_swap_city2)
            self.best_solutions = [self.solution]
            best_swap_city1 = 0
            best_swap_city2 = 0
                        
                        

                    
    def reverse_sublist(self, lst, start, end):
        """
        Reverse a slice of the @lst elements between
        @start and @end
        """
        lst[start:end+1] = reversed(lst[start:end+1])
        return lst