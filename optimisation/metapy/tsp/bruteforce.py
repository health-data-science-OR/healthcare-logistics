# -*- coding: utf-8 -*-
"""
Functions and classes to enable bruteforce solution of the TSP.
Note Bruteforce is inefficient after tours exceed 5 cities
"""


import itertools as ite
import numpy as np
from sympy.utilities.iterables import multiset_permutations

from metapy.tsp.objective import tour_cost
from metapy.tsp.init_solutions import random_tour
from metapy.tsp.tsp_utility import append_base, trim_base
    

class BruteForceSolver(object):
    """
    Enumerates all permutations of a given list of cities and calculates
    waits. Note for n cities there are n! permutations!  BruteForceSolver is
    too slow for tours beyond 5 cities.
    """
    def __init__(self, init_solution, matrix):
        """
        Constructor Method
        @init_solution = initial tour
        @matrix = matrix of travel costs
        """
        self.matrix = matrix
        self.init_solution = init_solution
        #list of best solutions in case of multiple optima
        self.best_solutions = [init_solution]
        self.best_cost = tour_cost(init_solution, matrix)
    
    def all_permutations(self, tour, base_city):
        """
        Returns a list of lists containing all permutations of a
        tour.  The base_city is appended to each_list
        """
        return [append_base(list(x), base_city) for x in ite.permutations(tour)]
    
        
    
    def solve(self):
        """
        Enumerate all costs to find the minimum.
        Store solution(s)
        """
        trimmed_tour, base_city = trim_base(self.init_solution)
        perms = self.all_permutations(trimmed_tour, base_city)
        
        #print(p) for p in perms]  #uncomment if want to see all perms
    
        for current_tour in perms:
            cost = tour_cost(current_tour, self.matrix)
            #print(cost)
            
            if self.best_cost == cost:
                self.best_cost = cost
                self.best_solutions.append(current_tour)
                
            elif self.best_cost > cost:
                self.best_cost = cost
                self.best_solutions = [current_tour]


class BruteForceSolverNew(object):
    """
    Enumerates all permutations of a given list of cities and calculates
    waits. Note for n cities there are n! permutations!  BruteForceSolver is
    too slow for tours beyond 5 cities.
    """
    def __init__(self, init_solution, objective, maximisation=False):
        """
        Constructor Method
        @init_solution = initial tour
        @matrix = matrix of travel costs
        """
        self._objective = objective
        self._init_solution = init_solution

        if maximisation:
            self._negate = 1.0
        elif not maximisation:
            self._negate = -1.0
        else:
            raise ValueError('parameter maximisation must be bool True|False')

        #list of best solutions in case of multiple optima
        self.best_solutions = [init_solution]
        self._best_cost = self._objective.evaluate(init_solution) * self._negate

    def _get_best_cost(self):
        return self._best_cost * self._negate
       
    def solve(self):
        """
        Enumerate all costs to find the minimum.
        Store solution(s)
        """
        #perms = self._all_permutations(self._init_solution)
        
        #print(p) for p in perms]  #uncomment if want to see all perms
    
        #better to pick a random start city and then permute 
        #the rest and rejoin.
        for current_tour in ite.permutations(self._init_solution):
            cost = self._objective.evaluate(current_tour) * self._negate
                        
            if self._best_cost == cost:
                self._best_cost = cost
                self.best_solutions.append(current_tour)
                
            elif cost > self._best_cost:
                self._best_cost = cost
                self.best_solutions = [current_tour]

    def _all_permutations(self, tour):
        """
        Returns a list of lists containing all permutations of a
        tour.  The base_city is appended to each_list
        """
        return [np.array(x) for x in ite.permutations(tour)]

    best_cost = property(_get_best_cost)
        
                
                
            
class RandomSearch(object):
    """
    A simple global optimisation algorithm - encapsulates Random Search.  
    The algorithm is completely explorative and randomly
    samples a tour and compares if it is better than the current
    best.
    """
    def __init__(self, init_solution, objective, max_iter=1000, maximisation=True):
        """
        Constructor Method

        Parameters:
        ---------
        init_solution -- initial tour
        matrix -- matrix of travel costs
        """
        self._objective = objective
        self._init_solution = init_solution
        self._max_iter = max_iter

        if maximisation:
            self._negate = 1.0
        elif not maximisation:
            self._negate = -1.0
        else:
            raise ValueError('parameter maximisation must be bool True|False')

        #list of best solutions in case of multiple optima
        self.best_solutions = [init_solution]
        self._best_cost = self._objective.evaluate(init_solution) * self._negate

    def _get_best_cost(self):
        return self._best_cost * self._negate
        
    def solve(self):
        '''
        Random search.
        
        Loop until all iterations are complete.  
        Sample a random tour on each iterations and compare to best.
        
        '''
        
        for iteration in range(self._max_iter):
            sample_tour = np.random.permutation(self._init_solution)
            sample_cost = self._objective.evaluate(sample_tour) * self._negate

            if self.best_cost == sample_cost:
                self.best_solutions.append(sample_tour)

            elif sample_cost > self._best_cost:
                self.best_solutions = [sample_tour]
                self._best_cost = sample_cost
        
    best_cost = property(_get_best_cost)
