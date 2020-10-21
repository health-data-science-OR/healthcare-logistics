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
    too slow for tours beyond 10 cities.
    """
    def __init__(self, init_solution, objective, maximisation=False):
        """
        Constructor Method
        
        Params:
        -------
        init_solution, np.ndarray
            initial tour
        objective, Object
            Class that implements .evaluate() interface
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
    
        #pick a random start city and then permute the rest and rejoin.
        origin = np.array([self._init_solution[0]])
        for current_tour in ite.permutations(self._init_solution[1:]):
            current_tour = np.concatenate([origin, current_tour])
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
    def __init__(self, init_solution, objective, max_iter=1000, maximisation=False):
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
