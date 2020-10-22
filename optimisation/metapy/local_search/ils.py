# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 20:59:46 2017

@author: tm3y13
"""

from abc import ABC, abstractmethod
import numpy as np

from metapy.tsp.init_solutions import random_tour


class ILSPertubation(ABC):
    @abstractmethod
    def perturb(self, tour):
        pass


class ILSHomeBaseAcceptanceLogic(ABC):
    @abstractmethod
    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        pass
    
    
class AbstractCoolingSchedule(ABC):
    '''
    Encapsulates a cooling schedule for
    a SA algorithm.
    Abstract base class.
    Concrete implementations can be
    customised to extend the range of
    cooling schedules available to the SA.
    '''
    def __init__(self, starting_temp):
        self.starting_temp = starting_temp

    @abstractmethod
    def cool_temperature(self, k):
        pass
    

class HigherQualityHomeBase(ILSHomeBaseAcceptanceLogic):
    '''
    Accept if candidate is better than home_base
    '''

    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        if candidate_cost > home_cost:
            return candidate, candidate_cost
        else:
            return home_base, home_cost

class RandomHomeBase(ILSHomeBaseAcceptanceLogic):
    '''
    Random walk homebase
    '''
    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        return candidate, candidate_cost
    
    
class EpsilonGreedyHomeBase(ILSHomeBaseAcceptanceLogic):
    '''
    Accept if candidate is better than home_base of 
    if sampled u > epsilon otherwise explore.
    '''
    def __init__(self, epsilon=0.2, exploit=None, explore=None):
        self.epsilon = epsilon
        if exploit is None:
            self.exploit = HigherQualityHomeBase()
        else:
            self.exploit = exploit

        if explore is None:
            self.explore = RandomHomeBase()
        else:
            self.explore = explore

    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        
        u = np.random.rand()

        if u > self.epsilon:
            return self.exploit.new_home_base(home_base, 
                                              home_cost, 
                                              candidate, 
                                              candidate_cost)
        else:
            return self.explore.new_home_base(home_base, 
                                              home_cost, 
                                              candidate, 
                                              candidate_cost)
            
            
class AnnealingEpsilonGreedyHomeBase(ILSHomeBaseAcceptanceLogic):
    '''
    Accept if candidate is better than home_base of 
    if sampled u > epsilon otherwise explore.
    
    Epsilon decreases from 1 to 0.
    
    '''
    def __init__(self, maxiter_per_temp=1, annealing=None, exploit=None, 
                 explore=None, verbose=False):
        
        #create epsilon greedy hombase
        self.accept = EpsilonGreedyHomeBase(1.0, exploit, explore)
        self.actions = 0
        self.verbose = verbose
        self.maxiter_per_temp = maxiter_per_temp
        
        #scheme to reduce epsilon
        if annealing is None:
            #default is basic approach from SA. 0.95 * temp
            self.annealing = ExponentialCoolingSchedule(1.0)
            
        else:
            self.annealing = annealing
        
        #iterations at current temperature
        self.iters_at_temp = 0
       

    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        
        #new home base
        candidate, candidate_cost = \
            self.accept.new_home_base(home_base, 
                                      home_cost, 
                                      candidate, 
                                      candidate_cost)
       
        #track number of updates to home base and iters at temp
        self.iters_at_temp += 1
        
        
        if self.iters_at_temp == self.maxiter_per_temp: 
            #update epsilon
            self.actions += 1
            self.accept.epsilon = self.annealing.cool_temperature(self.actions) 
            self.iters_at_temp = 0
            if self.verbose:
                print(f'epsilon: {self.accept.epsilon}')
        
        return candidate, candidate_cost




class ExponentialCoolingSchedule(AbstractCoolingSchedule):
    '''
    Expenontial Cooling Scheme.
    Source:
    https://uk.mathworks.com/help/gads/how-simulated-annealing-works.html
    '''
    def cool_temperature(self, k):
        '''
        Cool the temperature using the scheme
        T = T0 * 0.95^k.
        Where
        T = temperature after cooling
        T0 = starting temperature
        k = iteration number (within temperature?)
        Keyword arguments:
        ------
        k -- int, iteration number (within temp?)
        Returns:
        ------
        float, new temperature
        '''
        return self.starting_temp * (0.95**k)
    
    
class TempFastCoolingSchedule(AbstractCoolingSchedule):
    '''
    Cooling the temperature quickly!
    Source:
    https://uk.mathworks.com/help/gads/how-simulated-annealing-works.html
    '''
    def cool_temperature(self, k):
        '''
        Cool the temperature using the scheme
        T = T0 / k.
        Where
        T = temperature after cooling
        T0 = starting temperature
        k = iteration number 
        Keyword arguments:
        ------
        k -- int, iteration number (within temp?)
        Returns:
        ------
        float, new temperature
        '''
        return self.starting_temp / k
    
    
    
class BoltzCoolingSchedule(AbstractCoolingSchedule):
    '''
    Boltzman Cooling Scheme.
    Source:
    https://uk.mathworks.com/help/gads/how-simulated-annealing-works.html
    '''
    def cool_temperature(self, k):
        '''
        Cool the temperature using the scheme
        T = T0 / ln(k).
        Where
        T = temperature after cooling
        T0 = starting temperature
        k = iteration number (within temperature?)
        Keyword arguments:
        ------
        k -- int, iteration number
        Returns:
        ------
        float, new temperature
        '''
        return self.starting_temp / np.log(k)
    
    

class DoubleBridgePertubation(ILSPertubation):
    '''
        Perform a random 4-opt ("double bridge") move on a tour.
        
         E.g.
        
            A--B             A  B
           /    \           /|  |\
          H      C         H------C
          |      |   -->     |  |
          G      D         G------D
           \    /           \|  |/
            F--E             F  E
        
        Where edges AB, CD, EF and GH are chosen randomly.

    '''

    def perturb(self, tour):
        '''
        Perform a random 4-opt ("double bridge") move on a tour.
        
        Returns:
        --------
        numpy.array, vector. representing the tour

        Parameters:
        --------
        tour - numpy.array, vector representing tour between cities e.g.
               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        '''

        n = len(tour)
        end = int(n/4)+1
        end = int(n/3)+1
        pos1 = np.random.randint(1, end) 
        pos2 = pos1 + np.random.randint(1, end) 
        pos3 = pos2 + np.random.randint(1, end) 

        #print(tour[:pos1], tour[pos1:pos2], tour[pos2:pos3], tour[pos3:])

        p1 = np.concatenate((tour[:pos1] , tour[pos3:]), axis=None)
        p2 = np.concatenate((tour[pos2:pos3] , tour[pos1:pos2]), axis=None)
        #this bit will need updating if switching to alternative operation.
        #return np.concatenate((p1, p2, p1[0]), axis=None)
        return np.concatenate((p1, p2), axis=None)
    
    
class TabuDoubleBridgeTweak(object):
    '''
    Wraps a DoubleBridgePertubation with a limited size memory
    
    The memory acts a tabu list and prevents the tweak from returning
    to previous solutions
    
    '''
    def __init__(self, tabu_size, init_solution):
        '''
        constructor
        
        Params:
        -------
        tabu_size: int
            Size of tabu list
            
        init_solution: np.ndarray
            First solution to add to the memory
            
        '''
        self.tabu_size = tabu_size
        
        #double bridge tweak
        self.perturber = DoubleBridgePertubation()
        
        self.history = [init_solution.tolist()]
       
        
    def perturb(self, tour):
        '''
        apply the perturbation to the tour.
        
        Params:
        ------
        tour: np.ndarray
            vector representing tour
            
        Returns:
        -------
        np.ndarray
            perturbed tour.
        '''
        candidate = self.perturber.perturb(tour)
    
        while candidate.tolist() in self.history:
            print('loop')
            candidate = self.perturber.perturb(tour)
            
        self.history.append(candidate.tolist())
        if len(self.history) > self.tabu_size:
            del self.history[0]
        
        return candidate


if __name__ == '__main__':
    tour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    d = DoubleBridgePertubation()
    d.perturb(tour)


class IteratedLocalSearch(object):
    '''
    Iterated Local Search metaheuristic 
    
    Intelligent search of local optima provided by a local search
    algorithm.
    
    Implemented to work with a fixed number of iterations.
    '''

    def __init__(self, local_search, accept=None, perturb=None, 
                 verbose=False):
        '''
        ILS constructor Method

        Parameters:
        --------
        
        local_search: object,
            hill climbing solver or similar
            
        accept: ILSHomeBaseAcceptanceLogic, optional (default=RandomHomeBase())
            contains the logic for accepting a new homebase
        
        perturb: ILSPertubation, 
            contains the logic for pertubation from the local optimimum in each 
            iteration
            
        verbose: bool, optional (default=False)
            When verbose=True the results of each iteration are printed
            Useful for learning how the algorithm works.
        '''

        self._local_search = local_search
        if accept == None:
            self._accepter = RandomHomeBase()
        else:
            self._accepter = accept

        if perturb == None:
            self._perturber = DoubleBridgePertubation()
        else:
            self._perturber = perturb
            
        self.best_solutions = []
        self.best_cost = self._local_search.best_cost
        self.verbose = verbose
        
        if verbose:
            print(f'ITERATED LOCAL SEARCH')
            print(f'Initial solution cost: {self.best_cost}')
                
    def run(self, n):
        """
        Re-run solver n times using a different initial solution
        each time.  Init solution is generated randomly each time.

        The potential power of iterated local search lies in its biased 
        sampling of the set of local optima.

        """

        current = self._local_search.solution
                
        home_base = current
        home_base_cost = self._local_search.best_cost
        self.best_cost = home_base_cost 
        
        self.best_solutions.append(current)
                
        for x in range(n):

            #Hill climb from new starting point
            self._local_search.set_init_solution(current)
            self._local_search.solve()
            current = self._local_search.best_solutions[0]

            #best cost
            iteration_best_cost = self._local_search.best_cost
            
            if iteration_best_cost > self.best_cost:
                self.best_cost = iteration_best_cost
                self.best_solutions = self._local_search.best_solutions

            elif iteration_best_cost == self.best_cost:
                [self.best_solutions.append(i) 
                     for i in self._local_search.best_solutions]

            #update homebase
            home_base, home_base_cost = \
                self._accepter.new_home_base(home_base, home_base_cost, 
                                             current, iteration_best_cost)
            if self.verbose:
                print(f'{x+1}', end=': ')
                print(f'iteration cost: {iteration_best_cost}', end='; ')
                print(f'homebase cost: {home_base_cost}', end='; ')
                print(f'best found {self.best_cost}')
                
            current = self._perturber.perturb(home_base)
        
    def get_best_solutions(self):
        return self.best_cost, self.best_solutions

  
            
            
            