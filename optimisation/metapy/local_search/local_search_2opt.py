# -*- coding: utf-8 -*-
"""
local search implemented with 2-opt swap

2-opt = Switch 2 edges

@author: Tom Monks
"""

from metapy.tsp.objective import tour_cost


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