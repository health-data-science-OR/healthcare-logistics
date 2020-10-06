import numpy as np
from abc import ABC, abstractmethod

class AbstractPopulationGenerator(ABC):
    @abstractmethod
    def generate(self, population_size):
        pass

class AbstractMutator(ABC):
    @abstractmethod
    def mutate(self, individual):
        pass

class AbstractEvolutionStrategy(ABC):
    @abstractmethod
    def evolve(self, population, fitness):
        pass

class AbstractSelector(ABC):
    @abstractmethod
    def select(self, population, fitness):
        pass

class AbstractCrossoverOperator(ABC):
    @abstractmethod
    def crossover(self, parent_a, parent_b):
        pass


class PartiallyMappedCrossover(AbstractCrossoverOperator):
    '''
    Partially Mapped Crossover operator
    '''
    def __init__(self):
        pass

    def crossover(self, parent_a, parent_b):
    
        child_a = self._pmx(parent_a.copy(), parent_b)
        child_b = self._pmx(parent_b.copy(), parent_a)

        return child_a, child_b

    def _pmx(self, child, parent_to_cross):
        x_indexes = np.sort(np.random.randint(0, len(child), size=2))
        
        for index in range(x_indexes[0], x_indexes[1]):
            city = parent_to_cross[index]
            swap_index = np.where(child == city)[0][0]
            child[index], child[swap_index] = child[swap_index], child[index]

        return child
            

class TruncationSelector(AbstractSelector):
    '''
    Simple truncation selection of the mew fittest 
    individuals in the population
    '''
    def __init__(self, mew):
        self._mew = mew

    def select(self, population, fitness):
        fittest_indexes = np.argpartition(fitness, fitness.size - self._mew)[-self._mew:]
        return population[fittest_indexes]

class TournamentSelector(AbstractSelector):
    '''
    Encapsulates a popular GA selection algorithm called
    Tournament Selection.  An individual is selected at random
    (with replacement) as the best from the population and competes against
    a randomly selected (with replacement) challenger.  If the individual is
    victorious they compete in the next round.  If the challenger is successful
    they become the best and are carried forward to the next round. This is repeated
    for t rounds.  Higher values of t are more selective.  
    '''
    def __init__(self, tournament_size=2):
        '''
        Constructor

        Parameters:
        ---------
        tournament_size, int, must be >=1, (default=2)
        '''
        if tournament_size < 1:
            raise ValueError('tournamant size must int be >= 1')
        
        self._tournament_size = tournament_size
        
    def select(self, population, fitness):
        '''
        Select individual from population for breeding using
        a tournament approach.  t tournaments are conducted.

        Parameters:
        ---------
        population -    numpy.array.  Matrix of chromosomes 
        fitness -       numpy.array, vector of floats representing the
                        fitness of individual chromosomes

        Returns:
        --------
        numpy.array, vector (1D array) representing the chromosome
        that won the tournament.

        '''

        tournament_participants = np.random.randint(0, population.shape[0], size=self._tournament_size)
        best = population[np.argmax(fitness[tournament_participants])]

        return best


   

class TwoCityMutator(AbstractMutator):
    '''
    Mutates an individual tour by
    randomly swapping two cities.

    Designed to work with the TSP.
    '''
    def mutate(self, tour):
        '''
        Randomly swap two cities
        
        Parameters:
        --------
        tour, np.array, tour

        '''
        #remember that index 0 and len(tour) are start/end city
        to_swap = np.random.randint(0, len(tour) - 1, 2)

        tour[to_swap[0]], tour[to_swap[1]] = \
            tour[to_swap[1]], tour[to_swap[0]]

        return tour


class TwoOptMutator(AbstractMutator):
    '''
    Mutates an individual tour by
    randomly swapping two cities.

    Designed to work with the TSP
    '''
    def mutate(self, tour):
        '''
        Randomly reverse a section of the route
        
        Parameters:
        --------
        tour, np.array, tour

        '''
        #remember that index 0 and len(tour) are start/end city
        to_swap = np.random.randint(0, len(tour) - 1, 2)

        if to_swap[1] < to_swap[0]:
            to_swap[0], to_swap[1] = to_swap[1], to_swap[0]

        return self._reverse_sublist(tour, to_swap[0], to_swap[1])


    def _reverse_sublist(self, lst, start, end):
        """
        Reverse a slice of the @lst elements between
        @start and @end
        """
        lst[start:end] = lst[start:end][::-1]
        return lst


class GeneticAlgorithmStrategy(AbstractEvolutionStrategy):
    '''
    The Genetic evolution
    Individual chromosomes in the population
    compete to cross over and breed children.  
    Children are randomly mutated.

    Each generation is of size lambda.
    '''
    def __init__(self, _lambda, selector, xoperator, mutator):
        '''
        Constructor

        Parameters:
        --------

        _lambda -   int, controls the size of each generation. (make it even)

        selector -  AbstractSelector, selects an individual chromosome for crossover

        xoperator - AbstractCrossoverOperator, encapsulates the logic
                    crossover two selected parents
        
        mutator -   AbstractMutator, encapsulates the logic of mutation for a 
                    selected individual
        '''
 
        self._lambda = _lambda
        self._selector = selector
        self._xoperator = xoperator
        self._mutator = mutator

    
    def evolve(self, population, fitness):
        '''
        Truncation selection - only mew fittest individuals survive.  
        
        Each of these breed lambda/mew children who are mutations
        of the parent.

        Parameters:
        --------
        population -- numpy array, matrix representing a generation of tours
                      size (lambda, len(individual))

        fitness --    numpy.array, vector, size lambda, representing the cost of the 
                      tours in population

        Returns:
        --------
        numpy.array - matrix of new generation, size (lambda, len(individual))
        '''

        next_gen = np.full((self._lambda, len(population[0])),
                             fill_value=-1, dtype=np.byte)

        index = 0
        for crossover in range(int(self._lambda / 2)):
            
            parent_a = self._selector.select(population, fitness)
            parent_b = self._selector.select(population, fitness)
            
            c_a, c_b = self._xoperator.crossover(parent_a, parent_b)
           
            self._mutator.mutate(c_a)
            self._mutator.mutate(c_b)

            next_gen[index], next_gen[index+1] = c_a, c_b
            
            index += 2
        return next_gen


class ElitistGeneticAlgorithmStrategy(AbstractEvolutionStrategy):
    '''
    The Genetic evolution
    Individual chromosomes in the population
    compete to cross over and breed children.  
    Children are randomly mutated.

    Each generation is of size lambda.
    '''
    def __init__(self, mew, _lambda, selector, xoperator, mutator):
        '''
        Constructor

        Parameters:
        --------

        _lambda -   int, controls the size of each generation. (make it even)

        selector -  AbstractSelector, selects an individual chromosome for crossover

        xoperator - AbstractCrossoverOperator, encapsulates the logic
                    crossover two selected parents
        
        mutator -   AbstractMutator, encapsulates the logic of mutation for a 
                    selected individual
        '''
 
        self._mew = mew
        self._lambda = _lambda
        self._selector = selector
        self._xoperator = xoperator
        self._mutator = mutator
        self._trunc_selector = TruncationSelector(mew)

    
    def evolve(self, population, fitness):
        '''
        Truncation selection - only mew fittest individuals survive.  
        
        Each of these breed lambda/mew children who are mutations
        of the parent.

        Parameters:
        --------
        population -- numpy array, matrix representing a generation of tours
                      size (lambda, len(individual))

        fitness --    numpy.array, vector, size lambda, representing the cost of the 
                      tours in population

        Returns:
        --------
        numpy.array - matrix of new generation, size (lambda, len(individual))
        '''

        next_gen = np.full((self._mew + self._lambda, len(population[0])),
                             fill_value=-1, dtype=np.byte)


        #the n fittest chromosomes in the population (breaking ties at random)
        #this is the difference from the standard GA strategy
        fittest = self._trunc_selector.select(population, fitness)
        next_gen[:len(fittest),] = fittest                     

        index = self._mew
        for crossover in range(int(self._lambda / 2)):
            
            parent_a = self._selector.select(population, fitness)
            parent_b = self._selector.select(population, fitness)
            
            c_a, c_b = self._xoperator.crossover(parent_a, parent_b)
           
            self._mutator.mutate(c_a)
            self._mutator.mutate(c_b)

            next_gen[index], next_gen[index+1] = c_a, c_b
            
            index += 2
        return next_gen


class MuLambdaEvolutionStrategy(AbstractEvolutionStrategy):
    '''
    The (mu, lambda) evolution strategy
    The fittest mew of each generation
    produces lambda/mew offspring each of which is
    mutated.

    Each generation is of size lambda.
    '''
    def __init__(self, mu, _lambda, mutator):
        '''
        Constructor

        Parameters:
        --------
        mu -       int, controls how selectiveness the algorithm.  
                    Low values of mew relative to _lambsa mean that only the best 
                    breed in each generation and the algorithm becomes 
                    more exploitative.

        _lambda -   int, controls the size of each generation.


        mutator -   AbstractMutator, encapsulates the logic of mutation for a 
                    selected individual
        '''
        self._mu = mu
        self._lambda = _lambda
        self._selector = TruncationSelector(mu)
        self._mutator = mutator

    
    def evolve(self, population, fitness):
        '''
        Truncation selection - only mew fittest individuals survive.  
        
        Each of these breed lambda/mew children who are mutations
        of the parent.

        Parameters:
        --------
        population -- numpy array, matrix representing a generation of tours
                      size (lambda, len(individual))

        fitness --    numpy.array, vector, size lambda, representing the cost of the 
                      tours in population

        Returns:
        --------
        numpy.array - matrix of new generation, size (lambda, len(individual))
        '''

        fittest = self._selector.select(population, fitness)
        population = np.full((self._lambda, fittest[0].shape[0]),
                             fill_value=-1, dtype=np.byte)

        index = 0
        for parent in fittest:
            for child_n in range(int(self._lambda/self._mu)):
                child = self._mutator.mutate(parent.copy())
                population[index] = child
                index += 1

        return population
        

class MuPlusLambdaEvolutionStrategy(AbstractEvolutionStrategy):
    '''
    The (mu+lambda) evolution strategy
    The fittest mew of each generation
    produces lambda/mew offspring each of which is
    mutated.  The mew fittest parents compete with 
    their offspring int he new generation.

    The first generation is of size lambda.
    The second generation is of size mew+lambda
    '''
    def __init__(self, mu, _lambda, mutator):
        '''
        Constructor

        Parameters:
        --------
        mu -       int, controls how selectiveness the algorithm.  
                    Low values of mew relative to _lambsa mean that only the best 
                    breed in each generation and the algorithm becomes 
                    more exploitative.

        _lambda -   int, controls the size of each generation.

        mutator -   AbstractMutator, encapsulates the logic of mutation for an indiviudal
        '''
        self._mu = mu
        self._lambda = _lambda
        self._selector = TruncationSelector(mu)
        self._mutator = mutator

    
    def evolve(self, population, fitness):
        '''
        Only mew fittest individual survice.
        Each of these breed lambda/mew children who are mutations
        of the parent.

        Parameters:
        --------
        population -- numpy array, matrix representing a generation of tours
                      size (lambda+mu, len(tour))

        fitness     -- numpy.array, vector, size lambda, representing the fitness of the 
                       individuals in the population

        Returns:
        --------
        numpy.arrays - matric a new generation of individuals, 
                       size (lambda+mu, len(individual))
        '''

        fittest = self._selector.select(population, fitness)
        
        #this is the difference from (mew, lambda)
        #could also use np.empty - quicker for larger populations...
        population = np.full((self._lambda+self._mu, fittest[0].shape[0]),
                             0, dtype=np.byte)

        population[:len(fittest),] = fittest
    
        index = self._mu
        for parent in range(len(fittest)):
            for child_n in range(int(self._lambda/self._mu)):
                child = self._mutator.mutate(fittest[parent].copy())
                population[index] = child
                index += 1
            
        return population


    
class EvolutionaryAlgorithm(object):
    '''
    Encapsulates a simple Evolutionary algorithm
    with mutation at each generation.
    '''
    def __init__(self, initialiser, objective, _lambda, 
                 strategy, maximisation=True, generations=1000):
        '''
        Parameters:
        ---------
        tour        - np.array, cities to visit
        matrix      - np.array, cost matrix travelling from city i to city j
        _lambda     - int, initial population size
        strategy    - AbstractEvolutionStrategy, evolution stratgy
        maximisation- bool, True if the objective is a maximisation and 
                      False if objective is minimisation (default=True)
        generations - int, maximum number of generations  (default=1000)
        '''
        self._initialiser = initialiser
        self._max_generations = generations
        self._objective = objective
        self._strategy = strategy
        self._lambda = _lambda
        self._best = None
        self._best_fitness = np.inf
        
        if maximisation:
            self._negate = 1
        else:
            self._negate = -1

    def _get_best(self):
        return self._best
    
    def _get_best_fitness(self):
        return self._best_fitness * self._negate

    def solve(self):

        #population = initiation_population(self._lambda, self._tour)
        population = self._initialiser.generate(self._lambda)
        fitness = None
    
        for generation in range(self._max_generations):
            fitness = self._fitness(population)
            
            max_index = np.argmax(fitness)

            if self._best is None or (fitness[max_index] > self._best_fitness):
                self._best = population[max_index]
                self._best_fitness = fitness[max_index]
            
            population = self._strategy.evolve(population, fitness)
            

    
    def _fitness(self, population):
        fitness = np.full(len(population), -1.0, dtype=float)
        for i in range(len(population)):
            
            #specific to the TSP - needs to be encapsulated...
            #fitness[i] = tour_cost(population[i], self._matrix)
            fitness[i] = self._objective.evaluate(population[i])

        return fitness * self._negate
            
    best_solution = property(_get_best)
    best_fitness = property(_get_best_fitness)



class BasicFacilityLocationMutator(AbstractMutator):
    '''
    Mutates an individual solution by
    randomly resampling each element with constant probability

    Designed to work with Facility location problems
    
    A recommended mutation rate is 1/n where n is the solution size
    '''
    
    def __init__(self, n_candidates, solution_size, mutation_rate=None,
                 verbose=False):
        '''
        Init the mutator
        
        Parameters:
        -----------
        n_candidates: int
            Number of discrete locations available
            
        solution_size: int
            Length of a solution vector (chromosome)
            
        mutation_rate: float (optional, default = None)
            If None then set to 1 / solution_size
            
        verbose: bool (optional default = False)
            Useful for education only.  Prints out each stage of the mutation.

        '''
        self.rng = np.random.default_rng()
        self.solution_size = solution_size
        self.n_candidates = n_candidates
        self.verbose = verbose
        
        if mutation_rate is None:
            self.mutation_rate = 1 / solution_size
        else:    
            self.mutation_rate = mutation_rate
    
    def mutate(self, solution):
        '''
        Randomly mutate a facility location solution
        
        Parameters:
        --------
        solution, np.ndarray, 
            A facility allocation e.g. [10, 2, 15, 20]

        '''
        #1 = mutate element
        mask = self.rng.binomial(n=1, p=self.mutation_rate, 
                                 size=len(solution))
        
        if self.verbose:
            print(f'elements mutated: {mask}')
        
        #elements keep the same (reverse of mask)
        mutant = solution[~mask.astype(bool)]
        
        if self.verbose:
            print(f'mutant_part1: {mutant}')
        
        #remaining candidate solutions to sample from...
        candidates = np.arange(self.n_candidates)
        
        if self.verbose:
            print(f'candidates1: {candidates}')
        
        mask = np.isin(candidates, solution)
        candidates = candidates[~mask]
        
        if self.verbose:
            print(f'candidates: {candidates}')
    
        
        #new locations to add to mutant
        new_locations = self.rng.choice(candidates, 
                                        size=self.solution_size - len(mutant), 
                                        replace=False)      
        if self.verbose:
            print(f'new_locations: {new_locations}')
            
        #concate the mutant 
        mutant = np.concatenate([mutant, new_locations])

        return mutant


            
class FacilityLocationPopulationGenerator(AbstractPopulationGenerator):
    '''
    Logic for generating a random finite sized population
    of facility locations.
    '''
    def __init__(self, n_candidates, n_facilities, random_seed=None):
        self.n_candidates = n_candidates
        self.n_facilities = n_facilities
        self.rng = np.random.default_rng(random_seed)

    def generate(self, population_size):
        '''
        Generate a list of @population_size solutions. Solutions
        are randomly generated and unique to maximise
        diversity of the population.

        Parameters:
        ---------
        population_size -- the size of the population

        Returns:
        ---------
        np.ndarray. 
        
            matrix size = (population_size, n_facilities). 
            Contains the initial generation of facility locations
        '''

        #fast lookup to check if solution already exists
        population = {}

        #return data as
        population_arr = np.full((population_size, self.n_facilities), -1, 
                                 dtype=np.byte)

        i = 0
        while i < population_size:
            #sample a permutation
            new_solution = self.random_solution()
            
            #check its unique to maximise diversity
            if str(new_solution) not in population:
                population[str(new_solution)] = new_solution
                i = i + 1

        #save unique permutation
        population_arr[:,] = list(population.values())

        return population_arr
    
    
    def random_solution(self):
        '''
        construct a random solution for the facility location
        problem.  Returns vector of length p
        '''
        #create array of candidate indexes
        candidates = np.arange(self.n_candidates, dtype=np.byte)

        #sample without replacement and return array
        return self.rng.choice(candidates, size=self.n_facilities, 
                               replace=False)



class WeightedAverageObjective:
    '''
    Encapsulates logic for calculation of 
    weighted average in a simple facility location problem
    '''
    def __init__(self, demand, travel_matrix):
        '''store the demand and travel times'''
        self.demand = demand
        self.travel_matrix = travel_matrix
        
    def evaluate(self, solution):
        '''calculate the weighted average travel time for solution'''

        #only select clinics encoded with 1 in the solution (cast to bool) 
        
        mask = self.travel_matrix.columns[solution]
        active_facilities = self.travel_matrix[mask]
        
        #merge demand and travel times into a single DataFrame
        problem = self.demand.merge(active_facilities, on='sector', how='inner')
        
        #assume travel to closest facility
        problem['min_cost'] = problem.min(axis=1)

        #return weighted average
        return np.average(problem['min_cost'], 
                          weights=problem['n_patients'])



class FacilityLocationSinglePointCrossOver():
    '''
    Single point cross over for a facility location problem
    with non-binary representation.
    '''
    def __init__(self):
        self.rng = np.random.default_rng()
    
    def crossover(self, parent_a, parent_b):
        
        #generate exchange vectors
        ex_vector_a = parent_a[~np.isin(parent_a, parent_b)]
        ex_vector_b = parent_b[~np.isin(parent_b, parent_a)]
        
        #cross over points
        #copy parents if equal and no exchange possible.
        if len(ex_vector_a) > 0:
            x_point = self.rng.integers(len(ex_vector_a))

            #child a
            child_a = parent_a[np.isin(parent_a, parent_b)]
            child_a = np.concatenate([child_a, 
                                      ex_vector_a[:x_point],
                                      ex_vector_b[x_point:]])           
            #child b
            child_b = parent_b[np.isin(parent_b, parent_a)]
            child_b = np.concatenate([child_b, 
                                      ex_vector_b[:x_point],
                                      ex_vector_a[x_point:]])
        else:
            child_a = parent_a
            child_b = parent_b
               
        return child_a, child_b



    


