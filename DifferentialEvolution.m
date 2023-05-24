classdef DifferentialEvolution
    % Differential Evolution algorithm implementation class
    % This class contains the Differential Evolution (DE) optimization algorithm.
    % DE is a population-based optimization algorithm that uses differences
    % of vector population members to explore the search-space.
    
    properties(Constant)
        POPULATION_SIZE = 100;           % Size of the population
        MAX_FITNESS_EVALUATIONS = 3000;  % Maximum number of fitness evaluations
        CROSSOVER_RATE = 0.9;            % Crossover probability
        SCALE_FACTOR = 0.8;              % Scaling factor for mutation
        DIMENSION_RANGE = [-10 10];      % Search space boundaries
    end
     
    properties
        objectiveFunction;   % Function to be optimized
        numDimensions;       % Dimensionality of the optimization problem
        population;          % Current population of candidate solutions
        currentfitness;      % Current fitness values of the population
        bestFitness;         % Best fitness value found so far
        bestIndividual;      % Best individual found so far
        generation;          % Current generation number
        numFitnessCalls;     % Number of fitness function evaluations so far
    end
    
    methods
        
        function obj = DifferentialEvolution(objectiveFunction, numDimensions)
            % Constructor for the DifferentialEvolution class.
            obj.objectiveFunction = objectiveFunction;
            obj.numDimensions = numDimensions;
            
            % Initialize population, fitness and generation
            obj.population = zeros(obj.POPULATION_SIZE, obj.numDimensions);
            obj.currentfitness = inf(1, obj.POPULATION_SIZE);
            obj.bestFitness = Inf;
            obj.bestIndividual = [];
            obj.generation = 0;
            obj.numFitnessCalls = 0;
        end
        
        function [obj, bestIndividual, bestFitnesses] = run(obj, log)
            % Main method to run the Differential Evolution algorithm.
            if nargin < 2
                log = false;  % Default log option to false if not provided
            end
            
            % Initialize the best fitness and individual records
            bestFitnesses = zeros(ceil(obj.MAX_FITNESS_EVALUATIONS * obj.numDimensions/ obj.POPULATION_SIZE), 1);
            
            % Run the algorithm until the stopping condition is met
            while obj.bestFitness > 0 && obj.numFitnessCalls < obj.MAX_FITNESS_EVALUATIONS * obj.numDimensions
                obj = obj.runSingleStep(log);
                bestIndividual = obj.bestIndividual;
                bestFitnesses(obj.generation) = obj.bestFitness;
            end
        end
        
        function obj = runSingleStep(obj, log)
            % Run a single step of the Differential Evolution algorithm.
            if nargin < 2
                log = false;  % Default log option to false if not provided
            end
            
            % Initialize population for the first generation
            % For the next generations, evolve the current population
            if obj.generation == 0
                obj = obj.initializePopulation();
                obj = obj.evaluatePopulation();
            else
                obj = obj.evolvePopulation();
                obj = obj.evaluatePopulation(false);
            end
            
            % Log the progress of the algorithm if the log option is true
            if log
                fprintf('Generation: %d, Number of Fitness Calls: %d\n', obj.generation, obj.numFitnessCalls);
                fprintf('Best individual: Genes: %f , Fitness: %f\n', obj.bestIndividual, obj.bestFitness);
                fprintf('----------------------------------------\n');
            end
        end
        
        function obj = initializePopulation(obj)
            % Initialize the population with random individuals.
            obj.population = rand(obj.POPULATION_SIZE, obj.numDimensions) ...
                * (obj.DIMENSION_RANGE(2) - obj.DIMENSION_RANGE(1)) + obj.DIMENSION_RANGE(1);
            obj.generation = 1;
        end
        
        function obj = evaluatePopulation(obj, calculateFitness)
            % Evaluate the fitness of each individual in the population.
            if nargin < 2
                calculateFitness = true;
            end
            
            if calculateFitness
                for i = 1:obj.POPULATION_SIZE
                    obj.currentfitness(i) = obj.fitness(obj.population(i, :));
                    obj.numFitnessCalls = obj.numFitnessCalls + 1;
                end
            end
            
            % Sort the population by fitness
            [obj.currentfitness, index] = sort(obj.currentfitness);
            obj.population = obj.population(index, :);
            
            % Update the best fitness and individual
            obj.bestFitness = obj.currentfitness(1);
            obj.bestIndividual = obj.population(1, :);
        end
        
        function obj = evolvePopulation(obj)
            % Evolve the current population through mutation and crossover.
            obj.generation = obj.generation + 1;
            
            % Initialize trial population and fitness
            trialPopulation = obj.population;
            trialFitness = inf(1, obj.POPULATION_SIZE);
            
            for i = 1:obj.POPULATION_SIZE
                % Select parents and create a trial individual through mutation and crossover
                parents = obj.selectParents(i);
                mutant = obj.mutate(parents(1, :), parents(2, :), parents(3, :));
                trial = obj.crossover(obj.population(i, :), mutant);
                
                % Evaluate the fitness of the trial individual
                trialFitness(i) = obj.fitness(trial);
                obj.numFitnessCalls = obj.numFitnessCalls + 1;
                
                % Replace current individual if trial individual is better
                if trialFitness(i) < obj.currentfitness(i)
                    obj.population(i, :) = trial;
                    obj.currentfitness(i) = trialFitness(i);
                end
            end
        end
        
        function parents = selectParents(obj, targetIndex)
            % Select three parents for mutation (excluding the target individual).
            indices = setdiff(1:obj.POPULATION_SIZE, targetIndex); % Indices without the target
            randomIndices = randperm(obj.POPULATION_SIZE - 1, 3);
            parents = obj.population(indices(randomIndices), :);
        end
        
        function mutant = mutate(obj, parent1, parent2, parent3)
            % Generate a mutant individual through mutation operation.
            mutant = parent1 + obj.SCALE_FACTOR * (parent2 - parent3);
            
            % Wrap around the search space if a mutant exceeds the boundaries
            mask = mutant < obj.DIMENSION_RANGE(1);
            mutant(mask) = obj.DIMENSION_RANGE(2) - (obj.DIMENSION_RANGE(1) - mutant(mask));
            mask = mutant > obj.DIMENSION_RANGE(2);
            mutant(mask) = obj.DIMENSION_RANGE(1) + (mutant(mask) - obj.DIMENSION_RANGE(2));

        end
        
        function trial = crossover(obj, target, mutant)
            % Generate a trial individual through crossover operation.
            mask = rand(1, obj.numDimensions) < obj.CROSSOVER_RATE;
            if ~any(mask), mask(randi(obj.numDimensions)) = true; end
            trial = target;
            trial(mask) = mutant(mask);
        end
        
        function fitness = fitness(obj, individual)
            % Evaluate the fitness of an individual.
            fitness = obj.objectiveFunction(individual);
        end
    end
end
