% This class represents a Genetic Algorithm with binary gene representation
% and single point crossover. It includes methods for running the algorithm,
% evolving the population, performing mutation, crossover, and selection operations.
classdef GeneticAlgorithm
    
    properties(Constant)
        % Parameters of the Genetic Algorithm
        POPULATION_SIZE = 100;          % The size of the population
        MAX_FITNESS_EVALUATIONS = 3000; % Maximum number of times the fitness function can be called
        CROSSOVER_RATE = 0.9;           % Probability of crossover operation
        MUTATION_RATE = 0.01;           % Probability of mutation operation
        ELITISM = true;                 % If true, the best individual is preserved across generations
        TOURNAMENT_SIZE = 2;            % The number of individuals participating in tournament selection
        ALTENATE_ENCODING = false;      % If true, use alternate encoding; otherwise, use adjacent encoding
        CHROMASOME_LENGTH_PER_DIMENSION = 24; % Length of binary representation for each dimension
        DIMENSION_RANGE = [-10 10];     % Range of values each dimension can take
    end
    
    properties
        % Variables used by the Genetic Algorithm
        objectiveFunction;  % Function to be optimized
        numDimensions;      % Number of dimensions in the problem space
        population;         % Matrix representing the current population
        currentfitness;     % Fitness values of the current population
        bestFitness;        % Fitness value of the best individual in the population
        bestIndividual;     % Best individual in the population
        generation;         % Current generation number
        numFitnesscalls;    % Number of times the fitness function has been called
    end
    
    methods
        % Constructor for the Genetic Algorithm class
        function obj = GeneticAlgorithm(objectiveFunction, numDimensions)
            % Initialize instance variables
            obj.objectiveFunction = objectiveFunction;
            obj.numDimensions = numDimensions;
            
            % Create an initial population
            obj.population = zeros(obj.POPULATION_SIZE, obj.CHROMASOME_LENGTH_PER_DIMENSION * obj.numDimensions);
            obj.currentfitness = inf(1, obj.POPULATION_SIZE);
            obj.bestFitness = Inf;
            obj.bestIndividual = [];
            obj.generation = 0;
            obj.numFitnesscalls = 0;
        end
        
        % Method to run the Genetic Algorithm
        function [obj, bestIndividual, bestFitnesses] = run(obj, log)
            % Check if log argument was supplied
            if nargin < 2
                log = false;
            end
            
            % Initialize array to store best fitness value from each generation
            bestFitnesses = zeros(ceil(obj.MAX_FITNESS_EVALUATIONS * obj.numDimensions/obj.POPULATION_SIZE), 1);

            % Main loop of the algorithm
            while obj.bestFitness > 0 && obj.numFitnesscalls < obj.MAX_FITNESS_EVALUATIONS * obj.numDimensions
                obj = obj.runSingleStep(log);       % Run a single step of the algorithm
                bestIndividual = obj.decode(obj.bestIndividual);
                bestFitnesses(obj.generation) = obj.bestFitness;
            end
        end

        % Method to run a single generation of the Genetic Algorithm
        function obj = runSingleStep(obj, log)
            % Check if log argument was supplied
            if nargin < 2
                log = false;
            end
            
            % If this is the first generation, initialize the population; otherwise, evolve the current population
            if obj.generation == 0
                obj = obj.initializePopulation();
            else
                obj = obj.evolvePopulation();
            end
            
            % Evaluate the fitness of the individuals in the population
            obj = obj.evaluatePopulation();
            
            % Display log info if logging is enabled
            if log
                fprintf('Generation: %d, Number of Fitness Calls: %d\n', obj.generation, obj.numFitnessCalls);
                fprintf('Best individual: Genes: %d , Dimensions: %f , Fitness: %f\n', ...
                    obj.bestIndividual, obj.decode(obj.bestIndividual), obj.bestFitness);
                fprintf('----------------------------------------\n');
            end
        end

        % Method to initialize the population with random individuals
        function obj = initializePopulation(obj)
            % Initialize population with random binary individuals
            obj.population = randi([0 1], obj.POPULATION_SIZE, obj.numDimensions * obj.CHROMASOME_LENGTH_PER_DIMENSION);
            obj.generation = 1;
        end

        % Method to evaluate the fitness of the population
        function obj = evaluatePopulation(obj)
            for i = 1:obj.POPULATION_SIZE
                obj.currentfitness(i) = obj.fitness(obj.population(i, :)); % Evaluate fitness of each individual
                obj.numFitnesscalls = obj.numFitnesscalls + 1;
            end
            % Sort population based on fitness
            [obj.currentfitness, index] = sort(obj.currentfitness);
            obj.population = obj.population(index, :);
            % Store fitness and individual of the best individual in the population
            obj.bestFitness = obj.currentfitness(1);
            obj.bestIndividual = obj.population(1, :);
        end

        % Method to evolve the population through selection, crossover, and mutation operations
        function obj = evolvePopulation(obj)
            obj.generation = obj.generation + 1;
            newPopulation = obj.population;
            for i = 1:floor(obj.POPULATION_SIZE/2)
                % Select two parents using tournament selection
                parent1 = obj.selectParent();
                parent2 = obj.selectParent();
                % Perform crossover to generate two children
                [child1, child2] = obj.crossover(parent1, parent2);
                % Perform mutation on the children
                child1 = obj.mutate(child1);
                child2 = obj.mutate(child2);
                % Add the children to the new population
                newPopulation(i*2-1, :) = child1;
                newPopulation(i*2, :) = child2;
            end
            
            % If elitism is enabled, include the best individual in the new population
            if obj.ELITISM
                newPopulation(end, :) = obj.bestIndividual;
            end

            obj.population = newPopulation;
        end
        
        % Tournament selection
        function parent = selectParent(obj)
            % Select a subset of the population for the tournament
            tournament = randi(obj.POPULATION_SIZE, 1, obj.TOURNAMENT_SIZE);
            % The individual with the smallest index wins the tournament (since the population is sorted by fitness)
            i = min(tournament);
            parent = obj.population(i, :);
        end
        
        % Single point crossover operation
        function [child1, child2] = crossover(obj, parent1, parent2)
            if rand < obj.CROSSOVER_RATE
                % Single point crossover
                if obj.ALTENATE_ENCODING
                    crossoverPoint = randi(obj.numDimensions * obj.CHROMASOME_LENGTH_PER_DIMENSION);
                else
                    possibleCrossoverPoints = [1:obj.CHROMASOME_LENGTH_PER_DIMENSION*obj.numDimensions ...
                        (1:obj.numDimensions-1)*obj.CHROMASOME_LENGTH_PER_DIMENSION];
                    crossoverPoint = possibleCrossoverPoints(randi(length(possibleCrossoverPoints)));
                end
                % Swap genes of parents after the crossover point to produce children
                child1 = parent1;
                child2 = parent2;
                child1(1:crossoverPoint) = parent2(1:crossoverPoint);
                child2(1:crossoverPoint) = parent1(1:crossoverPoint);
            else
                % No crossover occurs
                child1 = parent1;
                child2 = parent2;
            end
        end

        % Mutation operation
        function individual = mutate(obj, individual)
            % Perform mutation with a certain probability
            if rand < obj.MUTATION_RATE
                % Randomly select a gene for mutation
                mutationPoint = randi(length(individual));
                % Flip the selected gene (0 -> 1, 1 -> 0)
                individual(mutationPoint) = ~individual(mutationPoint);
            end
        end

        % Fitness evaluation function
        function fitness = fitness(obj, individual)
            % Decode the binary gene to its phenotypic representation
            x = obj.decode(individual);
            % Evaluate the objective function for the decoded individual
            fitness = obj.objectiveFunction(x);
        end

        % Method to decode the binary gene representation to its phenotypic representation
        function x = decode(obj, individual)
            if obj.ALTENATE_ENCODING
                dimensions = reshape(individual, obj.CHROMASOME_LENGTH_PER_DIMENSION, obj.numDimensions);
            else
                dimensions = reshape(individual, obj.numDimensions, obj.CHROMASOME_LENGTH_PER_DIMENSION);
            end
            geneMax = 2^obj.CHROMASOME_LENGTH_PER_DIMENSION - 1;
            range = obj.DIMENSION_RANGE(2) - obj.DIMENSION_RANGE(1);
            
            x = zeros(1, obj.numDimensions);
            for i = 1:obj.numDimensions
                xi = 0;
                for j = 1:obj.CHROMASOME_LENGTH_PER_DIMENSION
                    if obj.ALTENATE_ENCODING
                        xi = xi + dimensions(j, i) * 2^(obj.CHROMASOME_LENGTH_PER_DIMENSION - j);
                    else
                        xi = xi + dimensions(i, j) * 2^(obj.CHROMASOME_LENGTH_PER_DIMENSION - j);
                    end
                end
                % Scale the decoded value to the given dimension range
                x(i) = obj.DIMENSION_RANGE(1) + range * xi / geneMax;
            end
        end
    end
end
