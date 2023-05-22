% Genetic Algorithm
% with binary gene representation
% and single point crossover

classdef GeneticAlgorithm
    
    properties(Constant)
        % Genetic Algorithm parameters
        POPULATION_SIZE = 100;
        MAX_GENERATIONS = 1000;
        MAX_FITNESS_EVALUATIONS = 3000;
        CROSSOVER_RATE = 0.9;
        MUTATION_RATE = 0.01;
        ELITISM = true; % preserve the best individual
        TOURNAMENT_SIZE = 2; % 1 = random, >1 = tournament
        ALTENATE_ENCODING = false; % use alternate encoding or adjacent encoding
        CHROMASOME_LENGTH_PER_DIMENSION = 24;
        DIMENSION_RANGE = [-10 10];
    end
    
    properties
        % Genetic Algorithm variables
        objectiveFunction;
        numDimensions;
        population;
        bestFitness;
        bestIndividual;
        generation;
        numFitnesscalls;
    end
    
    methods
        
        function obj = GeneticAlgorithm(objectiveFunction, numDimensions)
            % GeneticAlgorithm constructor
            obj.bestFitness = 0;
            obj.bestIndividual = 0;
            obj.generation = 0;
            obj.numFitnesscalls = 0;
            obj.objectiveFunction = objectiveFunction;
            obj.numDimensions = numDimensions;
            obj.population = zeros(obj.POPULATION_SIZE ...
                , obj.CHROMASOME_LENGTH_PER_DIMENSION * obj.numDimensions);
        end
        
        % Run the Genetic Algorithm
        function [obj, bestIndividual, bestFitnesses] = run(obj, log)
            if nargin < 2
                log = false;
            end
            % Run the Genetic Algorithm
            obj = obj.runSingleStep(log);
            bestFitnesses = inf(obj.MAX_GENERATIONS);
            % evolve and evaluate
            while obj.generation < obj.MAX_GENERATIONS
                % store the best fitness and individual
                bestIndividual = obj.decode(obj.bestIndividual);
                bestFitnesses(obj.generation) = obj.bestFitness;
                if obj.numFitnesscalls >= obj.MAX_FITNESS_EVALUATIONS * obj.numDimensions
                    break;
                end
                obj = runSingleStep(obj, log);
            end
            bestFitnesses = bestFitnesses(1:obj.generation);
        end
        
        % Run a single step of the Genetic Algorithm
        function obj = runSingleStep(obj, log)
            if nargin < 2
                log = false;
            end
            % Generate the initial population or evolve the current population
            if obj.generation == 0
                obj = obj.initializePopulation();
                obj = obj.evaluatePopulation();
            else
                obj = obj.evolvePopulation();
                obj = obj.evaluatePopulation();
            end
            
            % logging
            if log
                fprintf('Generation: %d , NFC: %d\n', obj.generation, obj.numFitnesscalls);
                fprintf('Best individual:\n')
                fprintf('\tGenes: ');
                fprintf('%d', obj.bestIndividual);
                fprintf('\tdimensions: ');
                fprintf('%f ', obj.decode(obj.bestIndividual));
                fprintf('\n\tFitness: %f', obj.bestFitness);
                fprintf('\n')
                fprintf('----------------------------------------\n');
            end
        end
        
        % Initialize the population with random individuals
        function obj = initializePopulation(obj)
            % Generate a new population
            obj.population = randi([0 1], obj.POPULATION_SIZE ...
                , obj.numDimensions * obj.CHROMASOME_LENGTH_PER_DIMENSION);
            obj.generation = 1;
        end
        
        % Evaluate and sort the population by fitness
        function obj = evaluatePopulation(obj)
            % Evaluate the population
            popFitness = zeros(1, obj.POPULATION_SIZE);
            for i = 1:obj.POPULATION_SIZE
                popFitness(i) = obj.fitness(obj.population(i, :));
                obj.numFitnesscalls = obj.numFitnesscalls + 1;
            end
            % Sort the population by fitness
            [popFitness, index] = sort(popFitness);
            obj.population = obj.population(index, :);
            % Update the best fitness and individual
            obj.bestFitness = popFitness(1);
            obj.bestIndividual = obj.population(1, :);
        end
        
        function obj = evolvePopulation(obj)
            % Evolve the population
            obj.generation = obj.generation + 1;
            newPopulation = obj.population;
            
            for i = 1:floor(obj.POPULATION_SIZE/2)
                % Select parents
                parent1 = obj.selectParent();
                parent2 = obj.selectParent();
                % Crossover
                [child1, child2] = obj.crossover(parent1, parent2);
                % Mutate
                child1 = obj.mutate(child1);
                child2 = obj.mutate(child2);
                % Add children to new population
                newPopulation(i*2-1, :) = child1;
                newPopulation(i*2, :) = child2;
            end

            % Elitism
            if obj.ELITISM
                newPopulation(end, :) = obj.bestIndividual;
            end

            obj.population = newPopulation;
        end
        
        % Tournament selection
        function parent = selectParent(obj)
            % Select a parent for crossover using tournament selection
            tournament = randi(obj.POPULATION_SIZE, 1, obj.TOURNAMENT_SIZE);
            % find the best individual in the tournament
            % (its the lowest index as the population is sorted)
            i = min(tournament);
            parent = obj.population(i, :);
        end
        
        function [child1, child2] = crossover(obj, parent1, parent2)
            if rand < obj.CROSSOVER_RATE
                % Single point crossover 
                if obj.ALTENATE_ENCODING
                    crossoverPoint = randi(obj.numDimensions * obj.CHROMASOME_LENGTH_PER_DIMENSION);
                else
                    % with an additional chance at dimension boundaries
                    possibleCrossoverPoints = [1:obj.CHROMASOME_LENGTH_PER_DIMENSION*obj.numDimensions ...
                        (1:obj.numDimensions-1)*obj.CHROMASOME_LENGTH_PER_DIMENSION];
                    crossoverPoint = possibleCrossoverPoints(randi(length(possibleCrossoverPoints)));
                end
                child1 = parent1;
                child2 = parent2;
                child1(1:crossoverPoint) = parent2(1:crossoverPoint);
                child2(1:crossoverPoint) = parent1(1:crossoverPoint);
            else
                child1 = parent1;
                child2 = parent2;
            end
        end
        
        function individual = mutate(obj, individual)
            % Mutate
            if rand < obj.MUTATION_RATE
                mutationPoint = randi(length(individual));
                individual(mutationPoint) = ~individual(mutationPoint);
            end
        end
        
        function fitness = fitness(obj, individual)
            % Fitness function
            % Decode the gene using number of dimnesions and range
            x = obj.decode(individual);
            % Evaluate the objective function
            fitness = obj.objectiveFunction(x);
        end
        
        function x = decode(obj, individual)

            % Decode the gene
            % split the gene into dimensions
            if obj.ALTENATE_ENCODING
                dimensions = reshape(individual, obj.CHROMASOME_LENGTH_PER_DIMENSION ...
                    , obj.numDimensions);
            else
                dimensions = reshape(individual, obj.numDimensions ...
                    , obj.CHROMASOME_LENGTH_PER_DIMENSION);
            end
            geneMax = 2^obj.CHROMASOME_LENGTH_PER_DIMENSION - 1;
            range = obj.DIMENSION_RANGE(2) - obj.DIMENSION_RANGE(1);
            
            x = zeros(1, obj.numDimensions);
            for i = 1:obj.numDimensions
                % binary to decimal
                xi = 0;
                for j = 1:obj.CHROMASOME_LENGTH_PER_DIMENSION
                    if obj.ALTENATE_ENCODING
                        xi = xi + dimensions(j, i) * 2^(obj.CHROMASOME_LENGTH_PER_DIMENSION - j);
                    else
                        xi = xi + dimensions(i, j) * 2^(obj.CHROMASOME_LENGTH_PER_DIMENSION - j);
                    end
                end
                
                % scale to the range
                x(i) = obj.DIMENSION_RANGE(1) + range * xi / geneMax;
            end
        end
    end
end

