classdef DifferentialEvolution
    
    properties(Constant)
        % Differential Evolution parameters
        POPULATION_SIZE = 100;
        MAX_FITNESS_EVALUATIONS = 3000;
        CROSSOVER_RATE = 0.9;
        SCALE_FACTOR = 0.8;
        DIMENSION_RANGE = [-10 10];
    end
     
    properties
        % Differential Evolution variables
        objectiveFunction;
        numDimensions;
        population;
        currentfitness;
        bestFitness;
        bestIndividual;
        generation;
        numFitnessCalls;
    end
    
    methods
        
        function obj = DifferentialEvolution(objectiveFunction, numDimensions)
            % DifferentialEvolution constructor
            obj.bestFitness = 0;
            obj.bestIndividual = 0;
            obj.generation = 0;
            obj.numFitnessCalls = 0;
            obj.objectiveFunction = objectiveFunction;
            obj.numDimensions = numDimensions;
            obj.population = zeros(obj.POPULATION_SIZE, obj.numDimensions);
            obj.currentfitness = inf(1, obj.POPULATION_SIZE);
        end
        
        % Run the Differential Evolution algorithm
        function [obj, bestIndividual, bestFitnesses] = run(obj, log)
            if nargin < 2
                log = false;
            end
            % Run the Differential Evolution algorithm
            obj = obj.runSingleStep(log);
            bestFitnesses = zeros(ceil(obj.MAX_FITNESS_EVALUATIONS * obj.numDimensions/ obj.numDimensions), 1);
            bestIndividual = obj.bestIndividual;
            bestFitnesses(obj.generation) = obj.bestFitness;
            % evolve and evaluate
            while obj.bestFitness > 0 && obj.numFitnessCalls < obj.MAX_FITNESS_EVALUATIONS * obj.numDimensions
                % store the best fitness and individual
                if obj.numFitnessCalls >= obj.MAX_FITNESS_EVALUATIONS * obj.numDimensions
                    break;
                end
                obj = obj.runSingleStep(log);
                bestIndividual = obj.bestIndividual;
                bestFitnesses(obj.generation) = obj.bestFitness;
            end
            bestFitnesses = bestFitnesses(1:obj.generation);
        end
        
        % Run a single step of the Differential Evolution algorithm
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
                % no fitness calculation for evolve
                obj = obj.evaluatePopulation(false);
            end

            % logging
            if log
                fprintf('Generation: %d, NFC: %d\n', obj.generation, obj.numFitnessCalls);
                fprintf('Best individual:\n');
                fprintf('\tGenes: ');
                fprintf('%f ', obj.bestIndividual);
                fprintf('\n\tFitness: %f', obj.bestFitness);
                fprintf('\n');
                fprintf('----------------------------------------\n');
            end
        end
        
        % Initialize the population with random individuals
        function obj = initializePopulation(obj)
            % Generate a new population
            obj.population = rand(obj.POPULATION_SIZE, obj.numDimensions) ...
                * (obj.DIMENSION_RANGE(2) - obj.DIMENSION_RANGE(1)) + obj.DIMENSION_RANGE(1);
            obj.generation = 1;
        end
        
        % Evaluate and sort the population by fitness
        function obj = evaluatePopulation(obj, calculateFitness)
            if nargin < 2
                calculateFitness = true;
            end
            if calculateFitness
                % Evaluate the population
                for i = 1:obj.POPULATION_SIZE
                    obj.currentfitness(i) = obj.fitness(obj.population(i, :));
                    obj.numFitnessCalls = obj.numFitnessCalls + 1; % finess calcualtion for gen
                end
            end
            % Sort the population by fitness
            [obj.currentfitness, index] = sort(obj.currentfitness);
            obj.population = obj.population(index, :);
            % Update the best fitness and individual
            obj.bestFitness = obj.currentfitness(1);
            obj.bestIndividual = obj.population(1, :);
        end
        
        % Evolve the population
        function obj = evolvePopulation(obj)
            % Evolve the population
            obj.generation = obj.generation + 1;
            trialPopulation = obj.population;
            trialFitness = inf(1, obj.POPULATION_SIZE);
            
            % Generate trial population
            for i = 1:obj.POPULATION_SIZE
                % Select parents
                parents = obj.selectParents(i);
                parent1 = parents(1, :);
                parent2 = parents(2, :);
                parent3 = parents(3, :);
                % Mutate
                mutant = obj.mutate(parent1, parent2, parent3);
                % Crossover
                trial = obj.crossover(obj.population(i, :), mutant);
                trialPopulation(i, :) = trial;

                % Evaluate fitness for main population and trial population
                trialFitness(i) = obj.fitness(trial);
                obj.numFitnessCalls = obj.numFitnessCalls + 1; % finess calcualtion for trial
            end
            
            % Select between trial population and main population using logical indexing
            replaceIndices = trialFitness < obj.currentfitness;
            obj.population(replaceIndices, :) = trialPopulation(replaceIndices, :);
            obj.currentfitness(replaceIndices) = trialFitness(replaceIndices);
        
        end
        
        % Select parents for mutation
        function parents = selectParents(obj, targetIndex)
            % Select parents for mutation (rand/1/bin)
            indices = 1:obj.POPULATION_SIZE;
            indices(targetIndex) = [];
            randomIndices = randperm(obj.POPULATION_SIZE - 1, 3);
            parents = obj.population(indices(randomIndices), :);
        end
        
        % Mutate the target individual
        function mutant = mutate(obj, parent1, parent2, parent3)
            % Mutate the target individual (rand/1/bin)
            difference = obj.SCALE_FACTOR * (parent2 - parent3);
            mutant = parent1 + difference;

            % WRAPPING: enter from the otherside if outside the range
            mask = mutant < obj.DIMENSION_RANGE(1);
            mutant(mask) = obj.DIMENSION_RANGE(2) - (obj.DIMENSION_RANGE(1) - mutant(mask));
            mask = mutant > obj.DIMENSION_RANGE(2);
            mutant(mask) = obj.DIMENSION_RANGE(1) + (mutant(mask) - obj.DIMENSION_RANGE(2));

        end
        
        % Perform crossover between the target and mutant individuals
        function trial = crossover(obj, target, mutant)
            % Perform crossover between the target and mutant individuals
            mask = rand(size(target)) < obj.CROSSOVER_RATE;
            % atleast one gene should be different
            if ~any(mask)
                mask(randi(obj.numDimensions)) = true;
            end
            trial = target;
            trial(mask) = mutant(mask);
        end
        
        % Evaluate the fitness of an individual
        function fitness = fitness(obj, individual)
            % Evaluate the fitness of an individual
            fitness = obj.objectiveFunction(individual);
        end
        
    end
end
