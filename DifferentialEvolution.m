classdef DifferentialEvolution
    
    properties (Constant)
        % Differential Evolution parameters
        POPULATION_SIZE = 100;
        MAX_GENERATIONS = 1000;
        MAX_FITNESS_EVALUATIONS = 3000;
        CROSSOVER_RATE = 0.9;
        SCALE_FACTOR = 0.5;
        DIMENSION_RANGE = [-10 10];
    end
    
    properties
        % Differential Evolution variables
        objectiveFunction;
        numDimensions;
        population;
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
        end
        
        % Run the Differential Evolution algorithm
        function [obj, bestIndividuals, bestFitness] = run(obj, log)
            if nargin < 2
                log = false;
            end
            % Run the Differential Evolution algorithm
            obj = obj.initializePopulation();
            obj = obj.evaluatePopulation();
            
            bestIndividuals = [];
            bestFitness = [];
            
            while obj.generation < obj.MAX_GENERATIONS
                if obj.numFitnessCalls >= obj.MAX_FITNESS_EVALUATIONS
                    break;
                end
                obj = obj.evolvePopulation();
                obj = obj.evaluatePopulation();
                
                if log
                    fprintf('Generation: %d, NFC: %d\n', obj.generation, obj.numFitnessCalls);
                    fprintf('Best individual:\n');
                    fprintf('\tgenes: ');
                    fprintf('%f ', obj.bestIndividual);
                    fprintf('\n\tfitness: %f', obj.bestFitness);
                    fprintf('\n');
                    fprintf('----------------------------------------\n');
                end
                
                bestIndividuals = [bestIndividuals; obj.bestIndividual];
                bestFitness = [bestFitness; obj.bestFitness];
            end
        end
        
        function obj = evolvePopulation(obj)
            obj.generation = obj.generation + 1;
            newPopulation = obj.population;
            
            for i = 1:obj.POPULATION_SIZE
                targetIdx = i;
                
                % Select three distinct individuals for mutation
                r = randperm(obj.POPULATION_SIZE, 3);
                a = obj.population(r(1), :);
                b = obj.population(r(2), :);
                c = obj.population(r(3), :);
                
                % Perform mutation
                mutated = a + obj.SCALE_FACTOR * (b - c);
                mutated = obj.clipToBounds(mutated);
                
                % Perform crossover
                crossed = obj.crossover(obj.population(targetIdx, :), mutated);
                
                % Select between the mutated and original individual
                trial = obj.selectTrialIndividual(obj.population(targetIdx, :), crossed);
                
                % Replace the target individual if the trial is better
                if obj.fitness(trial) < obj.fitness(obj.population(targetIdx, :))
                    newPopulation(targetIdx, :) = trial;
                end
            end
            
            obj.population = newPopulation;
        end
        
        function individual = crossover(obj, target, mutated)
            individual = target;
            
            for j = 1:obj.numDimensions
                if rand < obj.CROSSOVER_RATE
                    individual(j) = mutated(j);
                end
            end
        end
        
        function trial = selectTrialIndividual(obj, target, crossed)
            trial = target;
            
            j = randi(obj.numDimensions);
            for k = 1:obj.numDimensions
                if k == j || rand < obj.CROSSOVER_RATE
                    trial(k) = crossed(k);
                end
            end
        end
        
        function obj = initializePopulation(obj)
            obj.population = rand(obj.POPULATION_SIZE, obj.numDimensions) * ...
                (obj.DIMENSION_RANGE(2) - obj.DIMENSION_RANGE(1)) + obj.DIMENSION_RANGE(1);
            
            obj.generation = 1;
        end
        
        function obj = evaluatePopulation(obj)
            popFitness = zeros(1, obj.POPULATION_SIZE);
            
            for i = 1:obj.POPULATION_SIZE
                popFitness(i) = obj.fitness(obj.population(i, :));
                obj.numFitnessCalls = obj.numFitnessCalls + 1;
            end
            
            [obj.bestFitness, bestIdx] = min(popFitness);
            obj.bestIndividual = obj.population(bestIdx, :);
        end
        
        function fitness = fitness(obj, individual)
            fitness = obj.objectiveFunction(individual);
        end
        
        function clipped = clipToBounds(obj, individual)
            clipped = min(max(individual, obj.DIMENSION_RANGE(1)), obj.DIMENSION_RANGE(2));
        end
        
    end
    
end
