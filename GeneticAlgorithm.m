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
        MUTATION_RATE = 0.1;
        CHROMASOME_LENGTH_PER_DIMENSION = 10;
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
            obj.population = obj.generatePopulation();
        end

        function obj = run(obj)
            % Run the Genetic Algorithm
            obj.evaluatePopulation();
            while obj.generation < obj.MAX_GENERATIONS
                obj = obj.evolvePopulation();
                obj.evaluatePopulation();
            end
        end

        function obj = evaluatePopulation(obj)
            % Evaluate the population
            for i = 1:obj.POPULATION_SIZE
                obj.population(i).fitness = obj.fitness(obj.population(i).gene);
                obj.numFitnesscalls = obj.numFitnesscalls + 1;
            end
            [obj.bestFitness, index] = max([obj.population.fitness]);
            obj.bestIndividual = obj.population(index);
        end

        function obj = evolvePopulation(obj)
            % Evolve the population
            obj.generation = obj.generation + 1;
            newPopulation = obj.population;
            for i = 1:obj.POPULATION_SIZE/2
                % Select parents
                parent1 = obj.selectParent();
                parent2 = obj.selectParent();
                % Crossover
                [child1, child2] = obj.crossover(parent1, parent2);
                % Mutate
                child1 = obj.mutate(child1);
                child2 = obj.mutate(child2);
                % Add children to new population
                newPopulation(i*2-1) = child1;
                newPopulation(i*2) = child2;
            end
            obj.population = newPopulation;
        end

        function parent = selectParent(obj)
            % Select a parent for crossover
            % Tournament selection for minimization
            % randomize the tournament size to give a chance for all
            tournamentSize = randi(obj.POPULATION_SIZE/2);
            tournament = randi(obj.POPULATION_SIZE, 1, tournamentSize);
            [~, index] = min([obj.population(tournament).fitness]);
            parent = obj.population(tournament(index));
        end

        function [child1, child2] = crossover(obj, parent1, parent2)
            if rand < obj.CROSSOVER_RATE
                % Single point crossover
                crossoverPoint = randi(length(parent1.gene));
                child1 = parent1;
                child2 = parent2;
                child1.gene(crossoverPoint:end) = parent2.gene(crossoverPoint:end);
                child2.gene(crossoverPoint:end) = parent1.gene(crossoverPoint:end);
            else
                child1 = parent1;
                child2 = parent2;
            end
         end

        function individual = mutate(obj, individual)
            % Mutate
            if rand < obj.MUTATION_RATE
                mutationPoint = randi(length(individual.gene));
                individual.gene(mutationPoint) = ~individual.gene(mutationPoint);
            end
        end

        function obj = generatePopulation(obj)
            % Generate a population
            for i = 1:obj.POPULATION_SIZE
                obj.population(i) = randi([0 1], 1, obj.numDimensions * obj.CHROMASOME_LENGTH_PER_DIMENSION);
            end
        end

        function fitness = fitness(obj, gene)
            % Fitness function
            % Decode the gene using number of dimnesions and range
            x = obj.decode(gene);
            % Evaluate the objective function
            fitness = obj.objectiveFunction(x);
        end

        function x = decode(obj, gene)
            % Decode the gene
            % split the gene into dimensions
            dimensions = reshape(gene, obj.numDimensions, obj.chromosomeLength/obj.numDimensions);
            geneMax = 2^obj.chromosomeLength/obj.numDimensions - 1;
            range = obj.DIMENSION_RANGE(2) - obj.DIMENSION_RANGE(1);

            x = zeros(1, obj.numDimensions);
            for i = 1:obj.numDimensions
                xi = bi2de(dimensions(i, :));
                % scale to the range
                x(i) = obj.DIMENSION_RANGE(1) + range * xi / geneMax;
            end
        end
    end
end