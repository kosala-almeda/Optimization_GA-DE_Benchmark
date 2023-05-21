% The main Script for the project

% Clear the workspace
clear;
close all;


fun = @Benchmark.rosenbrock;
alg = @DifferentialEvolution;
D = 2;

% runMultipleTimes(fun, alg, D);
runAndPlot(fun, alg, D);

function runMultipleTimes(fun, alg, D)

    % run 31 times and plot the best individual in  each iteration
    figure;
    hold on;
    overallBestFitness = zeros(1,31);
    for i = 1:31
        % ge = DifferentialEvolution(fun, 2);
        ge = alg(fun, D);
        [ge, bestIndividuals, bestFitness] = ge.run();

        nfc = (0:length(bestFitness)-1)*ge.POPULATION_SIZE;

        % plot the best fitness chart in each iteration
        % do not clear existing figure
        plot(nfc,bestFitness, 'DisplayName', sprintf('Run %d', i));
        title('Best Fitness in each iteration');
        subtitle(sprintf('%s , dimensions = %d', func2str(fun), ge.numDimensions));
        xlabel('Number of fitness calls');
        ylabel('Fitness (log scale)');
        set(gca, 'YScale', 'log');

        % show legend outside the plot
        legend('Location', 'eastoutside');
        
        overallBestFitness(i) = bestFitness(end);
        fprintf('Run %d: Best fitness = %f\n', i, bestFitness(end));
    end
    hold off;

    % calculate the best, mean and standard deviation of the best fitness across all runs
    best = min(overallBestFitness);
    avg = mean(overallBestFitness);
    stdv = std(overallBestFitness);
    
    fprintf('\nOverall statistics:\n')
    fprintf('\tBest = %f\n', best);
    fprintf('\tMean = %f\n', avg);
    fprintf('\tStD = %f\n', stdv);

end

function runAndPlot(fun, alg, D)
    
    ge = alg(fun, 2);

    % Open maximized figure window
    figure('units','normalized','outerposition',[0 0 1 1]);
    Plotting.plotFunction3Dto2D(ge.objectiveFunction,[-10:.1:+10], [-10:.1:+10]);
    
    % run a step plot ant wait
    while true
        ge = ge.runSingleStep();
        plotall(ge);
        % pause(0.1);
        % waitforbuttonpress;
    end
end

function plotall(ge)

    hold on;
    % print class name of the algorithm in subtitle
    subtitle(sprintf('%s , Generation: %d , Best Fitness: %f', class(ge), ge.generation, ge.bestFitness));
    
    scatterObjects = findobj(gca, 'Type', 'scatter');
    delete(scatterObjects);
    % decode entire population
    for i = 1:ge.POPULATION_SIZE
        individual = ge.population(i, :);
        % check if decode function exists to decode
        if ismethod(ge, 'decode')
            individual = ge.decode(individual);
        end
        scatter(individual(1), individual(2),'ko')
    end
    
    bestIndividual = ge.bestIndividual;
    % check if decode function exists to decode
    if ismethod(ge, 'decode')
        bestIndividual = ge.decode(bestIndividual);
    end
    scatter(bestIndividual(1), bestIndividual(2),'r*')

    hold off;
end
