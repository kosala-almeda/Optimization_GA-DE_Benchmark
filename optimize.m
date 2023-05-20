% The main Script for the project

% Clear the workspace
clear;
close all;


fun = @Benchmark.rosenbrock;

% run 31 times and plot the best individual in each iteration
figure;
hold on;
overallBestFitness = zeros(1,31);
for i = 1:31
    ge = GeneticAlgorithm(fun, 10);
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
mean = mean(overallBestFitness);
std = std(overallBestFitness);

fprintf('\nOverall statistics:\n')
fprintf('\tBest = %f\n', best);
fprintf('\tMean = %f\n', mean);
fprintf('\tSTD = %f\n', std);

function runAndPlot(ge)
    % Open maximized figure window
    figure('units','normalized','outerposition',[0 0 1 1]);
    Plotting.plotFunction3Dto2D(ge.objectiveFunction,[-10:.1:+10], [-10:.1:+10]);
    
    % run a step plot ant wait
    while true
        ge = ge.runSingleStep();
        plotall(ge);
        pause(0.5);
        % waitforbuttonpress;
    end
end

function plotall(ge)

    hold on;
    scatterObjects = findobj(gca, 'Type', 'scatter');
    delete(scatterObjects);
    % decode entire population
    for i = 1:ge.POPULATION_SIZE
        individual = ge.decode(ge.population(i, :));
        scatter(individual(1), individual(2),'ko')
    end
    bestIndividual = ge.decode(ge.bestIndividual);
    scatter(bestIndividual(1), bestIndividual(2),'r*')

    % plot the population in 3D

    % clear only the scatter plot
    % plot3(population(:,1), population(:,2), fitness, 'bo');
    % darken the color of population with lower fitness
    % show gridlines in light color

    hold off;
end
