% The main Script for the project

% Clear the workspace
clear;
close all;


fun = @Benchmark.weierstrass;
alg = @DifferentialEvolution;
D = 20;

time = tic;

runMultipleTimes(fun, alg, D);
% runAndPlot(fun, alg);
% summary = runAll(); display(summary);

toc(time)


function summary = runAll()
    funcs = {
            @Benchmark.elliptic ...
            @Benchmark.bentcigar ...
            @Benchmark.discus ...
            @Benchmark.rosenbrock ...
            @Benchmark.ackley ...
            @Benchmark.weierstrass ...
            @Benchmark.griewank ...
            @Benchmark.rastrigin
        };

    dims = [ 2 10 20 ];
    algs = { @GeneticAlgorithm @DifferentialEvolution };
    
    summary = [];
    
    figure('units','normalized','outerposition',[0.1 0 0.8 1]);
    for d = dims
        for f = funcs
            for a = algs
                [bf, af, sf] = runMultipleTimes(cell2mat(f), cell2mat(a), d);
                summary = [summary; [ d, f, a, bf, af, sf]];
                saveas(gcf, sprintf('plots/fitness_%d_%s_%s.png', d, func2str(cell2mat(f)) ...
                   , func2str(cell2mat(a))));
            end
        end
    end
end



function [best, avg, stdv] = runMultipleTimes(fun, alg, D)

    fprintf('\n%s , %s , dimensions = %d\n', func2str(fun), func2str(alg), D);

    % run 31 times and plot the best individual in  each iteration
    hold on;
    cla;
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
        funcstr = regexprep(func2str(fun), '.*\.', '');
        funcstr(1) = upper(funcstr(1));
        subtitle(sprintf('%s , %s , dimensions = %d',  funcstr , ...
             class(ge), ge.numDimensions));
        xlabel('Number of fitness calls');
        ylabel('Best Fitness (log scale)');
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

function runAndPlot(fun, alg)
    
    ge = alg(fun, 2);

    % Open maximized figure window
    figure('units','normalized','outerposition',[0.1 0 0.7 1]);
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

    % save to file
    % saveas(gcf, sprintf('plots/%s_%s_%d.png', func2str(ge.objectiveFunction) ...
    %     , class(ge), ge.generation));

    % save as an animated gif
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    outfile = sprintf('plots/%s_%s.gif', func2str(ge.objectiveFunction) ...
        , class(ge));
    if ge.generation == 1
        imwrite(imind,cm,outfile,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,outfile,'gif','WriteMode','append');
    end


end
