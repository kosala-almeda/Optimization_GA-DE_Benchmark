% The main Script for the project

% Clear the workspace
clear;
close all;


fun = @Benchmark.rastrigin;
alg = @GeneticAlgorithm;
D = 20;

time = tic;

% runMultipleTimes(fun, alg, D);
% runAndPlot(fun, alg);
sm = runAll();

toc(time)


function summaries = runAll()
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
    
    summaries = cell(length(funcs)*length(dims)*length(algs), 26);
    
    figure('units','normalized','outerposition',[0.1 0 0.8 1]);

    for di = 1:length(dims)
        d = dims(di);

        for fi = 1:length(funcs)
            f = funcs(fi);

            % plotting inside runMultipleTimes for each algorithm
            hold on;
            cla;

            for ai = 1:length(algs)
                a = algs(ai);
                [bf, af, sf, solution] = runMultipleTimes(cell2mat(f), cell2mat(a), d);
                summaries(di+fi+ai, 1:6) = { d, func2str(cell2mat(f)), func2str(cell2mat(a)), bf, af, sf };
                summaries(di+fi+ai, 7:6+d) = num2cell(solution);
            end
            
            % show legend outside the plot
            legend('Location', 'southwest');
            hold off;

            saveas(gcf, sprintf('plots/new/fitness_%d_%s.png', d, func2str(cell2mat(f))));
        end
    end
end



function [best, avg, stdv, bestSolution] = runMultipleTimes(fun, alg, D)

    fprintf('\n%s , %s , dimensions = %d\n', func2str(fun), func2str(alg), D);

    % run 31 times and capture the best fitness, individual and fitness history
    overallBestFitness = inf(1,31);
    bestIndividuals = cell(1,31);
    bestFitnessHistories = cell(1,31);
    for i = 1:31
        % ge = DifferentialEvolution(fun, 2);
        ge = alg(fun, D);
        [ge, bestIndividual, bestFitnessHistory] = ge.run();
        bestFitnessHistories{i} = bestFitnessHistory;
        overallBestFitness(i) = bestFitnessHistory(end);
        bestIndividuals{i} = bestIndividual;
        fprintf('Run %d: Best fitness = %f , Solution: %s\n', i, bestFitnessHistory(end), ...
                 strjoin(arrayfun(@(x) sprintf('%f', x), bestIndividual, 'UniformOutput', false), ', '));
    end

    % calculate the best, mean and standard deviation of the best fitness across all runs
    [best, bi] = min(overallBestFitness);
    bestSolution = bestIndividuals{bi};
    avg = mean(overallBestFitness);
    stdv = std(overallBestFitness);
    [~, wi] = max(overallBestFitness);
    mi = overallBestFitness == median(overallBestFitness);
    
    fprintf('\nOverall statistics:\n')
    fprintf('\tBest = %f\n', best);
    fprintf('\tMean = %f\n', avg);
    fprintf('\tStD = %f\n', stdv);

    medFitnessHistory = cell2mat(bestFitnessHistories(mi));
    minFitnessHistory = cell2mat(bestFitnessHistories(bi));
    maxFitnessHistory = cell2mat(bestFitnessHistories(wi));

    %  pick random color for each algorithm
    color = rand(3,3);
    algstr = func2str(alg);
    if strcmp(algstr, 'DifferentialEvolution')
        color = [ 1 0.3 0; 0.9 0.9 0; 1 0 0.2 ]; % reds
    elseif strcmp(algstr, 'GeneticAlgorithm')
        color = [ 0 0.3 1; 0 0.9 0.9; 0.2 0 1 ]; % blues
    end
    
    % plot the best fitness chart in each iteration
    % do not clear existing figure
    plot((1:length(bestFitnessHistory))*ge.POPULATION_SIZE, medFitnessHistory, ...
        'DisplayName', strtrim(regexprep(algstr, '([A-Z])', ' $1')), 'LineStyle','-', 'Color', color(1,:), 'LineWidth' , 1);
    plot((1:length(bestFitnessHistory))*ge.POPULATION_SIZE, minFitnessHistory, ...
        'DisplayName', sprintf('Best of %s', regexprep(algstr, '[^A-Z]', '')), 'LineStyle','-.', 'Color', color(2,:));
    plot((1:length(bestFitnessHistory))*ge.POPULATION_SIZE, maxFitnessHistory, ...
        'DisplayName', sprintf('Worst of %s', regexprep(algstr, '[^A-Z]', '')), 'LineStyle',':', 'Color', color(3,:));
    
    title('Evolution of best fitness');
    funcstr = regexprep(func2str(fun), '.*\.', '');
    funcstr(1) = upper(funcstr(1));
    subtitle(sprintf('%s , dimensions = %d',  funcstr , ge.numDimensions));
    xlabel('Number of fitness calls');
    ylabel('Best Fitness (log scale)');
    set(gca, 'YScale', 'log');
    set(gca, 'XMinorTick', 'on');

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
