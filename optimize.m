% Clear the workspace and close all figures
clear;
close all;

% Define the objective function, algorithm, and number of dimensions
fun = @Benchmark.rastrigin;
alg = @GeneticAlgorithm;
D = 20;

% Start the timer
time = tic;

% Run all combinations of functions, dimensions, and algorithms
summaries = runAll();

% Stop the timer and display the elapsed time
toc(time)

% Function to run all combinations of functions, dimensions, and algorithms
function summaries = runAll()
    % Define the benchmark functions, dimensions, and algorithms
    funcs = {
        @Benchmark.elliptic
        @Benchmark.bentcigar
        @Benchmark.discus
        @Benchmark.rosenbrock
        @Benchmark.ackley
        @Benchmark.weierstrass
        @Benchmark.griewank
        @Benchmark.rastrigin
    };
    dims = [2, 10, 20];
    algs = {@GeneticAlgorithm @DifferentialEvolution};

    % Create a cell array to store the summaries
    summaries = cell(length(funcs)*length(dims)*length(algs), 26);
    
    % Create a figure to plot the results
    figure('units','normalized','outerposition',[0.2 0.2 0.5 0.7]);

    % Iterate over dimensions
    for di = 1:length(dims)
        d = dims(di);

        % Iterate over benchmark functions
        for fi = 1:length(funcs)
            f = funcs(fi);

            % Plotting inside runMultipleTimes for each algorithm
            hold on;
            cla;

            % Iterate over algorithms
            for ai = 1:length(algs)
                a = algs(ai);
                
                % Run multiple times and capture the best fitness, individual, and fitness history
                [bf, af, sf, solution] = runMultipleTimes(cell2mat(f), cell2mat(a), d);
                
                % Store the summary information
                i = (di-1)*length(funcs)*length(algs) + (fi-1)*length(algs) + ai;
                summaries(i, 1:6) = {d, func2str(cell2mat(f)), func2str(cell2mat(a)), bf, af, sf};
                summaries(i, 7:6+d) = num2cell(solution);
            end
            
            % Show the legend outside the plot
            legend('Location', 'southwest');
            hold off;

            % Save the plot as an image
            saveas(gcf, sprintf('plots/new/fitness_%d_%s.png', d, func2str(cell2mat(f))));
        end
    end
end

% Function to run multiple times and calculate the best, average, and standard deviation of the best fitness
function [best, avg, stdv, bestSolution] = runMultipleTimes(fun, alg, D)
    fprintf('\n%s , %s , dimensions = %d\n', func2str(fun), func2str(alg), D);

    % Run the algorithm multiple times and capture the best fitness, individual, and fitness history
    overallBestFitness = inf(1, 31);
    bestIndividuals = cell(1, 31);
    bestFitnessHistories = cell(1, 31);
    for i = 1:31
        ge = alg(fun, D);
        [ge, bestIndividual, bestFitnessHistory] = ge.run();
        bestFitnessHistories{i} = bestFitnessHistory;
        overallBestFitness(i) = bestFitnessHistory(end);
        bestIndividuals{i} = bestIndividual;
        fprintf('Run %d: Best fitness = %f , Solution: %s\n', i, bestFitnessHistory(end), ...
            strjoin(arrayfun(@(x) sprintf('%f', x), bestIndividual, 'UniformOutput', false), ', '));
    end

    % Calculate the best, mean, and standard deviation of the best fitness across all runs
    [best, bi] = min(overallBestFitness);
    bestSolution = bestIndividuals{bi};
    avg = mean(overallBestFitness);
    stdv = std(overallBestFitness);
    [~, wi] = max(overallBestFitness);
    mi = overallBestFitness == median(overallBestFitness);
    
    fprintf('\nOverall statistics:\n');
    fprintf('\tBest = %f\n', best);
    fprintf('\tMean = %f\n', avg);
    fprintf('\tStD = %f\n', stdv);

    medFitnessHistory = cell2mat(bestFitnessHistories(mi));
    minFitnessHistory = cell2mat(bestFitnessHistories(bi));
    maxFitnessHistory = cell2mat(bestFitnessHistories(wi));

    % Pick a random color for each algorithm
    color = rand(3, 3);
    algstr = func2str(alg);
    if strcmp(algstr, 'DifferentialEvolution')
        color = [1 0.3 0; 0.9 0.9 0; 1 0 0.2]; % Reds
    elseif strcmp(algstr, 'GeneticAlgorithm')
        color = [0 0.3 1; 0 0.9 0.9; 0.2 0 1]; % Blues
    end
    
    % Plot the best fitness chart in each iteration
    % Do not clear the existing figure
    plot((1:length(bestFitnessHistory))*ge.POPULATION_SIZE, medFitnessHistory, ...
        'DisplayName', strtrim(regexprep(algstr, '([A-Z])', ' $1')), 'LineStyle', '-', 'Color', color(1,:), 'LineWidth', 1);
    plot((1:length(bestFitnessHistory))*ge.POPULATION_SIZE, minFitnessHistory, ...
        'DisplayName', sprintf('Best of %s', regexprep(algstr, '[^A-Z]', '')), 'LineStyle', '-.', 'Color', color(2,:));
    plot((1:length(bestFitnessHistory))*ge.POPULATION_SIZE, maxFitnessHistory, ...
        'DisplayName', sprintf('Worst of %s', regexprep(algstr, '[^A-Z]', '')), 'LineStyle', ':', 'Color', color(3,:));
    
    title('Evolution of best fitness');
    funcstr = regexprep(func2str(fun), '.*\.', '');
    funcstr(1) = upper(funcstr(1));
    subtitle(sprintf('%s , dimensions = %d', funcstr, ge.numDimensions));
    xlabel('Number of fitness calls');
    ylabel('Best Fitness (log scale)');
    set(gca, 'YScale', 'log');
    set(gca, 'XMinorTick', 'on');
end

% Function to run the algorithm and plot the results
function runAndPlot(fun, alg)
    ge = alg(fun, 2);

    % Open maximized figure window
    figure('units','normalized','outerposition',[0.1 0 0.7 0.9]);
    Plotting.plotFunction3Dto2D(ge.objectiveFunction, [-10:0.1:+10], [-10:0.1:+10]);
    
    % Run a step plot and wait for user interaction
    while true
        ge = ge.runSingleStep();
        plotall(ge);
        % pause(0.1);
        % waitforbuttonpress;
    end
end

% Function to plot the current generation of the algorithm
function plotall(ge)
    hold on;
    % Print the class name of the algorithm in the subtitle
    subtitle(sprintf('%s , Generation: %d , Best Fitness: %f', class(ge), ge.generation, ge.bestFitness));
    
    scatterObjects = findobj(gca, 'Type', 'scatter');
    delete(scatterObjects);
    
    % Decode the entire population
    for i = 1:ge.POPULATION_SIZE
        individual = ge.population(i, :);
        
        % Check if the decode function exists to decode the individual
        if ismethod(ge, 'decode')
            individual = ge.decode(individual);
        end
        scatter(individual(1), individual(2), 'ko')
    end
    
    bestIndividual = ge.bestIndividual;
    
    % Check if the decode function exists to decode the best individual
    if ismethod(ge, 'decode')
        bestIndividual = ge.decode(bestIndividual);
    end
    scatter(bestIndividual(1), bestIndividual(2), 'r*');

    

    % add next button on the figure
    % uicontrol('Style', 'pushbutton', 'String', 'Next', 'Position', [20 20 50 20], 'Callback', 'uiresume(gcbf)');
    % wait for user to press next button
    % uiwait(gcf);

    hold off;

    % Save the plot as an image
    % saveas(gcf, sprintf('plots/%s_%s_%d.png', func2str(ge.objectiveFunction), class(ge), ge.generation));

    % Save as an animated GIF
    frame = getframe(1);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    outfile = sprintf('plots/%s_%s.gif', func2str(ge.objectiveFunction), class(ge));
    if ge.generation == 1
        imwrite(imind, cm, outfile, 'gif', 'Loopcount', inf);
    else
        imwrite(imind, cm, outfile, 'gif', 'WriteMode', 'append');
    end
end
