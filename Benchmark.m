% Benchmark Class:
% A collection of benchmark functions used for optimization problems.
% The class exports a list of the following benchmark functions:
% - High Conditioned Elliptic Function
% - Bent Cigar Function
% - Discus Function
% - Rosenbrock’s Function
% - Ackley’s Function
% - Weierstrass Function
% - Griewank’s Function
% - Rastrigin’s Function

classdef Benchmark
    % Benchmark functions for optimization problems
    
    methods (Static)
        
        % High Conditioned Elliptic Function
        % This is a unimodal, non-separable, and quadratic ill-conditioned function.
        function result = elliptic(input)
            dimension = length(input);
            result = sum((10^6) .^ ((0:dimension-1) / (dimension-1)) .* (input.^2));
        end
        
        % Bent Cigar Function
        % This is a unimodal, non-separable function with a smooth but narrow ridge.
        function result = bentcigar(input)
            dimension = length(input);
            result = input(1)^2 + 10^6 * sum(input(2:dimension).^2);
        end
        
        % Discus Function
        % This is a unimodal, non-separable function with one sensitive direction.
        function result = discus(input)
            dimension = length(input);
            result = 10^6 * input(1)^2 + sum(input(2:dimension).^2);
        end
        
        % Rosenbrock’s Function
        % This is a multi-modal, non-separable function, having a very narrow valley from local optimum to global optimum.
        function result = rosenbrock(input)
            dimension = length(input);
            result = sum(100 * (input(1:dimension-1).^2 - input(2:dimension)).^2 + (input(1:dimension-1) - 1).^2);
        end
        
        % Ackley’s Function
        % This is a multi-modal, non-separable function.
        function result = ackley(input)
            dimension = length(input);
            sumSquares = sum(input.^2);
            result = -20 * exp(-0.2 * sqrt(sumSquares/dimension)) - exp(sum(cos(2*pi*input))/dimension) + 20 + exp(1);
        end
        
        % Weierstrass Function
        % This is a multi-modal, non-separable, continuous but differentiable only on a set of points.
        function result = weierstrass(input)
            a = 0.5; b = 3; kmax = 20;
            dimension = length(input);
            ak = a .^ (0:kmax);
            bk = b .^ (0:kmax);
            result = sum(sum(ak .* cos(2*pi*bk .* (input' + 0.5)))) - dimension * sum(ak .* cos(2*pi*bk*0.5));
        end
        
        % Griewank’s Function
        % This is a multi-modal, rotated, non-separable function.
        function result = griewank(input)
            dimension = length(input);
            sumSquares = sum(input.^2);
            product = prod(cos(input ./ sqrt(1:dimension)));
            result = sumSquares/4000 - product + 1;
        end
        
        % Rastrigin’s Function
        % This is a multi-modal, separable function with a huge number of local optima.
        function result = rastrigin(input)
            dimension = length(input);
            result = sum(input.^2 - 10 * cos(2*pi*input) + 10);
        end
        
    end
    
end
