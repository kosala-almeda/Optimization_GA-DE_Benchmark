% This class exports the following list of benchmark functions:
% 1) High Conditioned Elliptic Function
%   - Unimodal
%   - Non-separable
%   - Quadratic ill-conditioned
% 2) Bent Cigar Function
%   - Unimodal
%   - Non-separable
%   - Smooth but narrow ridge
% 3) Discus Function
%   - Unimodal
%   - Non-separable
%   - With one sensitive direction
% 4) Rosenbrock’s Function
%   - Multi-modal
%   - Non-separable
%   - Having a very narrow valley from local optimum to global optimum
% 5) Ackley’s Function
%   - Multi-modal
%   - Non-separable
% 6) Weierstrass Function
%   - Multi-modal
%   - Non-separable
%   - Continuous but differentiable only on a set of points
% 7) Griewank’s Function
%   - Multi-modal
%   - Rotated
%   - Non-separable
% 8) Rastrigin’s Function
%   - Multi-modal
%   - Separable
%   - Local optima’s number is huge
%
% Reinventing the wheel out of quriousity :)
%%

classdef Benchmark
    % Benchmark functions for optimization problems
    
    methods (Static)
        
        %% High Conditioned Elliptic Function
        function y = elliptic(x)
            y = 0;
            D = length(x);
            for i = 1:D
                y = y + (10^6)^((i-1)/(D-1)) * x(i)^2;
            end
        end
        
        %% Bent Cigar Function
        function y = bentcigar(x)
            y = 0;
            D = length(x);
            for i = 2:D
                y = y + x(i)^2;
            end
            y = x(1)^2 + 10^6 * y;
        end
        
        %% Discus Function
        function y = discus(x)
            y = 10^6 * x(1)^2;
            D = length(x);
            for i = 2:D
                y = y + x(i)^2;
            end
        end
        
        %% Rosenbrock’s Function
        function y = rosenbrock(x)
            y = 0;
            D = length(x);
            for i = 1:D-1
                y = y + 100 * (x(i)^2 - x(i+1))^2 + (x(i) - 1)^2;
            end
        end
        
        %% Ackley’s Function
        function y = ackley(x)
            y = 0;
            D = length(x);
            for i = 1:D
                y = y + x(i)^2;
            end
            y = -20 * exp(-0.2 * sqrt(y/D)) - exp(sum(cos(2*pi*x))/D) + 20 + exp(1);
        end
        
        %% Weierstrass Function
        function y = weierstrass(x)
            y = 0;
            a = 0.5; b = 3; kmax = 20;
            D = length(x);
            for i = 1:D
                for k = 0:kmax
                    y = y + a^k * cos(2*pi*b^k*(x(i)+0.5));
                end
            end
            for k = 0:kmax
                y = y - D * a^k * cos(2*pi*b^k*0.5);
            end
        end
        
        %% Griewank’s Function
        function y = griewank(x)
            y = 0;
            D = length(x);
            for i = 1:D
                y = y + x(i)^2;
            end
            y = y/4000;
            p = 1;
            for i = 1:D
                p = p * cos(x(i)/sqrt(i));
            end
            y = y - p + 1;
        end
        
        %% Rastrigin’s Function
        function y = rastrigin(x)
            y = 0;
            D = length(x);
            for i = 1:D
                y = y + x(i)^2 - 10 * cos(2*pi*x(i)) + 10;
            end
        end
        
    end
    
end