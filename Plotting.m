% Plotting Class:
% A collection of static methods for different types of plots. 
% The class provides functionalities for the following types of plots:
% - 3D plot of a 2D function
% - 2D color plot of a 2D function
% - 2D plot of a 1D function

classdef Plotting
    
    methods(Static)
        
        % Plots a 2D function in 3D coordinates.
        % Input:
        %   - f: the function to be plotted
        %   - xx and yy: vectors defining the grid on which the function will be evaluated
        function plotFunction3D(f, xx, yy)
            f2 = @(x,y) f([x, y]);   % Converts 2D function f to a form that can be used with arrayfun
            
            [X, Y] = meshgrid(xx, yy);   % Create a 2D grid for function evaluation
            Z = arrayfun(f2, X, Y);      % Evaluate the function on the grid
            
            % Plot the function using a surface plot with specified transparency and edge color
            surf(X, Y, Z, "FaceAlpha", 0.9, "EdgeAlpha", 0);
            colormap turbo;
            
            % Label the axes and the plot
            xlabel('x');
            ylabel('y');
            zlabel('f(x, y)');
            title(func2str(f));
        end
        
        % Plots a 2D function in 2D coordinates using colors to indicate the third dimension.
        % Input:
        %   - f: the function to be plotted
        %   - xx and yy: vectors defining the grid on which the function will be evaluated
        function plotFunction3Dto2D(f, xx, yy)
            f2 = @(x,y) f([x, y]);   % Converts 2D function f to a form that can be used with arrayfun
            
            [X, Y] = meshgrid(xx, yy);   % Create a 2D grid for function evaluation
            Z = arrayfun(f2, X, Y);      % Evaluate the function on the grid
            
            % Plot the function using filled contours
            contourf(X, Y, Z, 1000,  'LineColor', 'none');
            colormap turbo;

            % Label the axes, add a color bar, and title the plot
            xlabel('x');
            ylabel('y');
            colorbar;
            title(func2str(f));
        end
        
        % Plots a 1D function in 2D coordinates.
        % Input:
        %   - f: the function to be plotted
        %   - xx: the vector of x-values at which the function will be evaluated
        function plotFunction2D(f, xx)
            
            Y = arrayfun(f, xx);   % Evaluate the function at the given x-values
            
            % Plot the function
            plot(xx, Y);
            
            % Label the axes and the plot
            xlabel('x');
            ylabel('f(x)');
            title(func2str(f));
        end
    end
    
end
