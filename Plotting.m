% Plotting

classdef Plotting
    
    methods(Static)
        
        %% Plotting 2D function in 3D coordinates
        function plotFunction3D(f, xx, yy)
            f2 = @(x,y) f([x, y]);
            
            [X, Y] = meshgrid(xx, yy);
            Z = arrayfun(f2, X, Y);
            
            % surface with alternate grid colors
            surf(X, Y, Z, "EdgeAlpha", 0.3, "FaceAlpha", 0.7);
            xlabel('x');
            ylabel('y');
            zlabel('f(x, y)');
            title(func2str(f));
        end
        
        %% Plotting 2D function in 2D coordinates with color
        function plotFunction3Dto2D(f, xx, yy)
            f2 = @(x,y) f([x, y]);
            
            [X, Y] = meshgrid(xx, yy);
            Z = arrayfun(f2, X, Y);
            
            contourf(X, Y, Z, 1000,  'LineColor', 'none');
            colormap jet;
            
            xlabel('x');
            ylabel('y');
            colorbar;
            title(func2str(f));
        end
        
        %% Plotting 1D function in 2D coordinates
        function plotFunction2D(f, xx)
            
            Y = arrayfun(f, xx);
            
            plot(xx, Y);
            xlabel('x');
            ylabel('f(x)');
            title(func2str(f));
        end
    end
    
end