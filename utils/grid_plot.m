function grid_plot(figures_corr, highlight_index, num_columns, x_axis, z_axis,threshold)
   
    
    correlations=figures_corr.correlations;
    figures=figures_corr.spatial_maps;
    figure;
    num_figures = length(figures);
    num_rows = ceil(num_figures / num_columns);
    for i = 1:num_figures
        ax = subplot(num_rows, num_columns, i);
        currfig = figures(i);
        
        data = currfig.data;
        data(abs(data) < threshold*max(abs(data))) = 0;

        imagesc(x_axis, z_axis, data);
        colormap(jet(256));
        title(sprintf('corr(%d) = %.2f', i, correlations(i)));
        xlabel(currfig.axis_labels(1));
        ylabel(currfig.axis_labels(2));
        colorbar;

        caxis([-max(abs(data(:))), max(abs(data(:)))]);
        if ismember(i, highlight_index)
            hold on;
            rectangle('Position', [min(x_axis), min(z_axis), range(x_axis), range(z_axis)], ...
                  'EdgeColor', 'black', 'LineWidth', 4);
            hold off;
        end
    end
    

end
