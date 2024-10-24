%%%%%%%%%%%%%%%%%% Clear previous work and add paths %%%%%%%%%%%%%%%%%%
close all; 
clear; 
clc; 

addpath(genpath('given'));
addpath(genpath('data'));

%%%%%%%%%%%%%%%% Load experimental data and set parameters %%%%%%%%%%%%%%%%
load('params.mat', 'params'); 
x_axis = params.x_axis; % Pixel locations along width [mm]
z_axis = params.z_axis; % Pixel locations along depth [mm]
Fs = params.Fs; % Sampling rate of the experiment

% Load the power-Doppler images
load('pdi.mat', 'PDI'); 

% Load the binary stimulus vector
load('stim.mat', 'stim');

Nz = size(PDI, 1); % Number of pixels along the depth dimension
Nx = size(PDI, 2); % Number of pixels along the width dimension
Nt = size(PDI, 3); % Number of timestamps
t_axis = 0 : 1 / Fs : (Nt - 1) / Fs; % Time axis of the experiment


%%%%%%%%%%%%%%%%%% Get to know the data %%%%%%%%%%%%%%%%%%
% Choose a timestamp and show the PDI for this timestamp
idx_t = 100;       
%
figure;
imagesc(x_axis, z_axis, PDI(:, :, idx_t)); 
xlabel('Width [mm]');
ylabel('Depth [mm]'); 
title(['PDI at ' num2str(t_axis(100)) ' seconds']);

% Choose a pixel and plot the time series for this pixel
idx_z = 30;
idx_x = 30;
pxl_time_series = squeeze(PDI(idx_z, idx_x, :));
offset = min(pxl_time_series); 
wid = max(pxl_time_series) - min(pxl_time_series);
%
figure; 
plot(t_axis, pxl_time_series); 
hold on; 
s = plot(t_axis, offset + wid * stim); % the stimulus is offset and multiplied 
        % to visualize it at the same y-axis scale as the fUS time series
xlabel('Time [s]'); 
ylabel('Power Doppler amplitude [-]');
title(['Time series of the pixel at (' num2str(z_axis(idx_z)) 'mm, ' num2str(x_axis(idx_x)) 'mm)']); 

% Show the mean PDI
% Calculate the mean PDI
mean_PDI = mean(PDI, 3);
mean_PDI = mean_PDI./(max(mean_PDI(:)));

% Display the log of mean_PDI to enhance the contrast
figure; 
imagesc(x_axis, z_axis, log(mean_PDI));  
title('Mean PDI')
ylabel('Depth [mm]')
xlabel('Width [mm]')

%%%%%%%%%%%%%%%%%%%%% Data preprocessing %%%%%%%%%%%%%%%%%%%%%
% Standardize the time series for each pixel
P = (PDI - mean(PDI, 3)) ./ std(PDI, [], 3); 

% Spatial Gaussian smoothing
ht = fspecial('gaussian', [4 4], 2);
Pg = double(convn(P, ht, 'same'));

% Filter the pixel time-series with a temporal low pass filter at 0.3 Hz 
f1 = 0.3;
[b, a] = butter(5, f1 / (Fs / 2), 'low');
PDImatrix = reshape(Pg, Nz * Nx, Nt);
Pgf = reshape(filtfilt(b, a, PDImatrix')', size(PDI));
PDI = Pgf;
PDI_matrix = reshape(PDI, Nz * Nx, Nt);
clear P Pg Pgf

%%%% Calculate the best correlation lag and show the correlation image %%%%
pc_image = [];

max_delay = round(10*Fs);
max_corr = 0;
best_delay = 0;
for delay=0:max_delay
    shifted_stim = [zeros(delay, 1); stim(1:end-delay)];
    r = corr(shifted_stim, PDI_matrix')';
    if mean(abs(r))>max_corr
        best_delay = delay;
        max_corr = mean(abs(r));
        pc_image = r;
    end
end

% Threshold resulting corrolation images
corr_thresh = 0.36;
pc_image(abs(pc_image)<corr_thresh) = 0;
pc_image = reshape(pc_image, Nz, Nx);

% Visualize results
plot_version = 1;
display_brain_img(pc_image(:,:,1), log(mean_PDI), z_axis, x_axis,...
    'Significantly Correlated Regions', plot_version);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%% SVD %%%%%%%%%%%%%%%%%%%%%%%%%%%
[U,S,V] = svd(PDI_matrix, "econ");

shifted_stim = [zeros(best_delay, 1); stim(1:end-best_delay)];

svd_corrs = abs(corr(shifted_stim, V));

figure;
% figures_corr_svd = cell(1,maxk(svd_corrs,3));
% Spatial_data = struct( 'data', [], 'title', [],'axis_labels',[]);
bar(svd_corrs(1:10));
for c = maxk(svd_corrs,3)
    indx = find(svd_corrs == c);
    
    figure;
    subplot(1,2,1);
    offset = min(V(:,indx)); 
    wid = max(V(:,indx)) - min(V(:,indx));
    plot(shifted_stim*wid+offset, 'DisplayName', 'Shifted Stimulus');
    hold on
    plot(V(:,indx), 'DisplayName', ['Component ' num2str(indx)]);
    legend('show');
    hold off
    subplot(1,2,2);
    imagesc(x_axis, z_axis, reshape(U(:,indx), Nz, Nx));

    % Spatial_data(c).data=reshape(U(:,indx), Nz, Nx);
    % Spatial_data(c).axis_labels=["Width [mm]","Depth [mm]"];
    % Spatial_data(c).title=string(['Component ' num2str(c)]);
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%% CPD %%%%%%%%%%%%%%%%%%%%%%%%%%%
% You can use hidden_cpd_als_3d.m for this part.
% Include plots for all your claims (you can use display_brain_img.m to 
% help with the visualization of the spatial maps)
R1 = 5:6; % Range of ranks to test
options.maxiter = 300; 
options.th_relerr = 0.6;

num_ranks = length(R1);
num_rows = num_ranks;
num_cols = max(R1)+1;

shifted_stim = [zeros(best_delay, 1); stim(1:end-best_delay)];
figures_corr_cpd = cell(1, num_ranks);
figure;
for idx = 1:num_ranks
    r = R1(idx);
    [B1, B2, B3, c, output] = cpd_als_3d(PDI, r, options);

    % Correlate the stim with the best delay value with each of the columns of B3
    correlations = abs(corr(shifted_stim, B3));

    % Display the correlations and spatial map in subplots
    subplot(num_rows, num_cols, (idx-1)*num_cols + 1);
    bar(correlations);
    xlabel('Component');
    ylabel('Temporal Correlation with Stimulus');
    title(['Rank = ' num2str(r)]);
    Spatial_data = struct( 'data', [], 'title', [],'axis_labels',[]);
    % Reconstruct and display the spatial map
    for comp = 1:r
        subplot(num_rows, num_cols, (idx-1)*num_cols + 1 + comp);
        data=B1(:, comp) * (B2(:, comp).');
        imagesc(x_axis, z_axis, data);
        xlabel('Width [mm]');
        ylabel('Depth [mm]');
        title(['Spatial Map Rank' num2str(r) ' Component ' num2str(comp)]);

        
        Spatial_data(comp).data= data;
        Spatial_data(comp).axis_labels=["Width [mm]","Depth [mm]"];
        Spatial_data(comp).title=string(['Component ' num2str(comp)]);
    end
end

% Identify and plot the most corrolated time courses from the CPD
for comp = 1:r 
    cor = correlations(comp);
    if cor > corr_thresh

        % Processing
        time_comp = B3(:, comp);
        if corr(shifted_stim, time_comp)<0
            time_comp = time_comp.*-1;
        end
        offset = min(time_comp); 
        wid = max(time_comp) - min(time_comp);
        
        % Plotting
        figure();
        plot(shifted_stim*wid+offset, 'DisplayName', 'Shifted Stimulus');
        hold on
        plot(time_comp, 'DisplayName', ['Component ' num2str(comp)]);
        hold off
        
        % Boilerplate
        title(['Component ' num2str(comp) ' vs Shifted Stimulus']);
        xlabel('Time');
        ylabel('Normalized Amplitude');
        legend('show');
        grid on;
    end
    figures_corr_cpd{idx} = struct('spatial_maps', Spatial_data, 'correlations', correlations);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%% BTD %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fill in btd_ll1_als_3d.m.
% Include plots for all your claims (you can use display_brain_img.m to 
% help with the visualization of the spatial maps)

R2 = 5:10; % Range of ranks to test
options.maxiter = 6000; 
options.th_relerr = 0.6;
Lr=4;

num_ranks = length(R2);
num_rows = num_ranks;
num_cols = max(R2)+1;
shifted_stim=[zeros(best_delay,1);stim(1:end-best_delay)];


figures_corr_btd = cell(1, num_ranks);
figure;
for idx = 1:num_ranks
    r = R2(idx);
    [A, B, C, const, output] = btd_ll1_als_3d(PDI, r, Lr, options);

    % Correlate the stim with the best delay value with each of the columns of B3
    correlations = abs(corr(shifted_stim, C));

    % Display the correlations and spatial map in subplots
    subplot(num_rows, num_cols, (idx-1)*num_cols + 1);
    bar(correlations);
    xlabel('Component');
    ylabel('Temporal Correlation with Stimulus');
    title(['Rank = ' num2str(r)]);
    Spatial_data = struct( 'data', [], 'title', [],'axis_labels',[]);
    % Reconstruct and display the spatial maps
    for comp = 1:r
        subplot(num_rows, num_cols, (idx-1)*num_cols + 1 + comp);
        data=A(:, (comp-1)*Lr+1:comp*Lr) * (B(:, (comp-1)*Lr+1:comp*Lr).');
        imagesc(x_axis, z_axis,data);
        xlabel('Width [mm]');
        ylabel('Depth [mm]');
        title(['Spatial Map Rank' num2str(r) ' Component ' num2str(comp)]);

        Spatial_data(comp).data= data;
        Spatial_data(comp).axis_labels=["Width [mm]","Depth [mm]"];
        Spatial_data(comp).title=string(['Component ' num2str(comp)]);
    end
     figures_corr_btd{idx} = struct('spatial_maps', Spatial_data, 'correlations', correlations);
end

%%
% Visualise BTD plots 
load("figures_corr.mat")
grid_plot(figures_corr{6}, [1],3,x_axis, z_axis,0.001);

%%
% Identify and plot the most corrolated time courses from the BTD
for comp = 1:r 
    cor = correlations(comp);
    if cor > corr_thresh

        % Processing
        time_comp = C(:, comp);
        if corr(shifted_stim, time_comp)<0
            time_comp = time_comp.*-1;
        end
        offset = min(time_comp); 
        wid = max(time_comp) - min(time_comp);
        
        % Plotting
        figure();
        plot(shifted_stim*wid+offset, 'DisplayName', 'Shifted Stimulus');
        hold on
        plot(time_comp, 'DisplayName', ['Component ' num2str(comp)]);
        hold off
        
        % Boilerplate
        title(['Component ' num2str(comp) ' vs Shifted Stimulus']);
        xlabel('Time');
        ylabel('Normalized Amplitude');
        legend('show');
        grid on;
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%% ICA %%%%%%%%%%%%%%%%%%%%%%%%%%%
R=10;

[icasig, A, W] = icatb_fastICA(PDI_matrix', 'lastEig', 10, 'numOfIC', R);


correlations = abs(corr(shifted_stim, A));
figure();
bar(correlations);
title('Correlations between Shifted Stimulus and Components');
xlabel('Component Index');
ylabel('Correlation Magnitude');


figures_corr_ICA = cell(1,1);
% Loop through each component to create individual plots
Spatial_data = struct( 'data', [], 'title', [],'axis_labels',[]);
for i = 1:R
    figure();
    
    % Spatial component plot
    subplot(1, 2, 2);
    spatial_sig = icasig(i, :);

    
    Spatial_data(i).data= reshape(spatial_sig, Nz, Nx);
    Spatial_data(i).axis_labels=["Width [mm]","Depth [mm]"];
    Spatial_data(i).title=string(['Component ' num2str(i)]);

    spatial_sig(abs(spatial_sig) < 0.25*max(abs(spatial_sig))) = 0;
    imagesc(x_axis, z_axis, reshape(spatial_sig, Nz, Nx));
    clim([-max(abs(spatial_sig)), max(abs(spatial_sig))]);
    title(['Spatial Map of Component ' num2str(i)]);
    colorbar;

    
    subplot(1, 2, 1);
    time_sig = A(:, i);
    offset = min(time_sig);
    wid = max(time_sig) - min(time_sig);

    plot(shifted_stim * wid + offset, 'DisplayName', 'Shifted Stimulus');
    hold on;
    plot(time_sig, 'DisplayName', ['Component ' num2str(i)]);
    hold off;
    
    
    
    title(['Temporal Signal Comparison for Component ' num2str(i) ' with Corrolation ' num2str(correlations(i))]);
    xlabel('Time'); % Label for x-axis
    ylabel('Signal Amplitude'); % Label for y-axis
    legend(); % Display legend for the two plots
end
figures_corr_ICA{1} = struct('spatial_maps', Spatial_data, 'correlations', correlations);
%%
grid_plot(figures_corr_ICA{1}, [1],2,x_axis, z_axis,0.001)