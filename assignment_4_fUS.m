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
%%
%%%% Calculate the best correlation lag and show the correlation image %%%%


max_delay = 10; % Maximum delay to consider
PCC={};
avg_abs=zeros([max_delay,1]);
for delay = 1:max_delay
    shifted_stim = [zeros(delay, 1); stim(1:end-delay)];
    pc_image = zeros([Nz,Nx]);  
    for i = 1:Nz
        for j = 1:Nx
            slice = squeeze(PDI(i, j, :));
            R = corr(slice, shifted_stim,'Type','Pearson');
            if abs(R) > 0.36
                pc_image(i, j) = abs(R);
            else
                pc_image(i, j) = 0; 
            end
            avg_abs(delay) = avg_abs(delay) + abs(R)/(Nz*Nx);
        end
    end
    fprintf('Average absolute correlation at delay %d: %f\n', delay, avg_abs);
    PCC{delay}=pc_image;
end

[~,best_delay]=max(avg_abs);

% Plot the correlation image for the current delay
    plot_version = 1;
    display_brain_img(PCC{best_delay}, log(mean_PDI), z_axis, x_axis, ...
        sprintf('Significantly Correlated Regions (Delay = %d)', best_delay), plot_version);
% % Two ways to visualize the correlation image are provided -->
% plot_version = 1;
% display_brain_img(pc_image, log(mean_PDI), z_axis, x_axis,...
%     'Significantly Correlated Regions', plot_version);
% 
% plot_version = 2;
% display_brain_img(pc_image, log(mean_PDI), z_axis, x_axis,...
%     'Significantly Correlated Regions', plot_version);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%% CPD %%%%%%%%%%%%%%%%%%%%%%%%%%%
% You can use hidden_cpd_als_3d.m for this part.
% Include plots for all your claims (you can use display_brain_img.m to 
% help with the visualization of the spatial maps)
R1 = 5:20; % Range of ranks to test
options.maxiter = 300; 
options.th_relerr = 0.6;

num_ranks = length(R1);
num_cols = 4; % Number of columns for subplots
num_rows = ceil(num_ranks / num_cols); % Number of rows for subplots

figure;
for idx = 1:num_ranks
    r = R1(idx);
    [B1, B2, B3, c, output] = cpd_als_3d(PDI, r, options);
    
    % Plot the relative error vs iterations
    figure;
    semilogy(output.relerr)
    grid on
    xlabel('Iteration number')
    ylabel('Relative error $\frac{\| T-T_{dec} \|_F}{\| T\|_F}$','interpreter','latex');
    title(['Relative error of the decomposition (Rank = ' num2str(r) ')'])
    disp(['The algorithm stopped after # iterations: ' num2str(output.numiter)])

    % Correlate the stim with the best delay value with each of the columns of B3
    shifted_stim = [zeros(best_delay, 1); stim(1:end-best_delay)];
    correlations = zeros(size(B3, 2), 1);

    for k = 1:size(B3, 2)
        correlations(k) = abs(corr(shifted_stim, B3(:, k), 'Type', 'Pearson'));
    end

    % Display the correlations in subplots
    subplot(num_rows, num_cols, idx);
    bar(correlations);
    xlabel('Component');
    ylabel('Correlation with Stimulus');
    title(['Rank = ' num2str(r)]);
end

figure;
for idx = 1:num_ranks
    r = R1(idx);
    [B1, B2, B3, c, output] = cpd_als_3d(PDI, r, options);
    
    % Reconstruct and display the spatial map in subplots
    spatial_map_cpd = zeros(size(B1, 1), size(B2, 1));
    for i = 1:r
        spatial_map_cpd = spatial_map_cpd + B1(:, i) * (B2(:, i).');
    end
    
    subplot(num_rows, num_cols, idx);
    imagesc(x_axis, z_axis, spatial_map_cpd);
    xlabel('Width [mm]');
    ylabel('Depth [mm]');
    title(['Rank = ' num2str(r)]);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%% BTD %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fill in btd_ll1_als_3d.m.
% Include plots for all your claims (you can use display_brain_img.m to 
% help with the visualization of the spatial maps)
