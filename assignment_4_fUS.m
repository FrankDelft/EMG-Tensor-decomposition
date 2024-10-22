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
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%% CPD %%%%%%%%%%%%%%%%%%%%%%%%%%%
% You can use hidden_cpd_als_3d.m for this part.
% Include plots for all your claims (you can use display_brain_img.m to 
% help with the visualization of the spatial maps)
R1 = 5:7; % Range of ranks to test
options.maxiter = 300; 
options.th_relerr = 0.6;

num_ranks = length(R1);
num_rows = num_ranks;
num_cols = max(R1)+1;

shifted_stim = [zeros(best_delay, 1); stim(1:end-best_delay)];

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
    
    % Reconstruct and display the spatial map
    for comp = 1:r
        subplot(num_rows, num_cols, (idx-1)*num_cols + 1 + comp);
        imagesc(x_axis, z_axis, B1(:, comp) * (B2(:, comp).'));
        xlabel('Width [mm]');
        ylabel('Depth [mm]');
        title(['Spatial Map Rank' num2str(r) ' Component ' num2str(comp)]);
    end
end

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
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%% BTD %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fill in btd_ll1_als_3d.m.
% Include plots for all your claims (you can use display_brain_img.m to 
% help with the visualization of the spatial maps)

options.maxiter = 300; 
options.th_relerr = 0.6;
R2 = 10:20;
% R2=6;
Lr=2;
% [A, B, C, const, output] = btd_ll1_als_3d(PDI, R2, Lr, options);
% % Plot the loss curve
% figure;
% plot(output.relerr, 'LineWidth', 2);
% xlabel('Iteration');
% ylabel('Loss');
% set(gca, 'YScale', 'log');
% title('BTD Loss Curve');



num_ranks = length(R2);

for idx = 1:num_ranks
    r = R2(idx);
    [A, B, C, const, output] = btd_ll1_als_3d(PDI, r, Lr, options);

    % Correlate the stim with the best delay value with each of the columns of B3
    correlations = abs(corr(shifted_stim, C));
    fig2=figure;
    bar(correlations);
    xlabel('Component');
        ylabel('Temporal Correlation with Stimulus');
    title(['Rank = ' num2str(r)]);
  

    dirPath = ['./BTD/rank' num2str(r)];
    if ~exist(dirPath, 'dir')
        status = mkdir(dirPath);
        if status == 0
            error('Directory creation failed');
        end
    end

    saveas(fig2, ['./BTD/rank' num2str(r) '/barcoeff.jpg']);



    % Reconstruct and display the spatial map
    for comp = 1:r
        fig1=figure;
        imagesc(x_axis, z_axis, A(:, (comp-1)*Lr+1:comp*Lr) * (B(:, (comp-1)*Lr+1:comp*Lr).'));
        xlabel('Width [mm]');
        ylabel('Depth [mm]');
        title(['Spatial Map Rank' num2str(r) ' Component ' num2str(comp)]);
        saveas(fig1, ['./BTD/rank' num2str(r) '/' num2str(comp) '.jpg']);
    end
end
