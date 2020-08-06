% convolution_demo
% Christopher Zahasky
% 13/3/2019
clear all
close all
set(0,'DefaultAxesFontSize',13, 'defaultlinelinewidth', 2,...
    'DefaultAxesTitleFontWeight', 'normal')

% Determine where your m-file's folder is.
current_folder = fileparts(which('convolution_demo')); 
% Add path to data folder
addpath([current_folder, '\data_generation']);

load('field_C01_5_and_1_50x25')

% filter
filter_size = 15;
filter = zeros(filter_size);
% filter(:,6) = 1;
filter(8,:) = 1;

% sharpen filter
% filter = [0 -1 0; -1 5 -1; 0 -1 0];

F_mat_filtered = conv2(F_mat, filter, 'valid');


figure('position', [118         558        1759         237])
subplot(1,3,1)
imagesc(x,y, F_mat);
axis equal
axis tight
colorbar;
colormap(gray)
xlabel('distance [cm]')
ylabel('distance [cm]')
title('synthetic permeability field')

subplot(1,3,2)
imagesc(filter);
axis equal
axis tight
colorbar;
colormap(gray)
% xlabel('distance [cm]')
% ylabel('distance [cm]')
title('kernel')

subplot(1,3,3)
imagesc(x,y, F_mat_filtered);
axis equal
axis tight
colorbar;
colormap(gray)
xlabel('distance [cm]')
ylabel('distance [cm]')
title('filtered permeability field')