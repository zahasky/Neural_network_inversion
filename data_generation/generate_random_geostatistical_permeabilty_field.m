% generate_geostatistical_permeability_field
% Christopher Zahasky
% 13/3/2019
clear all
close all
set(0,'DefaultAxesFontSize',13, 'defaultlinelinewidth', 2,...
    'DefaultAxesTitleFontWeight', 'normal')

%  Example 2 (2-D):
% build the correlation struct
corr.name = 'exp'; %Specifies the correlation type from 'gauss', 'exp', or 'turbulent'.

% corr.c0:    The scaling parameters for the correlation function. c0 may
% be a scalar for isotropic correlation or a vector for
% anisotropic correlation. In the anisotropic case, the
% vector must have d elements, where d is the dimesion of a mesh point.
corr.c0 = [1.5 0.1]; % anisotropic correlation

x = linspace(0, 10, 50);
y = linspace(0,5, 25);
[X,Y] = meshgrid(x,y); 
mesh = [X(:) Y(:)]; % 2-D mesh

% set a spatially varying variance (must be positive!)
corr.sigma = 10^2;
mean_app = 100;

tic
[F,KL] = randomfield(corr,mesh);
toc
% plot the realization
% reshape array and scale to mean
F_mat = ((reshape(F, length(y), length(x)).*0.8)+ mean_app);
% F_mat2 = reshape(KL.F2, length(y), length(x))+ mean_app;

figure
% subplot(1,2,1)
imagesc(x,y, F_mat);
axis equal
axis tight
colorbar;
colormap(gray)
xlabel('distance [cm]')
ylabel('distance [cm]')
title('synthetic permeability field')
% save('field_C01_5_and_1_50x25', 'x', 'y', 'corr', 'F_mat')
% caxis([0 1])