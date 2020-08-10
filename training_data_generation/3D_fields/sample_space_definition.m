% sample_space_definition
% Christopher Zahasky
% 08/06/2019 
clear all
close all
set(0,'DefaultAxesFontSize',14, 'defaultlinelinewidth', 2,...
    'DefaultAxesTitleFontWeight', 'normal')
% adjust paths depending on computer
current_folder = pwd;
str_index = strfind(pwd, '\Dropbox');

%% INPUT
n = 25;
number_parameters = 3;

X = lhsdesign(n,number_parameters);
Xrand = rand(n,number_parameters);

figure
subplot(1,2,1)
scatter3(Xrand(:,1), Xrand(:,2), Xrand(:,3), 'k', 'filled')
axis equal
axis([0 1 0 1 0 1])
box on
title('Random')
view(-75, 10)

subplot(1,2,2)
scatter3(X(:,1), X(:,2), X(:,3), 'k', 'filled')
axis equal
axis([0 1 0 1 0 1])
box on
title('Latin Hypercube')
view(-75, 10)

%% Plot in 2D
gridlines = 6;
figure
subplot(3,2,1)
scatter(Xrand(:,1), Xrand(:,2), 'k', 'filled')
axis equal
axis([0 1 0 1])
box on
grid on 
% grid minor
set(gca,'xtick',linspace(0,1,gridlines))
set(gca,'ytick',linspace(0,1,gridlines))
set(gca, 'XTickLabel', [])
xlabel('Z plane')
title('Random')

subplot(3,2,3)
scatter(Xrand(:,1), Xrand(:,3), 'k', 'filled')
axis equal
axis([0 1 0 1])
box on
grid on 
% grid minor
set(gca,'xtick',linspace(0,1,gridlines))
set(gca,'ytick',linspace(0,1,gridlines))
set(gca, 'XTickLabel', [])
xlabel('Y plane')

subplot(3,2,5)
scatter(Xrand(:,2), Xrand(:,3), 'k', 'filled')
axis equal
axis([0 1 0 1])
box on
grid on 
% grid minor
set(gca,'xtick',linspace(0,1,gridlines))
set(gca,'ytick',linspace(0,1,gridlines))
set(gca, 'XTickLabel', [])
xlabel('X plane')

% Hypercube
subplot(3,2,2)
scatter(X(:,1), X(:,2), 'k', 'filled')
axis equal
axis([0 1 0 1])
box on
grid on 
% grid minor
set(gca,'xtick',linspace(0,1,gridlines))
set(gca,'ytick',linspace(0,1,gridlines))
set(gca, 'XTickLabel', [])
xlabel('Z plane')
title('Hypercube')

subplot(3,2,4)
scatter(X(:,1), X(:,3), 'k', 'filled')
axis equal
axis([0 1 0 1])
box on
grid on 
% grid minor
set(gca,'xtick',linspace(0,1,gridlines))
set(gca,'ytick',linspace(0,1,gridlines))
set(gca, 'XTickLabel', [])
xlabel('Y plane')

subplot(3,2,6)
scatter(X(:,2), X(:,3), 'k', 'filled')
axis equal
axis([0 1 0 1])
box on
grid on 
% grid minor
set(gca,'xtick',linspace(0,1,gridlines))
set(gca,'ytick',linspace(0,1,gridlines))
set(gca, 'XTickLabel', [])
xlabel('X plane')

%% Gaussian CDF illustration
% x = linspace(0,6);
% y = normpdf(x,2.2,0.7);
% p = normcdf(x,2.2,0.7);
% 
% figure
% subplot(2,1,1)
% plot(x,y, 'k')
% title('Gaussian PDF [\mu=3, \sigma=0.6]')
% 
% subplot(2,1,2)
% plot(x,p, 'k')
% title('Gaussian CDF [\mu=3, \sigma=0.6]')

%% Parameter file generation
number_realizations = 10000;
number_of_hyperparameters = 5;

% X = lhsdesign(number_realizations, number_of_hyperparameters);
% 
% R = [(X(:,1).*100)+0.1, (X(:,2).*100)+0.1, (X(:,3).*100)+0.1, ...
%     X(:,4).*5, (X(:,5).*1.8)+0.2];
% 
% fileID = fopen('perm_field3D_parameter_space.dat','w');
% fprintf(fileID,'%4.3E %4.3E %4.3E %4.3E %4.3E\n', R');
% fclose(fileID);

tic
fid = fopen('perm_field3D_parameter_space.dat');
Rr = textscan(fid,'%f %f %f %f %f');
fclose(fid);

Rm = cell2mat(Rr);
toc