% sample_space_definition
% Christopher Zahasky
% 04/01/2019
clear all
close all
set(0,'DefaultAxesFontSize',14, 'defaultlinelinewidth', 2,...
    'DefaultAxesTitleFontWeight', 'normal')
% adjust paths depending on computer
current_folder = pwd;
str_index = strfind(pwd, '\Dropbox');

%% INPUT
n = 10;
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
view(-38, 17)

subplot(1,2,2)
scatter3(X(:,1), X(:,2), X(:,3), 'k', 'filled')
axis equal
axis([0 1 0 1 0 1])
box on
title('Latin Hypercube')
view(-38, 17)

%% Plot in 2D
figure
subplot(3,2,1)
scatter(Xrand(:,1), Xrand(:,2), 'k', 'filled')
axis equal
axis([0 1 0 1])
box on
grid on 
% grid minor
set(gca,'xtick',linspace(0,1,n+1))
set(gca,'ytick',linspace(0,1,n+1))
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
set(gca,'xtick',linspace(0,1,n+1))
set(gca,'ytick',linspace(0,1,n+1))
set(gca, 'XTickLabel', [])
xlabel('Y plane')

subplot(3,2,5)
scatter(Xrand(:,2), Xrand(:,3), 'k', 'filled')
axis equal
axis([0 1 0 1])
box on
grid on 
% grid minor
set(gca,'xtick',linspace(0,1,n+1))
set(gca,'ytick',linspace(0,1,n+1))
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
set(gca,'xtick',linspace(0,1,n+1))
set(gca,'ytick',linspace(0,1,n+1))
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
set(gca,'xtick',linspace(0,1,n+1))
set(gca,'ytick',linspace(0,1,n+1))
set(gca, 'XTickLabel', [])
xlabel('Y plane')

subplot(3,2,6)
scatter(X(:,2), X(:,3), 'k', 'filled')
axis equal
axis([0 1 0 1])
box on
grid on 
% grid minor
set(gca,'xtick',linspace(0,1,n+1))
set(gca,'ytick',linspace(0,1,n+1))
set(gca, 'XTickLabel', [])
xlabel('X plane')

%% Gaussian CDF illustration
x = linspace(0,6);
y = normpdf(x,2.2,0.7);
p = normcdf(x,2.2,0.7);

figure
subplot(2,1,1)
plot(x,y, 'k')
xlabel('Radius [vox]')
title('Gaussian PDF [\mu=3, \sigma=0.6]')

subplot(2,1,2)
plot(x,p, 'k')
xlabel('Radius [vox]')
title('Gaussian CDF [\mu=3, \sigma=0.6]')

% porosity
x = linspace(0.10, 0.30);
y = normcdf(x, 0.205, 0.025);
figure
plot(x,y, 'k')
xlabel('porosity')


% lambda
x = linspace(0, 150);
y = normpdf(x, 80, 2.5);
figure
plot(x,y, 'k')
xlabel('lambda')

%% Parameter file generation
number_realizations = 1000;
number_seg_steps = 3;

X = lhsdesign(number_realizations,number_seg_steps);

Xrand = rand(number_realizations,7);

R = [X(:,1), Xrand(:,1).*4, Xrand(:,2).*4, Xrand(:,3).*2.5, ...
    X(:,2), Xrand(:,4).*4, Xrand(:,5).*0.9, ...
    X(:,3), Xrand(:,6), (Xrand(:,7).*140)+10];

% Sample the phi threhold
x = linspace(0.10, 0.30, 1000);
p = normcdf(x, 0.205, 0.025);

for i=1:number_realizations
    R(i,9) = x(find(p>R(i,9), 1, 'first'));
end

% Convert LHS to integers 
% Filters
R(R(:,1)>(2/3), 1)=3;
R(R(:,1)<(1/3), 1)=1;
R(R(:,1)<1, 1)=2;
% Edges
R(R(:,5)>(1/2), 5)=2;
R(R(:,5)<1, 5)=1;
% Seg
R(R(:,8)>(3/4), 8) = 4;
R(R(:,8)>(1/2) & R(:,8)< 1, 8) = 3;
R(R(:,8)>(1/4) & R(:,8)< 1, 8) = 2;
R(R(:,8)<1, 8)=1;

% fill zeros in cells with parameters defined for unused
R(R(:,1)~=1, 2)=0;
R(R(:,1)~=2, 3)=0;
R(R(:,1)~=3, 4)=0;

R(R(:,5)~=1, 6:7)=0;

R(R(:,8)~=3, 9)=0;
R(R(:,8)~=1, 10)=0;

% % porosity validation
% figure
% ptest = R(:,9);
% ptest(ptest==0)=[];
% histogram(ptest)

fileID = fopen('sample_space.dat','w');
fprintf(fileID,'%u %4.3E %4.3E %4.3E %u %4.3E %4.3E %u %3.2f %4.3E\n', R');
fclose(fileID);

tic
fid = fopen('sample_space.dat');
Rr = textscan(fid,'%f %f %f %f %f %f %f %f %f %f');
fclose(fid);

Rm = cell2mat(Rr);
toc
