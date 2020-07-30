% generate_geostatistical_permeability_field
% Christopher Zahasky
% 13/3/2019
clear all
close all
set(0,'DefaultAxesFontSize',13, 'defaultlinelinewidth', 2,...
    'DefaultAxesTitleFontWeight', 'normal')

%  Example 2 (2-D):
% build the correlation struct
corr.name = 'gauss'; %Specifies the correlation type from 'gauss', 'exp', or 'turbulent'.

nrow = 20;
ncol = 40;

max_shape = max([nrow, ncol]);

% x = linspace(0, 10, round(max_shape*1.5));
% y = linspace(0,10, round(max_shape*1.5));
x = linspace(0, 10, ncol);
y = linspace(0,5, nrow);
% z = linspace(0,5, nrow);
[X,Y] = meshgrid(x,y);
mesh = [X(:) Y(:)]; % 3-D mesh

tnreal = 20;
extra_factor = 2;

% perm_range in mDarcy [1 1000] = [-3 3] in log space
log_min_mD = 0;
log_max_mD = 6;


% matrix of training data information, col#1 = corr.c0(1), col2 =
% corr.c0(2), col3 = mean_k, col4 = sig_value_multiplier
% load('training_data_input.mat')
tdp_mat =zeros(tnreal*extra_factor, 5);

% 
% anistropy
tdp_mat(:,1) = rand(tnreal*extra_factor,1).*2;
% tdp_mat(:,1) = linspace(0.1, 50, tnreal*extra_factor);
tdp_mat(:,2) = rand(tnreal*extra_factor,1).*10;
%  
% fill in all perm
tdp_mat(:,3)= log_min_mD + rand(tnreal*extra_factor,1).*(log_max_mD-log_min_mD);
%  
% fill in all sig
tdp_mat(:,4)= rand(tnreal*extra_factor,1).*10;
% tdp_mat(:,4) = linspace(0.1, 50, tnreal*extra_factor);
% 
% fill in all rotation
tdp_mat(:,5)= rand(tnreal*extra_factor,1).*180;

% iteration counter
n = 1;
% n=tnreal;
% rejection counter
nn = 1;

for i=1:tnreal
    % corr.c0:    The scaling parameters for the correlation function. c0 may
    % be a scalar for isotropic correlation or a vector for
    % anisotropic correlation. In the anisotropic case, the
    % vector must have d elements, where d is the dimesion of a mesh point.
    corr.c0 = [tdp_mat(n,1), tdp_mat(n,2)]; % anisotropic correlation
%     corr.c0 = [50, 0.1]; % anisotropic correlation
    
    % Mean permeability [mD]
    %     ks = log_min_D+rand(1)*(log_max_D-log_min_D);
    mean_k = 10^tdp_mat(n,3);
    % mean_k = 500;
    
    % set a spatially varying variance (must be positive!)
%     corr.sigma = 10^tdp_mat(n,4)*mean_k;
    corr.sigma = tdp_mat(n,4)*mean_k;
    
    [F,KL] = randomfield(corr,mesh, 'mean', mean_k);
    % If there are values below zero then set them equal to 0.1 mD
    F(F<0)= 0.1;
    
    % Calculate aspect ratio
    aspect_ratio = max(F)/min(F);
    
    
    F_mat = reshape(F, length(y), length(x));
    
    % Rotate matrix
%     RF_mat = imrotate(F_mat, tdp_mat(n,5), 'bilinear', 'loose');
%     r_size = size(RF_mat);
%     r_offset = floor((r_size(1)-nrow)/2);
%     c_offset = floor((r_size(2)-ncol)/2);
%     RF_crop = RF_mat(r_offset:r_offset+nrow-1, c_offset:c_offset+ncol-1);
%     % Convert from mD to D
%     F_mat_D = RF_crop./1000;
    
    % Convert from mD to D
    F_mat_D = F_mat./1000;
    % Convert to m^2
    F_mat_m2 = F_mat_D*9.869233E-13;
    % Transpose matrix to make easier to load in python
    % F_matt = F_mat_m2';
    % Save to csv
    %         filename = ['td_km2_', num2str(n), '.csv'];
    %         csvwrite(filename, [F_matt(:); nrow; ncol])
    
    
    
    % RF_mat = imrotate(F_mat_D, 45, 'bilinear', 'loose');
    % r_size = size(RF_mat);
    % r_offset = floor((r_size(1)-nrow)/2);
    % c_offset = floor((r_size(2)-ncol)/2);
    % RF_crop = RF_mat(r_offset:r_offset+nrow-1, c_offset:c_offset+ncol-1);
    % size(RF_crop)
    aspect_ratio = max(F)/min(F);
    
    if aspect_ratio < 100 && aspect_ratio > 1.1
        %     figure
        % subplot(1,2,1)
        % imagesc(squeeze(F_mat_m2))
        % colormap(gray)
        imagesc(F_mat_D);
        % axis equal
        % axis tight
        % subplot(1,2,2)
        % imagesc(squeeze(F_mat_m2))
        % imagesc(x,y, F_mat_D);
        axis equal
        axis tight
        colorbar;
        colormap(gray)
        xlabel('distance [cm]')
        ylabel('distance [cm]')
        title('synthetic permeability field [D]')
        drawnow
        pause
        % caxis([0.4500    0.5939])
    elseif aspect_ratio > 100
        disp(['Heterogenity too high, realization rejected'])
    elseif aspect_ratio < 1.1
        disp(['Heterogenity too low, realization rejected'])
    end
    
    n = n+1;
    
end

% figure
% imagesc(x,y, F_mat_D);
% axis equal
% axis tight
% colorbar;
% colormap(gray)
% xlabel('distance [cm]')
% ylabel('distance [cm]')
% title('synthetic permeability field [D]')


% writematrix([F_mat(:); nrow; ncol] ,'testd.txt','Delimiter',',')
% save('field_C01_5_and_1_50x25', 'x', 'y', 'corr', 'F_mat')
% caxis([0 1])

% figure
% edges = [0:0.5:10];
% histogram(aspect_ratio,edges)
