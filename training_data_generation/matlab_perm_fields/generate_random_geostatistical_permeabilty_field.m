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

x = linspace(0, 10, ncol);
y = linspace(0,5, 20);
[X,Y] = meshgrid(x,y);
mesh = [X(:) Y(:)]; % 2-D mesh

tnreal = 1000;
extra_factor = 3;

% perm_range in mDarcy [1 1000] = [-3 3] in log space
log_min_mD = 0;
log_max_mD = 6;


% matrix of training data information, col#1 = corr.c0(1), col2 =
% corr.c0(2), col3 = mean_k, col4 = sig_value_multiplier
load('training_data_input.mat')
% tdp_mat =zeros(tnreal*extra_factor, 4);
% 
% % number of realizations for each anistropy set
% set_nreal = 250;
% % horizontal anistropy
% tdp_mat(1:set_nreal,1) = rand(set_nreal,1).*50;
% tdp_mat(1:set_nreal,2) = rand(set_nreal,1);
% 
% % vertical anistropy
% tdp_mat(set_nreal+1:set_nreal+set_nreal,1) = rand(set_nreal,1).*50;
% tdp_mat(set_nreal+1:set_nreal+set_nreal,2) = rand(set_nreal,1);
% 
% % fill in the rest
% [rows_left, c] = size(tdp_mat(set_nreal+set_nreal+2:end,1:2));
% tdp_mat(set_nreal+set_nreal+2:end,1:2) = rand(rows_left, 2);
% 
% % fill in all perm
% tdp_mat(:,3)= log_min_mD + rand(tnreal*extra_factor,1).*(log_max_mD-log_min_mD);
% 
% % fill in all sig
% tdp_mat(:,4)= rand(tnreal*extra_factor,1);

% iteration counter
% n = 1;
n=394;
% rejection counter
nn = 109;
while n < tnreal+1
    
    % corr.c0:    The scaling parameters for the correlation function. c0 may
    % be a scalar for isotropic correlation or a vector for
    % anisotropic correlation. In the anisotropic case, the
    % vector must have d elements, where d is the dimesion of a mesh point.
    corr.c0 = [tdp_mat(n,1), tdp_mat(n,2)]; % anisotropic correlation
    
    % Mean permeability [mD]
    %     ks = log_min_D+rand(1)*(log_max_D-log_min_D);
    mean_k = 10^tdp_mat(n,3);
    
    % set a spatially varying variance (must be positive!)
    corr.sigma = 10*tdp_mat(n,4)*mean_k;
    
    [F,KL] = randomfield(corr,mesh, 'mean', mean_k);
    % If there are values below zero then set them equal to 0.1 mD
    F(F<0)= 0.1;
    
    % Calculate aspect ratio
    aspect_ratio = max(F)/min(F);
    if aspect_ratio < 100
        
        F_mat = reshape(F, length(y), length(x));
        % Convert from mD to D
        F_mat_D = F_mat./1000;
        % Convert to m^2
        F_mat_m2 = F_mat_D*9.869233E-13;
        % Transpose matrix to make easier to load in python
        F_matt = F_mat_m2';
        % Save to csv
        filename = ['td_km2_', num2str(n), '.csv'];
        csvwrite(filename, [F_matt(:); nrow; ncol])
        
%         imagesc(x,y, F_mat_D);
%         axis equal
%         axis tight
%         colorbar;
%         colormap(gray)
%         xlabel('distance [cm]')
%         ylabel('distance [cm]')
%         title('synthetic permeability field [D]')
        
        n = n+1;
    else
        % If aspect ratio is too high ignore this iteration and remove from
        % input matrix. Then resave input matrix
        tdp_mat(n,:)=[];
        save('training_data_input', 'tdp_mat')
        % output rejection notice
        disp([num2str(nn), ' rejected'])
        nn = nn+1;
    end
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
