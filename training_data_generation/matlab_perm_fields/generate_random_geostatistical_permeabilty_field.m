% generate_geostatistical_permeability_field
% Christopher Zahasky
% 13/3/2019
clear all
close all
set(0,'DefaultAxesFontSize',13, 'defaultlinelinewidth', 2,...
    'DefaultAxesTitleFontWeight', 'normal')

cd('D:\training_data\gauss_fields\no_rotation')
addpath('D:\Dropbox\Matlab\Deep_learning_computer_vision\Neural_network_inversion\training_data_generation\matlab_perm_fields')
%  Example 2 (2-D):
% build the correlation struct
% corr.name = 'gauss'; %Specifies the correlation type from 'gauss', 'exp', or 'turbulent'.

nrow = 20;
ncol = 40;
% Determine maximum dimension
max_shape = max([nrow, ncol]);
% Make sure grid is plenty big enough to rotate matrix
% x = linspace(0, 10, round(max_shape*1.5));
% y = linspace(0,5, round(max_shape*1.5));
x = linspace(0, 10, ncol);
y = linspace(0,5, nrow);
[X,Y] = meshgrid(x,y);
mesh = [X(:) Y(:)]; % 2-D mesh

tnreal = 10000;
extra_factor = 50;

% perm_range in mDarcy [1 1000] = [-3 3] in log space
log_min_mD = 0;
log_max_mD = 6;


% matrix of training data information, col#1 = corr.c0(1), col2 =
% corr.c0(2), col3 = mean_k, col4 = sig_value_multiplier, col5 = rotation
% in degrees
% load('rot_training_data_input_1.mat')

tdp_mat =zeros(tnreal*extra_factor, 4);
%
% anistropy
tdp_mat(:,1) = rand(tnreal*extra_factor,1).*35;
tdp_mat(:,2) = rand(tnreal*extra_factor,1).*2;
%
% fill in all perm
tdp_mat(:,3)= log_min_mD + rand(tnreal*extra_factor,1).*(log_max_mD-log_min_mD);
%
% fill in all sig
tdp_mat(:,4)= rand(tnreal*extra_factor,1);
%
% fill in all rotation
% tdp_mat(:,5)= 89+rand(tnreal*extra_factor,1).*90;

% % Save data for future reference
tdp_mat_save =zeros(tnreal, 4);
% save('gauss_10k_training_data_input', 'tdp_mat')

% iteration counter
n = 1;

% realization counter
nn = 1;
while nn < 1 %tnreal+1
    
    if nn<5000
        % corr.c0:    The scaling parameters for the correlation function. c0 may
        % be a scalar for isotropic correlation or a vector for
        % anisotropic correlation. In the anisotropic case, the
        % vector must have d elements, where d is the dimesion of a mesh point.
        corr.c0 = [tdp_mat(n,1), tdp_mat(n,2)]; % anisotropic correlation
    elseif nn>=5001
        corr.c0 = [tdp_mat(n,2), tdp_mat(n,1)]; % anisotropic correlation
    end
    
    % Mean permeability [mD]
    %     ks = log_min_D+rand(1)*(log_max_D-log_min_D);
    mean_k = 10^tdp_mat(n,3);
    corr.name = 'gauss'; %Specifies the correlation type from 'gauss', 'exp', or 'turbulent'.
    % set a spatially varying variance (must be positive!)
    corr.sigma = tdp_mat(n,4)*mean_k;
    % set a spatially varying variance (must be positive!)
    %     corr.sigma = 10*tdp_mat(n,4)*mean_k;
    
    [F,KL] = randomfield(corr,mesh, 'mean', mean_k);
    % If there are values below zero then set them equal to 0.1 mD
    F(F<0)= 0.1;
    
    % Calculate aspect ratio
    aspect_ratio = max(F)/min(F);
    
    if aspect_ratio < 100 && aspect_ratio > 1.1
        
        F_mat = reshape(F, length(y), length(x));
        % Rotate matrix
        %         RF_mat = imrotate(F_mat, tdp_mat(n,5), 'bilinear', 'loose');
        %         r_size = size(RF_mat);
        %         r_offset = floor((r_size(1)-nrow)/2);
        %         c_offset = floor((r_size(2)-ncol)/2);
        %         RF_crop = RF_mat(r_offset:r_offset+nrow-1, c_offset:c_offset+ncol-1);
        %
        %         % Convert from mD to D
        %         F_mat_D = RF_crop./1000;
        % Convert from mD to D
        F_mat_D = F_mat./1000;
        % Convert to m^2
        F_mat_m2 = F_mat_D*9.869233E-13;
        % Transpose matrix to make easier to load in python
        F_matt = F_mat_m2';
        
        % Save to csv
        filename = ['tdg_km2_', num2str(nn), '.csv'];
        csvwrite(filename, [F_matt(:); nrow; ncol])
        
        %         RF_mat = imrotate(F_mat, 45, 'bilinear', 'loose');
        %         figure
        %         pcolor(F_mat_D);
        %         axis equal
        %         axis tight
        %         colorbar;
        %         colormap(gray)
        %         xlabel('distance [cm]')
        %         ylabel('distance [cm]')
        %         title('synthetic permeability field [D]')
        %         drawnow
        %         caxis([330 365])
        
        tdp_mat_save(nn,:) = tdp_mat(n,:);
        
        % save data every 10 iterations
        if rem(nn,20) == 0
            save('gauss_10k_training_data_input', 'tdp_mat_save')
        end
        nn = nn+1;
        
    elseif aspect_ratio > 100
        disp(['Heterogenity too high, realization rejected'])
    elseif aspect_ratio < 1.1
        disp(['Heterogenity too low, realization rejected'])
    end
    
    n = n+1;
    
    %         % If aspect ratio is too high ignore this iteration and remove from
    %         % input matrix. Then resave input matrix
    %         tdp_mat(n,:)=[];
    %         save('rot_training_data_input_2', 'tdp_mat')
    %         % output rejection notice
    %         disp([num2str(nn), ' rejected'])
    %         nn = nn+1;
    %     end
end

% save('gauss_10k_training_data_input', 'tdp_mat_save')



