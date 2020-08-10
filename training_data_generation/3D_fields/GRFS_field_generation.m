% GRFS_field_generation
% Christopher Zahasky
% 08/06/2019 

% This script relies on the GRFS: Gaussian Random Field Simulator -
% Exponential covariance from Raess, L., Kolyukhin, D., and Minakov, A.
% (2019). Efficient Implementation of 3-D Random Field Simulator for Geophysics.

clear all
close all

fid = fopen('perm_field3D_parameter_space.dat');
Rr = textscan(fid,'%f %f %f %f %f');
fclose(fid);

% This is the hypercube sample data with:[x correlation length, y
% correlation length, z correlation length, permeability exponent,
% heterogeneity factor]
Rm = cell2mat(Rr);

% Sample size and resolution
F.Lx  = 50;              % domain size in x [mm]
F.Ly  = 50;              % domain size in y [mm]
F.Lz  = 100;              % domain size in z [mm]

% numerics
F.nx  = 20;               % numerical grid resolution in x
F.ny  = 20;               % numerical grid resolution in y
F.nz  = 40;               % numerical grid resolution in z

for i = 1:1 %length(Rm)
    
    % Correlation length
    F.If  = [Rm(i,1), Rm(i,2), Rm(i,3)]; % correlation lengths in [x,y,z]
    
    [Yf]= GRFS_exp_function(F);
    
    % scale GRF down to a range of 1
    Yf_norm = Yf./range(Yf(:));
    
    
    
    % Permeabiltiy range in mDarcy [1 100,000] = [0 5] in log space
    % log_min_mD = 0;
    % log_max_mD = 5;
    % converting to mD out of log space
    mean_k = 10^Rm(i,4);
    
    % Heterogeneity factor is the range divided by the mean (0.2 to 2)
    Hf = Rm(i,5);
    
    % scaled permeability field
    Yfs = (Hf*mean_k.*Yf_norm) + mean_k;
    
    % Save for Python loading
    % Convert from mD to D
    Yfs_D = Yfs./1000;
    % Convert to m^2
    Yfs_m2 = Yfs_D*9.869233E-13;
    % Transpose matrix to make easier to load in python
    % F_matt = Yfs_m2';
    
    % Save to csv
%     filename = ['td_3dk_m2_', num2str(i), '.csv'];
%     Yfs_m2 = permute(Yfs_m2,[3 2 1]);
%     csvwrite(filename, [Yfs_m2(:); F.nx; F.ny; F.nz])
    
    figure
    imagesc(squeeze(Yfs(10,:,:)))
    axis image
    colormap('gray')
    colorbar
    set(gca,'Ydir','normal')
end


%% Slice plot for python
% This plot should align with orientation plotted in python when read in
% from the .csv. In python this is imported via:
% Yfs = tdata_km2.reshape(nlay, nrow, ncol)
% and plotted with:
% raw_km2[9,:,:] (note the slice is offset by 1 due to differences between
% python and matlab indexing)

% figure
% imagesc(squeeze(Yfs(10,:,:)))
% axis image
% colormap('gray')
% colorbar
% set(gca,'Ydir','normal')

%% 3D plot for inspection
figure
subplot(1,2,1)
slice(Yf,fix(F.ny/2),fix(F.nx/2),fix(F.nz/2))
title('Unscaled Field')
shading flat
axis image
colorbar

subplot(1,2,2)
slice(Yfs,fix(F.ny/2),fix(F.nx/2),fix(F.nz/2))
title('Scaled Field')
shading flat
axis image
colorbar