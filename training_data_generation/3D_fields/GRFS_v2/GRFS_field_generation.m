% GRFS_field_generation
% 

% This script relies on the GRFS: Gaussian Random Field Simulator - 
% Exponential covariance from Raess, L., Kolyukhin, D., and Minakov, A. 
% (2019). Efficient Implementation of 3-D Random Field Simulator for Geophysics.

clear all
close all

% Sample size and resolution
F.Lx  = 50;              % domain size in x [mm]     
F.Ly  = 50;              % domain size in y [mm]
F.Lz  = 100;              % domain size in z [mm]
% sf  = 1;                % standard deviation

% numerics
F.nx  = 20;               % numerical grid resolution in x
F.ny  = 20;               % numerical grid resolution in y
F.nz  = 40;               % numerical grid resolution in z

% Correlation length
F.If  = [2.0, 1.0, 100]; % correlation lengths in [x,y,z]

[Yf]= GRFS_exp_function(F);

% scale GRF down to a range of 1
Yf_norm = Yf./range(Yf(:));



% Permeabiltiy range in mDarcy [1 100,000] = [0 5] in log space
log_min_mD = 0;
log_max_mD = 5;
% converting to mD out of log space
mean_k = 10^1.2;

% Heterogeneity factor is the range divided by the mean (0.1 to 2)
Hf = 2;

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
filename = ['td_3dk_m2_', num2str(1), '.csv'];
csvwrite(filename, [Yfs_m2(:); F.nx; F.ny; F.nz])


% plot stuff
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

%% Slice plot for python
% This plot should align with orientation plotted in python when read in
% from the .csv. In python this is imported via:
% Yfs = tdata_km2.reshape(nlay, nrow, ncol, order='F')
% and plotted with:
% raw_km2[9,:,:] (note the slice is offset by 1 due to differences between
% python and matlab indexing)

figure
imagesc(squeeze(Yfs(10,:,:)))
axis image
colormap('gray')