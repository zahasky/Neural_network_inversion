% data_matlab2python
% Christopher Zahasky
% This script is used to load the previously analyzed PET data and save it
% in the same format as the CNN training data
clear all
close all



% Berea
% addpath('C:\Users\zahas\Dropbox\Research\Experiment stuff\Data\Stanford_data\BSS_c1\june_17_pet\6_12_single_phase')
% load('BSS_c1_2ml_2_3mm_vox')
% timestep length in seconds
% timestep_length = 60;
% injected pulse volume [mL]
% inj_pv = 4;
% Berea porosity
addpath('B:\Experimental_data\Stanford_data\PET_CT\Berea_sandston_C1\january_16_CT_imbibe\BS_C1_SImb')
load('dry_average.mat')
dry = AVG;
load('full_sat_average.mat')
wet = AVG;
vox_size = [0.03125 0.03125 0.125];
filename = 'berea_porosity_uncoarsened';
% Calculate porosity
PET_4D_coarse = (wet-dry)./1000;

% Bentheimer
% addpath('C:\Users\zahas\Dropbox\Research\Experiment stuff\Data\Stanford_data\Bentheimer_imperial')
% filename = 'Estaillades_3ml_2_3mm_cropped';
% load(filename)
% % % timestep length in seconds
% timestep_length = 85;
% % % injected pulse volume [mL]
% inj_pv = 2;
% % % injected flow rate [mL/min]
% q = 3;

% Image voxel size
% vox_size = [0.2329 0.2329 0.2388];

% Grid definition
s = size(PET_4D_coarse);

% Calculate size
PET_size = size(PET_4D_coarse);

%% Visual check
figure
imagesc(squeeze(PET_4D_coarse(10,:,:)))
axis image
colorbar
set(gca,'Ydir','normal')

figure
imagesc(squeeze(PET_4D_coarse(:,:,80)))
axis image
% colormap('gray')
colorbar
set(gca,'Ydir','normal')

%% WRITE DATA
% Permute matrix to make easier to load in python
PET_4D_coarse_python = permute(PET_4D_coarse,[3 2 1]);
% PET_4D_coarse_python = permute(PET_4D_coarse,[4 3 2 1]);

% Save to csv
filename_csv = [filename, '_nan.csv'];
% csvwrite(filename_csv, [PET_4D_coarse_python(:); PET_size(1); PET_size(2); PET_size(3); PET_size(4); timestep_length; q; inj_pv; vox_size(:)])
csvwrite(filename_csv, [PET_4D_coarse_python(:); PET_size(1); PET_size(2); PET_size(3); vox_size(:)])

