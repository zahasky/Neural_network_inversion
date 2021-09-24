% data_matlab2python
% Christopher Zahasky
% This script is used to load the previously analyzed CT data and save it
% in the same format as the CNN training data
clear all
close all

filename = 'Edwards3Dporo';
load(filename)

% Image voxel size
vox_size = [0.2 0.2 0.2];

% Grid definition
s = size(Poro3D_1);

Poro3D_1 = Poro3D_1(2:end-1,:,2:end-1);

% Calculate size
CT_size = size(Poro3D_1);

%% Visual check
figure
imagesc(squeeze(Poro3D_1(10,:,:)))
axis image
colorbar
set(gca,'Ydir','normal')

figure
imagesc(squeeze(Poro3D_1(:,:,10)))
axis image
% colormap('gray')
colorbar
set(gca,'Ydir','normal')

%% WRITE DATA
% Permute matrix to make easier to load in python
Poro3D_1_python = permute(Poro3D_1,[2 1 3]);

% Save to csv
filename_csv = [filename, '_nan.csv'];
csvwrite(filename_csv, [Poro3D_1_python(:); CT_size(1); CT_size(3); CT_size(2); vox_size(:)])

