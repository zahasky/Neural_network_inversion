% load_micropet_core_crop_only
% Christopher Zahasky
% 3/30/2016
clear all
% close all
% tic
% This is the Matlab code for extracting both the PET image data
% (filename.v format)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input_format = 'raw';

path_str='C:\Users\zahas\Dropbox\Research\Experiment stuff\Data\reprocessed_PET_for_inversion\navajo_ss';
pet_filename = 'bh21_2ml_scan3_66_66_159_43.raw';
save_filename = 'Navajo_2ml_1_2_3mm_cropped';
% If loading raw input size
input_size = [66 66 159 43];

% Bentheimer
% crop_dim = [1, 66, 1, 66, 16, 144];
% cropped_ts = [2:24];
% Berea
% crop_dim = [1, 66, 1, 66, 17, 142];
% cropped_ts = [1:24];
% % Edwards
% crop_dim = [1, 66, 1, 66, 16, 144];
% cropped_ts = [2:40];
% % Estalles (may be too much bypass
% crop_dim = [1, 66, 1, 66, 15, 140];
% cropped_ts = [2:19];
% 
% % Indiana
% crop_dim = [1, 66, 1, 66, 16, 144];
% cropped_ts = [1:58];
% 
% Ketton (day 1)
% crop_dim = [1, 66, 1, 66, 22, 144];
% cropped_ts = [1:38];
% 
% % Navajo
crop_dim = [1, 66, 1, 66, 17, 139];
cropped_ts = [1:40];


% coarsen factor
% original pixel size in x is 0.776383 mm
cfx = 3;
% original pixel size in y is 0.776383 mm
cfy = cfx ;
% original pixel size in z is 0.796 mm
cfz = cfx ;
plot_slice = 3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END INPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if strncmpi(input_format, 'raw',1)
    % read PET image data from file named 'pet_filename' to 1D array of 32-bit
    % single precision, little-endian-formatted values, number of elements
    % determined by 'size'
    cd(path_str)
    fid = fopen(pet_filename); %open file identifier
    PET_4D = fread(fid, input_size(1)*input_size(2)*input_size(3)*input_size(4),'single','ieee-le');
    fclose(fid);
    PET_4D = reshape(PET_4D, input_size);
    PET_4D = flip(PET_4D);
    PET_4D = flip(PET_4D,1);
    PET_4D = permute(PET_4D,[2 1 3 4]);
    
end
% figure
% Coarsen data
% PET_4D_cropped = PET_4D(8:122, 6:120, 1:145, cropped_ts);
PET_4D_cropped = PET_4D(crop_dim(1):crop_dim(2), crop_dim(3):crop_dim(4),...
    crop_dim(5):crop_dim(6), cropped_ts);
% PET_4D = flip(PET_4D,3);
PET_4D_cropped = flip(PET_4D_cropped,3);
% save(save_filename, 'PET_4D_cropped')
% pause
% % original pixel size in x is 0.776383 mm
% cfx = 3;
% % original pixel size in y is 0.776383 mm
% cfy = 3;
% % original pixel size in z is 0.796 mm
% cfz = 3;  % cf of 7 rounds up voxel size to approximately 5 mm
crop_size = size(PET_4D_cropped);
PET_4D_coarse = zeros(crop_size(1)/cfx, crop_size(2)/cfy, crop_size(3)/cfz, crop_size(4));
% PET_4D_coarse = zeros(crop_size(1)/cfx, crop_size(2)/cfy, crop_size(3)/cfz);
% coarse_size = size(PET_4D_coarse);
% first coarsen in slices
% average original rows down into original/cfx
v=repmat({ones(cfx,1)/cfx},1, floor(crop_size(1)/cfx));
A = blkdiag(v{:});
for t=1:crop_size(4)
    for k = 1:crop_size(3)/cfz
        sum_slices = zeros(floor(crop_size(1)/cfx),floor(crop_size(2)/cfy));
        for i = 1:cfz
            slice = squeeze(squeeze(PET_4D_cropped(:,:,k*cfz+i-cfz,t)));
            smaller_slice=A'*slice*A;
            sum_slices = sum_slices+smaller_slice;
        end
        PET_4D_coarse(:,:,k,t) = sum_slices./cfz;
    end
    
    % Get core outline
    core_size = size(PET_4D_coarse);
    
    % crop_size = size(PET_4D_cropped);
    r = (core_size(1)-1)/2;
    cx = r+1;
    cy = r+1;
    dia = core_size(1);
    
    for i=1:core_size(1)
        yp = round(cy + sqrt(r^2 - (i-cx)^2));
        ym = round(cy - sqrt(r^2 - (i-cx)^2));
        
        if yp < dia
            PET_4D_coarse(i,yp:end,:,t) = nan;
        end
        
        if ym > 0
            PET_4D_coarse(i,1:ym,:,t) = nan;
        end
        
    end
end
PET_4D_coarse(:,end,:,:) = nan;

figure
subplot(1,3,1)
slice_exam = squeeze(squeeze(PET_4D_coarse(:,10,:,1)));

h=imagesc(slice_exam);
axis equal
axis tight
shading flat
colorbar
caxis([0 max(max(max(max(PET_4D_coarse))))*0.8])
set(h,'alphadata',~isnan(slice_exam))

subplot(1,3,2)
slice_exam = squeeze(squeeze(PET_4D_coarse(:,10,:,5)));

h=imagesc(slice_exam);
axis equal
axis tight
shading flat
colorbar
caxis([0 max(max(max(max(PET_4D_coarse))))*0.8])
set(h,'alphadata',~isnan(slice_exam))

subplot(1,3,3)
slice_exam = squeeze(squeeze(PET_4D_coarse(:,10,:,end)));

h=imagesc(slice_exam);
axis equal
axis tight
shading flat
colorbar
caxis([0 max(max(max(max(PET_4D_coarse))))*0.8])
set(h,'alphadata',~isnan(slice_exam))

% Plot total activity
figure
M0 = squeeze(nansum(nansum(nansum(PET_4D_coarse))));
plot(M0)

% Plot total activity
figure
M0 = squeeze(nansum(nansum(PET_4D_coarse(:,:,1,:))));
plot(M0, 'or')
title('inlet slice BTC')

% Plot total activity
figure
M0 = squeeze(nansum(nansum(PET_4D_coarse(:,:,end,:))));
plot(M0)
title('outlet slice BTC')



save(save_filename, 'PET_4D_coarse')
% save(save_filename, 'PET_4D_cropped', 'PET_4D_coarse')