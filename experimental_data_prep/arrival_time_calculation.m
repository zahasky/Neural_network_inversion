% arrival_time_calculation
% Christopher Zahasky
% 14/24/2018
clear all
close all
set(0,'DefaultAxesFontSize',17, 'defaultlinelinewidth', 2,...
    'DefaultAxesTitleFontWeight', 'normal')

addpath('C:\Users\zahas\Dropbox\Matlab\high_res_images')
arrival_color = cbrewer('seq', 'YlGn', 60 , 'linear');
velo_color = flipud(cbrewer('div', 'RdYlBu', 60 , 'linear'));

% Load data
% Navajo Sandstone
% addpath('C:\Users\zahas\Dropbox\Research\Experiment stuff\Data\Stanford_data\BH21\bh21_pet_figure')
% load('BH21_4ml_coarse_cropped')
% % timestep length in seconds
% timestep_length = 40;
% injected pulse volume [mL]
% inj_pv = 4;

% Berea
% addpath('C:\Users\zahas\Dropbox\Research\Experiment stuff\Data\Stanford_data\BSS_c1\june_17_pet\6_12_single_phase')
% load('BSS_c1_2ml_2_3mm_vox')
% timestep length in seconds
% timestep_length = 60;
% injected pulse volume [mL]
% inj_pv = 4;

% Bentheimer
addpath('C:\Users\zahas\Dropbox\Research\Experiment stuff\Data\Stanford_data\Bentheimer_imperial')
load('Bentheimer_4ml_2_3mm_vox')
% timestep length in seconds
timestep_length = 47;
% injected pulse volume [mL]
inj_pv = 2;

% flow rate [mL/min]
q = 4;
% Image voxel size
vox_size = [0.2329 0.2329 0.2388];
% noise threshold
noise_thresh = 0.003;
% normalized time (1 = yes, 0 = no)
norm_time = 1;

%%%%% END INPUT %%%%%%%%%
% approx velocity
r = 2.54;
a = pi*r^2;
phi = 0.16;
v_inj = q/(a*phi);

% Grid definition
s = size(PET_4D_coarse);
gridX = ([1:s(1)].*vox_size(1) - vox_size(1)/2);
gridY = ([1:s(2)].*vox_size(2) - vox_size(2)/2);
gridZ = ([1:s(3)].*vox_size(3) - vox_size(3)/2);

% Filter out low rad noise
conditioned_PET =  PET_4D_coarse;
conditioned_PET(conditioned_PET< noise_thresh)=0;
% Calculate size
PET_size = size(PET_4D_coarse);

% Call arrival time calculation function 
[Xt, Mt0, St]= arrival_time_calculation_function(conditioned_PET, ...
    timestep_length, 0);
% convert seconds to minutes in mean arrival time
Xt = Xt./60;
% shift mean arrival time so that mean of first slice is mean at x=0
inj_t = inj_pv/q;
mean_arrival_inlet = nanmean(nanmean(Xt(:,:,1)));
Xt = Xt -(mean_arrival_inlet - (inj_t/2));



%% Arrival time difference map calculation
% calculate mean arrival time at the inlet
mean_arrival_inlet = nanmean(nanmean(Xt(:,:,1)));
% calculate mean arrival time at the outlet
mean_arrival_outlet = nanmean(nanmean(Xt(:,:,end)));
% Calculate velocity from difference in mean arrival time between inlet and
% outlet (distance is known)
v = (gridZ(end)-gridZ(1))/(mean_arrival_outlet-mean_arrival_inlet);
% Now calculate what the mean arrival would be if the velocity was constant
% everywhere
% vector of ideal mean arrival time based average v
mean_xt = gridZ./v;
% shifted
mean_xt = mean_xt -(mean_xt(1) - (inj_t/2));
% Turn this vector into a matrix so that it can simple be subtracted from
% the arrive time map calculated from the PET data
mean_xt3d(1,1,1:PET_size(3)) = mean_xt;
Xth = repmat(mean_xt3d, PET_size(1),PET_size(1),1);
% Arrival time difference map
Xt_diff = (Xth - Xt);

% if norm_time == 1
% normalized
Xt_diff_norm = Xt_diff.*(v./gridZ(end));
Xt_norm = Xt.*(v./gridZ(end));
% end

%% Plot data
% Plot center half of core mean arrival times
if norm_time == 1
    slice_plane = squeeze(Xt_norm);
else
    slice_plane = squeeze(Xt);
end
slice_plane(1:end,11:end,:) = nan;
slice_plane = flip(slice_plane);
slice_plane = flip(slice_plane,2);
slice_plane = permute(slice_plane,[3 2 1]);

if norm_time == 1
    clim = [0, 1.1];
else
    clim = [timestep_length*2./60, max(max(max(Xt)))];
end
% create figure
figure('position', [274         432        1492         420])
subplot(1,2,1)
PATCH_3Darray(slice_plane, gridZ, gridY, gridX, arrival_color, clim, 'col')
title(['Tracer mean breakthrough time map'],'FontWeight', 'Normal')
% format figure
axis equal
axis([0 10 max(gridY)/2 max(gridY)+0.1 0 max(gridX)])
grid on
xticks([0, 2, 4, 6, 8, 10])
xlabel('Distance from inlet [cm]')
set(gca,'ZTickLabel',[]);
set(gca,'YTickLabel',[]);
view(-30,27)
set(gca,'color','none')

% Add colorbar
colormap(gca, arrival_color)
h1 = colorbar('eastoutside');
if norm_time == 1
ylabel(h1, 'Time [PV]', 'FontSize',14)
else
    ylabel(h1, 'Time [min]', 'FontSize',14)
end

%% velocity approximation
% figure
subplot(1,2,2)

% Plot center half of core mean arrival times
if norm_time == 1
    flow_slice_plane = Xt_diff_norm;
else
    flow_slice_plane = Xt_diff;
end
flow_slice_plane(1:end,11:end,:) = nan;
flow_slice_plane = flip(flow_slice_plane);
flow_slice_plane = flip(flow_slice_plane,2);
flow_slice_plane = permute(flow_slice_plane,[3 2 1]);

PATCH_3Darray(flow_slice_plane, gridZ, gridY, gridX, velo_color, 'col')
title(['Breakthrough time difference map'],'FontWeight', 'Normal')

axis equal
axis([0 10 max(gridY)/2 max(gridY)+0.1 0 max(gridX)])
grid on
xticks([0, 2, 4, 6, 8, 10])

xlabel('Distance from inlet [cm]')
set(gca,'ZTickLabel',[]);
set(gca,'YTickLabel',[]);
view(-30,27)
set(gca,'color','none')

% Add colorbar
colormap(gca, velo_color)
% colorbar axis label
h2 = colorbar('eastoutside');
if norm_time == 1
    ylabel(h2, 'Difference [PV]', 'FontSize',14)
else
    ylabel(h2, 'Difference [min]', 'FontSize',14)
end


subplot(1,2,1)
colormap(gca, arrival_color)

%% Voxel-level flow rate calculation, very noisy!
% flow_rate_matrix = nan(PET_size(1:3));
% for k = 3:PET_size(3)-2
%     flow_rate_matrix(:,:,k) = (4*vox_size(3)./(Xt(:,:,k+2) -Xt(:,:,k-2))).*(vox_size(1)*vox_size(2));
% end
% flow_rate_matrix(:,:,1) = flow_rate_matrix(:,:,3);
% flow_rate_matrix(:,:,2) = flow_rate_matrix(:,:,3);
% flow_rate_matrix(:,:,PET_size(3)-1) = flow_rate_matrix(:,:,PET_size(3)-2);
% flow_rate_matrix(:,:,PET_size(3)) = flow_rate_matrix(:,:,PET_size(3)-2);

%% profile plotting testing
% figure
% hold on
% plot(time_vec, squeeze(PET_4D_coarse(11, 11, 2, :)), 'k')
% plot([Xt(11, 11, 2), Xt(11, 11, 2)], [0 0.05], ':k')
% plot(time_vec, squeeze(PET_4D_coarse(11, 11, 7, :)), 'g')
% plot([Xt(11, 11, 7), Xt(11, 11, 7)], [0 0.05], ':g')
% plot(time_vec, squeeze(PET_4D_coarse(11, 11, 38, :)), 'r')
% plot([Xt(11, 11, 38), Xt(11, 11, 38)], [0 0.05], ':r')

% f = fit(time_vec,squeeze(PET_4D_coarse(11, 11, 38, :)),'gauss1')

% xlabel('Time [sec]')
% xlabel('Concentration')
% box on

%% correct for variation in measure mass along the axis of the scanner
% axis_mass = squeeze(nanmean(nanmean(nanmean(conditioned_PET)),4));
% figure
% hold on
% plot([1:PET_size(3)], axis_mass)
% 
% % AM = repmat(axis_mass,[PET_size(1), PET_size(2), 1]);
% for i=1:PET_size(1)
%     for j=1:PET_size(2)
%         for k = 1:PET_size(4)
%             
%             axis_vox_n = squeeze(conditioned_PET(i,j,:,k));
%             conditioned_PET(i,j,:,k) = axis_vox_n./axis_mass;
%         end
%     end
% end
% 
% axis_mass2 = squeeze(nanmean(nanmean(nanmean(conditioned_PET)),4));
% plot([1:PET_size(3)], axis_mass2)

%% Plot mean arrival time as a function of distance
% hold on
% ccc = jet(PET_size(1)^2);
% n=1;
% for i = 1:PET_size(1)
%     for j = 1:PET_size(2)
%         if ~isnan(Xt(i,j,1))
%     plot(gridZ, squeeze(Xt(i,j,:)), '-', 'color', ccc(n,:))
%     n=n+1;
%         end
%     end
% end

% Call arrival time calculation function 
[Xt, Mt0, St]= arrival_time_calculation_function(conditioned_PET, ...
    timestep_length, 0);
r = 10;
c = 9;
figure
hold on
time_vec = timestep_length.*[1:PET_size(4)]'-(timestep_length/2);
cc = gray(35);
for i = [5 15 25]
    plot(time_vec./60, squeeze(conditioned_PET(r,c,i,:)), '-o', 'color', cc(i,:), 'linewidth', 2.5)
    plot([Xt(r,c,i), Xt(r,c,i)]./60, [0 0.1], ':', 'color', cc(i,:))
%     pause
end
box on
fs = 28
title('Mean breakthrough time', 'fontsize', fs)
xlabel('Time [min]', 'fontsize', fs)
ylabel('Radioactivity [mCi]', 'fontsize', fs)
set(gca, 'Color', 'none');
set(gca,'linewidth',1.8)
axis([0 10 0 0.06])

