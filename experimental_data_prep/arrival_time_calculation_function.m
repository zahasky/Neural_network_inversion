% arrival_time_calculation_function
function [Xt, Mt0, St]= arrival_time_calculation_function(PET_4D_cc, ...
    timestep_length, gauss_fit)
% Christopher Zahasky
% 7/3/2017
% This script is used to calculate mean arrival time at each voxel in the
% PET image

% PET_4D_cc = PET_4D_coarse;
PET_dim = size(PET_4D_cc);

% preallocate zero moment matrix
Mt0 = nan(PET_dim(1), PET_dim(2), PET_dim(3));
% preallocate first moment matrix
Xt = nan(PET_dim(1), PET_dim(2), PET_dim(3));
% preallocate second moment matrix
St = nan(PET_dim(1), PET_dim(2), PET_dim(3));

if length(timestep_length) == 1
    time_vec = timestep_length.*[1:PET_dim(4)]'-(timestep_length/2);
elseif length(timestep_length) == PET_dim(4)
    time_vec = timestep_length;
else
    error('timestep length must either be constant value or vector with dimensions equal to PET_dim(4)')
end
% neg_time = -time_vec;
% time_vec = [-flipud(time_vec); time_vec];
for i=1:PET_dim(1)
    for j=1:PET_dim(2)
        for k=1:PET_dim(3)
            % isolate one voxel
            vox_n = squeeze(PET_4D_cc(i,j,k,:));
            
            % if gauss_fit input equals 1 (meaning fit to gauss) and the
            % voxel contains NO nans
            if gauss_fit == 1 && sum(isnan(vox_n)) == 0
                f = fit(time_vec, vox_n,'gauss1');
                vox_n = f(time_vec);
            end

            % check to make sure tracer is in tube
            if sum(vox_n) > 0
                % calculate zero moment of voxel
                m0 = trapz(time_vec, vox_n);
                Mt0(i,j,k) = m0;
                
                % calculate first moment of voxel 
                m1 = trapz(time_vec, vox_n.*time_vec);
                % calculate center of mass (mean arrival time)
                xc = m1/m0;
                Xt(i,j,k) = xc;
                
                % calculate second moment of voxel (distribution of
                % temporal moment)
                m2 = trapz(time_vec, vox_n.*(time_vec-xc).^2);
                % calculate spread of tracer
                sxx = m2/m0;
                St(i,j,k) = sxx;
            end
        end
    end
end