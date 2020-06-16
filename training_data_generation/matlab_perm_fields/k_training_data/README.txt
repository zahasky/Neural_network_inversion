The permeability fields generated in this folder were generated with the 'randomfield' and 'correlation_fun' functions called from the script 'generate_random_geostatistical_permeability_field.m'. The input to generate these data is available in the 'training_data_input.mat'. The matrix of permeability field generation was generated as follows: column #1 = corr.c0(1), column #2 = corr.c0(2), column #3 = mean_k, column #4 = sig_value_multiplier

Essentially 250 horizontal bedding realiztions were created, followed by 250 vertical bedding realization, followed by 1500 realizations with variable anisotropy (0-1 in x and y). The fields were generated with the 'gauss' option.

The matrix was populated with the following script
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