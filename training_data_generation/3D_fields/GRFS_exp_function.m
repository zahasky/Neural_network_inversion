function [Yf]= GRFS_exp_function(F)
% =====================================================================
% GRFS: Gaussian Random Field Simulator - Exponential covariance
% 
% Copyright (C) 2019  Ludovic Raess, Dmitriy Kolyukhin and Alexander Minakov.
% 
% GRFS is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% GRFS is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with GRFS. If not, see <http://www.gnu.org/licenses/>.
% =====================================================================


% reset(RandStream.getGlobalStream);
% physics
% Lx  = 20;              % domain size in x     
% Ly  = 20;              % domain size in y
% Lz  = 100;              % domain size in z
sf  = 1;                % standard deviation
% If  = [2.0, 1.0, 100]; % correlation lengths in [x,y,z]
% numerics
Nh  = 3000;             % inner parameter, number of harmonics
% nx  = 20;               % numerical grid resolution in x
% ny  = 20;               % numerical grid resolution in y
% nz  = 40;               % numerical grid resolution in z
F.dx  = F.Lx/F.nx;            % numerical grid step size in x
F.dy  = F.Ly/F.ny;            % numerical grid step size in y
F.dz  = F.Lz/F.nz;            % numerical grid step size in z
% preprocessing
C   = sf/sqrt(Nh);
Yf  = zeros(F.nx,F.ny,F.nz);
tmp = zeros(F.nx,F.ny,F.nz);
% action
tic % timer
for ih = 1:Nh
    fi = 2.0*pi*rand;
    %   Gaussian spectrum
    flag = true;
    while flag
        k = tan(pi*0.5*rand);
        d = (k*k)/(1.0+(k*k));
        if rand<d
            flag = false;
        end
    end   
    theta = acos(1-2*rand);
    V1 = k*sin(fi)*sin(theta)/F.If(1);
    V2 = k*cos(fi)*sin(theta)/F.If(2);
    V3 = k*cos(theta)/F.If(3);
    a  = randn;
    b  = randn;
    for iz=1:F.nz
    for iy=1:F.ny
    for ix=1:F.nx
        tmp(ix,iy,iz) = F.dx*(ix-0.5)*V1 + F.dy*(iy-0.5)*V2 + F.dz*(iz-0.5)*V3;
    end
    end
    end 
    Yf = Yf + a*sin(tmp) + b*cos(tmp);
end
Yf = C*Yf;
toc