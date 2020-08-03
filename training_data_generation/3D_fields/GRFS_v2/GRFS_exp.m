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
clear all
% close all

reset(RandStream.getGlobalStream);
% physics
Lx  = 20;              % domain size in x     
Ly  = 20;              % domain size in y
Lz  = 100;              % domain size in z
sf  = 1;                % standard deviation
If  = [2.0, 1.0, 100]; % correlation lengths in [x,y,z]
% numerics
Nh  = 3000;             % inner parameter, number of harmonics
nx  = 20;               % numerical grid resolution in x
ny  = 20;               % numerical grid resolution in y
nz  = 40;               % numerical grid resolution in z
dx  = Lx/nx;            % numerical grid step size in x
dy  = Ly/ny;            % numerical grid step size in y
dz  = Lz/nz;            % numerical grid step size in z
% preprocessing
C   = sf/sqrt(Nh);
Yf  = zeros(nx,ny,nz);
tmp = zeros(nx,ny,nz);
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
    V1 = k*sin(fi)*sin(theta)/If(1);
    V2 = k*cos(fi)*sin(theta)/If(2);
    V3 = k*cos(theta)/If(3);
    a  = randn;
    b  = randn;
    for iz=1:nz
    for iy=1:ny
    for ix=1:nx
        tmp(ix,iy,iz) = dx*(ix-0.5)*V1 + dy*(iy-0.5)*V2 + dz*(iz-0.5)*V3;
    end
    end
    end 
    Yf = Yf + a*sin(tmp) + b*cos(tmp);
end
Yf = C*Yf;
toc
% visu
figure
slice(Yf,fix(nx/2),fix(ny/2),fix(nz/2))
shading flat
axis image
colorbar
