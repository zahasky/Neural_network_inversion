% =====================================================================
% GRFS: Gaussian Random Field Simulator - Gaussian covariance
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
clear
reset(RandStream.getGlobalStream);
% physics
Lx  = 100;              % domain size in x     
Ly  = 100;              % domain size in y
Lz  = 100;              % domain size in z
sf  = 1;                % standard deviation
If  = 5.0;              % correlation lengths in [x,y,z]
% numerics
Nh  = 5000;             % inner parameter, number of harmonics
k_m = 100;              % maximum value of the wave number
nx  = 64;               % numerical grid resolution in x
ny  = 64;               % numerical grid resolution in y
nz  = 64;               % numerical grid resolution in z
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
    fi = 2*pi*rand;
    lf = 2*If/sqrt(pi);
    %   Gaussian spectrum
    flag = true;
    while flag
        k = k_m*rand;
        d = k*k*exp(-0.5*k*k);
        if (rand*2*exp(-1)<d)
            flag = false;
        end
    end   
    k     = sqrt(2)*k/lf;    
    theta = acos(1-2*rand);
    V1 = k*sin(fi)*sin(theta);
    V2 = k*cos(fi)*sin(theta);
    V3 = k*cos(theta); 
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
figure(1),clf
slice(Yf,fix(nx/2),fix(ny/2),fix(nz/2)),shading flat,axis image,colorbar
drawnow
