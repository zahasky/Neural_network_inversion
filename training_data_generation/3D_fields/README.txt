
—————————————————————————
 GRFS README | 28.05.2019
—————————————————————————

GRFS: Gaussian Random Field Simulator

Copyright (C) 2018  Ludovic Raess, Dmitriy Kolyukhin and Alexander Minakov.

GRFS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GRFS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GRFS. If not, see <http://www.gnu.org/licenses/>.

================================================================================

https://bitbucket.org/lraess/grfs/
http://www.unil.ch/geocomputing/software/


Please cite us if you use our routine:
Raess, L., Kolyukhin, D., and Minakov, A. (2019). Efficient Implementation 
of 3-D Random Field Simulator for Geophysics. XXX.

================================================================================
Distributed software, files in this directory:
================================================================================

GRFS_exp.m 		Gaussian Random Field with exponential covariance generation
			routine based on a MATLAB serial implementation.

GPU_GRFS_exp.cu	    	GPU-based parallel implementation of the GRFS_exp algorithm.

GRFS_gauss.m 		Gaussian Random Field with Gaussian covariance generation
			routine based on a MATLAB serial implementation.

GPU_GRFS_gauss.cu   	GPU-based parallel implementation of the GRFS_gauss algorithm.

cuda_scientific.h 	Supporting function library for the GPU application.

vizme.m		   	MATLAb-based visualisation script for the GPU codes.

================================================================================
QUICK START: 	MATLAB > select the routine and enjoy!

		CUDA   > compile the GPU_GRFS.cu routine on a system hosting an
			 Nvidia GPU using the compilation line displayed on the
			 top line of the .cu file. Run it (./a.out) and use the
			 vizme.m MATLAB script to visualise the output. 

================================================================================
Contact: ludovic.rass@gmail.com
