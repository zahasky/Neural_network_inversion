/* GRFS: Gaussian Random Field Simulator - Exponential covariance

Copyright (C) 2019  Ludovic Raess, Dmitriy Kolyukhin and Alexander Minakov.
 
GRFS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
GRFS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License
along with GRFS. If not, see <http://www.gnu.org/licenses/>. */
       
// -------------------------------------------------------------------------
// Compile as:    nvcc GPU_GRFS_exp.cu -O3 -arch=sm_61 -DSAVE
//  > arch=sm_52 (Titan X), arch=sm_61 (TitanXp), arch=sm_70 (Titan V)
// Run as:        ./a.out
// -------------------------------------------------------------------------

//////////////////////////////////
//  PROGRAM  --  GPU_GRFS_exp   //
//////////////////////////////////

#define USE_SINGLE_PRECISION    /* Comment this line using "//" if you want to use double precision.  */
#define GPU_ID 0

#ifdef USE_SINGLE_PRECISION
#define DAT     float
#define PRECIS  4
#else
#define DAT     double
#define PRECIS  8
#endif
////////// ========== Simulation Initialisation ========== //////////
#define NDIMS    3
#define BLOCKS_X         32
#define BLOCKS_Y         16
#define BLOCKS_Z         2
#define GRID_X           1*4
#define GRID_Y           1*8
#define GRID_Z           1*64
// maximum overlap in x, y, z direction. x : Vx is nx+1, so it is 1; y: Vy is ny+1, so it is 1; z: Vz is nz+1, so it is 1.
#define MAX_OVERLENGTH_X 0
#define MAX_OVERLENGTH_Y 0
#define MAX_OVERLENGTH_Z 0

#define PI 3.14159265358979323846
#define RND(rnd)  scaled  = ((double)rand() / (double)RAND_MAX); \
                  rnd     = (DAT)scaled;
#define RNDN(rnd) scaled  = ((double)rand() / (double)RAND_MAX); \
                  scaled2 = ((double)rand() / (double)RAND_MAX); \
                  scaled  = sqrt(-(double)2.0*log(scaled))*cos((double)2.0*PI*scaled2); \
                  rnd     = (DAT)scaled;
////////// ========== Simulation Params ========== //////////
const int nx  = GRID_X*BLOCKS_X - MAX_OVERLENGTH_X;        // we want to have some threads available for all cells of any array, also the ones that are bigger than nx.
const int ny  = GRID_Y*BLOCKS_Y - MAX_OVERLENGTH_Y;        // we want to have some threads available for all cells of any array, also the ones that are bigger than ny.
const int nz  = GRID_Z*BLOCKS_Z - MAX_OVERLENGTH_Z;        // we want to have some threads available for all cells of any array, also the ones that are bigger than nz.
// --------------------------------------------------------//
const DAT Lx    = 100.0;
const DAT Ly    = Lx;
const DAT Lz    = Lx;
// ------------------------
const DAT sf    = 1.0;            // standard deviation
const DAT If[3] = {10.0,8.0,5.0}; // correlation length x,y,z
const int Nh    = 10000;          // inner parameter, number of harmonics
const DAT nIO   = 2;              // number of in out operations, here 2*2 ReadWrite
// Preprocessing ------------------------------------------------------------------------
const DAT C   = sf/sqrt(Nh);
const DAT dx  = Lx/(DAT)nx;
const DAT dy  = Ly/(DAT)ny;
const DAT dz  = Lz/(DAT)nz;
// Include (ATTENTION: "cuda_scientific.h" must be included after the definition the basic parameters and the GPU parametrs as they are needed there!)
#include "cuda_scientific.h"
////////// ========================================  Physics  __CUDAkernels  ======================================== //////////
__global__ void compute1(DAT V1,DAT V2,DAT V3,DAT a,DAT b, DAT*Yf, const DAT dx,const DAT dy,const DAT dz, const int nx,const int ny,const int nz){  
  // CUDA specific
  def_sizes(Yf  ,nx,ny,nz);
  
  int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
  int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
  int iz  = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

  DAT tmp  = dx*((DAT)(ix+1)-(DAT)0.5)*V1 + dy*((DAT)(iy+1)-(DAT)0.5)*V2 + dz*((DAT)(iz+1)-(DAT)0.5)*V3;
  if (participate_a(Yf))  all(Yf) = all(Yf) + a*sin(tmp) + b*cos(tmp);
}
__global__ void compute2(const DAT C, DAT*Yf, const int nx,const int ny,const int nz){  
  // CUDA specific
  def_sizes(Yf  ,nx,ny,nz);
  
  int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
  int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
  int iz  = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z

  if (participate_a(Yf))  all(Yf) = C*all(Yf);
}
////////// ========================================  MAIN  ======================================== //////////
int main(){
  int    i, N;
  double mem, gbs;
  set_up_gpu();
  if (me==0){ printf("\n   ------------------------  ");
              printf("\n   |   3D Random Field    |  ");
              printf("\n   ------------------------  \n\n"); }

  N = nx*ny*nz; mem = (double)1e-9*(double)N*sizeof(DAT);
  
  if (me==0) printf("Local size: %dx%dx%d, %1.4f GB.\n", nx, ny, nz, mem);
  if (me==0) printf("Launching (%dx%dx%d) grid of (%dx%dx%d) blocks.\n\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
  
  srand(time(NULL));
  // allocate memory (on host + device), initialize to 0, copy from host to device
  zeros(Yf ,nx,ny,nz);
  
  int    flag, ih;
  double scaled=0.0, scaled2=0.0, time_s=0.0;
  DAT    fi=0.0, k=0.0, d=0.0, theta=0.0, V1=0.0, V2=0.0, V3=0.0, a=0.0, b=0.0, rand1=0.0;
  ///////////============================================================================ ACTION START ====////
  tic();
  for (ih=0; ih < Nh; ih++){
    RND(rand1);
    fi = (DAT)2.0*PI*rand1;
    // Gaussian spectrum
    flag = 1;
    while(flag==1){
      RND(rand1);
      k = tan(PI*(DAT)0.5*rand1);
      d = (k*k)/((DAT)1.0 + (k*k));
      RND(rand1);
      if(rand1 < d){ flag = 0; }
    }
    RND(rand1);
    theta = acos((DAT)1.0-(DAT)2.0*rand1);

    V1 = k*sin(fi)*sin(theta) / If[0];
    V2 = k*cos(fi)*sin(theta) / If[1];
    V3 = k*cos(theta) / If[2];
    RNDN(a);
    RNDN(b);
    compute1<<<grid, block>>>(V1,V2,V3, a,b, Yf_d, dx,dy,dz, nx,ny,nz);  cudaDeviceSynchronize();
  }
  compute2<<<grid, block>>>(C, Yf_d, nx,ny,nz);  cudaDeviceSynchronize();
  ///////////============================================================================ ACTION END ====////
  time_s = toc();
  gbs    = mem/time_s;
  if(me==0){ printf("-> Perf: %d Nh iterations took %1.4f seconds @ %3.3f GB/s \n\n",Nh,time_s,gbs*(Nh+((DAT)1.0))*nIO ); }
  #ifdef SAVE
  SaveArray(Yf , "RND_3D")
  #endif  
  // clear host memory & clear device memory
  free_all(Yf);
  clean_cuda();
}
