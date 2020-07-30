/* GRFS: Gaussian Random Field Simulator

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
along with GRFS. If not, see <http://www.gnu.org/licenses/>. */

// cuda_scientific include file  -  © L.Räss  -  Stanford

#include <math.h>
#define min(a,b)               ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a < _b ? _a : _b; })
#define max(a,b)               ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a > _b ? _a : _b; })
#define mod(a,b)               (a % b)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Definition of basic macros
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define NB_THREADS             (BLOCKS_X*BLOCKS_Y)
#define NB_BLOCKS              (GRID_X*GRID_Y)
#define def_sizes(A,nx,ny,nz)  const int sizes_##A[] = {nx,ny,nz};                            
#define      size(A,dim)       (sizes_##A[dim-1])
#define     numel(A)           (size(A,1)*size(A,2)*size(A,3))
#define       end(A,dim)       (size(A,dim)-1)
#define     zeros(A,nx,ny,nz)  def_sizes(A,nx,ny,nz);                                         \
                               DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(numel(A)*sizeof(DAT)); \
                               for(i=0; i < (nx)*(ny)*(nz); i++){ A##_h[i]=(DAT)0.0; }        \
                               cudaMalloc(&A##_d      ,numel(A)*sizeof(DAT));                 \
                               cudaMemcpy( A##_d,A##_h,numel(A)*sizeof(DAT),cudaMemcpyHostToDevice);
#define gather(A)              cudaMemcpy( A##_h,A##_d,numel(A)*sizeof(DAT),cudaMemcpyDeviceToHost);
#define free_all(A)            free(A##_h);cudaFree(A##_d);
#define    all(A)              ( A[ ix    +  iy   *size(A,1) +  iz   *size(A,1)*size(A,2)] )
#define select(A,ix,iy,iz)     ( A[ix + iy*size(A,1) + iz*size(A,1)*size(A,2)] )
// participate_a: Test if the thread (ix,iy,iz) has to participate for the following computation of (all) A.
#define participate_a(A)       (ix< size(A,1) && iy< size(A,2) && iz< size(A,3) )

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Variables for cuda, performance measurement and Multi-GPU applications
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int nprocs=1, me=0;
dim3 grid, block;
int gpu_id=-1;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions (host code)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_up_gpu(){
  block.x = BLOCKS_X; block.y = BLOCKS_Y; block.z = BLOCKS_Z;
  grid.x  = GRID_X;   grid.y  = GRID_Y;   grid.z  = GRID_Z;
  gpu_id = GPU_ID;
  cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
  cudaDeviceReset();                                // Reset the device to avoid problems caused by a crash in a previous run (does still not assure proper working in any case after a crash!).
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
}

void  clean_cuda(){ 
  cudaError_t ce = cudaGetLastError();
  if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset();}
}

// Timer
#include "sys/time.h"
double timer_start = 0;
double cpu_sec(){ struct timeval tp; gettimeofday(&tp,NULL); return tp.tv_sec+1e-6*tp.tv_usec; }
void   tic(){ timer_start = cpu_sec(); }
double toc(){ return cpu_sec()-timer_start; }
void   tim(const char *what, double n){ double s=toc();if(me==0){ printf("%s: %8.3f seconds",what,s);if(n>0)printf(", %8.3f GB/s", n/s); printf("\n"); } }
////////// ========== Save & Read Data functions ========== //////////
void save_info(){
  FILE* fid;
  if (me==0){ fid=fopen("infos.inf","w"); fprintf(fid,"%d %d %d %d",PRECIS,nx,ny,nz); fclose(fid);}
}

void save_array(DAT* A, size_t nb_elems, const char A_name[]){
  char* fname; FILE* fid;
  asprintf(&fname, "%s.res" , A_name); 
  fid=fopen(fname, "wb"); fwrite(A, PRECIS, nb_elems, fid); fclose(fid); free(fname);
}
#define SaveArray(A,A_name)  gather(A); save_info(); save_array(A##_h, numel(A), A_name);
