//
// Created by u1590812 on 27/08/16.
//

#ifndef DTWM_MATRIXKERNELS_H
#define DTWM_MATRIXKERNELS_H


__device__
double getCost(size_t i, size_t j, double* dX, double* dY);

__global__
void initCuda(size_t *I, double* C, double* D,
              double* dX, double* dY,
              const size_t nx, const size_t ny);

__global__
void initCudaOp(size_t *I, double* C, double* D,
              double* dX, double* dY,
              const size_t nx, const size_t ny);

__device__
void dtwm_task(size_t i, size_t j,
               size_t* I, double* C, double* D, size_t* L,
               size_t* Rsi, size_t* Rsj, size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj,
               double t, size_t o, const size_t ny);

__global__
void dtwmCuda(size_t* I, double* C, double* D, size_t* L,
              size_t* Rsi, size_t* Rsj, size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj,
              double t, size_t o, const size_t nx, const size_t ny);

__device__
void findPath_task(size_t i, size_t j,
                   size_t* I, size_t* L,
                   size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj, bool* OP,
                   size_t w, size_t ny);

__global__
void findPathCuda(size_t* I, size_t* L,
                  size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj, bool* OP,
                  size_t w, size_t nx, size_t ny);


#endif //DTWM_MATRIXKERNELS_H
