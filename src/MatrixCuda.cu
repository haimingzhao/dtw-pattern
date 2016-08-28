//
// Created by u1590812 on 20/08/16.
//

#include "MatrixCuda.h"
#include "MatrixKernels.h"
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

MatrixCuda::MatrixCuda(const std::vector<double> &X, const std::vector<double> &Y): Matrix(X,Y){}

void MatrixCuda::allocate() {
    std::cout << "Cuda allocate" << std::endl;

#ifdef TIME
    cudaEvent_t start , stop ;
    cudaEventCreate (& start) ;
    cudaEventCreate (& stop) ;
    cudaEventRecord ( start ) ;
    float milliseconds = 0.0 ;
#endif
    // allocate matrix
    cudaMalloc(&I, (nx*ny)*sizeof(size_t));

    cudaMalloc(&dX, nx*sizeof(double));
    cudaMalloc(&dY, ny*sizeof(double));

    cudaMalloc(&C, (nx*ny)*sizeof(double));
    cudaMalloc(&D, (nx*ny)*sizeof(double));

    cudaMalloc(&L, (nx*ny)*sizeof(size_t));
    cudaMalloc(&Rsi, (nx*ny)*sizeof(size_t));
    cudaMalloc(&Rsj, (nx*ny)*sizeof(size_t));
    cudaMalloc(&Rli, (nx*ny)*sizeof(size_t));
    cudaMalloc(&Rlj, (nx*ny)*sizeof(size_t));
    cudaMalloc(&Pi, (nx*ny)*sizeof(size_t));
    cudaMalloc(&Pj, (nx*ny)*sizeof(size_t));

    cudaMalloc(&visited, (nx*ny)*sizeof(bool));
    cudaMalloc(&OP, (nx*ny)*sizeof(bool));
#ifdef TIME
    cudaEventRecord ( stop ) ;
    cudaEventSynchronize ( stop ) ;
    cudaEventElapsedTime(&milliseconds, start, stop ) ;
    std::cout << "Matrix, on cudaMalloc, "<< milliseconds << std::endl;
#endif


#ifdef TIME
//    cudaEvent_t start , stop ;
    cudaEventCreate (& start) ;
    cudaEventCreate (& stop) ;
    cudaEventRecord ( start ) ;
    milliseconds = 0.0 ;
#endif
    // copy X Y to device
    cudaMemcpy(dX, X.data(), nx*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y.data(), ny*sizeof(double),cudaMemcpyHostToDevice);
#ifdef TIME
    cudaEventRecord ( stop ) ;
    cudaEventSynchronize ( stop ) ;
    cudaEventElapsedTime(&milliseconds, start, stop ) ;
    std::cout << "Matrix, on cudaMemcpy time series, "<< milliseconds << std::endl;
#endif

    allocated = true;
}

void MatrixCuda::deallocate() {
#ifdef TIME
    cudaEvent_t start , stop ;
    cudaEventCreate (& start) ;
    cudaEventCreate (& stop) ;
    cudaEventRecord ( start ) ;
    float milliseconds = 0.0 ;
#endif
    cudaFree(C);
    cudaFree(D);
    cudaFree(L);
    cudaFree(Rsi);
    cudaFree(Rsj);
    cudaFree(Rli);
    cudaFree(Rlj);

    cudaFree(Pi);
    cudaFree(Pj);

    cudaFree(visited);
    cudaFree(OP);

#ifdef TIME
    cudaEventRecord ( stop ) ;
    cudaEventSynchronize ( stop ) ;
    cudaEventElapsedTime(&milliseconds, start, stop ) ;
    std::cout << "Matrix, on cudaFree, "<< milliseconds << std::endl;
#endif
}

MatrixCuda::~MatrixCuda() {
    deallocate();
}

void MatrixCuda::init() {
    std::cout <<"Cuda init"<< std::endl;

    cudaMemset(I, 0, (nx*ny)*sizeof(size_t));          // init I to 0 for debug
    // init D in initCUDA since cudaMemset only handle bytes
    cudaMemset(L, 0, (nx*ny)*sizeof(size_t));          // init L to 0 for empty lengths

    //todo: are these needed?
    cudaMemset(Rsi, 0, (nx*ny)*sizeof(size_t));
    cudaMemset(Rsj, 0, (nx*ny)*sizeof(size_t));
    cudaMemset(Rli, 0, (nx*ny)*sizeof(size_t));
    cudaMemset(Rlj, 0, (nx*ny)*sizeof(size_t));
    cudaMemset(Pi, 0, (nx*ny)*sizeof(size_t));
    cudaMemset(Pj, 0, (nx*ny)*sizeof(size_t));

    cudaMemset(visited, 0, (nx*ny)*sizeof(bool)); // init visited to 0 (false)
    cudaMemset(OP,      0, (nx*ny)*sizeof(bool)); // init OptimalPath marks to 0 (false)


    // CUDA calculate anti-diagonal index in I and calculate cost in parallel for all cells
    const size_t num_blocks = (nx*ny + BLOCK_SIZE-1)/BLOCK_SIZE; // rounding up dividing by BLOCK_SIZE

    initCuda<<<num_blocks, BLOCK_SIZE>>>(I, C, D, dX, dY, nx, ny);
/*//    size_t idx;
//    for (size_t i = 0; i < nx; ++i) {
//        for (size_t j = 0; j < ny; ++j) {
//            if ( i >= j ){
//                I[i*ny+ j] = getI_bl(i,j, nx, ny);
//            }else{
//                size_t uj0= getI_bl(0,0, nx, ny) ;
//                I[i*ny+ j] = getI_ur(i,j, nx, ny, uj0);
//            }
//            idx = I[i*ny+ j];
//            C[idx] = getCost(i, j);
//        }
//    }

//    std::cout<< "indexes: " << std::endl;
//    for (size_t i = 0; i < nx; ++i) {
//        for (size_t j = 0; j < ny; ++j) {
//            std::cout<< I[i][j] << " " ;
//        }
//        std::cout << std::endl;
//    }
//
//    // C and D matrix initialisation anti-diagonal --not need to
//    size_t idx;
//    for (size_t si = 0; si < nx; ++si) {
//        size_t i = si + 1; // because while loop has i--
//        size_t j = 0 ;
//        while (i-- && j < ny){
//            idx = I[i][j];
//            C[idx] = getCost(i, j);
//            D[idx] = cuda_inf;
//            j = j + 1;
//        }
//    }
//
//    for (size_t sj = 1; sj < ny; ++sj) {
//        size_t i = nx ;  // which is nx = i end index +1, because we need it for i--
//        size_t j = sj ;
//        while (i-- && j < ny){
//            idx = I[i][j];
//            C[idx] = getCost(i, j);
//            D[idx] = cuda_inf;
//            j = j + 1;
//        }
//    }*/
}

void MatrixCuda::dtwm(double t, size_t o) {
    std::cout <<"Cuda dtwm"<< std::endl;

    // run CUDA in parallel in an anti-diagonal strip way
    const size_t num_blocks = (ny + BLOCK_SIZE-1)/BLOCK_SIZE; // rounding up dividing by BLOCK_SIZE

    dtwmCuda<<<num_blocks, BLOCK_SIZE>>>(I, C, D, L, Rsi, Rsj, Rli, Rlj, Pi, Pj, t, o, nx, ny);

/*//    for (size_t si = 0; si < nx; ++si) {
//        size_t i = si + 1; // because while loop has i--
//        size_t j = 0 ;
//        while (i-- && j < ny){
//            dtwm_task(i, j, I, t, o,
//                      C, D, L, Rsi, Rsj, Rli, Rlj, Pi, Pj);
//            j = j + 1;
//        }
//    }
//
//    for (size_t sj = 1; sj < ny; ++sj) {
//        size_t i = nx ;  // which is nx = i end index +1, because we need it for i--
//        size_t j = sj ;
//        while (i-- && j < ny){
//            dtwm_task(i, j, I, t, o,
//                      C, D, L, Rsi, Rsj, Rli, Rlj, Pi, Pj);
//            j = j + 1;
//        }
//    }*/
}

void MatrixCuda::findPath(size_t w) {
    std::cout <<"Cuda findPath"<< std::endl;

    const size_t num_blocks = (nx*ny + BLOCK_SIZE-1)/BLOCK_SIZE; // rounding up dividing by BLOCK_SIZE
    findPathCuda<<<num_blocks, BLOCK_SIZE>>>(I, L, Rli, Rlj, Pi, Pj, OP, w, nx, ny);

/*//    for (size_t si = 0; si < nx; ++si) {
//        size_t i = si + 1; // because while loop has i--
//        size_t j = 0 ;
//        while (i-- && j < ny){
//            findPath_task(i,j, I, ny, w,
//                    L, Rli, Rlj, Pi, Pj, OP);
//            j = j + 1;
//        }
//    }
//
//    for (size_t sj = 1; sj < ny; ++sj) {
//        size_t i = nx ;  // which is nx = i end index +1, because we need it for i--
//        size_t j = sj ;
//        while (i-- && j < ny){
//            findPath_task(i,j, I, ny, w,
//                          L, Rli, Rlj, Pi, Pj, OP);
//            j = j + 1;
//        }
//    }*/
}

double *MatrixCuda::getC() {
    double *hC = new double[nx*ny];
    cudaMemcpy(hC, C, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    return hC;
}

double *MatrixCuda::getD() {
    double *hD = new double[nx*ny];
    cudaMemcpy(hD, D, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    return hD;
}

size_t *MatrixCuda::getL() {
    size_t *hL = new size_t[nx*ny];
    cudaMemcpy(hL, L, nx*ny*sizeof(size_t), cudaMemcpyDeviceToHost);
    return hL;
}

bool *MatrixCuda::getOP() {
    bool *hOP = new bool[nx*ny];
    cudaMemcpy(hOP, OP, nx*ny*sizeof(bool), cudaMemcpyDeviceToHost);
    return hOP;     // no need to rearrange since OP is arranged normally
}

size_t *MatrixCuda::getI() {
    size_t *hI = new size_t[nx*ny];
    cudaMemcpy(hI, I, nx*ny*sizeof(size_t), cudaMemcpyDeviceToHost);
    return hI;     // no need to rearrange since OP is arranged normally
}
