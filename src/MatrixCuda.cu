//
// Created by u1590812 on 20/08/16.
//

#include "MatrixCuda.h"

#include <iostream>
#include <cuda_runtime.h>

MatrixCuda::MatrixCuda(const std::string datafile): Matrix(datafile){ }

void MatrixCuda::allocate() {

    std::cout << "Cuda allocate" << std::endl;
    // todo copy X Y to device
#ifdef TIME
    cudaEvent_t start , stop ;
    cudaEventCreate (& start) ;
    cudaEventCreate (& stop) ;
    cudaEventRecord ( start ) ;
    float milliseconds = 0.0 ;
#endif

#ifdef TIME
    cudaEventRecord ( stop ) ;
    cudaEventSynchronize ( stop ) ;
    cudaEventElapsedTime(&milliseconds, start, stop ) ;
    std::cout << "Matrix, on copySeries, "<< milliseconds << std::endl;
#endif


#ifdef TIME
//    cudaEvent_t start , stop ;
    cudaEventCreate (& start) ;
    cudaEventCreate (& stop) ;
    cudaEventRecord ( start ) ;
    milliseconds = 0.0 ;
#endif

    // allocate matrix
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

    /* TODO Allocate and initialise anti-diagonal coordinate arrays */

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
}

void MatrixCuda::dtwm(double t, size_t o) {
    std::cout <<"Cuda dtwm"<< std::endl;
}

void MatrixCuda::findPath(size_t w) {
    std::cout <<"Cuda findPath"<< std::endl;
}


void MatrixCuda::markPath(size_t si, size_t sj, size_t li, size_t lj) {
    std::cout <<"Cuda markPath"<< std::endl;
}
