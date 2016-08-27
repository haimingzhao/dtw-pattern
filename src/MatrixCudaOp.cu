//
// Created by u1590812 on 20/08/16.
//

#include "MatrixCudaOp.h"
#include "MatrixKernels.h"
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// infinity value for host to copy to cuda
//#define inf DBL_MAX
//double inf = (std::numeric_limits<double>::max());
//#define cuda_inf CUDART_INF

MatrixCudaOp::MatrixCudaOp(const std::string datafile): Matrix(datafile){

//    // Allocate and initialise 2D diagonal index matrix - move to cuda
//    I = new size_t*[nx];
//    for (size_t i = 0; i < nx; ++i) {
//        I[i] = new size_t[ny]();
//    }

    /* Initialise anti-diagonal memory location coordinates using while loop
     * i is row -> X of length nx,
     * j is column -> Y of length ny
     * careful for size_t being unsigned */
//    size_t idx = 0;
//    for (size_t si = nx; si--; ) {
//        size_t i = si;
//        size_t j = 0 ;
//        while (i < nx && j < ny){
//            I[i][j] = idx;
//            ++ idx;
//            i = i + 1;
//            j = j + 1;
//        }
//    }
//    for (size_t sj = 1; sj < ny; ++sj) {
//        size_t i =  0;
//        size_t j = sj;
//        while (i < nx && j < ny){
//            I[i][j] = idx;
//            ++ idx;
//            i = i + 1;
//            j = j + 1;
//        }
//    }

}

void MatrixCudaOp::allocate() {
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

void MatrixCudaOp::deallocate() {
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

MatrixCudaOp::~MatrixCudaOp() {
    deallocate();
}

void MatrixCudaOp::MatrixCudaOp::init() {
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

    initCudaOp<<<num_blocks, BLOCK_SIZE>>>(I, C, D, dX, dY, nx, ny);
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

void MatrixCudaOp::dtwm(double t, size_t o) {
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

void MatrixCudaOp::findPath(size_t w) {
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

double *MatrixCudaOp::getC() {
    double *hC = new double[nx*ny];
    double *hCn = new double[nx*ny];
    cudaMemcpy(hC, C, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    // copy host copy to a normal index arrangement
    size_t idx = 0;
    for (size_t si = nx; si--; ) {
        size_t i = si;
        size_t j = 0 ;
        while (i < nx && j < ny){
            hCn[i*ny +j] = hC[idx];

            ++ idx;
            i = i + 1;
            j = j + 1;
        }
    }
    for (size_t sj = 1; sj < ny; ++sj) {
        size_t i =  0;
        size_t j = sj;
        while (i < nx && j < ny){
            hCn[i*ny +j] = hC[idx];

            ++ idx;
            i = i + 1;
            j = j + 1;
        }
    }
    return hCn;
}

double *MatrixCudaOp::getD() {
    double *hD = new double[nx*ny];
    double *hDn = new double[nx*ny];
    cudaMemcpy(hD, D, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);
    // copy host copy to a normal index arrangement
    size_t idx = 0;
    for (size_t si = nx; si--; ) {
        size_t i = si;
        size_t j = 0 ;
        while (i < nx && j < ny){
            hDn[i*ny +j] = hD[idx];

            ++ idx;
            i = i + 1;
            j = j + 1;
        }
    }
    for (size_t sj = 1; sj < ny; ++sj) {
        size_t i =  0;
        size_t j = sj;
        while (i < nx && j < ny){
            hDn[i*ny +j] = hD[idx];

            ++ idx;
            i = i + 1;
            j = j + 1;
        }
    }
    return hDn;
}

size_t *MatrixCudaOp::getL() {
    size_t *hL = new size_t[nx*ny];
    size_t *hLn = new size_t[nx*ny];
    cudaMemcpy(hL, L, nx*ny*sizeof(size_t), cudaMemcpyDeviceToHost);
    // copy host copy to a normal index arrangement
    size_t idx = 0;
    for (size_t si = nx; si--; ) {
        size_t i = si;
        size_t j = 0 ;
        while (i < nx && j < ny){
            hLn[i*ny +j] = hL[idx];

            ++ idx;
            i = i + 1;
            j = j + 1;
        }
    }
    for (size_t sj = 1; sj < ny; ++sj) {
        size_t i =  0;
        size_t j = sj;
        while (i < nx && j < ny){
            hLn[i*ny +j] = hL[idx];

            ++ idx;
            i = i + 1;
            j = j + 1;
        }
    }
    return hLn;
}

bool *MatrixCudaOp::getOP() {
    bool *hOP = new bool[nx*ny];
    cudaMemcpy(hOP, OP, nx*ny*sizeof(bool), cudaMemcpyDeviceToHost);
    return hOP;     // no need to rearrange since OP is arranged normally
}

size_t *MatrixCudaOp::getI() {
    size_t *hI = new size_t[nx*ny];
    cudaMemcpy(hI, I, nx*ny*sizeof(size_t), cudaMemcpyDeviceToHost);
    return hI;     // no need to rearrange since OP is arranged normally
}
