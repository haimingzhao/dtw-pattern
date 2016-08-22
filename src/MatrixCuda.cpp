//
// Created by u1590812 on 20/08/16.
//

#include "MatrixCuda.h"

#include <iostream>
#include <limits>
#include <math_constants.h>
#include <cuda_runtime.h>

#define min3(x,y,z) ( x<y ? ( x<z ? x:z) : (y<z ? y:z) );

double cuda_inf = std::numeric_limits<double>::infinity();

MatrixCuda::MatrixCuda(const std::string datafile): Matrix(datafile){

    // Allocate and initialise 2D diagonal index matrix
    I = new size_t*[nx];
    for (size_t i = 0; i < nx; ++i) {
        I[i] = new size_t[ny]();
    }

    /* Initialise anti-diagonal memory location coordinates,
     * i is row -> X of length nx,
     * j is column -> Y of length ny
     * careful for size_t being unsigned */
    size_t idx = 0;
    for (size_t si = nx; si--; ) {
        size_t i = si;
        size_t j = 0 ;
        while (i < nx && j < ny){
            I[i][j] = idx;
            ++ idx;
            i = i + 1;
            j = j + 1;
        }
    }
    for (size_t sj = 1; sj < ny; ++sj) {
        size_t i =  0;
        size_t j = sj;
        while (i < nx && j < ny){
            I[i][j] = idx;
            ++ idx;
            i = i + 1;
            j = j + 1;
        }
    }

    std::cout<< "indexes: " << std::endl;
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            std::cout<< I[i][j] << " " ;
        }
        std::cout << std::endl;
    }

//    // TODO anti-diagonal calculation order put to kernel
//    idx = 0;
//    for (size_t si = 0; si < nx; ++si) {
//        size_t i = si + 1; // because while loop has i--
//        size_t j = 0 ;
//        while (i-- && j < ny){
//            d_index[i][j] = idx;
//            j = j + 1;
//            ++ idx;
//        }
//    }
//
//    for (size_t sj = 1; sj < ny; ++sj) {
//        size_t i = nx ;  // which is nx = i end index +1, because we need it for i--
//        size_t j = sj ;
//        while (i-- && j < ny){
//            d_index[i][j] = idx;
//            ++ idx;
//            j = j + 1;
//        }
//    }
//
//    std::cout<< "indexes: " << std::endl;
//    for (size_t i = 0; i < nx; ++i) {
//        for (size_t j = 0; j < ny; ++j) {
//            std::cout<< d_index[i][j] << " " ;
//        }
//        std::cout << std::endl;
//    }

}

void MatrixCuda::allocate() {
    std::cout << "Cuda allocate" << std::endl;

#ifdef TIME
    cudaEvent_t start , stop ;
    cudaEventCreate (& start) ;
    cudaEventCreate (& stop) ;
    cudaEventRecord ( start ) ;
    float milliseconds = 0.0 ;
#endif
    // todo  copy X Y to device
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

double MatrixCuda::getCost(size_t i, size_t j) {
    return Matrix::getCost(i, j);
}

void MatrixCuda::init() {
    std::cout <<"Cuda init"<< std::endl;

//    cudaFree(C);
//
//    cudaMemset(D);
//    cudaMemset(L);
//    cudaMemset(Rsi);
//    cudaMemset(Rsj);
//    cudaMemset(Rli);
//    cudaMemset(Rlj);
//
//    cudaMemset(Pi);
//    cudaMemset(Pj);
//
//    cudaFree(visited);
//    cudaFree(OP);

//    for (size_t i = 0; i < nx; ++i) {
//        for (size_t j = 0; j < ny; ++j) {
//            size_t idx = I[i][j];
//            C[idx] = getCost(i, j);
//            D[idx] = inf;
////            L[getIndex(i,j)] = 0;   // zero length
////            OP[getIndex(i,j)] = 0;   // default to false
//            // todo : is this needed?
////            Rsi[getIndex(i,j)] = 0;// not in path and not start of path
////            Rsj[getIndex(i,j)] = 0;// not in path and not start of path
////            Rli[getIndex(i,j)] = 0;// not start of path
////            Rlj[getIndex(i,j)] = 0;// not start of path
////            // Pi, Pj
//        }
//    }

    // todo anti diagonal cuda
    size_t idx;
    for (size_t si = 0; si < nx; ++si) {
        size_t i = si + 1; // because while loop has i--
        size_t j = 0 ;
        while (i-- && j < ny){
            idx = I[i][j];
            C[idx] = getCost(i, j);
            D[idx] = cuda_inf;
            j = j + 1;
        }
    }

    for (size_t sj = 1; sj < ny; ++sj) {
        size_t i = nx ;  // which is nx = i end index +1, because we need it for i--
        size_t j = sj ;
        while (i-- && j < ny){
            idx = I[i][j];
            C[idx] = getCost(i, j);
            D[idx] = cuda_inf;
            j = j + 1;
        }
    }
}

/* We still need to pass i, j because we need to calculate distance form start cell */
void dtwm_task(size_t i, size_t j, size_t** I, double t, size_t o,
               double* C, double* D, size_t* L,
               size_t* Rsi, size_t* Rsj, size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj){
    double minpre, dtwm;

    size_t idx   = I[i  ][j  ];
    size_t min_idx = idx;
    size_t mini = i; size_t minj = j;

    if ( i==0 || j==0 ){    // special case where index is 0 there is no previous
        minpre = 0.0;
//      mini = i; minj = j;
    } else{
        size_t idx_d = I[i-1][j-1];
        size_t idx_t = I[i-1][j  ];
        size_t idx_l = I[i  ][j-1];

        minpre = min3( D[idx_d], D[idx_t], D[idx_l] );

        // mini, minj are the index of the min previous cells
        if (minpre == D[idx_d]){
            mini = i-1;
            minj = j-1;
            min_idx = idx_d;
        }else if(minpre == D[idx_t]){
            mini = i-1;
            minj = j;
            min_idx = idx_t;
        }else if(minpre == D[idx_l]){
            mini = i;
            minj = j-1;
            min_idx = idx_l;
        }
        if (minpre == cuda_inf){ minpre = 0.0; }
    }

    // calculated average cost for the path adding the current cell
    dtwm = (minpre + C[idx]) / (L[min_idx] + 1.0);

    // only consider this cell if average cost dtwm smaller than t
    if (dtwm < t && (L[min_idx] == 0)) {
//            if ( dtwm<t && L[getIndex(i-1,j-1)]==0
//                        && L[getIndex(i-1,j  )]==0
//                        && L[getIndex(i  ,j-1)]==0) {

        // if previous cell not in a path, start new path

        D[idx] = C[idx];  // update current cell dtw distance
        L[idx] = 1;                 // update current cell dtw length

        Rsi[idx] = i; // this path start at i
        Rsj[idx] = j; // this path start at j
        Rli[idx] = i; // this path ends at i
        Rlj[idx] = j; // this path ends at j

        // else add to the previous cell's path
        // if the current cell is not diverge more than the offset o
    }else if (dtwm < t) {
        size_t si = Rsi[min_idx];
        size_t sj = Rsj[min_idx];

        // Note: have to use comparison since size_t is unsigned !!
        // guarantee si is smaller than i for this implementation but watch out
        size_t offset = (i-si)>(j-sj) ? (i-si)-(j-sj) : (j-sj)-(i-si);
        if ( offset < o){
            D  [idx] = minpre + C[idx];  // update current cell dtw distance
            L  [idx] = L[min_idx] + 1;// update current cell dtw length

            Rsi[idx] = si; // this path start at same as previous cell
            Rsj[idx] = sj; // this path start at same as previous cell

            Pi [idx] = mini; // mark path
            Pj [idx] = minj; // mark path

            // update last position further away
            size_t s_idx = I[si][sj];
            size_t li = Rli[ s_idx ]; // Path end i, only stored in start cell
            size_t lj = Rlj[ s_idx ]; // Path end j, only stored in start
            if ( i > li && j > lj ){
                Rli[s_idx] = i;
                Rlj[s_idx] = j;
            }
        }
    }
}

void MatrixCuda::dtwm(double t, size_t o) {
    std::cout <<"Cuda dtwm"<< std::endl;

    // todo anti diagonal cuda
    for (size_t si = 0; si < nx; ++si) {
        size_t i = si + 1; // because while loop has i--
        size_t j = 0 ;
        while (i-- && j < ny){
            dtwm_task(i, j, I, t, o,
                      C, D, L, Rsi, Rsj, Rli, Rlj, Pi, Pj);
            j = j + 1;
        }
    }

    for (size_t sj = 1; sj < ny; ++sj) {
        size_t i = nx ;  // which is nx = i end index +1, because we need it for i--
        size_t j = sj ;
        while (i-- && j < ny){
            dtwm_task(i, j, I, t, o,
                      C, D, L, Rsi, Rsj, Rli, Rlj, Pi, Pj);
            j = j + 1;
        }
    }
}


void findPath_task(size_t i, size_t j, size_t** I, size_t ny, size_t w,
                   size_t* L, size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj, bool* OP){
    size_t idx = I[i][j];

    size_t li = Rli[idx]; // Path end i, only stored in start cell
    size_t lj = Rlj[idx]; // Path end j, only stored in start
    size_t idx_l = I[li][lj];

    // only look at start cells and end length longer than w, and mark the path
    if (L[idx] == 1 && L[idx_l] > w){

        while(li > i && lj > j){    // while the current last if further away from start
            OP[li*ny + lj] = true;  // use normal horizontal indexing in report paths
            size_t mi = Pi[idx_l];
            size_t mj = Pj[idx_l];
            li=mi;  lj=mj;          // has to do it this way, otherwise weird errors
            idx_l = I[mi][mj];      // not forget to update idx_l as well
        }
        OP[li*ny + lj] = true;
    }
}

void MatrixCuda::findPath(size_t w) {

    std::cout <<"Cuda findPath"<< std::endl;
    for (size_t si = 0; si < nx; ++si) {
        size_t i = si + 1; // because while loop has i--
        size_t j = 0 ;
        while (i-- && j < ny){
            findPath_task(i,j, I, ny, w,
                    L, Rli, Rlj, Pi, Pj, OP);
            j = j + 1;
        }
    }

    for (size_t sj = 1; sj < ny; ++sj) {
        size_t i = nx ;  // which is nx = i end index +1, because we need it for i--
        size_t j = sj ;
        while (i-- && j < ny){
            findPath_task(i,j, I, ny, w,
                          L, Rli, Rlj, Pi, Pj, OP);
            j = j + 1;
        }
    }
}