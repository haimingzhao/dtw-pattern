//
// Created by u1590812 on 20/08/16.
//

#include "MatrixCuda.h"

#include <iostream>
#include <limits>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

#define min3(x,y,z) ( x<y ? ( x<z ? x:z) : (y<z ? y:z) );

#define getI_bl(i,j, nx, ny) ((nx-(i-j)-1) < ny) ? ( (nx-(i-j)-1)*(nx-(i-j))/2 + j ):( (ny-1)*(ny)/2 + (nx-(i-j)- ny)*ny + j ) ;
#define getI_ur(i,j, nx, ny, uj0) (ny<=nx) ? (uj0+(ny+ny-j+i+1)*(j-i)/2+i) : ( (j-i<=ny-nx)?(uj0+nx*(j-i)+i):(uj0+nx*(ny-nx)+(nx+ny-j+i+1)*(j-i-ny+nx)/2+i) );

// infinity value for CUDA implementation
double cuda_inf = (std::numeric_limits<double>::max());

MatrixCuda::MatrixCuda(const std::string datafile): Matrix(datafile){

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

//double MatrixCuda::getCost(size_t i, size_t j) {
//    return Matrix::getCost(i, j);
//}

// cost function, can be changed
__device__
double getCost(size_t i, size_t j, double* dX, double* dY){
    return dX[i] > dY[j] ? dX[i] - dY[j] : dY[j] - dX[i];
}

__global__
void initCuda(double* C, size_t *I, double* dX, double* dY, const size_t nx, const size_t ny) {
    const size_t global_index = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (global_index < nx*ny){
        const size_t i = global_index / nx;
        const size_t j = global_index % nx;

        // calculate anti diagnonal index using derived function in macro
        if ( i >= j ){
            I[i*ny+ j] = getI_bl(i,j, nx, ny);
        }else{
            size_t uj0= getI_bl(0,0, nx, ny) ;
            I[i*ny+ j] = getI_ur(i,j, nx, ny, uj0);
        }

        // calculate cost matrix using the anti diagonal index just got
        size_t idx = I[i*ny+ j];
        C[idx] = getCost(i, j, dX, dY);
    }
}

void MatrixCuda::init() {
    std::cout <<"Cuda init"<< std::endl;

    cudaMemset(D, cuda_inf, (nx*ny)*sizeof(double));   // init D to inf for min comparison
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

    /* TODO initialise anti-diagonal coordinate arrays */
    // todo put anti diagnoal index in I
    const size_t num_blocks = (nx*ny + BLOCK_SIZE-1)/BLOCK_SIZE; // rounding up dividing by BLOCK_SIZE

    initCuda<<<num_blocks, BLOCK_SIZE>>>(C, I, dX, dY, nx, ny);
//    size_t idx;
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
//    }
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