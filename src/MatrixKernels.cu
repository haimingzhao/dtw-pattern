//
// Created by u1590812 on 27/08/16.
//

#include "MatrixKernels.h"
#include <math_constants.h>
#include <stddef.h>

#define min2(x,y) (x<y? x : y)
#define min3(x,y,z) ( x<y ? ( x<z ? x:z) : (y<z ? y:z) )

//#define getI_bl(i,j, nx, ny) ((nx-(i-j)-1) < ny) ? ( (nx-(i-j)-1)*(nx-(i-j))/2 + j ):( (ny-1)*(ny)/2 + (nx-(i-j)- ny)*ny + j )
//#define getI_ur(i,j, nx, ny, uj0) (ny<nx) ? (uj0+(ny+ny-j+i+1)*(j-i)/2+i) : ( (j-i<=ny-nx)?(uj0+nx*(j-i)+i):(uj0+nx*(ny-nx)+(nx+ny-j+i+1)*(j-i-ny+nx)/2+i) );

#define getI_tlOp(i,j, nx, ny) ( (i+j < ny) ? ( (i+j)*(i+j+1)/2 + j ):( (ny-1)*(ny)/2 + (i+j-ny+1)*ny + j ) )
#define getI_brOp(i,j, nx, ny, uj0) (ny<=nx) ? (uj0+(ny+ny-j+nx-i)*(j-nx+1+i)/2+nx-1-i) : ( (j-nx+1+i<=ny-nx)?(uj0+nx*(j-nx+1+i)+(nx-1-i)):(uj0+nx*(ny-nx)+(nx+ny-j+nx-i)*(j-nx+1+i-ny+nx)/2+(nx-1-i)) )

// cost function, can be changed
__device__
double getCost(size_t i, size_t j, double* dX, double* dY){
    return dX[i] > dY[j] ? dX[i] - dY[j] : dY[j] - dX[i];
}


__global__
void initCuda(size_t *I, double* C, double* D,
              double* dX, double* dY,
              const size_t nx, const size_t ny) {
    const size_t global_index = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (global_index < nx*ny){
        const size_t i = global_index / ny;
        const size_t j = global_index % ny;

        // calculate cost matrix using the anti diagonal index just got
        size_t idx = i*ny+ j;
        I[i*ny+ j] = idx;
        C[idx] = getCost(i, j, dX, dY);
        D[idx] = CUDART_INF;
    }
}

__global__
void initCudaOp(size_t *I, double* C, double* D,
                double* dX, double* dY,
                const size_t nx, const size_t ny){
    const size_t global_index = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (global_index < nx*ny){
        const size_t i = global_index / ny;
        const size_t j = global_index % ny;

        // calculate anti diagnonal index using derived function in macro
        if ( j < nx-i ){
            I[i*ny+ j] = getI_tlOp(i,j, nx, ny);    // top left optimised mem address
        }else{
            size_t uj0= getI_tlOp(nx-1,0, nx, ny);
            I[i*ny+ j] = getI_brOp(i,j, nx, ny, uj0); // bottom right optimised mem address
        }

        // calculate cost matrix using the anti diagonal index just got
        size_t idx = I[i*ny+ j];
        C[idx] = getCost(i, j, dX, dY);
        D[idx] = CUDART_INF;
    }
}

//__device__
//inline size_t getIndexOp(i,j, nx, ny){
//    size_t idx;
//    if ( j < nx-i ){
//        idx = getI_tlOp(i,j, nx, ny);    // top left optimised mem address
//    }else{
//        size_t uj0= getI_tlOp(nx-1,0, nx, ny);
//        idx = getI_brOp(i,j, nx, ny, uj0); // bottom right optimised mem address
//    }
//    return idx;
//}
//
//__device__
//inline size_t getIndex(i,j, nx, ny){
//    return i*ny + j;
//}

/* sub function to call for calculating individual cells
 * We still need to pass i
 * because we need to calculate distance form start cell in dtwm_task */
__device__
void dtwm_task(size_t i, size_t j,
               size_t* I, double* C, double* D, size_t* L,
               size_t* Rsi, size_t* Rsj, size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj,
               double t, size_t o, const size_t ny){
    double minpre, dtwm;

    size_t idx   = I[i*ny +j];
    size_t min_idx = idx;
    size_t mini = i; size_t minj = j;

    if ( i==0 || j==0 ){    // special case where index is 0 there is no previous
        minpre = 0.0;
//      mini = i; minj = j;
    } else{
        size_t idx_d = I[(i-1)*ny +(j-1)];
        size_t idx_t = I[(i-1)*ny +(j  )];
        size_t idx_l = I[(i  )*ny +(j-1)];

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

        // todo: cannot call clib inf from here, use cuda math_constrains.h
        if (minpre==CUDART_INF){ minpre = 0.0; }
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
            L  [idx] = ( i==0 || j==0 ) ? 1 : L[min_idx] + 1;// update current cell dtw length

            Rsi[idx] = si; // this path start at same as previous cell
            Rsj[idx] = sj; // this path start at same as previous cell

            Pi [idx] = mini; // mark path
            Pj [idx] = minj; // mark path

            // update last position further away
            size_t s_idx = I[si*ny +sj];
            size_t li = Rli[ s_idx ]; // Path end i, only stored in start cell
            size_t lj = Rlj[ s_idx ]; // Path end j, only stored in start
            if ( i > li && j > lj ){
                Rli[s_idx] = i;
                Rlj[s_idx] = j;
            }
        }
    }
}

__global__
void dtwmCuda(size_t* I, double* C, double* D, size_t* L,
              size_t* Rsi, size_t* Rsj, size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj,
              double t, size_t o, const size_t nx, const size_t ny){
    const size_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < ny){
        __syncthreads(); //todo: it maynot be needed
        size_t i,j;
        for (size_t si = 0; si < nx; ++si) {
            // only the thread within the anti-diagonal region is called
            if ( tid <= min2(si, ny-1)){    // careful with case nx > ny
                i = si - tid; // start i position (si) walking up tid times
                j = tid;
                dtwm_task(i, j,
                          I, C, D, L,
                          Rsi, Rsj, Rli, Rlj, Pi, Pj,
                          t, o, ny);
            }
            __syncthreads();
        }

        for (size_t sj = 1; sj < ny; ++sj) {
            // only the thread within the anti-diagonal region is called
            if ( tid >= sj ){                  // careful with case ny > nx
                i = nx-1 - min2(tid-sj, nx-1); // last i - step from cell position (tid) to sj
                j = tid;
                dtwm_task(i, j,
                          I, C, D, L,
                          Rsi, Rsj, Rli, Rlj, Pi, Pj,
                          t, o, ny);
            }
            __syncthreads();
        }
    } // run total for nx+ny-1 times in parallel

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
//    }
*/
}

__device__
void findPath_task(size_t i, size_t j,
                   size_t* I, size_t* L,
                   size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj, bool* OP,
                   size_t w, size_t ny){
    size_t idx = I[i*ny +j];

    size_t li = Rli[idx]; // Path end i, only stored in start cell
    size_t lj = Rlj[idx]; // Path end j, only stored in start
    size_t idx_l = I[li*ny +lj];

    // only look at start cells and end length longer than w, and mark the path
    if (L[idx] == 1 && L[idx_l] > w){

        while(li > i && lj > j){    // while the current last if further away from start
            OP[li*ny + lj] = true;  // use normal horizontal indexing in report paths
            size_t mi = Pi[idx_l];
            size_t mj = Pj[idx_l];
            li=mi;  lj=mj;          // has to do it this way, otherwise weird errors
            idx_l = I[mi*ny +mj];      // not forget to update idx_l as well
        }
        OP[li*ny + lj] = true;
    }
}

__global__
void findPathCuda(size_t* I, size_t* L,
                  size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj, bool* OP,
                  size_t w, size_t nx, size_t ny){
    const size_t global_index = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (global_index < nx*ny){
        const size_t i = global_index / ny;
        const size_t j = global_index % ny;
        findPath_task(i, j, I, L, Rli, Rlj, Pi, Pj, OP, w, ny);
    }
}