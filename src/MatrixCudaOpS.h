//
// Created by u1590812 on 20/08/16.
//

#ifndef DTWM_MATRIXCUDAOPS_H
#define DTWM_MATRIXCUDAOPS_H

#include <stddef.h>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "Matrix.h"

class MatrixCudaOpS : public Matrix{
private:

    /* anti-diagonal indexes, 2D array corresponds to index in 1D array
     * to store indexes for use of matrix stored in diagonal consecutive way */
    size_t* I;  // index matrix. use 1 D array and access 2D index like i*ny + j

    // device copy of data
    double* dX;
    double* dY;

    void allocate();
    void deallocate();

//    inline size_t getIndex(size_t i, size_t j);
//    double getCost(size_t i, size_t j); // calculate the cost of position i, j

public:
    MatrixCudaOpS(const std::string datafile);
    virtual ~MatrixCudaOpS();

// getters
//    size_t getNx() const { return nx; }
//    size_t getNy() const { return ny; }
    double *getC();
    double *getD();
    size_t *getL();
    bool  *getOP();
    size_t *getI();

    // the 3 method to run
    void init();

    //    double t:  minimum threshold for average cost
    //    size_t o:  threshold for o: diagonal offset,
    void dtwm(double t, size_t o);

    void findPath(size_t w); // w: threshold for o: diagonal offset, w: window for report length

};
#endif //DTWM_MATRIXCUDAOPS_H
