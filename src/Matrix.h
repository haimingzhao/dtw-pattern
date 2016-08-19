//
// Created by u1590812 on 18/08/16.
//

#ifndef DTWM_MATRIX_H
#define DTWM_MATRIX_H

#include <string>
#include <vector>

#include "Util.h"

// macro to get the 1D index from 2D index
//#define getIndex(i, j, ny) (i*ny + j)

class Matrix {
public:
    const std::string datafile;

    // the 2 comparing time series
    size_t nx, ny;
    std::vector<double> X;
    std::vector<double> Y;

    // Matrices: 1D array to sim 2D array
    double* C; // Cost matrix
    double* D; // DTW matrix
    size_t * L; // marks the lengths
    size_t* Rsi; // Region marking matrix start position i
    size_t* Rsj; // Region marking matrix start position j
    size_t* Rli; // Region marking matrix last position i
    size_t* Rlj; // Region marking matrix last position j
    bool*   P;  // report Paths

    inline int getIndex(size_t i, size_t j);

    void allocate();
    bool allocated;
    double getCost(size_t i, size_t j); // calculate the cost of position i, j


public:
    Matrix(const std::string datafile);

    // getters
    size_t getNx() const { return nx; }
    size_t getNy() const { return ny; }
    double *getC() const { return C; }
    double *getD() const { return D; }
    size_t *getL() const { return L; }
    bool *getP() const { return P; }

    // the 3 method to run
    void init();

    //    double t:  minimum threshold for average cost
    //    size_t o:  threshold for o: diagonal offset,
    void dtwm(double t, size_t o);

    void findPath(size_t w); // w: threshold for o: diagonal offset, w: window for report length

    void runAll(double t, size_t o, size_t w);
};


#endif //DTWM_MATRIX_H
