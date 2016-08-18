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
private:
    const std::string datafile;

    // the 2 comparing time series
    size_t nx, ny;
    std::vector<double> X;
    std::vector<double> Y;

    // Matrices: 1D array to sim 2D array
    double* C; // Cost matrix
    double* D; // DTW matrix
    double* L; // marks the lengths
    double* Rsi; // Region marking matrix start position i
    double* Rsj; // Region marking matrix start position j
    double* Rli; // Region marking matrix last position i
    double* Rlj; // Region marking matrix last position j

    inline int getIndex(size_t i, size_t j);

    void allocate();
    bool allocated;
    double getCost(size_t i, size_t j); // calculate the cost of position i, j
    void init();

public:
    Matrix(const std::string datafile);

    void run();
};


#endif //DTWM_MATRIX_H
