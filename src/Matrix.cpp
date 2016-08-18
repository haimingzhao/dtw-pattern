//
// Created by u1590812 on 18/08/16.
//

#include "Matrix.h"

inline int Matrix::getIndex(size_t i, size_t j) {
    return i*this->ny + j;
}

Matrix::Matrix(const std::string datafile): datafile(datafile){
    allocated = false;

    X = Util::readSeries(datafile, 2, 1);
    Y = Util::readSeries(datafile, 2, 2);
    nx = X.size();
    ny = Y.size();

    allocate();
}

void Matrix::allocate() {
    // allocate matrix
    C = new double[nx*ny]; // Cost matrix
    D = new double[nx*ny]; // DTW matrix
    L = new double[nx*ny]; // marks the lengths
    Rsi = new double[nx*ny]; // Region marking matrix start position i
    Rsj = new double[nx*ny]; // Region marking matrix start position j
    Rli = new double[nx*ny]; // Region marking matrix last position i
    Rlj = new double[nx*ny]; // Region marking matrix last position j

    allocated = true;
}

double Matrix::getCost(size_t i, size_t j) {
    return std::abs(X[i]-Y[j]);
}

void Matrix::init() {
    // cost matrix
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {

            C[getIndex(i,j)] = getCost(i, j);
            D[getIndex(i,j)] = inf;
            L[getIndex(i,j)] = 0;   // zero length
            Rsi[getIndex(i,j)] = -1;// not in path and not start of path
            Rsj[getIndex(i,j)] = -1;// not in path and not start of path
            Rli[getIndex(i,j)] = -1;// not start of path
            Rlj[getIndex(i,j)] = -1;// not start of path


        }
    }

}


void Matrix::run() {
    // cost matrix
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            C[getIndex(i,j)] = getCost(i, j);
        }
    }

    // init all D to inf so that it automatically

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            D[getIndex(i,j)] = inf;
        }
    }

//    D[0] = C[0] or 0; // ??

    for (int i = 1; i < nx; ++i) {
        D[getIndex(i,0)] = C[getIndex(i,0)] + D[getIndex(i-1,0)];
    }
    for (int j = 1; j < ny; ++j) {
        D[getIndex(0,j)] = C[getIndex(0,j)] + D[getIndex(0,j-1)];
    }

    for (int i = 1; i < nx; ++i) {
        for (int j = 1; j < ny; ++j) {
            D[getIndex(i,j)] = C[getIndex(i,j)] +
                               min3( D[getIndex(i-1,j-1)],
                                     D[getIndex(i,  j-1)],
                                     D[getIndex(i-1,j  )] );
        }
    }

}



