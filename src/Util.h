//
// Created by u1590812 on 12/08/16.
//

#ifndef DTWM_UTIL_H
#define DTWM_UTIL_H

#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <boost/tokenizer.hpp>

#define min3(x,y,z) ( x<y ? ( x<z ? x:z) : (y<z ? y:z) )


class Util {
protected:
    size_t nx, ny;

    // Matrices: 1D array to sim 2D array
    double * C; // Cost matrix
    double * D; // Util matrix

    double cost(const double &x, const double &y );

public:
    int getIndex(size_t i, size_t j);

    // todo constructor destructor ??
    double *getD() const { return D; }
    double *getC() const { return C; }

    void dtw(double *S, double *T, size_t ns, size_t nt);
    void dtw(std::vector<double> a, std::vector<double> b);


    // helper functions for IO

    // print write matrix,  must be one of the class member matrix
    static void printMatrix(double *M, size_t nx, size_t ny, std::string title);
    static bool writeMatrix(double *M, size_t nx, size_t ny, std::string filename); // wt to CSV

    // read CSV file for time series
    static std::vector<double> readSeries(std::string filename, int row, int col);
};


#endif //DTWM_UTIL_H
