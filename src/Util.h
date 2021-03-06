//
// Created by u1590812 on 12/08/16.
//

#ifndef DTWM_UTIL_H
#define DTWM_UTIL_H

#include <stdlib.h>

#include <vector>
#include <string>


#define min3(x,y,z) ( x<y ? ( x<z ? x:z) : (y<z ? y:z) )


class Util {
protected:
    size_t nx, ny;
    std::vector<double> X;
    std::vector<double> Y;

//    // Matrices: 1D array to sim 2D array
//    double * C; // Cost matrix
//    double * D; // Util matrix

//    double cost(const double &x, const double &y );

public:
//    int getIndex(size_t i, size_t j);
//    void dtw(double *S, double *T, size_t ns, size_t nt);
//    void dtw(std::vector<double> a, std::vector<double> b);

    const std::vector<double> &getX() const {
        return X;
    }

    const std::vector<double> &getY() const {
        return Y;
    }

    void capVectors(int cap);

// helper functions for IO
    // print write matrix,  must be one of the class member matrix
    void readSeries(const std::string datafile, int start_row);

    static void printMatrix(double *M, size_t nx, size_t ny, std::string title);
    static bool writeMatrix(double *M, size_t nx, size_t ny, std::string filename); // wt to CSV
    static bool writeMatrixBool(bool *M, size_t nx, size_t ny, std::string filename);
    static bool writeMatrixSizet(size_t *M, size_t nx, size_t ny, std::string filename);

};


#endif //DTWM_UTIL_H
