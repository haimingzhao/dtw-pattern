//
// Created by u1590812 on 12/08/16.
//

#ifndef DTWM_DTW_H
#define DTWM_DTW_H

#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <boost/tokenizer.hpp>

#define min3(x,y,z) ( x<y ? ( x<z ? x:z) : (y<z ? y:z) )


class DTW {
    int nx, ny;

    double * D; // 1D array to sim 2D array
    double * C; // 1D array to sim 2D array

private:
    double cost(const double &x, const double &y );

public:
    int getIndex(int i, int j);

    // todo constructor destructor ??
    double *getD() const { return D; }
    double *getC() const { return C; }

    void run(double *S, double *T, const int ns, const int nt);
    void run(std::vector<double> a, std::vector<double> b);

    // helper functions for IO

    // print write matrix,  must be one of the class member matrix
    void printMatrix(double *M, std::string title);
    bool writeMatrix(double *M, std::string filename); // wt to CSV

    // read CSV file for time series
    static std::vector<double> readSeries(std::string filename, int row, int col);
};


#endif //DTWM_DTW_H
