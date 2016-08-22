//
// Created by u1590812 on 18/08/16.
//

#include <stdlib.h>

#include <algorithm>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>

#include "Matrix.h"

// todo where should I put this
#define min3(x,y,z) ( x<y ? ( x<z ? x:z) : (y<z ? y:z) )

inline size_t Matrix::getIndex(size_t i, size_t j) {
    return i*ny + j;
}

void Matrix::readSeries(const std::string datafile, int start_row) {

    std::ifstream file(datafile);
    if (file) {
        std::string line;

        int rowc = 1; // row start from 1

        while (getline(file, line)) {
            if (rowc >= start_row) {
                char delim = ',';
                std::string tok;
                std::istringstream input;
                input.str(line);
                std::getline(input, tok, delim);

                if(line[0]!=','){
                    X.push_back(strtod(tok.c_str(), 0 ));
                }
                if(std::getline(input, tok, delim)){
                    Y.push_back(strtod(tok.c_str(), 0 ));
                }
            }
            ++rowc;
        }
    } else {
        std::cerr << "Error: File not exist or cannot open: " << datafile << std::endl;
    }
}

Matrix::Matrix(const std::string datafile): datafile(datafile){
    allocated = false;

    readSeries(datafile, 2);
    nx = X.size();
    ny = Y.size();

//    allocate(); // move to runAll
}

void Matrix::allocate() {
    std::cout << "Matrix allocate" << std::endl;
    // allocate matrix
    C = new double[nx*ny]; // Cost matrix
    D = new double[nx*ny]; // DTW matrix
    L = new size_t[nx*ny](); // marks the lengths
    Rsi = new size_t[nx*ny](); // Region marking matrix start position i
    Rsj = new size_t[nx*ny](); // Region marking matrix start position j
    Rli = new size_t[nx*ny](); // Region marking matrix last position i
    Rlj = new size_t[nx*ny](); // Region marking matrix last position j

    Pi = new size_t[nx*ny](); // Path marking storing previous position i
    Pj = new size_t[nx*ny](); // Path marking storing previous position i

    visited = new bool[nx*ny](); // helper Matrix for storing marked path
    OP = new bool[nx*ny](); // Matrix for storing marked path
    allocated = true;
}

//todo normalise cost
double Matrix::getCost(size_t i, size_t j) {
    return std::abs(X[i]-Y[j]);
}

void Matrix::init() {
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            C[getIndex(i,j)] = getCost(i, j);
            D[getIndex(i,j)] = inf;
//            L[getIndex(i,j)] = 0;   // zero length
//            OP[getIndex(i,j)] = 0;   // default to false
            // todo : is this needed?
//            Rsi[getIndex(i,j)] = 0;// not in path and not start of path
//            Rsj[getIndex(i,j)] = 0;// not in path and not start of path
//            Rli[getIndex(i,j)] = 0;// not start of path
//            Rlj[getIndex(i,j)] = 0;// not start of path
//            // Pi, Pj
        }
    }
}

void Matrix::dtwm(double t, size_t o) {
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            double minpre, dtwm;
            size_t mini = i, minj = j;

            if ( i==0 || j==0 ){
                minpre = 0.0;
//                mini = i; minj = j;
            } else{
                minpre = std::min( std::min( D[getIndex(i-1,j-1)],
                                             D[getIndex(i-1,j  )] ),
                                             D[getIndex(i  ,j-1)] );
                // mini, minj are the index of the min previous cells
                if (minpre == D[getIndex(i-1,j-1)]){
                    mini = i-1;
                    minj = j-1;
                }else if(minpre == D[getIndex(i-1,j)]){
                    mini = i-1;
                    minj = j;
                }else if(minpre == D[getIndex(i,j-1)]){
                    mini = i;
                    minj = j-1;
                }
                if (minpre == inf){ minpre = 0.0; }
            }

            // calculated average cost for the path adding the current cell
            dtwm = (minpre + C[getIndex(i,j)]) / (L[getIndex(mini, minj)] + 1.0);

            // only consider this cell if average cost dtwm smaller than t
            if (dtwm < t && (L[getIndex(mini,minj)] == 0)) {
//            if ( dtwm<t && L[getIndex(i-1,j-1)]==0
//                        && L[getIndex(i-1,j  )]==0
//                        && L[getIndex(i  ,j-1)]==0) {

                // if previous cell not in a path, start new path

                D[getIndex(i, j)] = C[getIndex(i, j)];  // update current cell dtw distance
                L[getIndex(i, j)] = 1;                 // update current cell dtw length

                Rsi[getIndex(i, j)] = i; // this path start at i
                Rsj[getIndex(i, j)] = j; // this path start at j
                Rli[getIndex(i, j)] = i; // this path ends at i
                Rlj[getIndex(i, j)] = j; // this path ends at j

                // else add to the previous cell's path
                // if the current cell is not diverge more than the offset o
            }else if (dtwm < t) {
                size_t si = Rsi[getIndex(mini,minj)];
                size_t sj = Rsj[getIndex(mini,minj)];

                // Note: have to use comparison since size_t is unsigned !!
                // guarantee si is smaller than i for this implementation but watch out
                size_t offset = (i-si)>(j-sj) ? (i-si)-(j-sj) : (j-sj)-(i-si);
                if ( offset < o){
                    D  [getIndex(i,j)] = minpre + C[getIndex(i,j)];  // update current cell dtw distance
                    L  [getIndex(i,j)] = L[getIndex(mini, minj)] + 1;// update current cell dtw length



                    Rsi[getIndex(i,j)] = si; // this path start at same as previous cell
                    Rsj[getIndex(i,j)] = sj; // this path start at same as previous cell

                    Pi [getIndex(i,j)] = mini; // mark path
                    Pj [getIndex(i,j)] = minj; // mark path

                    // update last position further away
                    size_t li = Rli[getIndex(si, sj)]; // Path end i, only stored in start cell
                    size_t lj = Rlj[getIndex(si, sj)]; // Path end j, only stored in start
                    if ( i > li && j > lj ){
                        Rli[getIndex(si,sj)] = i;
                        Rlj[getIndex(si,sj)] = j;
                    }
                }
            }
        }
    }
}

void Matrix::markPath(size_t si, size_t sj, size_t li, size_t lj){
//    OP[getIndex(li, lj)] = true;
    while(li>si && lj>sj){
        OP[getIndex(li, lj)] = true;
        size_t mi = Pi[getIndex(li,lj)];
        size_t mj = Pj[getIndex(li,lj)];
        li=mi;  lj=mj;  // has to do it this way, otherwise weird errors
    }
    OP[getIndex(li, lj)] = true;
}

// Find the path that is suitable to report
// this implementation do not use the recursive method like in the prototype
// it uses sequential scan that is suitable to run in parallel
void Matrix::findPath(size_t w) {

    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            size_t li = Rli[getIndex(i, j)]; // Path end i, only stored in start cell
            size_t lj = Rlj[getIndex(i, j)]; // Path end j, only stored in start

            // only look at start cells and end length longer than w
            if (L[getIndex(i, j)] == 1 && L[getIndex(li, lj)] > w){
                markPath(i,j, li, lj);
            }
        }
    }
}

void Matrix::runAll(double t, size_t o, size_t w) {
    allocate();
    if (allocated){
        init();     // initialise all matrices
        std::cout <<"Initialised"<< std::endl;

        dtwm(t, o); // run DTW modified method with cost threshold: t and path offset: o
        std::cout <<"Solved Matrix"<< std::endl;

        findPath(w);// run find path an mark path with more than window threshold in P
        std::cout <<"Traced back"<< std::endl;
    }else{
        std::cerr << "Could not allocate matrices." << std::endl;
    }
}
