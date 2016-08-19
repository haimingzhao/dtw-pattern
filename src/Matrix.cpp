//
// Created by u1590812 on 18/08/16.
//

#include "Matrix.h"

// todo where should I put this
#define min3(x,y,z) ( x<y ? ( x<z ? x:z) : (y<z ? y:z) )

double inf = std::numeric_limits<double>::infinity();

inline int Matrix::getIndex(size_t i, size_t j) {
    return i*this->ny + j;
}

Matrix::Matrix(const std::string datafile): datafile(datafile){
    allocated = false;

    X = Util::readSeries(datafile, 2, 1);
    Y = Util::readSeries(datafile, 2, 2);
    nx = X.size();
    ny = Y.size();

    // todo read t, o, w

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
    // initialise all matrices
    init();

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double minpre;
            size_t mini, minj;
            if (i==0 || j ==0){
                minpre = 0;
                mini = i; minj = j;
            } else{
                minpre = min3( [D[getIndex(i-1,j-1)], D(i-1,j), D(i,j-1)] ) ;

                if minpre == D(i-1,j-1)
                mini = i-1;
                minj = j-1;
                elseif minpre == D(i-1,j)
                mini = i-1;
                minj = j;
                else
                mini = i;
                minj = j-1;
                end
            }
            double dtwm = (minpre + C[getIndex(i,j)]) / (L[getIndex(mini, minj)] + 1) ;
        }
    }

}



