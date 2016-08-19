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

    allocate();
}

void Matrix::allocate() {
    // allocate matrix
    C = new double[nx*ny]; // Cost matrix
    D = new double[nx*ny]; // DTW matrix
    L = new size_t[nx*ny]; // marks the lengths
    Rsi = new size_t[nx*ny]; // Region marking matrix start position i
    Rsj = new size_t[nx*ny]; // Region marking matrix start position j
    Rli = new size_t[nx*ny]; // Region marking matrix last position i
    Rlj = new size_t[nx*ny]; // Region marking matrix last position j
    P = new bool[nx*ny]; // Matrix for storing marked path

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
            L[getIndex(i,j)] = 0;   // zero length
            P[getIndex(i,j)] = 0;   // default to false
            // todo : is this needed?
//            Rsi[getIndex(i,j)] = 0;// not in path and not start of path
//            Rsj[getIndex(i,j)] = 0;// not in path and not start of path
//            Rli[getIndex(i,j)] = 0;// not start of path
//            Rlj[getIndex(i,j)] = 0;// not start of path
        }
    }
}


void Matrix::dtwm(double t, size_t o) {
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            double minpre;
            size_t mini, minj;
            if (i==0 || j ==0){
                minpre = 0;
                mini = i; minj = j;
            } else{
                minpre = min3( D[getIndex(i-1,j-1)],
                               D[getIndex(i-1,j  )],
                               D[getIndex(i  ,j-1)] ) ;

                // mini, minj are the index of the min previous cells
                if (minpre == D[getIndex(i-1,j-1)]){
                    mini = i-1;
                    minj = j-1;
                }else if(minpre == D[getIndex(i-1,j  )]){
                    mini = i-1;
                    minj = j;
                }else{
                    mini = i;
                    minj = j-1;
                }

                if (minpre == inf) minpre = 0;
            }

            // calculated average cost for the path adding the current cell
            double dtwm = (minpre + C[getIndex(i,j)]) / (L[getIndex(mini, minj)] + 1);

            // only consider this cell if average cost dtwm smaller than t
            if (dtwm < t && (L[getIndex(mini,minj)] == 0)) {

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
                if (std::abs( (i - si) - (j - sj) ) < o){
                    D[getIndex(i,j)] = minpre + C[getIndex(i,j)];  // update current cell dtw distance
                    L[getIndex(i,j)] = L[getIndex(mini, minj)] + 1;// update current cell dtw length
                    Rsi[getIndex(i,j)] = si; // this path start at same as previous cell
                    Rsj[getIndex(i,j)] = sj; // this path start at same as previous cell

                    // update last position further away
                    if ( i > Rli[getIndex(si,sj)] && j > Rlj[getIndex(si,sj)] ){
                        Rli[getIndex(si,sj)] = i;
                        Rlj[getIndex(si,sj)] = j;
                    }
                }
            }
        }
    }
}

// Find the path that is suitable to report
// this implementation do not use the recursive method like in the prototype
// it uses sequential scan that is suitable to run in parallel
void Matrix::findPath(size_t w) {
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            size_t si = Rsi[getIndex(i,j)];   // Path start i
            size_t sj = Rsj[getIndex(i,j)];   // Path start j
            size_t li = Rli[getIndex(si,sj)]; // Path end i, only stored in start cell
            size_t lj = Rlj[getIndex(si,sj)]; // Path end j, only stored in start cell
            // mark cell if warping path longer than window
            if (L[getIndex(li,lj)] > w) {
                P[getIndex(i, j)] = true;   //   mark cell as a match path
            }
        }
    }

}

void Matrix::runAll(double t, size_t o, size_t w) {
    init();     // initialise all matrices
    dtwm(t, o); // run DTW modified method with cost threshold: t and path offset: o
    findPath(w);// run find path an mark path with more than window threshold in P
    // todo output P
}
