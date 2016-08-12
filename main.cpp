#include <iostream>
#include <cmath>
#include <limits>

using namespace std;

#define getindex(i, j) (i*nx + j)
#define minelem(x,y,z) ( x<y ? ( x<z ? x:z) : (y<z ? y:z) )

class DTW {
    int nx, ny;
    double * D; // 1D array to sim 2D array
    double * C; // 1D array to sim 2D array

private:
    double cost(const double &x, const double &y ){
        return abs(x-y);
    }

public:
    void run(double *S, double *T, const int ns, const int nt){
        nx = ns;
        ny = nt;
        C = new double[ns*nt];
        D = new double[ns*nt];

        // cost matrixs
        for (int i = 0; i < ns; ++i) {
            for (int j = 0; j < nt; ++j) {
                C[getindex(i,j)] = cost(S[i], T[j]);
            }
        }

        // init
        D[0] = C[0];

        for (int i = 1; i < ns; ++i) {
            D[getindex(i,0)] = C[getindex(i,0)] + D[getindex(i-1,0)];
        }
        for (int j = 1; j < nt; ++j) {
            D[getindex(0,j)] = C[getindex(0,j)] + D[getindex(0,j-1)];
        }

        for (int i = 2; i < ns; ++i) {
            for (int j = 2; j < nt; ++j) {
                D[getindex(i,j)] = C[getindex(i,j)]
                                   + minelem( D[getindex(i-1,j-1)],
                                           D[getindex(i,  j-1)],
                                           D[getindex(i-1,j  )]);
            }
        }
    }

    void printD(){
        cout << "D:" << endl;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                cout << D[getindex(i,j)] << " ";
            }
            cout << endl;
        }
    }
};

//// Cost function to return the manhattan distance (diff between 2 data points)
//// can also use euclidean distance
//double cost(const double &x, const double &y ){
//    return abs(x-y);
//}
//
////todo:  what is window size and what is k
////todo:  not sure if this is right
//void dtw_c(double *S, double *T, int ns, int nt, double D[ns][nt]){
//    // cost matrix
//    double C[ns][nt];
//    for (int i = 0; i < ns; ++i) {
//        for (int j = 0; j < nt; ++j) {
//            C[i][j] = cost(S[i], T[j]);
//        }
//    }
//
//    // init
//    double **D = new [ns][nt];
//    double inf = std::numeric_limits<double>::infinity();
//
//    for (int i = 0; i < ns; ++i) {
//        for (int j = 0; j < nt; ++j) {
//            D[i][j] = 0;
//        }
//    }
//
//    D[0][0] = C[0][0];
//
//    for (int i = 1; i < ns; ++i) {
//        D[i][0] = C[i][0] + D[i-1][0];
//    }
//    for (int j = 1; j < nt; ++j) {
//        D[0][j] = C[0][j] + D[0][j-1];
//    }
//
//    for (int i = 2; i < ns; ++i) {
//        for (int j = 2; j < nt; ++j) {
//            D[i][j] = C[i][j] + min( min(D[i-1][j-1], D[i][j-1]), D[i-1][j]);
//        }
//    }
////    return D;
//}

//todo: extract dtw to class
//todo: add code for read data and write result
//todo: add code for validation
//todo: add code for visualisation
int main() {
    cout << minelem(3, 4, 1) << endl;
    int nx = 10;
    int ny = 10;
    double X[10] = {1,2,3,2,4,5,7,8,6,5};
    double Y[10] = {4,5,7,8,6,5,1,2,3,2};

    DTW d;
    d.run(X, Y, nx, ny);
    d.printD();

    return 0;
}