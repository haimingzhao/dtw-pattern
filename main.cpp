#include <iostream>
#include <cmath>
#include <limits>

using namespace std;
const int ns = 10;
const int nt = 10;

// Cost function to return the manhattan distance (diff between 2 data points)
// can also use euclidean distance
double cost(const double &x, const double &y ){
    return abs(x-y);
}

//todo:  what is window size and what is k
//todo:  not sure if this is right
void dtw_c(double S[ns], double T[nt], double D[ns][nt]){
    // cost matrix
    double C[ns][nt];
    for (int i = 0; i < ns; ++i) {
        for (int j = 0; j < nt; ++j) {
            C[i][j] = cost(S[i], T[j]);
        }
    }

    // init
//    double D[ns][nt];
    double inf = std::numeric_limits<double>::infinity();

    for (int i = 0; i < ns; ++i) {
        for (int j = 0; j < nt; ++j) {
            D[i][j] = 0;
        }
    }

    D[0][0] = 0;

    for (int i = 1; i < ns; ++i) {
        D[i][0] = C[i][0] + D[i-1][0];
    }
    for (int j = 1; j < nt; ++j) {
        D[0][j] = C[0][j] + D[0][j-1];
    }

    for (int i = 2; i < ns; ++i) {
        for (int j = 2; j < nt; ++j) {
            D[i][j] = C[i][j] + min( min(D[i-1][j-1], D[i][j-1]), D[i-1][j]);
        }
    }
//    return D;
}

//todo: extract dtw to class
//todo: add code for read data and write result
//todo: add code for validation
//todo: add code for visualisation
int main() {
    cout << cost(3, 4) << endl;
//    int nx = 10;
//    int ny = 10;
    double X[10] = {1,2,3,2,4,5,7,8,6,5};
    double Y[10] = {4,5,7,8,6,5,1,2,3,2};

    double D[ns][nt];
    dtw_c(X, Y, D);

    cout << "D:" << endl;
    for (int i = 0; i < ns; ++i) {
        for (int j = 0; j < nt; ++j) {
            cout << D[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}