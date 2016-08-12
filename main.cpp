#include <iostream>
#include "DTW.h"

using namespace std;

////todo:  what is window size and what is k
////todo:  not sure if this is right

//todo: extract dtw to class
//todo: add code for read data and write result
//todo: add code for validation
//todo: add code for visualisation

int main() {

    int nx = 10;
    int ny = 9;
    double X[10] = {1,2,3,2,4,5,7,8,6,5};
    double Y[10] = {4,5,7,8,6,5,1,2,3};

    DTW d;
    d.run(X, Y, nx, ny);
    d.printC();
    d.printD();

    return 0;
}