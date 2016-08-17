#include <iostream>
#include "DTW.h"

using namespace std;

//todo:  what is window size and what is k
//todo:  not sure if this is right

//todo: add code for read data and write result
//todo: add code for validation
//todo: add code for visualisation


int main() {

//    int nx = 10;
//    int ny = 9;
//    double X[10] = {1,2,3,2,4,5,7,8,6,5};
//    double Y[10] = {4,5,7,8,6,5,1,2,3};

    vector<double> a = DTW::readSeries("data/small.csv", 2, 1);
    vector<double> b = DTW::readSeries("data/small.csv", 2, 2);

    DTW d;
    d.run(a, b);
    d.printMatrix(d.getC(), "C");
    d.printMatrix(d.getD(), "D");
    d.writeMatrix(d.getD(), "out/D.csv");
    //DTW::readSeries("data/advertising-and-sales-data-36-co.csv", 2, 2);

    return 0;
}