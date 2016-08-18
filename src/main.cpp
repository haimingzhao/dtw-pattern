#include <iostream>
#include "Util.h"

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

    vector<double> a = Util::readSeries("data/small.csv", 2, 1);
    vector<double> b = Util::readSeries("data/small.csv", 2, 2);

    Util d;
    d.dtw(a, b);
    Util::printMatrix(d.getC(), a.size(), b.size(), "C");
    Util::printMatrix(d.getD(), a.size(), b.size(), "D");
    Util::writeMatrix(d.getD(), a.size(), b.size(), "out/D.csv");
    //Util::readSeries("data/advertising-and-sales-data-36-co.csv", 2, 2);

    return 0;
}