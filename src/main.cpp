#include <iostream>
#include "Util.h"
#include "Matrix.h"
#include <boost/tokenizer.hpp>

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

//    vector<double> a = Util::readSeries("data/small.csv", 2, 1);
//    vector<double> b = Util::readSeries("data/small.csv", 2, 2);

    Matrix* m = new Matrix("data/internet.csv");
//    Matrix* m = new Matrix("data/small.csv");
    cout <<"Read file to matrix"<< endl;
    cout << m->getNx() << " " << m->getNy() << endl;

    m->runAll(1.2e10, 15, 20);
//    m->runAll(1, 2, 2);
    cout <<"finished run"<< endl;

    Util::writeMatrix(m->getC(), m->getNx(), m->getNy(), "out/C.csv");
    Util::writeMatrix(m->getD(), m->getNx(), m->getNy(), "out/D.csv");
    Util::writeMatrixSizet(m->getL(), m->getNx(), m->getNy(), "out/L.csv");
    Util::writeMatrixBool(m->getOP(), m->getNx(), m->getNy(), "out/OP.csv");

    cout <<"Written to file"<< endl;

//    double inf = std::numeric_limits<double>::infinity();
//    cout << min(min(inf, inf), 0.0) << endl;

    return 0;
}