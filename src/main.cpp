#include <iostream>
#include "Matrix.h"
#include "MatrixCuda.h"
#include "Util.h"

#include <float.h>
#include <limits>

using namespace std;

//todo:  what is window size and what is k
//todo:  not sure if this is right

//todo: add code for read data and write result
//todo: add code for validation
//todo: add code for visualisation


int main() {

//    Matrix* m = new MatrixCuda("data/internet.csv");
//    m->runAll(1.2e10, 15, 20);

//    MatrixCuda* m = new MatrixCuda("data/small-d.csv");
//    m->runAll(1, 2, 2);
//
//    cout << m->getNx() <<  " " << m->getNy() << endl;
//
//    Util::writeMatrix(m->getCFromDivice(), m->getNx(), m->getNy(), "out/CUDA_C.csv");
//    Util::writeMatrix(m->getDFromDivice(), m->getNx(), m->getNy(), "out/CUDA_D.csv");
//    Util::writeMatrixSizet(m->getLFromDivice(), m->getNx(), m->getNy(), "out/CUDA_L.csv");
//    Util::writeMatrixBool(m->getOPFromDivice(), m->getNx(), m->getNy(), "out/CUDA_OP.csv");
//    cout <<"Written to file"<< endl;

    Matrix* m = new Matrix("data/small-d.csv");
    m->runAll(1, 2, 2);
    cout << m->getNx() <<  " " << m->getNy() << endl;

    Util::writeMatrix(m->getC(), m->getNx(), m->getNy(), "out/C.csv");
    Util::writeMatrix(m->getD(), m->getNx(), m->getNy(), "out/D.csv");
    Util::writeMatrixSizet(m->getL(), m->getNx(), m->getNy(), "out/L.csv");
    Util::writeMatrixBool(m->getOP(), m->getNx(), m->getNy(), "out/OP.csv");
    cout <<"Written to file"<< endl;

//    cout << " " << endl;

    delete m;
    return 0;
}