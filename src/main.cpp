#include <iostream>
#include <ctime>

#include "Matrix.h"
#include "MatrixCuda.h"
#include "MatrixCudaOp.h"
#include "MatrixCudaOpS.h"
#include "Util.h"


#include <cfloat>
#include <limits>

using namespace std;

//todo:  what is window size and what is k
//todo:  not sure if this is right

//todo: add code for read data and write result
//todo: add code for validation
//todo: add code for visualisation


int main() {

//    MatrixCuda* m = new MatrixCuda("data/internet.csv");
//    clock_t t = clock();
//    m->runAll(1.2e10, 15, 20);
//    t = clock()-t;
//    cout << "CUDA run: " << ((float)t)/CLOCKS_PER_SEC << endl;

    MatrixCudaOp* m = new MatrixCudaOp("data/small-d.csv");
    m->runAll(1, 2, 2);
    cout << m->getNx() <<  " " << m->getNy() << endl;

    Util::writeMatrixSizet(m->getI(), m->getNx(), m->getNy(), "out/CUDA_I.csv");
//    Util::writeMatrix(m->getC(), m->getNx(), m->getNy(), "out/CUDA_C.csv");
//    Util::writeMatrix(m->getD(), m->getNx(), m->getNy(), "out/CUDA_D.csv");
//    Util::writeMatrixSizet(m->getL(), m->getNx(), m->getNy(), "out/CUDA_L.csv");
//    Util::writeMatrixBool(m->getOP(), m->getNx(), m->getNy(), "out/CUDA_OP.csv");
    cout <<"Written to file"<< endl;
    delete m;

    /*

    Matrix* mh = new Matrix("data/internet.csv");
    t = clock();
    mh->runAll(1.2e10, 15, 20);
    t = clock()-t;
    cout << "Serial run: " << ((float)t)/CLOCKS_PER_SEC << endl;
//    Matrix* mh = new Matrix("data/small-d.csv");
//    mh->runAll(1, 2, 2);
    cout << mh->getNx() <<  " " << mh->getNy() << endl;

    Util::writeMatrix(mh->getC(), mh->getNx(), mh->getNy(), "out/C.csv");
    Util::writeMatrix(mh->getD(), mh->getNx(), mh->getNy(), "out/D.csv");
    Util::writeMatrixSizet(mh->getL(), mh->getNx(), mh->getNy(), "out/L.csv");
    Util::writeMatrixBool(mh->getOP(), mh->getNx(), mh->getNy(), "out/OP.csv");
    cout <<"Written to file"<< endl;

//    cout << std::numeric_limits<double>::infinity() << endl;
//    std::cout << DBL_MAX << std::endl;

    delete mh;
     */

    return 0;
}