#include <iostream>
#include <ctime>

#include "Matrix.h"
#include "MatrixCuda.h"
#include "MatrixCudaOp.h"
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

    Util *u = new Util();
    u->readSeries("data/small-d.csv", 2);  // read file for 2 column start with row 2

//    MatrixCuda* m = new MatrixCuda("data/internet.csv");
//    clock_t t = clock();
//    m->runAll(1.2e10, 15, 20);
//    t = clock()-t;
//    cout << "CUDA run: " << ((float)t)/CLOCKS_PER_SEC << endl;

    MatrixCudaOp *mo = new MatrixCudaOp(u->getX(), u->getY());
    mo->runAll(1, 2, 2);
    cout << mo->getNx() << " " << mo->getNy() << endl;

    Util::writeMatrixSizet(mo->getI(), mo->getNx(), mo->getNy(), "out/CUDAmo_I.csv");
    Util::writeMatrix(mo->getC(), mo->getNx(), mo->getNy(), "out/CUDAmo_C.csv");
    Util::writeMatrix(mo->getD(), mo->getNx(), mo->getNy(), "out/CUDAmo_D.csv");
    Util::writeMatrixSizet(mo->getL(), mo->getNx(), mo->getNy(), "out/CUDAmo_L.csv");
    Util::writeMatrixBool(mo->getOP(), mo->getNx(), mo->getNy(), "out/CUDAmo_OP.csv");
    cout <<"Written to file"<< endl;
    delete mo;

    MatrixCuda* mc = new MatrixCuda(u->getX(), u->getY());
    mc->runAll(1, 2, 2);
    cout << mc->getNx() <<  " " << mc->getNy() << endl;

    Util::writeMatrixSizet(mc->getI(), mc->getNx(), mc->getNy(), "out/CUDAmc_I.csv");
    Util::writeMatrix(mc->getC(), mc->getNx(), mc->getNy(), "out/CUDAmc_C.csv");
    Util::writeMatrix(mc->getD(), mc->getNx(), mc->getNy(), "out/CUDAmc_D.csv");
    Util::writeMatrixSizet(mc->getL(), mc->getNx(), mc->getNy(), "out/CUDAmc_L.csv");
    Util::writeMatrixBool(mc->getOP(), mc->getNx(), mc->getNy(), "out/CUDAmc_OP.csv");
    cout <<"Written to file"<< endl;
    delete mc;



//    Matrix* mh = new Matrix("data/internet.csv");
//    t = clock();
//    mh->runAll(1.2e10, 15, 20);
//    t = clock()-t;
//    cout << "Serial run: " << ((float)t)/CLOCKS_PER_SEC << endl;

    Matrix* mh = new Matrix(u->getX(), u->getY());
    mh->runAll(1, 2, 2);
    cout << mh->getNx() <<  " " << mh->getNy() << endl;

    Util::writeMatrix(mh->getC(), mh->getNx(), mh->getNy(), "out/C.csv");
    Util::writeMatrix(mh->getD(), mh->getNx(), mh->getNy(), "out/D.csv");
    Util::writeMatrixSizet(mh->getL(), mh->getNx(), mh->getNy(), "out/L.csv");
    Util::writeMatrixBool(mh->getOP(), mh->getNx(), mh->getNy(), "out/OP.csv");
    cout <<"Written to file"<< endl;


    delete mh;

    return 0;
}