#include <iostream>
#include <ctime>
#include <string>
#include <stdlib.h>

#include "Matrix.h"
#include "MatrixCuda.h"
#include "MatrixCudaOp.h"
#include "Util.h"

using namespace std;

int main(int argc, char *argv[]) {

    if (argc != 5){
        std::cerr << "Usage: main <filename> <threshold> <offset> <window>" << std::endl;
        exit(1);
    }

    const string filename = string(argv[1]);
    const double t = strtod(argv[2], NULL);
    const size_t o = (size_t) strtol(argv[3], NULL, 0);
    const size_t w = (size_t) strtol(argv[4], NULL, 0);

    Util *u = new Util();
    u->readSeries(filename, 2);  // read file for 2 column start with row 2
/*
    cout <<"CUDA Opt Matrix"<< endl;
    MatrixCudaOp *mo = new MatrixCudaOp(u->getX(), u->getY());
    mo->runAll(t, o, w);
    cout << mo->getNx() << " " << mo->getNy() << endl;

//    Util::writeMatrixSizet(mo->getI(), mo->getNx(), mo->getNy(), "out/CUDAmo_I.csv");
//    Util::writeMatrix(mo->getC(), mo->getNx(), mo->getNy(), "out/CUDAmo_C.csv");
//    Util::writeMatrix(mo->getD(), mo->getNx(), mo->getNy(), "out/CUDAmo_D.csv");
//    Util::writeMatrixSizet(mo->getL(), mo->getNx(), mo->getNy(), "out/CUDAmo_L.csv");
//    Util::writeMatrixBool(mo->getOP(), mo->getNx(), mo->getNy(), "out/CUDAmo_OP.csv");
//    cout <<"Written to file"<< endl;
    delete mo;
*/
/*
    cout <<"CUDA Matrix"<< endl;
    MatrixCuda* mc = new MatrixCuda(u->getX(), u->getY());
    mc->runAll(t, o , w);
    cout << mc->getNx() <<  " " << mc->getNy() << endl;

//    Util::writeMatrixSizet(mc->getI(), mc->getNx(), mc->getNy(), "out/CUDAmc_I.csv");
//    Util::writeMatrix(mc->getC(), mc->getNx(), mc->getNy(), "out/CUDAmc_C.csv");
//    Util::writeMatrix(mc->getD(), mc->getNx(), mc->getNy(), "out/CUDAmc_D.csv");
//    Util::writeMatrixSizet(mc->getL(), mc->getNx(), mc->getNy(), "out/CUDAmc_L.csv");
//    Util::writeMatrixBool(mc->getOP(), mc->getNx(), mc->getNy(), "out/CUDAmc_OP.csv");
//    cout <<"Written to file"<< endl;
//    delete mc;
*/
    cout <<"Device Matrix"<< endl;
    Matrix* mh = new Matrix(u->getX(), u->getY());
    mh->runAll(t, o, w);
    cout << mh->getNx() <<  " " << mh->getNy() << endl;

//    Util::writeMatrix(mh->getC(), mh->getNx(), mh->getNy(), "out/C.csv");
//    Util::writeMatrix(mh->getD(), mh->getNx(), mh->getNy(), "out/D.csv");
//    Util::writeMatrixSizet(mh->getL(), mh->getNx(), mh->getNy(), "out/L.csv");
//    Util::writeMatrixBool(mh->getOP(), mh->getNx(), mh->getNy(), "out/OP.csv");
//    cout <<"Written to file"<< endl;
//    delete mh;

    return 0;
}