//
// Created by u1590812 on 12/08/16.
//
#include "Util.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

//void Util::dtw(double *S, double *T, size_t ns, size_t nt){
//    this->nx = ns;
//    this->ny = nt;
//    C = new double[nx*ny];
//    D = new double[nx*ny];
//
//    // cost matrix
//    for (int i = 0; i < nx; ++i) {
//        for (int j = 0; j < ny; ++j) {
//            C[getIndex(i,j)] = cost(S[i], T[j]);
//        }
//    }
//
//    // init
//    D[0] = C[0];
//
//    for (int i = 1; i < nx; ++i) {
//        D[getIndex(i,0)] = C[getIndex(i,0)] + D[getIndex(i-1,0)];
//    }
//    for (int j = 1; j < ny; ++j) {
//        D[getIndex(0,j)] = C[getIndex(0,j)] + D[getIndex(0,j-1)];
//    }
//
//    for (int i = 1; i < nx; ++i) {
//        for (int j = 1; j < ny; ++j) {
//            D[getIndex(i,j)] = C[getIndex(i,j)] +
//                               min3( D[getIndex(i-1,j-1)],
//                                     D[getIndex(i,  j-1)],
//                                     D[getIndex(i-1,j  )] );
//        }
//    }
//}
//
//void Util::dtw(std::vector<double> a, std::vector<double> b) {
//    dtw(&a[0], &b[0], a.size(), b.size());
//}

void Util::readSeries(const std::string datafile, int start_row) {

    std::ifstream file(datafile);
    if (file) {
        std::string line;

        int rowc = 1; // row start from 1

        while (getline(file, line)) {
            if (rowc >= start_row) {
                char delim = ',';
                std::string tok;
                std::istringstream input;
                input.str(line);
                std::getline(input, tok, delim);

                if(line[0]!=','){
                    X.push_back(strtod(tok.c_str(), 0 ));
                }
                if(std::getline(input, tok, delim)){
                    Y.push_back(strtod(tok.c_str(), 0 ));
                }
            }
            ++rowc;
        }
    } else {
        std::cerr << "Error: File not exist or cannot open: " << datafile << std::endl;
    }
}


void Util::printMatrix(double *M, size_t nx, size_t ny, string title) {
    cout << title << ": " << endl;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            cout << M[i*ny + j] << " ";
        }
        cout << endl;
    }
}

bool Util::writeMatrix(double *M, size_t nx, size_t ny, std::string filename) {
    ofstream fout(filename);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            fout << M[i*ny + j] << ", ";
        }
        fout << "\n";
        fout.flush();
    }
    fout.close();
    return true;
}

bool Util::writeMatrixBool(bool *M, size_t nx, size_t ny, std::string filename) {
    ofstream fout(filename);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            fout << M[i*ny + j] << ", ";
        }
        fout << "\n";
        fout.flush();
    }
    fout.close();
    return true;
}

bool Util::writeMatrixSizet(size_t *M, size_t nx, size_t ny, std::string filename) {
    ofstream fout(filename);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            fout << M[i*ny + j] << ", ";
        }
        fout << "\n";
        fout.flush();
    }
    fout.close();
    return true;
}
