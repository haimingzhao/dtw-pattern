//
// Created by u1590812 on 12/08/16.
//
#include "Util.h"

using namespace std;

//inline int Util::getIndex(size_t i, size_t j) {
//    return i*this->ny + j;
//}

//double Util::cost(const double &x, const double &y ){
//    return std::abs(x-y);
//}

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

// assuming the time series is store vertically in csv
// specify start row and column number
// row and col start from 1
std::vector<double> Util::readSeries(std::string filename, int row, int col) {
    vector<double> values;
    ifstream file(filename);
    if (file)
    {
        // use boost tokenizer
        typedef boost::tokenizer< boost::char_separator<char> > Tokenizer;
        boost::char_separator<char> sep(",");
        string line;

        int rowc = 1; // row start from 1

        while (getline(file, line)) {
            if(rowc >= row){
                Tokenizer data(line, sep);   // tokenize the line of data
                Tokenizer::iterator it = data.begin(); // iterator of the line of data
                int colc = 1; // column counter

                while(it != data.end() && colc <= col) {
                    if (colc==col){
                        // convert string into double value and store
                        values.push_back(strtod(it->c_str(), 0));
                    }
                    ++colc;
                    ++it;
                }
            }
            ++rowc;
        }
    } else {
        cerr << "Error: File not exist or cannot open: " << filename << endl;
        return {};
    }
//    // test
//    cout << "test values read: ";
//    for (auto i = values.begin(); i != values.end(); ++i)
//        cout << *i << ' ';
//    cout << endl;

    return values;
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
    }
    fout.close();
    return true;
}
