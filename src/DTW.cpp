//
// Created by u1590812 on 12/08/16.
//
#include "DTW.h"

using namespace std;

inline int DTW::getIndex(int i, int j) {
    return i*this->ny + j;
}

double DTW::cost(const double &x, const double &y ){
    return std::abs(x-y);
}

void DTW::run(double *S, double *T, const int ns, const int nt){
    this->nx = ns;
    this->ny = nt;
    C = new double[ns*nt];
    D = new double[ns*nt];

    // cost matrix
    for (int i = 0; i < ns; ++i) {
        for (int j = 0; j < nt; ++j) {
            C[getIndex(i,j)] = cost(S[i], T[j]);
        }
    }

    // init
    D[0] = C[0];

    for (int i = 1; i < ns; ++i) {
        D[getIndex(i,0)] = C[getIndex(i,0)] + D[getIndex(i-1,0)];
    }
    for (int j = 1; j < nt; ++j) {
        D[getIndex(0,j)] = C[getIndex(0,j)] + D[getIndex(0,j-1)];
    }

    for (int i = 1; i < ns; ++i) {
        for (int j = 1; j < nt; ++j) {
            D[getIndex(i,j)] = C[getIndex(i,j)] +
                               min3( D[getIndex(i-1,j-1)],
                                     D[getIndex(i,  j-1)],
                                     D[getIndex(i-1,j  )] );
        }
    }
}

void DTW::run(std::vector<double> a, std::vector<double> b) {
    run(&a[0], &b[0], (const int)a.size(), (const int)b.size());
}

// assuming the time series is store vertically in csv
// specify start row and column number
// row and col start from 1
std::vector<double> DTW::readSeries(std::string filename, int row, int col) {
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

void DTW::printMatrix(double *M, string title) {
    cout << title << ": " << endl;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            cout << M[getIndex(i,j)] << " ";
        }
        cout << endl;
    }
}

bool DTW::writeMatrix(double *M, std::string filename) {
    ofstream fout(filename);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            fout << M[getIndex(i,j)] << ", ";
        }
        fout << "\n";
    }
    fout.close();
    return true;
}
