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

void DTW::printC(){
    cout << "C:" << endl;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            cout << C[getIndex(i,j)] << " ";
        }
        cout << endl;
    }
}

void DTW::printD(){
    cout << "D:" << endl;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            cout << D[getIndex(i,j)] << " ";
        }
        cout << endl;
    }
}

// assuming the time series is store vertically in
vector<double> DTW::readSeries(string filename) {
//    ifstream file(filename);
//    while(getline(file,line))
//    {
//        ++numline;
//    }
    vector<double> values;
    ifstream file(filename);
    if (file)
    {
        // use boost
        typedef boost::tokenizer< boost::char_separator<char> > Tokenizer;
        boost::char_separator<char> sep(",");
        string line;

        while (getline(file, line))
        {
            Tokenizer info(line, sep);   // tokenize the line of data
            vector<double> values;

            for (Tokenizer::iterator it = info.begin(); it != info.end(); ++it)
            {
                // convert data into double value, and store
                values.push_back(strtod(it->c_str(), 0));
            }

            // store array of values
            values.push_back(values);
        }
    }
    else
    {
        cerr << "Error: File not exist or cannot open: " << filename << endl;
        return null;
    }

    return values;
}

void DTW::writeC(string filename) {

}

void DTW::writeD(string filename) {

}
