# dtw-pattern

C++ code of the implementation of the DTW-based pattern mining method with CUDA acceleration.
This code implement the MATLAB prototype done in the project into a C++ version and CUDA.

- Matrix class contains the serial implementation.
- MatrixCuda class contains the parallel version 
- MatrixCudaOp class contains the parallel version with memory storage optimisation
- MatrixKernel class contains all the CUDA kernels
