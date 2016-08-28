cmake_minimum_required(VERSION 3.3)
project(dtwm)


include_directories("/usr/local/Cellar/boost/1.59.0/include/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES src/main.cpp src/Util.h src/Util.cpp src/Matrix.h src/Matrix.cpp src/MatrixCuda.cu src/MatrixCuda.h src/MatrixCudaOp.cu src/MatrixCudaOp.h src/MatrixKernels.cu src/MatrixKernels.h)
add_executable(dtwm ${SOURCE_FILES})