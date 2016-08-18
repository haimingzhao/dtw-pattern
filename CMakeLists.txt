cmake_minimum_required(VERSION 3.3)
project(dtwm)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES src/main.cpp src/Util.h src/Util.cpp src/Util.h src/Util.cpp src/Matrix.h src/Matrix.cpp)
add_executable(dtwm ${SOURCE_FILES})