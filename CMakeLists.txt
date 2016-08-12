cmake_minimum_required(VERSION 3.3)
project(dtwm)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp DTW.h DTW.cpp)
add_executable(dtwm ${SOURCE_FILES})