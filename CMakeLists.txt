cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

option(USE_BLAS "Use BLAS library for matrix multiplication (gemm)" ON)
message(STATUS "USE_BLAS ${USE_BLAS}")

add_subdirectory(TinyTorch)
add_subdirectory(demo)

option(BUILD_TEST "Whether or not to build the tests" ON)
if (${BUILD_TEST})
    message(STATUS "Building tests")
    enable_testing()
    add_subdirectory(test)
endif ()
