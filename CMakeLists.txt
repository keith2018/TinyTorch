cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

add_subdirectory(TinyTorch)
add_subdirectory(demo)

option(BUILD_TEST "Whether or not to build the tests" ON)
if (${BUILD_TEST})
    message(STATUS "Building tests")
    enable_testing()
    add_subdirectory(test)
endif ()
