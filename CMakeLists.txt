cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

option(USE_CUDA "Use CUDA" ON)
option(USE_BLAS "Use BLAS" ON)

if (APPLE)
    set(USE_CUDA OFF)
endif ()

message(STATUS "USE_CUDA ${USE_CUDA}")
message(STATUS "USE_BLAS ${USE_BLAS}")

if (${USE_CUDA})
    add_definitions(-DUSE_CUDA)
endif ()

add_subdirectory(TinyTorch)
add_subdirectory(demo)

option(BUILD_TEST "Whether or not to build the tests" OFF)
if (${BUILD_TEST})
    message(STATUS "Building tests")

    option(TEST_CUDA "Test CUDA" OFF)
    message(STATUS "TEST_CUDA ${TEST_CUDA}")

    if (${TEST_CUDA})
        add_definitions(-DDEFAULT_DEVICE_CUDA)
    endif ()

    enable_testing()
    add_subdirectory(test)
endif ()
