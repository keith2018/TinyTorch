cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

option(TINYTORCH_BUILD_DEMO "Whether or not to build demo" ON)
option(TINYTORCH_BUILD_TEST "Whether or not to build the tests" OFF)

option(TINYTORCH_USE_CUDA "Use CUDA" ON)
if (APPLE)
    set(TINYTORCH_USE_CUDA OFF)
endif ()

message(STATUS "TINYTORCH_BUILD_DEMO ${TINYTORCH_BUILD_DEMO}")
message(STATUS "TINYTORCH_BUILD_TEST ${TINYTORCH_BUILD_TEST}")
message(STATUS "TINYTORCH_USE_CUDA ${TINYTORCH_USE_CUDA}")

if (${TINYTORCH_USE_CUDA})
    add_definitions(-DUSE_CUDA)
endif ()

add_subdirectory(src)

if (${TINYTORCH_BUILD_DEMO})
    add_subdirectory(demo)
endif ()

if (${TINYTORCH_BUILD_TEST})
    message(STATUS "Building tests")

    option(TINYTORCH_TEST_CUDA "Test CUDA" OFF)
    message(STATUS "TINYTORCH_TEST_CUDA ${TINYTORCH_TEST_CUDA}")

    if (${TINYTORCH_TEST_CUDA})
        add_definitions(-DDEFAULT_DEVICE_CUDA)
    endif ()

    enable_testing()
    add_subdirectory(test)
endif ()
