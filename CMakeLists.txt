cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

set(TINYTORCH_BUILD_DEMO ON CACHE BOOL "Whether or not to build demo" FORCE)
set(TINYTORCH_BUILD_TEST OFF CACHE BOOL "Whether or not to build the tests" FORCE)

set(TINYTORCH_USE_CUDA ON CACHE BOOL "Use CUDA" FORCE)
set(TINYTORCH_USE_NCCL ON CACHE BOOL "Use NCCL" FORCE)

if (APPLE)
    set(TINYTORCH_USE_CUDA OFF)
endif ()

if (NOT TINYTORCH_USE_CUDA OR APPLE OR MSVC)
    set(TINYTORCH_USE_NCCL OFF)
endif ()

message(STATUS "TINYTORCH_BUILD_DEMO ${TINYTORCH_BUILD_DEMO}")
message(STATUS "TINYTORCH_BUILD_TEST ${TINYTORCH_BUILD_TEST}")
message(STATUS "TINYTORCH_USE_CUDA ${TINYTORCH_USE_CUDA}")
message(STATUS "TINYTORCH_USE_NCCL ${TINYTORCH_USE_NCCL}")

if (TINYTORCH_USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif ()

if (TINYTORCH_USE_NCCL)
    add_definitions(-DUSE_NCCL)
endif ()

add_subdirectory(src)

if (TINYTORCH_BUILD_DEMO)
    add_subdirectory(demo)
endif ()

if (TINYTORCH_BUILD_TEST)
    message(STATUS "Building tests")

    option(TINYTORCH_TEST_CUDA "Test CUDA" OFF)
    message(STATUS "TINYTORCH_TEST_CUDA ${TINYTORCH_TEST_CUDA}")

    if (TINYTORCH_TEST_CUDA)
        add_definitions(-DDEFAULT_DEVICE_CUDA)
    endif ()

    enable_testing()
    add_subdirectory(test)
endif ()
