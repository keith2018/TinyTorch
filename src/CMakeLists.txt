cmake_minimum_required(VERSION 3.24)
project(TinyTorch_lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (TINYTORCH_USE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
endif ()

if (TINYTORCH_USE_NCCL)
    find_path(NCCL_INCLUDE_DIRS nccl.h
            HINTS /usr/include /usr/local/include /usr/include/nccl /usr/local/include/nccl
    )
    find_library(NCCL_LIBRARY nccl
            HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/local/lib64
    )

    if (NOT NCCL_INCLUDE_DIRS OR NOT NCCL_LIBRARY)
        message(FATAL_ERROR "NCCL not found. Please install NCCL and set NCCL_INCLUDE_DIRS / NCCL_LIBRARY.")
    endif ()

    message(STATUS "Found NCCL include: ${NCCL_INCLUDE_DIRS}")
    message(STATUS "Found NCCL library: ${NCCL_LIBRARY}")
endif ()

file(GLOB_RECURSE TinyTorch_SRC_CPP "./*.cpp")
file(GLOB_RECURSE TinyTorch_SRC_CU "./*.cu")

if (NOT TINYTORCH_USE_NCCL)
    list(FILTER TinyTorch_SRC_CPP EXCLUDE REGEX "Distributed/.*")
    list(FILTER TinyTorch_SRC_CU EXCLUDE REGEX "Distributed/.*")
endif ()

if (TINYTORCH_USE_CUDA)
    add_library(${PROJECT_NAME} ${TinyTorch_SRC_CPP} ${TinyTorch_SRC_CU})
else ()
    add_library(${PROJECT_NAME} ${TinyTorch_SRC_CPP})
endif ()

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

# disable exceptions
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/EHs-c->
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fno-exceptions>
)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
    if (NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL Release)
    add_definitions(-DNDEBUG)
    if (MSVC)
        set_source_files_properties(${TinyTorch_SRC_CPP} PROPERTIES COMPILE_FLAGS "/Ox")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif ()
endif ()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# BLAS
if (APPLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ACCELERATE_NEW_LAPACK)
    target_link_libraries(${PROJECT_NAME} "-framework Accelerate")
else ()
    find_package(BLAS)
    if (BLAS_FOUND)
        find_path(CBLAS_INCLUDE_DIR cblas.h
                PATHS /usr/include /usr/include/openblas
        )
        target_include_directories(${PROJECT_NAME} PRIVATE ${CBLAS_INCLUDE_DIR})
        target_compile_definitions(${PROJECT_NAME} PRIVATE __BLAS__)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${BLAS_LIBRARIES})
    endif ()
endif ()

# CUDA
if (TINYTORCH_USE_CUDA)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES native)
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
endif ()

# NCCL
if (TINYTORCH_USE_NCCL)
    target_include_directories(${PROJECT_NAME} PUBLIC ${NCCL_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${NCCL_LIBRARY})
endif ()
