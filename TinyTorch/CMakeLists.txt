cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${TINYTORCH_USE_CUDA})
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80")
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)

    file(GLOB_RECURSE TinyTorch_src_cu "./TensorImpl/*.cu")
else ()
    set(TinyTorch_src_cu "")
endif ()

file(GLOB_RECURSE TinyTorch_src "./*.cpp" "./TensorImpl/*.cpp")
add_library(${PROJECT_NAME} ${TinyTorch_src} ${TinyTorch_src_cu})

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
elseif (CMAKE_BUILD_TYPE STREQUAL Release)
    if (MSVC)
        set_source_files_properties(${TinyTorch_src} PROPERTIES COMPILE_FLAGS "/Ox")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif ()
endif ()

if (${TINYTORCH_USE_CUDA})
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} CUDA::cudart CUDA::curand CUDA::cublas)
endif ()
