cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif ()

if (${USE_BLAS})
    add_definitions(-DUSE_BLAS)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif ()

file(GLOB_RECURSE TinyTorch_src "./*.cpp")

if (MSVC)
    set_source_files_properties(${TinyTorch_src} PROPERTIES COMPILE_FLAGS "/WX")
else ()
    set_source_files_properties(${TinyTorch_src} PROPERTIES COMPILE_FLAGS "-Werror -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif ()

add_library(${PROJECT_NAME} ${TinyTorch_src})

if (${USE_BLAS})
    if (MSVC)
        target_include_directories(${PROJECT_NAME} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/OpenBLAS/win64-64/include
        )
        target_link_libraries(${PROJECT_NAME}
                "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/OpenBLAS/win64-64/lib/libopenblas.dll.a"
        )
    elseif (APPLE)
        find_library(ACCELERATE_LIBRARY Accelerate)
        target_link_libraries(${PROJECT_NAME} ${ACCELERATE_LIBRARY})
    else ()
        target_include_directories(${PROJECT_NAME} PRIVATE
                "/usr/include/openblas"
        )
        target_link_libraries(${PROJECT_NAME} openblas)
    endif ()
endif ()