cmake_minimum_required(VERSION 3.10)
project(tinytorch_example_optimizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party
)

target_link_libraries(${PROJECT_NAME} TinyTorch_lib)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
