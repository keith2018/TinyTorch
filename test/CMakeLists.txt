cmake_minimum_required(VERSION 3.10)
project(TinyTorch_test)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(googletest)

include(GoogleTest)

add_executable(${PROJECT_NAME}
        test.cpp
        test_tensor.cpp
        test_operation.cpp
        test_autograd.cpp
        test_function.cpp
        test_module.cpp
        test_optimizer.cpp
        test_allocator.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
        googletest/googletest/include
        googletest/googlemock/include
)
target_link_libraries(${PROJECT_NAME} TinyTorch_lib gtest_main)

# add tests
gtest_discover_tests(${PROJECT_NAME} WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>)

# nccl distributed test
if (TINYTORCH_USE_NCCL)
    add_executable(TinyTorch_distributed_test test_distributed.cpp)
    target_include_directories(TinyTorch_distributed_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../src
            ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
            googletest/googletest/include
            googletest/googlemock/include
    )
    target_link_libraries(TinyTorch_distributed_test TinyTorch_lib gtest_main)
    gtest_discover_tests(TinyTorch_distributed_test WORKING_DIRECTORY $<TARGET_FILE_DIR:TinyTorch_distributed_test>)
endif ()
