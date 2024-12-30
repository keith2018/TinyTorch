cmake_minimum_required(VERSION 3.10)
project(TinyTorch_test)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif ()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(googletest)

add_executable(${PROJECT_NAME}
        main.cpp
        test.cpp
        test_autograd.cpp
        test_function.cpp
        test_module.cpp
        test_optimizer.cpp
        test_tensorimpl.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../TinyTorch
        googletest/googletest/include
        googletest/googlemock/include
)
target_link_libraries(${PROJECT_NAME} TinyTorch gtest_main)

include(GoogleTest)

if (${USE_BLAS})
    # copy libopenblas.dll
    if (MSVC)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/OpenBLAS/win64-64/bin/libopenblas.dll
                $<TARGET_FILE_DIR:${PROJECT_NAME}>/libopenblas.dll
        )
    endif ()
endif ()

# add tests
gtest_discover_tests(${PROJECT_NAME} WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>)
