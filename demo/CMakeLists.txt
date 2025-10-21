cmake_minimum_required(VERSION 3.10)
project(TinyTorch_demo)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DEMO_SRCS
        demo_autograd.cpp
        demo_module.cpp
        demo_optim.cpp
        demo_mnist.cpp
        main.cpp
)

if (TINYTORCH_USE_NCCL)
    list(APPEND DEMO_SRCS
            demo_nccl.cpp
            demo_ddp.cpp
    )
endif ()

add_executable(${PROJECT_NAME} ${DEMO_SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
)

target_link_libraries(${PROJECT_NAME} TinyTorch_lib)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)

# copy assets
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/data
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data $<TARGET_FILE_DIR:${PROJECT_NAME}>/data
)
