cmake_minimum_required(VERSION 3.18)
project(gpu_bandwidth_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# OpenCL headers and stub library from project root
set(OPENCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
set(OPENCL_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libs")

add_executable(gpu_bandwidth_test src/main.cpp)
target_include_directories(gpu_bandwidth_test PRIVATE "${OPENCL_INCLUDE_DIR}")
target_link_directories(gpu_bandwidth_test PRIVATE "${OPENCL_LIB_DIR}")
target_link_libraries(gpu_bandwidth_test PRIVATE OpenCL dl)
