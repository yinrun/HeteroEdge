cmake_minimum_required(VERSION 3.18)
project(fast_sync_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED QNN_SDK_ROOT)
  message(FATAL_ERROR "QNN_SDK_ROOT is not set. Use -DQNN_SDK_ROOT=/path/to/qairt")
endif()

set(OPENCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
set(OPENCL_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libs")

add_executable(fast_sync_test
  src/main.cpp
  src/gpu_engine.cpp
  src/npu_engine.cpp
  src/pipeline.cpp
)

target_include_directories(fast_sync_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${OPENCL_INCLUDE_DIR}"
  "${QNN_SDK_ROOT}/include"
  "${QNN_SDK_ROOT}/include/QNN"
)

target_link_directories(fast_sync_test PRIVATE "${OPENCL_LIB_DIR}")
target_link_libraries(fast_sync_test PRIVATE OpenCL dl)
