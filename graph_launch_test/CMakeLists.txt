cmake_minimum_required(VERSION 3.18)
project(graph_launch_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED QNN_SDK_ROOT)
  message(FATAL_ERROR "QNN_SDK_ROOT is not set. Use -DQNN_SDK_ROOT=/path/to/qairt")
endif()

add_executable(graph_launch_test
  src/main.cpp
  src/qnn_setup.cpp
  src/graph_builder.cpp
)

target_include_directories(graph_launch_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${QNN_SDK_ROOT}/include"
  "${QNN_SDK_ROOT}/include/QNN"
)

target_link_libraries(graph_launch_test PRIVATE dl)
