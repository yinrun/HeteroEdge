cmake_minimum_required(VERSION 3.14)
project(rmsnorm_custom_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QNN SDK
if(NOT DEFINED QNN_SDK_ROOT)
  message(FATAL_ERROR "QNN_SDK_ROOT not set")
endif()

include_directories(${QNN_SDK_ROOT}/include/QNN)

add_executable(rmsnorm_custom_test main.cpp)
target_link_libraries(rmsnorm_custom_test dl)
