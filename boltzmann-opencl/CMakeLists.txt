cmake_minimum_required(VERSION 3.10)

project(boltzmann-opencl LANGUAGES C)

find_library(Math m REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(boltzmann-opencl boltzmann-opencl.c)
target_link_libraries(boltzmann-opencl ${Math_LIBRARY} OpenCL::OpenCL)

configure_file(kernels.cl ${CMAKE_CURRENT_BINARY_DIR}/kernels.cl COPYONLY)