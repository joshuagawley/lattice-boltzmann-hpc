cmake_minimum_required(VERSION 3.10)

project(boltzmann-mpi LANGUAGES C)

find_library(Math m REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(boltzmann-openmp boltzmann-openmp.c)
target_link_libraries(boltzmann-openmp ${Math_LIBRARY} OpenMP::OpenMP_C)