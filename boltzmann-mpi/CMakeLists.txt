cmake_minimum_required(VERSION 3.10)

project(boltzmann-mpi LANGUAGES C)

find_library(Math m REQUIRED)
find_package(MPI REQUIRED)

add_executable(boltzmann-mpi boltzmann-mpi.c)
target_link_libraries(boltzmann-mpi ${Math_LIBRARY} MPI::MPI_C)