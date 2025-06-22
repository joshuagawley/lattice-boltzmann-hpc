cmake_minimum_required(VERSION 3.10)

project(hpc-lattice-boltzmann LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_subdirectory(boltzmann-openmp)
add_subdirectory(boltzmann-mpi)
add_subdirectory(boltzmann-opencl)

# Copy input files to the build directory
configure_file(input_128x128.params ${CMAKE_CURRENT_BINARY_DIR}/input_128x128.params COPYONLY)
configure_file(obstacles_128x128.dat ${CMAKE_CURRENT_BINARY_DIR}/obstacles_128x128.dat COPYONLY)
