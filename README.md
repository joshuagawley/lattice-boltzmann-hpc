# Optimizing and parallelizing fluid simulations using OpenMP, MPI, and OpenCL

## Introduction
This project contains C implementations of a Lattice-Boltzmann fluid simulation parallelized using OpenMP, MPI, and OpenCL, written for the High Performance Computing unit at the University of Bristol([COMS30053](https://cs-uob.github.io/COMS30053)) during the 2024/25 academic year.

## Building and running the code
Each implementation can be built and run as follows:
```bash
# OpenMP
cmake -S boltzmann-openmp -B build-openmp
cmake --build build-openmp
cd build-openmp
./boltzmann-openmp input_128x128.params obstacles_128x128.dat
```

```bash
# MPI
cmake -S boltzmann-mpi -B build-mpi
cmake --build build-mpi
cd build-mpi
mpirun ./boltzmann-mpi input_128x128.params obstacles_128x128.dat
```

```bash
# OpenCL
cmake -S boltzmann-opencl -B build-opencl
cmake --build build-opencl
cd build-opencl
./boltzmann-opencl input_128x128.params obstacles_128x128.dat
```

## Reports
Alongside the code are two reports outlining which optimizations were carried out and how and why they resulted in better performance.

[OpenMP report](report_openmp.pdf) (grade: 68%)

[MPI and OpenCL report](report_mpi_opencl.pdf) (grade: 75%)