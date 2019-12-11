# Sudoku Parallel Solver

This project is aimed at implementing sudoku parallel solver in nVidia CUDA.

## Getting Started & Building

Clone this repo and change your cuda location in main folder. While in main folder, type:
```
$ mkdir bin/ && make
```
in order to have parallel solver compiled. If you want to compile sequential solver, type:
```
$ cd sequential/ && mkdir bin/ && make
```

## Running

Command line syntax:
```
$ ./parallel_sudoku_solver path_to_sudoku ...
```
There are example data in `data/` folder. If you want to run sudoku solver on all example data, type:
```
$ cd Sudoku-parallel-solver/ && ./run.sh
``` 
Simillarly for sequential solver:
```
$ cd Sudoku-parallel-solver/sequential/ && ./run.sh
```
In case of `out_of_memory` errors, you can control amount of memory used to run the program by changing `kNBoards` variable inside `include/device_resource_manager.cuh`. By default, the algorithm uses 2^22 sudoku boards.

## Algorithm

TODO
