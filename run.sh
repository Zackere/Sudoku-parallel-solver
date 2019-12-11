#!/bin/bash
./parallel_sudoku_solver `for f in data/*; do echo $f; done`
