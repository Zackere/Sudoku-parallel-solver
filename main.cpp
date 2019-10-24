#include <iostream>

#include "include/cuda_kernel.cuh"
#include "include/sudoku_provider.hpp"

int main(int argc, char** argv) {
	sudoku::SudokuProvider sp;
	for (int i = 1; i < argc; ++i) {
		sp.Read(argv[i]);
		std::cout << (int)sp.IsValid();
	}
	return 0;
}
