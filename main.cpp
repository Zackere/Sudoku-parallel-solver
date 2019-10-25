#include <iostream>

#include "include/cuda_kernel.cuh"
#include "include/sudoku_provider.hpp"
#include "include/sudoku_validator.hpp"

int main(int argc, char** argv) {
	sudoku::SudokuProvider sp;
	for (int i = 1; i < argc; ++i) {
		sp.Read(argv[i]);
		std::cout << sudoku::sudoku_validator::IsValid(sp.Get());
	}
	std::cout << '\n';
	return 0;
}
