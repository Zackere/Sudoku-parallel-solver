#include <iostream>

#include "include/cuda_kernel.cuh"

#include "include/board.hpp"

int main(int argc, char const **argv) {
  sudoku::Board board;
  for (auto i = 1u; i < argc; ++i) {
    board.Read(argv[i]);
    if (board.Correct()) {
      std::cout << board << std::endl;
      auto data = sudoku::kernel::Run(board.Get());
      if (sudoku::Board::Validate(data)) {
        for (auto i = 0u;
             i < sudoku::Board::kBoardSize * sudoku::Board::kBoardSize; ++i) {
          if (i > 0 && i % sudoku::Board::kBoardSize == 0)
            std::cout << "|\n";
          if (i > 0 &&
              i % (sudoku::Board::kBoardSize * sudoku::Board::kQuadrantSize) ==
                  0)
            std::cout << "=============\n";
          if (i % sudoku::Board::kQuadrantSize == 0)
            std::cout << '|';
          std::cout << static_cast<int>(data[i]);
        }
        std::cout << "|\n";
      } else
        std::cout << "Sth went wrong\n";
    }
  }
  return 0;
}
