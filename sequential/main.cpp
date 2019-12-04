#include <iostream>
#include <list>

#include "include/sequential_solver.hpp"

#include "include/board.hpp"

int main(int argc, char const **argv) {
  std::list<sudoku::Board> boards;
  for (auto i = 1u; i < argc; ++i) {
    boards.emplace_back();
    boards.back().Read(argv[i]);
  }
  for (auto &board : boards) {
    if (board.Correct()) {
      std::cout << board;
      auto solved = sudoku::Solve(board.Get());
      bool b = sudoku::Board::Validate(solved);
      if (b)
        sudoku::Board::Print(std::cout, solved);
      else
        std::cout << "Sth went wrong@@@@@@@@@@@@@@@@@@@@@@@@\n";
    }
  }
  return 0;
}
