#include <iostream>
#include <list>

#include "./include/sequential_solver.hpp"

int main(int argc, char const **argv) {
	std::list<sudoku::Board> boards;
	  for (auto i = 1u; i < argc; ++i) {
	    boards.emplace_back();
	    boards.back().Read(argv[i]);
	  }
	  for (auto &board : boards) {
	    if (board.Correct()) {
	      std::cout << board << std::endl;
	      auto data = sudoku::Solve(board.Get());
	      bool b = sudoku::Board::Validate(data);
	      if (true) {
	        for (auto i = 0u;
	             i < sudoku::Board::kBoardSize * sudoku::Board::kBoardSize; ++i) {
	          if (i > 0 && i % sudoku::Board::kBoardSize == 0)
	            std::cout << "|\n";
	          if (i > 0 &&
	              i % (sudoku::Board::kBoardSize * sudoku::Board::kQuadrantSize) ==
	                  0)
	            std::cout << "==========================\n";
	          if (i % sudoku::Board::kQuadrantSize == 0)
	            std::cout << "| ";
	          std::cout << static_cast<int>(data[i]) << ' ';
	        }
	        std::cout << "|\n";
	        if (!b)
	          std::cout << "Sth went wrong@@@@@@@@@@@@@@@@@@@@@@@@\n";
	      }
	    }
	  }
	  return 0;
}
