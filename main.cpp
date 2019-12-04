#include <iostream>
#include <list>

#include "include/cuda_kernel.cuh"
#include "include/device_resource_manager.cuh"
#include "include/time_manager.hpp"

#include "include/board.hpp"

int main(int argc, char const **argv) {
  timeManager::ResetTime();
  std::list<sudoku::Board> boards;
  for (auto i = 1u; i < argc; ++i) {
    boards.emplace_back();
    boards.back().Read(argv[i]);
  }
  for (auto &board : boards) {
    if (board.Correct()) {
      std::cout << board;
      auto solved = sudoku::kernel::Run(board.Get());
      bool b = sudoku::Board::Validate(solved);
      if (b)
        sudoku::Board::Print(std::cout, solved);
      else
        std::cout << "Sth went wrong@@@@@@@@@@@@@@@@@@@@@@@@\n";
    }
  }
  deviceResourceManager::Release();
  std::cout << "Total time spent on computing: "
            << timeManager::GetElapsedTime() << " ms\n";
  return 0;
}
