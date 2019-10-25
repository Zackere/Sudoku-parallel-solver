#pragma once

#include <vector>

#include "../include/sudoku_provider.hpp"

namespace sudoku {
namespace sudoku_validator {
bool IsValid(SudokuProvider::Board const& board);
}
}
