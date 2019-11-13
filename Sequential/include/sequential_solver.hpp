#pragma once

#include <vector>

#include "./board.hpp"

namespace sudoku {
std::vector<Board::FieldValue>
Solve(std::vector<Board::FieldValue> const &board);
} // namespace sudoku
