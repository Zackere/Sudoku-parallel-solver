#pragma once

#include <vector>

#include "board.hpp"

namespace sudoku {
namespace kernel {
std::vector<Board::FieldValue> Run(std::vector<Board::FieldValue> const& board);
}
}
