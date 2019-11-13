#include "../include/sequential_solver.hpp"

#include <string.h>

namespace sudoku {
namespace {
struct PossibleValues {
  Board::FieldValue values[Board::kBoardSize] = {0};
  int len = 0;
  PossibleValues &operator=(PossibleValues const &other) {
    if (this != &other) {
      for (int i = 0; i < other.len; ++i)
        values[i] = other.values[i];
      len = other.len;
    }
    return *this;
  }
};

std::vector<Board::FieldValue> tmp_board;

PossibleValues GetPossibleValues(
                                 int x, int y) {
  bool free[Board::kBoardSize];
  memset(free, 1, sizeof free);
  for (auto i = 0u; i < Board::kBoardSize; ++i) {
    auto val = tmp_board[y * Board::kBoardSize + i];
    if (val > 0)
      free[val - 1] = false;
    val = tmp_board[i * Board::kBoardSize + x];
    if (val > 0)
      free[val - 1] = false;
  }
  auto pom_x = x - x % Board::kQuadrantSize;
  auto pom_y = y - y % Board::kQuadrantSize;
  for (int i = 0; i < Board::kQuadrantSize; ++i)
    for (int j = 0; j < Board::kQuadrantSize; ++j) {
      auto val =
          tmo_board[(pom_y + (y + j) % Board::kQuadrantSize) * Board::kBoardSize +
                pom_x + (x + i) % Board::kQuadrantSize];
      if (val > 0)
        free[val - 1] = false;
    }
  PossibleValues ret;
  for (int i = 0; i < Board::kBoardSize; ++i)
    if (free[i])
      ret.values[ret.len++] = i + 1;
  return ret;
}


bool SolveRec() {
  for (int i = 0; i < Board::kBoardSize; ++i)
    for (int j = 0; j < Board::kBoardSize; ++j) {
    if(!tmp_board[i * Board::kBoardSize + j]){
    auto pv = GetPossibleValues(j,i);
    }
    }
}
} // namespace
std::vector<Board::FieldValue>
Solve(std::vector<Board::FieldValue> const &board) {
  tmp_board = board;
  SolveRec();
  return tmp_board;
}
} // namespace sudoku
