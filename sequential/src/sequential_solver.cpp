#include "../include/sequential_solver.hpp"

#include <string.h>

namespace sudoku {
namespace {
std::vector<Board::FieldValue> tmp_board;

#define SetNthBit(number, n) ((number) |= (1ul << (n)))
#define ClearNthBit(number, n) ((number) &= ~(1ul << (n)))
#define GetNthBit(number, n) (((number) >> (n)) & 1u)

bool NotInRow(int row) {
  uint16_t st = 0;
  bool ret = true;
  for (int i = 0; i < Board::kBoardSize; i++) {
    if (tmp_board[Board::kBoardSize * row + i] &&
        GetNthBit(st, tmp_board[Board::kBoardSize * row + i]))
      return false;
    SetNthBit(st, tmp_board[Board::kBoardSize * row + i]);
  }
  return ret;
}

bool NotInCol(int col) {
  uint16_t st = 0;
  for (int i = 0; i < Board::kBoardSize; i++) {
    if (tmp_board[Board::kBoardSize * i + col] &&
        GetNthBit(st, tmp_board[Board::kBoardSize * i + col]))
      return false;
    SetNthBit(st, tmp_board[Board::kBoardSize * i + col]);
  }
  return true;
}

bool NotInBox(int row, int col) {
  row -= row % Board::kQuadrantSize;
  col -= col % Board::kQuadrantSize;
  uint16_t st = 0;
  auto pom_y = row - row % Board::kQuadrantSize;
  auto pom_x = col - col % Board::kQuadrantSize;
  for (int i = 0; i < Board::kQuadrantSize; ++i)
    for (int j = 0; j < Board::kQuadrantSize; ++j) {
      if (tmp_board[(pom_y + i) * Board::kBoardSize + pom_x + j] &&
          GetNthBit(st, tmp_board[(pom_y + i) * Board::kBoardSize + pom_x + j]))
        return false;
      SetNthBit(st, tmp_board[(pom_y + i) * Board::kBoardSize + pom_x + j]);
    }
  return true;
}

bool SolveBacktrack(int start) {
  for (int i = start; i < Board::kBoardSize * Board::kBoardSize; ++i) {
    if (!tmp_board[i]) {
      auto col = i % Board::kBoardSize;
      auto row = i / Board::kBoardSize;
      for (int j = 1; j <= Board::kBoardSize; ++j) {
        tmp_board[i] = j;
        if (NotInCol(col) && NotInRow(row) && NotInBox(row, col) &&
            SolveBacktrack(i + 1))
          return true;
      }
      tmp_board[i] = 0;
      return false;
    }
  }
  return true;
}
} // namespace
std::vector<Board::FieldValue>
Solve(std::vector<Board::FieldValue> const &board) {
  tmp_board = board;
  SolveBacktrack(0);
  return tmp_board;
}
} // namespace sudoku
