#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

namespace sudoku {
class Board {
public:
  using FieldValue = int8_t;
  using Size = unsigned;
  static bool Validate(std::vector<FieldValue> const &data);
  static constexpr Size kQuadrantSize = 3;
  static constexpr Size kBoardSize = kQuadrantSize * kQuadrantSize;
  Board();
  void Read(char const *path);
  std::vector<FieldValue> const &Get() const;
  bool Correct() const;
  static void Print(std::ostream &out, std::vector<FieldValue> const &board);

private:
  std::vector<FieldValue> data_;
  bool correct_ = true;
};
std::ostream &operator<<(std::ostream &out, Board const &board);
} // namespace sudoku
