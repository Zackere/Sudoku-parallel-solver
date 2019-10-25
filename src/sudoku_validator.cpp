#include "../include/sudoku_validator.hpp"

#include <set>

namespace sudoku {
namespace sudoku_validator {
namespace {
constexpr unsigned kQuadrantSize = 3;
constexpr unsigned kSudokuRowSize = kQuadrantSize * kQuadrantSize;
constexpr unsigned kSudokuColSize = kQuadrantSize * kQuadrantSize;
constexpr unsigned kSudokuSize = kSudokuRowSize * kSudokuColSize;

auto At(SudokuProvider::Board const& arr, unsigned x,
		unsigned y) -> decltype(arr.at(y * kSudokuRowSize + x)) {
	return arr.at(y * kSudokuRowSize + x);
}

bool CheckRow(SudokuProvider::Board const& board, unsigned row) {
	std::set<SudokuProvider::Board::value_type> used_values;
	for (auto i = 0u; i < kSudokuRowSize; ++i) {
		const auto val = At(board, row, i);
		if (val > 0 && val <= kSudokuRowSize) {
			if (!used_values.insert(val).second)
				return false;
		} else if (val > kSudokuRowSize) {
			return false;
		}
	}
	return true;
}

bool CheckCol(SudokuProvider::Board const& board, unsigned col) {
	std::set<SudokuProvider::Board::value_type> used_values;
	for (auto i = 0u; i < kSudokuColSize; ++i) {
		const auto val = At(board, i, col);
		if (val > 0 && val <= kSudokuColSize) {
			if (!used_values.insert(val).second)
				return false;
		} else if (val > kSudokuColSize) {
			return false;
		}
	}
	return true;
}

bool CheckQuadrant(SudokuProvider::Board const& board, unsigned x, unsigned y) {
	std::set<SudokuProvider::Board::value_type> used_values;
	for (auto i = 0u; i < kQuadrantSize; ++i)
		for (auto j = 0u; j < kQuadrantSize; ++j) {
			const auto val = At(board, x + i, y + j);
			if (val > 0 && val <= kQuadrantSize * kQuadrantSize) {
				if (!used_values.insert(val).second)
					return false;
			} else if (val > kQuadrantSize * kQuadrantSize) {
				return false;
			}
		}
	return true;
}
} // namespace

bool IsValid(std::vector<unsigned char> const& board) {
	if (board.size() != kSudokuRowSize * kSudokuColSize)
		return false;
	for (auto i = 0u; i < kSudokuRowSize; ++i)
		if (!CheckRow(board, i))
			return false;
	for (auto i = 0u; i < kSudokuColSize; ++i)
		if (!CheckCol(board, i))
			return false;
	for (auto i = 0u; i < kSudokuRowSize; i += kQuadrantSize)
		for (auto j = 0u; j < kSudokuColSize; j += kQuadrantSize) {
			if (!CheckQuadrant(board, i, j))
				return false;
		}
	return true;
}
} // namespace sudoku_vaidator
} // namespace sudoku
