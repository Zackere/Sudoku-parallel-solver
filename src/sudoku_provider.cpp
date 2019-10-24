#include "../include/sudoku_provider.hpp"

#include <fstream>
#include <set>
#include <iostream>

namespace sudoku {
namespace {
constexpr unsigned kSudokuRowSize = 9;
constexpr unsigned kSudokuColSize = 9;
constexpr unsigned kSudokuSize = kSudokuRowSize * kSudokuColSize;

bool CheckRow(SudokuProvider::Board const& board, unsigned row) {
	std::set<SudokuProvider::Board::value_type> used_values;
	for (auto i = 0u; i < kSudokuRowSize; ++i) {
		const auto val = board[kSudokuRowSize * row + i];
		if (val > 0 && val < 10) {
			if (!used_values.insert(val).second)
				return false;
		} else if (val > 9) {
			return false;
		}
	}
	return true;
}

bool CheckCol(SudokuProvider::Board const& board, unsigned col) {
	std::set<SudokuProvider::Board::value_type> used_values;
	for (auto i = 0u; i < kSudokuColSize; ++i) {
		const auto val = board[kSudokuColSize * i + col];
		if (val > 0 && val < 10) {
			if (!used_values.insert(val).second)
				return false;
		} else if (val > 9) {
			return false;
		}
	}
	return true;
}
}
SudokuProvider::SudokuProvider() :
		board_(kSudokuSize, 0) {
}

void SudokuProvider::Read(std::string_view path) {
	auto file = std::ifstream(path.data());
	if (!file.is_open()) {
		std::cerr << "Can't open " << path.data() << " for reading\n";
		return;
	}
	unsigned char c = 0;
	board_.clear();
	while (file >> c) {
		if (c < '0' || c > '9' || board_.size() == kSudokuSize) {
			std::cerr << "File " << path.data()
					<< " seems to be in incorrect format.\n";
			board_.clear();
			break;
		}
		board_.emplace_back(c - '0');
	}
	file.close();
}

SudokuProvider::Result SudokuProvider::IsValid() {
	if (board_.empty())
		return Result::NO_DATA;
	for (auto i = 0u; i < kSudokuRowSize; ++i)
		if (!CheckRow(board_, i))
			return Result::INVALID_SUDOKU;
	for (auto i = 0u; i < kSudokuColSize; ++i)
		if (!CheckCol(board_, i))
			return Result::INVALID_SUDOKU;
	return Result::VALID;
}

SudokuProvider::Board const& SudokuProvider::Get() {
	return board_;
}
}
