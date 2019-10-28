#include "../include/board.hpp"

#include <fstream>
#include <iostream>
#include <set>

namespace sudoku {
namespace {
Board::FieldValue At(std::vector<Board::FieldValue> const& arr, unsigned x,
		unsigned y) {
	return arr.at(y * Board::kBoardSize + x);
}
}
Board::Board() {
	data_.reserve(kBoardSize * kBoardSize);
}

void Board::Read(const char* path) {
	correct_ = false;
	data_.clear();
	std::ifstream file;
	file.open(path);
	if (!file.is_open()) {
		std::cerr << "Could not open " << path << '\n';
		return;
	}
	FieldValue val;
	while (file >> val) {
		if (data_.size() == kBoardSize * kBoardSize) {
			std::cerr << "File " << path
					<< " seems to be incorrect (file is too big).\n";
			return;
		}
		data_.emplace_back(val - '0');
	}
	if (data_.size() < kBoardSize * kBoardSize) {
		std::cerr << "File " << path
				<< " seems to be incorrect (file is too small).\n";
		return;
	}
	correct_ = Validate(data_);
}

std::vector<Board::FieldValue> const& Board::Get() const {
	return data_;
}

bool Board::Correct() const {
	return correct_;
}

bool Board::Validate(std::vector<FieldValue> const& data) {
	if (data.size() != kBoardSize * kBoardSize)
		return false;
	std::set<FieldValue> used_values;
	for (auto row = 0u; row < kBoardSize; ++row) {
		used_values.clear();
		for (auto i = 0u; i < kBoardSize; ++i) {
			auto val = At(data, row, i);
			if (val > 0 && val <= kBoardSize) {
				if (!used_values.insert(val).second)
					return false;
			} else if (val > kBoardSize) {
				return false;
			}
		}
	}
	for (auto col = 0u; col < kBoardSize; ++col) {
		used_values.clear();
		for (auto i = 0u; i < kBoardSize; ++i) {
			auto val = At(data, i, col);
			if (val > 0 && val <= kBoardSize) {
				if (!used_values.insert(val).second)
					return false;
			} else if (val > kBoardSize) {
				return false;
			}
		}
	}
	for (auto x = 0u; x < kBoardSize; x += kQuadrantSize) {
		for (auto y = 0u; y < kBoardSize; y += kQuadrantSize) {
			used_values.clear();
			for (auto i = 0u; i < kQuadrantSize; ++i)
				for (auto j = 0u; j < kQuadrantSize; ++j) {
					auto val = At(data, x + i, y + j);
					if (val > 0 && val <= kQuadrantSize * kQuadrantSize) {
						if (!used_values.insert(val).second)
							return false;
					} else if (val > kQuadrantSize * kQuadrantSize) {
						return false;
					}
				}
		}
	}
	return true;
}

std::ostream& operator<<(std::ostream& out, Board const& board) {
	auto data = board.Get();
	for (auto i = 0u; i < Board::kBoardSize * Board::kBoardSize; ++i) {
		if (i > 0 && i % Board::kBoardSize == 0)
			out << "|\n";
		if (i > 0 && i % (Board::kBoardSize * Board::kQuadrantSize) == 0)
			out << "=============\n";
		if (i % Board::kQuadrantSize == 0)
			out << '|';
		out << static_cast<int>(data[i]);
	}
	out << "|\n";
	return out;
}
}
