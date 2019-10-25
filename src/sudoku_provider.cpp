#include "../include/sudoku_provider.hpp"

#include <fstream>
#include <iostream>

namespace sudoku {
namespace {
constexpr unsigned kSudokuSize = 81;
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
		c -= '0';
		if (board_.size() == kSudokuSize) {
			std::cerr << "File " << path.data()
					<< " seems to be in incorrect format.\n";
			board_.clear();
			break;
		}
		board_.emplace_back(c);
	}
	file.close();
}

SudokuProvider::Board const& SudokuProvider::Get() {
	return board_;
}
}
