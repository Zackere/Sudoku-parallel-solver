#pragma once

#include <vector>
#include <string>

namespace sudoku {
class SudokuProvider {
public:
	using Board=std::vector<unsigned char>;

	SudokuProvider();
	void Read(std::string_view path);
	Board const& Get();
private:
	Board board_;
};
}
