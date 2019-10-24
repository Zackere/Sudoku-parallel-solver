#include <vector>
#include <string>

namespace sudoku {
class SudokuProvider {
public:
	using Board=std::vector<unsigned char>;
	enum class Result {
		VALID, NO_DATA, INVALID_SUDOKU,
	};

	SudokuProvider();
	void Read(std::string_view path);
	Result IsValid();
	Board const& Get();
private:
	Board board_;
};
}
