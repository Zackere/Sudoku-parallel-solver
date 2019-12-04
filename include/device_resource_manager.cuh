#pragma once
#include "./board.hpp"

namespace deviceResourceManager {
constexpr unsigned kNBoards = 1 << 22;
sudoku::Board::FieldValue *GetOldBoards();
int *GetOldBoardsCount();
sudoku::Board::FieldValue *GetNewBoards();
int *GetNewBoardsCount();
sudoku::Board::FieldValue *GetSolvedBoard();
int *GetSolvedBoardMutex();
uint8_t *GetEmptyFields();
uint8_t *GetEmptyFieldsCount();
void Release();
} // namespace deviceResourceManager
