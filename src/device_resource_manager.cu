#include "../include/device_resource_manager.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace deviceResourceManager {
namespace {
sudoku::Board::FieldValue *d_old_boards = nullptr;
int *d_old_boards_count = nullptr;
sudoku::Board::FieldValue *d_new_boards = nullptr;
int *d_new_boards_count = nullptr;
sudoku::Board::FieldValue *d_solved_board = nullptr;
int *d_solved_board_mutex = nullptr;
uint8_t *d_empty_fields = nullptr;
uint8_t *d_empty_fields_count = nullptr;

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
} // namespace
sudoku::Board::FieldValue *GetOldBoards() {
  if (!d_old_boards)
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_old_boards),
                         kNBoards * sudoku::Board::kBoardSize *
                             sudoku::Board::kBoardSize *
                             sizeof(sudoku::Board::FieldValue)));

  return d_old_boards;
}

int *GetOldBoardsCount() {
  if (!d_old_boards_count)
    cudaMalloc(reinterpret_cast<void **>(&d_old_boards_count), sizeof(int));
  return d_old_boards_count;
}

sudoku::Board::FieldValue *GetNewBoards() {
  if (!d_new_boards)
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_new_boards),
                         kNBoards * sudoku::Board::kBoardSize *
                             sudoku::Board::kBoardSize *
                             sizeof(sudoku::Board::FieldValue)));
  return d_new_boards;
}

int *GetNewBoardsCount() {
  if (!d_new_boards_count)
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_new_boards_count),
                         sizeof(int)));
  return d_new_boards_count;
}

sudoku::Board::FieldValue *GetSolvedBoard() {
  if (!d_solved_board)
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_solved_board),
                         sudoku::Board::kBoardSize * sudoku::Board::kBoardSize *
                             sizeof(sudoku::Board::FieldValue)));
  return d_solved_board;
}

int *GetSolvedBoardMutex() {
  if (!d_solved_board_mutex)
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_solved_board_mutex),
                         sizeof(int)));
  return d_solved_board_mutex;
}

uint8_t *GetEmptyFields() {
  if (!d_empty_fields)
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_empty_fields),
                         kNBoards * sudoku::Board::kBoardSize *
                             sudoku::Board::kBoardSize * sizeof(uint8_t)));
  return d_empty_fields;
}

uint8_t *GetEmptyFieldsCount() {
  if (!d_empty_fields_count)
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_empty_fields_count),
                         kNBoards * sizeof(uint8_t)));
  return d_empty_fields_count;
}

void Release() {
  cudaFree(d_empty_fields_count);
  cudaFree(d_empty_fields);
  cudaFree(d_solved_board_mutex);
  cudaFree(d_solved_board);
  cudaFree(d_new_boards_count);
  cudaFree(d_new_boards);
  cudaFree(d_old_boards_count);
  cudaFree(d_old_boards);
}
} // namespace deviceResourceManager
