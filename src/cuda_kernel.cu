#include "../include/cuda_kernel.cuh"

#include <cassert>
#include <cstdio>
#include <ctime>
#include <memory>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

namespace sudoku {
namespace kernel {
namespace {
constexpr unsigned kBlocks = 1024;
constexpr unsigned kThreadsPerBlock = 256;
constexpr unsigned kNBoards = 1 << 22;
constexpr unsigned kIterations = 9;

#define SetNthBit(number, n) ((number) |= (1ul << (n)))
#define ClearNthBit(number, n) ((number) &= ~(1ul << (n)))
#define GetNthBit(number, n) (((number) >> (n)) & 1u)

__device__ uint16_t GetPossibleValues(Board::FieldValue *board, int cell) {
  int row = cell / Board::kBoardSize;
  int col = cell - row * Board::kBoardSize;
  uint16_t free = 0x01ff;

  for (int i = 0; i < Board::kBoardSize; ++i) {
    auto val = board[row * Board::kBoardSize + i] - 1;
    if (val > -1)
      ClearNthBit(free, val);
    val = board[i * Board::kBoardSize + col] - 1;
    if (val > -1)
      ClearNthBit(free, val);
  }
  auto pom_y = row - row % Board::kQuadrantSize;
  auto pom_x = col - col % Board::kQuadrantSize;
  for (int i = 0; i < Board::kQuadrantSize; ++i)
    for (int j = 0; j < Board::kQuadrantSize; ++j) {
      auto val = board[(pom_y + i) * Board::kBoardSize + pom_x + j] - 1;
      if (val > -1)
        ClearNthBit(free, val);
    }
  return free;
}

__global__ void Generator(Board::FieldValue *old_boards, int *old_boards_count,
                          Board::FieldValue *new_boards, int *new_boards_count,
                          unsigned char *empty_fields,
                          unsigned char *empty_fields_count,
                          Board::FieldValue *solved_board,
                          int *solved_board_mutex) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < *old_boards_count; index += blockDim.x * gridDim.x) {
    if (*solved_board_mutex)
      return;
    for (int i = index * Board::kBoardSize * Board::kBoardSize;
         i < (index + 1) * Board::kBoardSize * Board::kBoardSize; ++i) {
      if (!old_boards[i]) {
        auto pv = GetPossibleValues(
            old_boards + index * Board::kBoardSize * Board::kBoardSize,
            i - index * Board::kBoardSize * Board::kBoardSize);
        for (int j = 0; j < Board::kBoardSize; ++j) {
          if (GetNthBit(pv, j)) {
            auto pos = atomicAdd(new_boards_count, 1);
            if (pos < kNBoards) {
              old_boards[i] = j + 1;
              unsigned char empty_index = static_cast<unsigned char>(-1);
              for (int k = 0; k < Board::kBoardSize * Board::kBoardSize; ++k) {
                if (!(new_boards[pos * Board::kBoardSize * Board::kBoardSize +
                                 k] = old_boards
                          [index * Board::kBoardSize * Board::kBoardSize + k]))
                  empty_fields[++empty_index +
                               pos * Board::kBoardSize * Board::kBoardSize] =
                      pos * Board::kBoardSize * Board::kBoardSize + k;
              }
              empty_fields_count[pos] = empty_index + 1;
            }
          }
        }
        goto NOT_SOLVED;
      }
    }
    atomicCAS(solved_board_mutex, 0, blockIdx.x * blockDim.x + threadIdx.x);
    if (*solved_board_mutex != blockIdx.x * blockDim.x + threadIdx.x)
      return;
    for (int i = 0; i < Board::kBoardSize * Board::kBoardSize; ++i)
      solved_board[i] =
          old_boards[index * Board::kBoardSize * Board::kBoardSize + i];
  NOT_SOLVED:;
  }
}

__device__ bool NotInRow(Board::FieldValue *board, int row) {
  uint16_t st = 0;

  for (int i = 0; i < Board::kBoardSize; i++)
    if (board[Board::kBoardSize * row + i]) {
      if (GetNthBit(st, board[Board::kBoardSize * row + i]))
        return false;
      SetNthBit(st, board[Board::kBoardSize * row + i]);
    }
  return true;
}

__device__ bool NotInCol(Board::FieldValue *board, int col) {
  uint16_t st = 0;
  for (int i = 0; i < Board::kBoardSize; i++)
    if (board[Board::kBoardSize * i + col]) {
      if (GetNthBit(st, board[Board::kBoardSize * i + col]))
        return false;
      SetNthBit(st, board[Board::kBoardSize * i + col]);
    }
  return true;
}

__device__ bool NotInBox(Board::FieldValue *board, int startRow, int startCol) {
  uint16_t st = 0;
  for (int row = 0; row < Board::kQuadrantSize; ++row)
    for (int col = 0; col < Board::kQuadrantSize; ++col)
      if (board[Board::kBoardSize * row + col]) {
        if (GetNthBit(st, board[Board::kBoardSize * row + col]))
          return false;
        SetNthBit(st, board[Board::kBoardSize * row + col]);
      }
  return true;
}

__device__ bool IsValid(Board::FieldValue *board, int row, int col) {
  return NotInRow(board, row) && NotInCol(board, col) &&
         NotInBox(board, row - row % 3, col - col % 3);
}

__device__ bool IsValidConfig(Board::FieldValue *board) {
  for (int i = 0; i < Board::kBoardSize; ++i)
    for (int j = 0; j < Board::kBoardSize; ++j)
      if (!IsValid(board, i, j))
        return false;
  return true;
}

__device__ bool Solve(Board::FieldValue *board, unsigned char *empty_fields,
                      unsigned char empty_fields_count) {
  unsigned char empty_index = 0;
  while (empty_index < empty_fields_count) {
    ++board[empty_fields[empty_index]];
    if (!IsValidConfig(board)) {
      if (board[empty_fields[empty_index]] >= 9) {
        board[empty_fields[empty_index]] = 0;
        --empty_index;
      }
    } else {
      ++empty_index;
    }
  }
  return empty_fields_count == empty_index;
}

__global__ void Backtracker(Board::FieldValue *old_boards,
                            int *old_boards_count, unsigned char *empty_fields,
                            unsigned char *empty_fields_count,
                            Board::FieldValue *solved_board,
                            int *solved_board_mutex) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < *old_boards_count; index += blockDim.x * gridDim.x) {
    if (*solved_board_mutex)
      return;
    if (Solve(old_boards + index * Board::kBoardSize * Board::kBoardSize,
              empty_fields + index * Board::kBoardSize * Board::kBoardSize,
              empty_fields_count[index])) {
      atomicCAS(solved_board_mutex, 0, blockIdx.x * blockDim.x + threadIdx.x);
      if (*solved_board_mutex != blockIdx.x * blockDim.x + threadIdx.x)
        return;
      for (int i = 0; i < Board::kBoardSize * Board::kBoardSize; ++i)
        solved_board[i] =
            old_boards[index * Board::kBoardSize * Board::kBoardSize + i];
    }
  }
}
} // namespace

std::vector<Board::FieldValue>
Run(std::vector<Board::FieldValue> const &board) {
  Board::FieldValue *d_old_boards = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_old_boards),
             kNBoards * Board::kBoardSize * Board::kBoardSize *
                 sizeof(Board::FieldValue));
  int *d_old_boards_count;
  cudaMalloc(reinterpret_cast<void **>(&d_old_boards_count), sizeof(int));
  Board::FieldValue *d_new_boards = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_new_boards),
             kNBoards * Board::kBoardSize * Board::kBoardSize *
                 sizeof(Board::FieldValue));
  int *d_new_boards_count = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_new_boards_count), sizeof(int));

  cudaMemset(d_old_boards, 0,
             kNBoards * Board::kBoardSize * Board::kBoardSize *
                 sizeof(Board::FieldValue));
  cudaMemset(d_new_boards, 0,
             kNBoards * Board::kBoardSize * Board::kBoardSize *
                 sizeof(Board::FieldValue));
  std::unique_ptr<int> one(new int(1));
  cudaMemcpy(d_old_boards_count, one.get(), sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_old_boards, board.data(),
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue),
             cudaMemcpyHostToDevice);

  Board::FieldValue *d_solved_board = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_solved_board),
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue));
  cudaMemset(d_solved_board, 0,
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue));

  int *d_solved_board_mutex = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_solved_board_mutex), sizeof(int));
  cudaMemset(d_solved_board_mutex, 0, sizeof(int));

  unsigned char *d_empty_fields = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_empty_fields),
             kNBoards * Board::kBoardSize * Board::kBoardSize *
                 sizeof(unsigned char));
  cudaMemset(d_empty_fields, 0,
             kNBoards * Board::kBoardSize * Board::kBoardSize *
                 sizeof(unsigned char));

  unsigned char *d_empty_fields_count = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_empty_fields_count),
             kNBoards * sizeof(unsigned char));
  cudaMemset(d_empty_fields_count, 0, kNBoards * sizeof(unsigned char));

  for (int i = 0; i < kIterations; ++i) {
    cudaMemset(d_new_boards_count, 0, sizeof(int));
    Generator<<<kBlocks, kThreadsPerBlock>>>(
        d_old_boards, d_old_boards_count, d_new_boards, d_new_boards_count,
        d_empty_fields, d_empty_fields_count, d_solved_board,
        d_solved_board_mutex);
    cudaDeviceSynchronize();
    cudaMemset(d_old_boards_count, 0, sizeof(int));
    Generator<<<kBlocks, kThreadsPerBlock>>>(
        d_new_boards, d_new_boards_count, d_old_boards, d_old_boards_count,
        d_empty_fields, d_empty_fields_count, d_solved_board,
        d_solved_board_mutex);
    cudaDeviceSynchronize();
  }

  int solved = 0;
  cudaMemcpy(&solved, d_solved_board_mutex, sizeof(int),
             cudaMemcpyDeviceToHost);
  if (!solved)
    Backtracker<<<kBlocks, kThreadsPerBlock>>>(
        d_old_boards, d_old_boards_count, d_empty_fields, d_empty_fields_count,
        d_solved_board, d_solved_board_mutex);
  std::unique_ptr<Board::FieldValue[]> ret(
      new Board::FieldValue[Board::kBoardSize * Board::kBoardSize]);
  cudaMemcpy(ret.get(), d_solved_board,
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue),
             cudaMemcpyDeviceToHost);

  cudaFree(d_empty_fields_count);
  cudaFree(d_empty_fields);
  cudaFree(d_solved_board_mutex);
  cudaFree(d_solved_board);
  cudaFree(d_new_boards_count);
  cudaFree(d_new_boards);
  cudaFree(d_old_boards_count);
  cudaFree(d_old_boards);
  return {ret.get(), ret.get() + Board::kBoardSize * Board::kBoardSize};
}
} // namespace kernel
} // namespace sudoku
