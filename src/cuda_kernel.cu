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
constexpr unsigned kBlocks = 500;
constexpr unsigned kThreadsPerBlock = 256;
constexpr unsigned kNumBanks = 16;
constexpr unsigned kLogNumBanks = 4;
constexpr unsigned kNBoards = 1 << 23;
constexpr unsigned kIterations = 32;

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
                          Board::FieldValue *new_boards,
                          int *new_boards_count) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < *old_boards_count; index += blockDim.x * gridDim.x) {
    for (int i = index * Board::kBoardSize * Board::kBoardSize;
         i < (index + 1) * Board::kBoardSize * Board::kBoardSize; ++i) {
      if (old_boards[i] == 0) {
        auto pv = GetPossibleValues(
            old_boards + index * Board::kBoardSize * Board::kBoardSize,
            i - index * Board::kBoardSize * Board::kBoardSize);

        for (int j = 0; j < Board::kBoardSize; ++j) {
          if (GetNthBit(pv, j)) {
            auto pos = atomicAdd(new_boards_count, 1);
            if (pos < kNBoards) {
              old_boards[i] = j + 1;
              for (int k = 0; k < Board::kBoardSize * Board::kBoardSize; ++k)
                new_boards[pos * Board::kBoardSize * Board::kBoardSize + k] =
                    old_boards[index * Board::kBoardSize * Board::kBoardSize +
                               k];
            } else {
              printf("%d could not schedule\n", threadIdx.x);
            }
          }
        }
        return;
      }
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

  for (int i = 0; i < kIterations; ++i) {
    cudaMemset(d_new_boards_count, 0, sizeof(int));
    Generator<<<kBlocks, kThreadsPerBlock>>>(d_old_boards, d_old_boards_count,
                                             d_new_boards, d_new_boards_count);
    cudaDeviceSynchronize();
    cudaMemset(d_old_boards_count, 0, sizeof(int));
    Generator<<<kBlocks, kThreadsPerBlock>>>(d_new_boards, d_new_boards_count,
                                             d_old_boards, d_old_boards_count);
    cudaDeviceSynchronize();
  }

  std::unique_ptr<Board::FieldValue[]> ret(
      new Board::FieldValue[Board::kBoardSize * Board::kBoardSize]);
  cudaMemcpy(ret.get(), d_old_boards,
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue),
             cudaMemcpyDeviceToHost);

  cudaFree(d_new_boards_count);
  cudaFree(d_new_boards);
  cudaFree(d_old_boards_count);
  cudaFree(d_old_boards);
  return {ret.get(), ret.get() + Board::kBoardSize * Board::kBoardSize};
}
} // namespace kernel
} // namespace sudoku
