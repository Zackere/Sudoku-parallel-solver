#include "../include/cuda_kernel.cuh"

#include <cassert>
#include <cstdio>
#include <ctime>
#include <memory>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

namespace sudoku {
namespace kernel {
namespace {
constexpr unsigned kBlocks = 30000;
constexpr unsigned kNumBanks = 16;
constexpr unsigned kLogNumBanks = 4;
constexpr unsigned kBufSize = kBlocks + 1;

#define SetNthBit(number, n) ((number) |= (1ul << (n)))
#define ClearNthBit(number, n) ((number) &= ~(1ul << (n)))
#define GetNthBit(number, n) (((number) >> (n)) & 1u)

__device__ uint16_t
GetPossibleValues(Board::FieldValue board[Board::kBoardSize][Board::kBoardSize],
                  int *nvalues) {
  uint16_t free = 0xffff;
  *nvalues = Board::kBoardSize;
#pragma unroll
  for (auto i = 0; i < Board::kBoardSize; ++i) {
    auto val = board[threadIdx.y][i];
    if (val > 0 && GetNthBit(free, --val)) {
      ClearNthBit(free, val);
      --*nvalues;
    }
    val = board[i][threadIdx.x];
    if (val > 0 && GetNthBit(free, --val)) {
      ClearNthBit(free, val);
      --*nvalues;
    }
  }
  auto pom_x = threadIdx.x - threadIdx.x % Board::kQuadrantSize;
  auto pom_y = threadIdx.y - threadIdx.y % Board::kQuadrantSize;
#pragma unroll
  for (int i = 0; i < Board::kQuadrantSize; ++i)
#pragma unroll
    for (int j = 0; j < Board::kQuadrantSize; ++j) {
      auto val = board[pom_y + (threadIdx.y + j) % Board::kQuadrantSize]
                      [pom_x + (threadIdx.x + i) % Board::kQuadrantSize];
      if (val > 0 && GetNthBit(free, --val)) {
        ClearNthBit(free, val);
        --*nvalues;
      }
    }
  return free;
}

__device__ int FindFreeBlock(int start, int *block_state) {
  while (++start < kBufSize - 1) {
    atomicCAS(&block_state[start], 0, blockIdx.x + 2);
    if (block_state[start] == blockIdx.x + 2)
      return start;
  }
  return -1;
}

__device__ constexpr int ConflictFreeIndex(int index) {
  return index + (index >> kLogNumBanks);
}

__device__ constexpr int AdjustSize(int size) {
  return size + size / kNumBanks;
}

/*
  __device__ bool Validate(Board::FieldValue board[kNumBanks][kNumBanks]) {
  if (threadIdx.x == 0) {
    bool used[Board::kBoardSize] = {0};
    for (int i = 0; i < Board::kBoardSize; ++i)
      used[board[threadIdx.y][i] - 1] = true;
    for (int i = 0; i < Board::kBoardSize; ++i)
      if (!used[i])
        return false;
  }
  __shared__ Board::FieldValue transpose[AdjustSize(kNumBanks * kNumBanks)];
  transpose[ConflictFreeIndex(threadIdx.x * Board::kBoardSize + threadIdx.y)] =
      board[threadIdx.y][threadIdx.x];
  if (threadIdx.x == 0) {
    bool used[Board::kBoardSize] = {0};
    for (int i = 0; i < Board::kBoardSize; ++i)
      used[transpose[ConflictFreeIndex(i * Board::kBoardSize + threadIdx.y)] -
           1] = true;
    for (int i = 0; i < Board::kBoardSize; ++i)
      if (!used[i])
        return false;
  }
  return true;
}
*/

__global__ void Kernel(int *block_state, Board::FieldValue *buffer) {
  __shared__ Board::FieldValue s_board[Board::kBoardSize][Board::kBoardSize];
  __shared__ int scheduling_thread_id;
  __shared__ int min_elems;
  __shared__ int available_block;

  if (block_state[blockIdx.x] != 1 || block_state[kBufSize - 1])
    return;
  const bool active =
      0 == (s_board[threadIdx.y][threadIdx.x] =
                buffer[Board::kBoardSize * Board::kBoardSize * blockIdx.x +
                       Board::kBoardSize * threadIdx.y + threadIdx.x]);

  // Try to simplify the board
  int nelems = 0;
  auto pv = active ? GetPossibleValues(s_board, &nelems) : 0;
  while (__syncthreads_or(active && nelems == 1)) {
    if (active && nelems == 1) {
      for (int i = 0; i < Board::kBoardSize; ++i)
        if (GetNthBit(pv, i)) {
          s_board[threadIdx.y][threadIdx.x] = i + 1;
          break;
        }
      *const_cast<bool *>(&active) = false;
    }
    __syncthreads();
    if (active)
      pv = GetPossibleValues(s_board, &nelems);
  }
  // Check if the board has been solved
  if (__syncthreads_and(!active)) {
    atomicCAS(&block_state[kBufSize - 1], 0, blockIdx.x + 2);
    if (block_state[kBufSize - 1] == blockIdx.x + 2)
      buffer[Board::kBoardSize * Board::kBoardSize * (kBufSize - 1) +
             Board::kBoardSize * threadIdx.y + threadIdx.x] =
          s_board[threadIdx.y][threadIdx.x];
    return;
  }
  // Check if the board is contradictory
  if (__syncthreads_or(active && !nelems)) {
    block_state[blockIdx.x] = 0;
    return;
  }
  // Find scheduler thread
  scheduling_thread_id = -1;
  min_elems = Board::kBoardSize;
  if (active)
    atomicMin(&min_elems, nelems);
  __syncthreads();
  if (active && nelems == min_elems)
    atomicMax(&scheduling_thread_id,
              Board::kBoardSize * threadIdx.y + threadIdx.x);
  __syncthreads();
  const bool is_scheduler =
      ((Board::kBoardSize * threadIdx.y + threadIdx.x) == scheduling_thread_id);

  // Schedule other blocks
  available_block = -1;
#pragma unroll
  for (int i = 0; i < Board::kBoardSize; ++i) {
    if (__syncthreads_and(!is_scheduler || GetNthBit(pv, i))) {
      if (is_scheduler)
        available_block = FindFreeBlock(available_block, block_state);
      if (__syncthreads_and(!is_scheduler || available_block != -1)) {
        buffer[Board::kBoardSize * Board::kBoardSize * available_block +
               Board::kBoardSize * threadIdx.y + threadIdx.x] =
            is_scheduler ? i + 1 : s_board[threadIdx.y][threadIdx.x];
        block_state[available_block] = 1;
      }
    }
  }
  block_state[blockIdx.x] = 0;
}
} // namespace

std::vector<Board::FieldValue>
Run(std::vector<Board::FieldValue> const &board) {
  Board::FieldValue *d_buffer = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_buffer),
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue) *
                 kBufSize);
  cudaMemset(d_buffer, 0,
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue) *
                 kBufSize);
  cudaMemcpy(d_buffer, board.data(),
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue),
             cudaMemcpyHostToDevice);
  int *d_block_state = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_block_state), kBufSize * sizeof(int));
  cudaMemset(d_block_state, 0, kBufSize * sizeof(int));
  cudaMemset(d_block_state, 1, 1);
  dim3 block(Board::kBoardSize, Board::kBoardSize);
  for (int i = 0; i < 81; ++i) {
    Kernel<<<kBlocks, block>>>(d_block_state, d_buffer);
    cudaDeviceSynchronize();
  }
  std::unique_ptr<Board::FieldValue[]> result_ptr(
      new Board::FieldValue[Board::kBoardSize * Board::kBoardSize]);
  cudaMemcpy(result_ptr.get(),
             d_buffer + (kBufSize - 1) * Board::kBoardSize * Board::kBoardSize,
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue),
             cudaMemcpyDeviceToHost);
  std::vector<Board::FieldValue> ret{result_ptr.get(),
                                     result_ptr.get() +
                                         Board::kBoardSize * Board::kBoardSize};
  cudaFree(d_block_state);
  cudaFree(d_buffer);
  return ret;
}
} // namespace kernel
} // namespace sudoku
