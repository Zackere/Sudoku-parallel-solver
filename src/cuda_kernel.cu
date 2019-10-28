#include "../include/cuda_kernel.cuh"

#include <cassert>
#include <cstdio>
#include <ctime>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

namespace sudoku {
namespace kernel {
namespace {
constexpr unsigned kBlocks = 999;
constexpr unsigned kNumBanks = 16;
constexpr unsigned kLogNumBanks = 4;

__device__ int Tid() {
  return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
         threadIdx.x;
}

__device__ constexpr int ConflictFreeIndex(int index) {
  return index;
}

__device__ constexpr int AdjustSize(int size) {
  return size;
}

struct PossibleValues {
  Board::FieldValue values[Board::kBoardSize] = {0};
  int len = 0;
  __device__ PossibleValues &operator=(PossibleValues const &other) {
    if (this != &other) {
      for (int i = 0; i < other.len; ++i)
        values[i] = other.values[i];
      len = other.len;
    }
    return *this;
  }
};

__device__ PossibleValues GetPossibleValues(Board::FieldValue board[AdjustSize(
    Board::kBoardSize)][AdjustSize(Board::kBoardSize)]) {
  bool free[Board::kBoardSize];
  memset(free, 1, sizeof free);
  for (auto i = 0u; i < Board::kBoardSize; ++i) {
    auto val = board[ConflictFreeIndex(i)][ConflictFreeIndex(threadIdx.y)];
    if (val > 0)
      free[val - 1] = false;
    val = board[ConflictFreeIndex(threadIdx.x)][ConflictFreeIndex(i)];
    if (val > 0)
      free[val - 1] = false;
  }
  auto pom_x = threadIdx.x - threadIdx.x % Board::kQuadrantSize;
  auto pom_y = threadIdx.y - threadIdx.y % Board::kQuadrantSize;
  for (int i = 0; i < Board::kQuadrantSize; ++i)
    for (int j = 0; j < Board::kQuadrantSize; ++j) {
      auto val = board
          [ConflictFreeIndex(pom_x + (threadIdx.x + i) % Board::kQuadrantSize)]
          [ConflictFreeIndex(pom_y + (threadIdx.y + j) % Board::kQuadrantSize)];
      if (val > 0)
        free[val - 1] = false;
    }
  PossibleValues ret;
  for (int i = 0; i < Board::kBoardSize; ++i)
    if (free[i])
      ret.values[ret.len++] = i + 1;
  return ret;
}

__device__ int2 RowPattern(int i) { return {i, threadIdx.x}; }

__device__ int2 ColPattern(int i) { return {threadIdx.x, i}; }

__device__ int2 QuaPattern(int i) {}

__device__ bool
Validate(Board::FieldValue board[AdjustSize(Board::kBoardSize)]
                                [AdjustSize(Board::kBoardSize)]) {
  auto pattern =
      threadIdx.y == 0 ? RowPattern : (threadIdx.y == 1 ? ColPattern : nullptr);
  if (!pattern)
    return true;
  bool used[Board::kBoardSize] = {0};
  for (int i = 0; i < Board::kBoardSize; ++i) {
    auto pos = pattern(i);
    used[board[ConflictFreeIndex(pos.x)][ConflictFreeIndex(pos.y)] - 1] = true;
  }
  for (int i = 0; i < Board::kBoardSize; ++i)
    if (!used[i])
      return false;
  return true;
}

__global__ void Kernel(Board::FieldValue *board, Board::FieldValue *solved,
                       curandState_t *states, int *done) {
  __shared__ Board::FieldValue s_board[AdjustSize(Board::kBoardSize)]
                                      [AdjustSize(Board::kBoardSize)];
  __shared__ bool s_done;

  s_done = true;
  const auto tid = Tid();
  const auto x = ConflictFreeIndex(threadIdx.x);
  const auto y = ConflictFreeIndex(threadIdx.y);
  bool active = 0 == (s_board[x][y] = board[threadIdx.x + threadIdx.y * Board::kBoardSize]);
  PossibleValues pv;
  do {
    pv = GetPossibleValues(s_board);
    __syncthreads();
    s_done = true;
    if (active && pv.len == 1) {
      s_board[x][y] = pv.values[0];
      active = false;
      s_done = false;
    }
    __syncthreads();
  } while (!s_done);
  __syncthreads();
  s_done = false;
  while (true) {
    __syncthreads();
    if (*done) {
      solved[tid] = s_done ? s_board[x][y] : 0;
      return;
    }
    if (active)
      s_board[x][y] = pv.values[curand(&states[tid]) % pv.len];
    s_done = true;
    __syncthreads();
    if (!Validate(s_board))
      s_done = false;
    __syncthreads();
    if (threadIdx.x + threadIdx.y == 0 && s_done)
      atomicAdd(done, 1);
  }
}
__global__ void InitCurand(unsigned seed, curandState_t *states) {
  curand_init(seed, Tid(), 0, &states[Tid()]);
}
} // namespace
std::vector<Board::FieldValue>
Run(std::vector<Board::FieldValue> const &board) {
  curandState_t *d_states;
  cudaMalloc(reinterpret_cast<void **>(&d_states), kBlocks * Board::kBoardSize *
                                                       Board::kBoardSize *
                                                       sizeof(curandState_t));
  dim3 block(Board::kBoardSize, Board::kBoardSize);
  InitCurand<<<kBlocks, block>>>(std::time(nullptr), d_states);

  Board::FieldValue *d_solved;
  cudaMalloc(reinterpret_cast<void **>(&d_solved),
             kBlocks * Board::kBoardSize * Board::kBoardSize *
                 sizeof(Board::FieldValue));
  cudaMemset(d_solved, 0,
             kBlocks * Board::kBoardSize * Board::kBoardSize *
                 sizeof(Board::FieldValue));

  Board::FieldValue *d_board = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_board),
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue));
  cudaMemcpy(d_board, board.data(),
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue),
             cudaMemcpyHostToDevice);
  int *d_done = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_done), sizeof(unsigned));
  cudaMemset(d_done, 0, sizeof(unsigned));
  cudaDeviceSynchronize();

  auto start = clock();
  Kernel<<<kBlocks, block>>>(d_board, d_solved, d_states, d_done);
  cudaDeviceSynchronize();
  auto end = clock();
  printf("%ld ms\n", end - start);

  std::vector<Board::FieldValue> solved(
      kBlocks * Board::kBoardSize * Board::kBoardSize, Board::FieldValue{0});
  cudaMemcpy(solved.data(), d_solved,
             kBlocks * Board::kBoardSize * Board::kBoardSize,
             cudaMemcpyDeviceToHost);
  std::vector<Board::FieldValue> ret;
  for (auto it = solved.begin(); it != solved.end();) {
    if (*it == 0) {
      it += Board::kBoardSize * Board::kBoardSize;
    } else {
      ret = std::vector<Board::FieldValue>{it, it + Board::kBoardSize *
                                                        Board::kBoardSize};
      break;
    }
  }
  cudaFree(d_states);
  cudaFree(d_board);
  cudaFree(d_solved);
  return ret;
}
} // namespace kernel
} // namespace sudoku
