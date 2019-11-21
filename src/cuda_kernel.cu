#include "../include/cuda_kernel.cuh"

#include "../include/device_resource_manager.cuh"
#include "../include/time_manager.hpp"

#include <cassert>
#include <cstdio>
#include <ctime>
#include <memory>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace sudoku {
namespace kernel {
namespace {
constexpr unsigned kBlocks = 512;
constexpr unsigned kThreadsPerBlock = 128;
constexpr unsigned kIterations = 13;

#define SetNthBit(number, n) ((number) |= (1ul << (n)))
#define ClearNthBit(number, n) ((number) &= ~(1ul << (n)))
#define GetNthBit(number, n) (((number) >> (n)) & 1u)

__device__ bool NotInRow(Board::FieldValue *board, int row) {
  uint16_t st = 0;
  bool ret = true;
  Board::FieldValue v;
  for (int i = 0; i < Board::kBoardSize; i++) {
    v = board[Board::kBoardSize * row + i];
    ret = v <= Board::kBoardSize && ret;
    ret = !(v && GetNthBit(st, v)) && ret;
    SetNthBit(st, v);
  }
  return ret;
}

__device__ bool NotInCol(Board::FieldValue *board, int col) {
  uint16_t st = 0;
  bool ret = true;
  Board::FieldValue v;
  for (int i = 0; i < Board::kBoardSize; i++) {
    v = board[Board::kBoardSize * i + col];
    ret = v <= Board::kBoardSize && ret;
    ret = !(v && GetNthBit(st, v)) && ret;
    SetNthBit(st, v);
  }
  return ret;
}

__device__ bool NotInBox(Board::FieldValue *board, int row, int col) {
  row -= row % Board::kQuadrantSize;
  col -= col % Board::kQuadrantSize;
  uint16_t st = 0;
  bool ret = true;
  auto pom_y = row - row % Board::kQuadrantSize;
  auto pom_x = col - col % Board::kQuadrantSize;
  Board::FieldValue v;
  for (int i = 0; i < Board::kQuadrantSize; ++i)
    for (int j = 0; j < Board::kQuadrantSize; ++j) {
      v = board[(pom_y + i) * Board::kBoardSize + pom_x + j];
      ret = v <= Board::kBoardSize && ret;
      ret = !(v && GetNthBit(st, v)) && ret;
      SetNthBit(st, v);
    }
  return ret;
}

__device__ bool IsValid(Board::FieldValue *board, int row, int col) {
  return NotInRow(board, row) && NotInCol(board, col) &&
         NotInBox(board, row, col);
}

__global__ void Generator(Board::FieldValue *old_boards, int *old_boards_count,
                          Board::FieldValue *new_boards, int *new_boards_count,
                          unsigned char *empty_fields,
                          unsigned char *empty_fields_count,
                          Board::FieldValue *solved_board,
                          int *solved_board_mutex) {
  __shared__ Board::FieldValue s_current_boards[kThreadsPerBlock *
                                                Board::kBoardSize *
                                                Board::kBoardSize];
  auto *my_board =
      s_current_boards + threadIdx.x * Board::kBoardSize * Board::kBoardSize;
  for (int index = blockIdx.x * blockDim.x; index < *old_boards_count;
       index += blockDim.x * gridDim.x) {
    __syncthreads();
    if (*solved_board_mutex)
      return;
    for (int i = 0; i < Board::kBoardSize * Board::kBoardSize; ++i) {
      auto j = i * kThreadsPerBlock +
               index * Board::kBoardSize * Board::kBoardSize + threadIdx.x;
      s_current_boards[i * kThreadsPerBlock + threadIdx.x] =
          j < *old_boards_count * Board::kBoardSize * Board::kBoardSize
              ? old_boards[j]
              : 1;
    }
    __syncthreads();
    for (int i = 0; i < Board::kBoardSize * Board::kBoardSize; ++i) {
      if (!my_board[i]) {
        auto row = i / Board::kBoardSize;
        auto col = i % Board::kBoardSize;
        for (int j = 1; j <= Board::kBoardSize; ++j) {
          my_board[i] = j;
          if (IsValid(my_board, row, col)) {
            auto pos = atomicAdd(new_boards_count, 1);
            if (pos < deviceResourceManager::kNBoards) {
              unsigned char empty_index = static_cast<unsigned char>(-1);
              for (int k = 0; k < Board::kBoardSize * Board::kBoardSize; ++k) {
                if (!(new_boards[pos * Board::kBoardSize * Board::kBoardSize +
                                 k] = my_board[k]))
                  empty_fields[++empty_index +
                               pos * Board::kBoardSize * Board::kBoardSize] = k;
              }
              empty_fields_count[pos] = empty_index + 1;
            } else {
              atomicMin(new_boards_count, deviceResourceManager::kNBoards);
              return;
            }
          }
        }
        goto NOT_SOLVED;
      }
    }
    if (threadIdx.x + index < *old_boards_count) {
      atomicCAS(solved_board_mutex, 0, blockIdx.x * blockDim.x + threadIdx.x);
      if (*solved_board_mutex == blockIdx.x * blockDim.x + threadIdx.x)
        for (int i = 0; i < Board::kBoardSize * Board::kBoardSize; ++i)
          solved_board[i] = my_board[i];
    }
  NOT_SOLVED:;
  }
}

__device__ bool Solve(Board::FieldValue *board, uint8_t *empty_fields,
                      uint8_t empty_fields_count) {
  unsigned char empty_index = 0;
  auto field = empty_fields[empty_index];
  auto row = field / Board::kBoardSize;
  auto col = field % Board::kBoardSize;
  while (empty_index < empty_fields_count) {
    ++board[field];
    if (IsValid(board, row, col)) {
      field = empty_fields[++empty_index];
      row = field / Board::kBoardSize;
      col = field % Board::kBoardSize;
    } else {
      if (board[field] >= Board::kBoardSize) {
        board[field] = 0;
        field = empty_fields[--empty_index];
        row = field / Board::kBoardSize;
        col = field % Board::kBoardSize;
      }
    }
  }
  return empty_index == empty_fields_count;
}

__global__ void Backtracker(Board::FieldValue *old_boards,
                            int *old_boards_count, uint8_t *empty_fields,
                            uint8_t *empty_fields_count,
                            Board::FieldValue *solved_board,
                            int *solved_board_mutex) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < *old_boards_count; index += blockDim.x * gridDim.x) {
    if (*solved_board_mutex)
      return;
    auto index_mul = index * Board::kBoardSize * Board::kBoardSize;
    if (Solve(old_boards + index_mul, empty_fields + index_mul,
              empty_fields_count[index])) {
      atomicCAS(solved_board_mutex, 0, blockIdx.x * blockDim.x + threadIdx.x);
      if (*solved_board_mutex != blockIdx.x * blockDim.x + threadIdx.x)
        return;
      for (int i = 0; i < Board::kBoardSize * Board::kBoardSize; ++i)
        solved_board[i] = old_boards[index_mul + i];
    }
  }
}

__device__ uint16_t GetPossibleValues(
    Board::FieldValue board[Board::kBoardSize][Board::kBoardSize]) {
  uint16_t free = 0x03ff;
  for (int i = 0; i < Board::kBoardSize; ++i) {
    ClearNthBit(free, board[threadIdx.y][i]);
    ClearNthBit(free, board[i][threadIdx.x]);
  }
  auto pom_x = threadIdx.x - threadIdx.x % Board::kQuadrantSize;
  auto pom_y = threadIdx.y - threadIdx.y % Board::kQuadrantSize;
  for (int i = 0; i < Board::kQuadrantSize; ++i)
    for (int j = 0; j < Board::kQuadrantSize; ++j)
      ClearNthBit(free,
                  board[pom_y + (threadIdx.y + j) % Board::kQuadrantSize]
                       [pom_x + (threadIdx.x + i) % Board::kQuadrantSize]);
  return free >> 1;
}

__global__ void Simplificator(Board::FieldValue *old_boards,
                              int *old_boards_count,
                              Board::FieldValue *new_boards,
                              int *new_boards_count) {
  __shared__ Board::FieldValue s_board[Board::kBoardSize][Board::kBoardSize];
  __shared__ int pos;
  pos = 0;
  for (int index = blockIdx.x; index < *old_boards_count; index += gridDim.x) {
    bool active =
        !(s_board[threadIdx.y][threadIdx.x] =
              (old_boards +
               index * Board::kBoardSize *
                   Board::kBoardSize)[Board::kBoardSize * threadIdx.y +
                                      threadIdx.x]);
    __syncthreads();
    auto pv = GetPossibleValues(s_board);
    auto nelems = __popc(pv);
    while (__syncthreads_or(active && nelems == 1)) {
      if (active && nelems == 1) {
        s_board[threadIdx.y][threadIdx.x] = __ffs(pv);
        active = false;
      }
      __syncthreads();
      if (active) {
        pv = GetPossibleValues(s_board);
        nelems = __popc(pv);
      }
    }
    if (__syncthreads_or(active && nelems == 0))
      continue;
    if (__syncthreads_and(
            IsValid(reinterpret_cast<Board::FieldValue *>(s_board), threadIdx.y,
                    threadIdx.x))) {
      if (threadIdx.x + threadIdx.y == 0)
        pos = atomicAdd(new_boards_count, 1);
      __syncthreads();
      (new_boards +
       pos * Board::kBoardSize *
           Board::kBoardSize)[Board::kBoardSize * threadIdx.y + threadIdx.x] =
          s_board[threadIdx.y][threadIdx.x];
    }
  }
}

class ScopedCudaEvent {
public:
  ScopedCudaEvent() { cudaEventCreate(&event_); }
  ~ScopedCudaEvent() { cudaEventDestroy(event_); }
  cudaEvent_t Get() { return event_; }
  void Record() { cudaEventRecord(event_); }
  void Sync() { cudaEventSynchronize(event_); }

private:
  cudaEvent_t event_;

  ScopedCudaEvent(ScopedCudaEvent const &) = delete;
  ScopedCudaEvent &operator=(ScopedCudaEvent const &) = delete;
};

class ScopedCudaStream {
public:
  ScopedCudaStream() { cudaStreamCreate(&stream_); }
  ~ScopedCudaStream() { cudaStreamDestroy(stream_); }
  cudaStream_t Get() { return stream_; }
  cudaError_t Query() { return cudaStreamQuery(stream_); }
  void Sync() { cudaStreamSynchronize(stream_); }

private:
  cudaStream_t stream_;

  ScopedCudaStream(ScopedCudaStream const &) = delete;
  ScopedCudaStream &operator=(ScopedCudaStream const &) = delete;
};
} // namespace

std::vector<Board::FieldValue>
Run(std::vector<Board::FieldValue> const &board) {
  Board::FieldValue *d_old_boards = deviceResourceManager::GetOldBoards();
  int *d_old_boards_count = deviceResourceManager::GetOldBoardsCount();
  Board::FieldValue *d_new_boards = deviceResourceManager::GetNewBoards();
  int *d_new_boards_count = deviceResourceManager::GetNewBoardsCount();
  Board::FieldValue *d_solved_board = deviceResourceManager::GetSolvedBoard();
  int *d_solved_board_mutex = deviceResourceManager::GetSolvedBoardMutex();
  uint8_t *d_empty_fields = deviceResourceManager::GetEmptyFields();
  uint8_t *d_empty_fields_count = deviceResourceManager::GetEmptyFieldsCount();
  ScopedCudaStream kernel_stream;
  ScopedCudaStream old_boards_set_stream, new_boards_set_stream,
      empty_fields_set_stream, empty_fields_count_set_stream;

  ScopedCudaEvent start, stop;
  start.Record();

  cudaMemsetAsync(d_old_boards, 0,
                  deviceResourceManager::kNBoards * Board::kBoardSize *
                      Board::kBoardSize * sizeof(Board::FieldValue),
                  old_boards_set_stream.Get());
  cudaMemsetAsync(d_new_boards, 0,
                  deviceResourceManager::kNBoards * Board::kBoardSize *
                      Board::kBoardSize * sizeof(Board::FieldValue),
                  new_boards_set_stream.Get());
  cudaMemsetAsync(d_empty_fields, 0,
                  deviceResourceManager::kNBoards * Board::kBoardSize *
                      Board::kBoardSize * sizeof(uint8_t),
                  empty_fields_set_stream.Get());
  cudaMemsetAsync(d_empty_fields_count, 0,
                  deviceResourceManager::kNBoards * sizeof(uint8_t),
                  empty_fields_count_set_stream.Get());
  std::unique_ptr<int> one(new int(1));
  cudaMemcpy(d_old_boards_count, one.get(), sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_old_boards, board.data(),
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue),
             cudaMemcpyHostToDevice);
  cudaMemset(d_solved_board, 0,
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue));
  cudaMemset(d_solved_board_mutex, 0, sizeof(int));

  old_boards_set_stream.Sync();
  new_boards_set_stream.Sync();
  empty_fields_set_stream.Sync();
  empty_fields_count_set_stream.Sync();

  for (int i = 0; i < kIterations; ++i) {
    cudaMemset(d_new_boards_count, 0, sizeof(int));
    Simplificator<<<kBlocks, dim3(Board::kBoardSize, Board::kBoardSize), 0,
                    kernel_stream.Get()>>>(d_old_boards, d_old_boards_count,
                                           d_new_boards, d_new_boards_count);
    kernel_stream.Sync();
    std::swap(d_old_boards, d_new_boards);
    std::swap(d_old_boards_count, d_new_boards_count);
    cudaMemset(d_new_boards_count, 0, sizeof(int));
    Generator<<<kBlocks, kThreadsPerBlock, 0, kernel_stream.Get()>>>(
        d_old_boards, d_old_boards_count, d_new_boards, d_new_boards_count,
        d_empty_fields, d_empty_fields_count, d_solved_board,
        d_solved_board_mutex);
    kernel_stream.Sync();
    std::swap(d_old_boards, d_new_boards);
    std::swap(d_old_boards_count, d_new_boards_count);
    cudaMemset(d_new_boards_count, 0, sizeof(int));
    Simplificator<<<kBlocks, dim3(Board::kBoardSize, Board::kBoardSize), 0,
                    kernel_stream.Get()>>>(d_old_boards, d_old_boards_count,
                                           d_new_boards, d_new_boards_count);
    kernel_stream.Sync();
    std::swap(d_old_boards, d_new_boards);
    std::swap(d_old_boards_count, d_new_boards_count);
    cudaMemset(d_new_boards_count, 0, sizeof(int));
    Generator<<<kBlocks, kThreadsPerBlock, 0, kernel_stream.Get()>>>(
        d_old_boards, d_old_boards_count, d_new_boards, d_new_boards_count,
        d_empty_fields, d_empty_fields_count, d_solved_board,
        d_solved_board_mutex);
    kernel_stream.Sync();
    std::swap(d_old_boards, d_new_boards);
    std::swap(d_old_boards_count, d_new_boards_count);
  }

  int solved = 0;
  cudaMemcpy(&solved, d_solved_board_mutex, sizeof(int),
             cudaMemcpyDeviceToHost);
  if (!solved) {
    Backtracker<<<kBlocks, kThreadsPerBlock, 0, kernel_stream.Get()>>>(
        d_old_boards, d_old_boards_count, d_empty_fields, d_empty_fields_count,
        d_solved_board, d_solved_board_mutex);
    kernel_stream.Sync();
  }
  std::unique_ptr<Board::FieldValue[]> ret(
      new Board::FieldValue[Board::kBoardSize * Board::kBoardSize]);
  cudaMemcpy(ret.get(), d_solved_board,
             Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue),
             cudaMemcpyDeviceToHost);

  stop.Record();
  stop.Sync();
  float ms = 0;
  cudaEventElapsedTime(&ms, start.Get(), stop.Get());
  timeManager::AddTimeElapsed(ms);

  return {ret.get(), ret.get() + Board::kBoardSize * Board::kBoardSize};
}
} // namespace kernel
} // namespace sudoku
