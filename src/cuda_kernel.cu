#include "../include/cuda_kernel.cuh"

#include <cassert>
#include <cstdio>
#include <ctime>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

namespace sudoku {
namespace kernel {
namespace {
constexpr unsigned kBlocks = 999;
constexpr unsigned kNumBanks = 16;
constexpr unsigned kLogNumBanks = 4;
__device__ constexpr unsigned ConflictFreeIndex(unsigned index) {
	return index + (index >> kLogNumBanks);
}

__device__ constexpr unsigned ExpandedSpace(unsigned size) {
	return size += size / kNumBanks;
}

template<class T>
__device__ constexpr T& At(T* arr, int x, int y) {
	return arr[ConflictFreeIndex(y * Board::kBoardSize + x)];
}

template<Board::Size size>
struct RowPattern {
	constexpr RowPattern() :
			pattern() {
		for (auto i = 0u; i < size; ++i)
			pattern[i] = {static_cast<int>(i), 0};
		}
		int2 pattern[size];
		Board::Size length = size;
	};

template<Board::Size size>
struct ColPattern {
	constexpr ColPattern() :
			pattern() {
		for (auto i = 0u; i < size; ++i)
			pattern[i] = {0, static_cast<int>(i)};
		}
		int2 pattern[size];
		Board::Size length = size;
	};

template<Board::Size size>
struct QuadrantPattern {
	constexpr QuadrantPattern() :
			pattern() {
		for (auto i = 0u; i < size; ++i)
			for (auto j = 0u; j < size; ++j)
				pattern[i + j * size] = {static_cast<int>(i), static_cast<int>(j)};
			}
			int2 pattern[size * size];
			Board::Size length = size * size;
		};

__device__ auto Tid() {
	return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x
			+ threadIdx.x;
}

__global__ void Kernel(Board::FieldValue* board, Board::FieldValue* solved,
		curandState_t* states, unsigned* done) {
	__shared__ Board::FieldValue s_board[ExpandedSpace(
			Board::kBoardSize * Board::kBoardSize)];
	__shared__ bool s_correct[ExpandedSpace(Board::kBoardSize)];

	const auto tid = Tid();
	const auto index = ConflictFreeIndex(
			threadIdx.x + Board::kBoardSize * threadIdx.y);
	s_board[index] = board[threadIdx.y * Board::kBoardSize + threadIdx.x];
	constexpr RowPattern<Board::kBoardSize> row_pattern;
	constexpr ColPattern<Board::kBoardSize> col_pattern;
	constexpr QuadrantPattern<Board::kQuadrantSize> q_pattern;
	bool active = s_board[index] == 0;
	Board::FieldValue* possible_values =
			new Board::FieldValue[Board::kBoardSize];
	unsigned int pvs = 0;
	if (active) {
		bool used[Board::kBoardSize];
		for (auto i = 0u; i < Board::kBoardSize; ++i)
			used[i] = true;
		for (auto i = 0u; i < Board::kBoardSize; ++i) {
			auto j = At(s_board, i, threadIdx.y);
			if (j > 0)
				used[j - 1] = false;
			j = At(s_board, threadIdx.x, i);
			if (j > 0)
				used[j - 1] = false;
		}
		auto pom_x = threadIdx.x - threadIdx.x % Board::kQuadrantSize;
		auto pom_y = threadIdx.y - threadIdx.y % Board::kQuadrantSize;
		for (auto x = 0u; x < Board::kQuadrantSize; ++x)
			for (auto y = 0u; y < Board::kQuadrantSize; ++y) {
				auto j = At(s_board, pom_x + x, pom_y + y);
				if (j > 0)
					used[j - 1] = false;
			}
		for (auto i = 0u; i < Board::kBoardSize; ++i)
			if (used[i])
				possible_values[pvs++] = i + 1;
		assert(pvs > 0);
		if (pvs == 1) {
			s_board[index] = possible_values[0];
			active = false;
		}
	}
	char used_values[Board::kBoardSize];
	bool b = false;
	while (true) {
		__syncthreads();
		if (*done > 0) {
			solved[tid] = s_board[index];
			if (threadIdx.x + threadIdx.y == 0 && !b)
				solved[tid] = 0;
			delete[] possible_values;
			return;
		}
		if (active)
			s_board[index] = possible_values[curand(&states[tid]) % pvs];
        __syncthreads();
		if (threadIdx.y == 0) {
			s_correct[index] = true;
			memset(used_values, 0, Board::kBoardSize * sizeof(char));
			for (auto i = 0u; i < Board::kBoardSize; ++i)
				used_values[At(s_board, row_pattern.pattern[i].x, threadIdx.x)
						- 1] = true;
			for (auto i = 0u; i < Board::kBoardSize; ++i)
				if (!used_values[i]) {
					s_correct[index] = false;
					break;
				}
			if (s_correct[index]) {
				memset(used_values, 0, Board::kBoardSize * sizeof(char));
				for (auto i = 0u; i < Board::kBoardSize; ++i)
					used_values[At(s_board, threadIdx.x,
							col_pattern.pattern[i].y) - 1] = true;
				for (auto i = 0u; i < Board::kBoardSize; ++i)
					if (!used_values[i]) {
						s_correct[index] = false;
						break;
					}
			}
			if (s_correct[index]) {
				memset(used_values, 0, Board::kBoardSize * sizeof(char));
				for (auto i = 0u; i < Board::kBoardSize; ++i)
					used_values[At(s_board,
							(threadIdx.x % Board::kQuadrantSize)
									* Board::kQuadrantSize
									+ q_pattern.pattern[i].x,
							(threadIdx.x / Board::kQuadrantSize)
									* Board::kQuadrantSize
									+ q_pattern.pattern[i].y) - 1] = true;
				for (auto i = 0u; i < Board::kBoardSize; ++i)
					if (!used_values[i]) {
						s_correct[index] = false;
						break;
					}
			}
		}
		__syncthreads();
		if (threadIdx.x + threadIdx.y == 0) {
			for (auto i = 0u; i < Board::kBoardSize; ++i)
				if (!(b = s_correct[ConflictFreeIndex(i)]))
					break;
			if (b)
				atomicAdd(done, 1);
		}
	}
}
__global__ void InitCurand(unsigned seed, curandState_t* states) {
	curand_init(seed, Tid(), 0, &states[Tid()]);
}
}
std::vector<Board::FieldValue> Run(
		std::vector<Board::FieldValue> const& board) {
	curandState_t* d_states;
	cudaMalloc(reinterpret_cast<void**>(&d_states),
			kBlocks * Board::kBoardSize * Board::kBoardSize
					* sizeof(curandState_t));
	dim3 block(Board::kBoardSize, Board::kBoardSize);
	InitCurand<<<kBlocks, block>>>(std::time(nullptr), d_states);

	Board::FieldValue* d_solved;
	cudaMalloc(reinterpret_cast<void**>(&d_solved),
			kBlocks * Board::kBoardSize * Board::kBoardSize
					* sizeof(Board::FieldValue));
	cudaMemset(d_solved, 0,
			kBlocks * Board::kBoardSize * Board::kBoardSize
					* sizeof(Board::FieldValue));

	Board::FieldValue* d_board = nullptr;
	cudaMalloc(reinterpret_cast<void**>(&d_board),
			Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue));
	cudaMemcpy(d_board, board.data(),
			Board::kBoardSize * Board::kBoardSize * sizeof(Board::FieldValue),
			cudaMemcpyHostToDevice);
	unsigned *d_done = nullptr;
	cudaMalloc(reinterpret_cast<void**>(&d_done), sizeof(unsigned));
	cudaMemset(d_done, 0, sizeof(unsigned));
	cudaDeviceSynchronize();

	auto start = clock();
	Kernel<<<kBlocks, block>>>(d_board, d_solved, d_states, d_done);
	cudaDeviceSynchronize();
	auto end = clock();
	printf("%ld ms\n", end - start);

	std::vector<Board::FieldValue> solved(
			kBlocks * Board::kBoardSize * Board::kBoardSize, Board::FieldValue {
					0 });
	cudaMemcpy(solved.data(), d_solved,
			kBlocks * Board::kBoardSize * Board::kBoardSize,
			cudaMemcpyDeviceToHost);
	std::vector<Board::FieldValue> ret;
	for (auto it = solved.begin(); it != solved.end();) {
		if (*it == 0) {
			it += Board::kBoardSize * Board::kBoardSize;
		} else {
			ret = std::vector<Board::FieldValue> { it, it
					+ Board::kBoardSize * Board::kBoardSize };
			break;
		}
	}
	cudaFree(d_states);
	cudaFree(d_board);
	cudaFree(d_solved);
	return ret;
}
}
}
