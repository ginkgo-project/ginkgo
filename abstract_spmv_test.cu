// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Complete test example for standalone abstract_spmv implementation
// This file demonstrates how to use the extracted kernel with a simple example

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// Include all the code from abstract_spmv_standalone.cu (or compile it separately)
// For simplicity, we'll include the declarations here

// ============================================================================
// Basic type definitions
// ============================================================================

using size_type = std::int64_t;
using int32 = std::int32_t;
using uint32 = std::uint32_t;

namespace config {
    constexpr uint32 warp_size = 32;
}

constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;

// ============================================================================
// Forward declarations (from abstract_spmv_standalone.cu)
// ============================================================================

template <typename T>
__host__ __device__ __forceinline__ constexpr T zero() { return T{}; }

template <typename T>
__host__ __device__ __forceinline__ T ceildivT(T nom, T denom) {
    return (nom + denom - 1ll) / denom;
}

template <typename T>
__host__ __device__ __forceinline__ T min(T a, T b) { return a < b ? a : b; }

template <typename T>
__host__ __device__ __forceinline__ T max(T a, T b) { return a > b ? a : b; }

__forceinline__ __device__ double atomic_add(double* __restrict__ addr, double val) {
    return atomicAdd(addr, val);
}

namespace group {
    using cooperative_groups::thread_block;
    using cooperative_groups::thread_block_tile;
    using cooperative_groups::this_thread_block;
    using cooperative_groups::tiled_partition;
}

namespace acc {

template <typename ValueType, typename IndexType>
class simple_row_major_2d {
public:
    using value_type = ValueType;
    using storage_type = ValueType;
    using arithmetic_type = ValueType;

    __host__ __device__ simple_row_major_2d(
        IndexType num_rows, IndexType num_cols,
        ValueType* data, IndexType stride)
        : num_rows_(num_rows), num_cols_(num_cols),
          data_(data), stride_(stride) {}

    __host__ __device__ ValueType& operator()(IndexType row, IndexType col) const {
        return data_[row * stride_ + col];
    }

    __host__ __device__ ValueType* get_storage_address(IndexType row, IndexType col) const {
        return &data_[row * stride_ + col];
    }

    __host__ __device__ IndexType length(int dimension) const {
        return dimension == 0 ? num_rows_ : num_cols_;
    }

private:
    IndexType num_rows_;
    IndexType num_cols_;
    ValueType* data_;
    IndexType stride_;
};

template <typename ValueType, typename IndexType>
class simple_1d {
public:
    using value_type = ValueType;
    using storage_type = ValueType;
    using arithmetic_type = ValueType;

    __host__ __device__ simple_1d(ValueType* data) : data_(data) {}

    __host__ __device__ ValueType operator()(IndexType idx) const {
        return data_[idx];
    }

private:
    ValueType* data_;
};

template <typename Accessor>
class range {
public:
    using accessor_type = Accessor;
    using value_type = typename Accessor::value_type;
    using storage_type = typename Accessor::storage_type;
    using arithmetic_type = typename Accessor::arithmetic_type;

    __host__ __device__ range(const Accessor& acc) : accessor_(acc) {}

    template <typename... Args>
    __host__ __device__ auto operator()(Args... args) const
        -> decltype(accessor_(args...)) {
        return accessor_(args...);
    }

    __host__ __device__ const Accessor* operator->() const {
        return &accessor_;
    }

    __host__ __device__ auto length(int dimension) const
        -> decltype(accessor_.length(dimension)) {
        return accessor_.length(dimension);
    }

private:
    Accessor accessor_;
};

}  // namespace acc

// ============================================================================
// Include all kernel code from abstract_spmv_standalone.cu
// (In practice, you would compile them together or link them)
// ============================================================================

// [All the kernel code would go here or be compiled separately]
// For brevity, we assume the kernels are available

template <unsigned subwarp_size, typename ValueType, typename IndexType,
          typename Operator>
__device__ __forceinline__ bool segment_scan(
    const group::thread_block_tile<subwarp_size>& group, const IndexType ind,
    ValueType& val, Operator op)
{
    bool head = true;
#pragma unroll
    for (int i = 1; i < subwarp_size; i <<= 1) {
        const IndexType add_ind = group.shfl_up(ind, i);
        ValueType add_val{};
        if (add_ind == ind && group.thread_rank() >= i) {
            add_val = val;
            if (i == 1) {
                head = false;
            }
        }
        add_val = group.shfl_down(add_val, i);
        if (group.thread_rank() < subwarp_size - i) {
            val = op(val, add_val);
        }
    }
    return head;
}

template <bool overflow, typename IndexType>
__device__ __forceinline__ void find_next_row(
    const IndexType num_rows, const IndexType data_size, const IndexType ind,
    IndexType& row, IndexType& row_end, const IndexType row_predict,
    const IndexType row_predict_end, const IndexType* __restrict__ row_ptr)
{
    if (!overflow || ind < data_size) {
        if (ind >= row_end) {
            row = row_predict;
            row_end = row_predict_end;
            while (ind >= row_end) {
                row_end = row_ptr[++row + 1];
            }
        }
    } else {
        row = num_rows - 1;
        row_end = data_size;
    }
}

template <unsigned subwarp_size, typename ValueType, typename IndexType,
          typename output_accessor, typename Closure>
__device__ __forceinline__ void warp_atomic_add(
    const group::thread_block_tile<subwarp_size>& group, bool force_write,
    ValueType& val, const IndexType row, acc::range<output_accessor>& c,
    const IndexType column_id, Closure scale)
{
    const bool need_write = segment_scan(
        group, row, val, [](ValueType a, ValueType b) { return a + b; });
    if (need_write && force_write) {
        atomic_add(c->get_storage_address(row, column_id), scale(val));
    }
    if (!need_write || force_write) {
        val = zero<ValueType>();
    }
}

template <bool last, unsigned subwarp_size, typename arithmetic_type,
          typename matrix_accessor, typename IndexType, typename input_accessor,
          typename output_accessor, typename Closure>
__device__ __forceinline__ void process_window(
    const group::thread_block_tile<subwarp_size>& group,
    const IndexType num_rows, const IndexType data_size, const IndexType ind,
    IndexType& row, IndexType& row_end, IndexType& nrow, IndexType& nrow_end,
    arithmetic_type& temp_val, acc::range<matrix_accessor> val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, acc::range<input_accessor> b,
    acc::range<output_accessor> c, const IndexType column_id, Closure scale)
{
    const auto curr_row = row;
    find_next_row<last>(num_rows, data_size, ind, row, row_end, nrow, nrow_end,
                        row_ptrs);
    if (group.any(curr_row != row)) {
        warp_atomic_add(group, curr_row != row, temp_val, curr_row, c,
                        column_id, scale);
        nrow = group.shfl(row, subwarp_size - 1);
        nrow_end = group.shfl(row_end, subwarp_size - 1);
    }

    if (!last || ind < data_size) {
        const auto col = col_idxs[ind];
        temp_val += val(ind) * b(col, column_id);
    }
}

template <typename IndexType>
__device__ __forceinline__ IndexType get_warp_start_idx(
    const IndexType nwarps, const IndexType nnz, const IndexType warp_idx)
{
    const long long cache_lines = ceildivT<IndexType>(nnz, config::warp_size);
    return (warp_idx * cache_lines / nwarps) * config::warp_size;
}

template <typename matrix_accessor, typename input_accessor,
          typename output_accessor, typename IndexType, typename Closure>
__device__ __forceinline__ void spmv_kernel(
    const IndexType nwarps, const IndexType num_rows,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c, Closure scale)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    const IndexType warp_idx = blockIdx.x * warps_in_block + threadIdx.y;
    const IndexType column_id = blockIdx.y;
    if (warp_idx >= nwarps) {
        return;
    }
    const IndexType data_size = row_ptrs[num_rows];
    const IndexType start = get_warp_start_idx(nwarps, data_size, warp_idx);
    constexpr IndexType wsize = config::warp_size;
    const IndexType end =
        min(get_warp_start_idx(nwarps, data_size, warp_idx + 1),
            ceildivT<IndexType>(data_size, wsize) * wsize);
    auto row = srow[warp_idx];
    auto row_end = row_ptrs[row + 1];
    auto nrow = row;
    auto nrow_end = row_end;
    auto temp_val = zero<arithmetic_type>();
    IndexType ind = start + threadIdx.x;
    find_next_row<true>(num_rows, data_size, ind, row, row_end, nrow, nrow_end,
                        row_ptrs);
    const IndexType ind_end = end - wsize;
    const auto tile_block =
        group::tiled_partition<wsize>(group::this_thread_block());
    for (; ind < ind_end; ind += wsize) {
        process_window<false>(tile_block, num_rows, data_size, ind, row,
                              row_end, nrow, nrow_end, temp_val, val, col_idxs,
                              row_ptrs, b, c, column_id, scale);
    }
    process_window<true>(tile_block, num_rows, data_size, ind, row, row_end,
                         nrow, nrow_end, temp_val, val, col_idxs, row_ptrs, b,
                         c, column_id, scale);
    warp_atomic_add(tile_block, true, temp_val, row, c, column_id, scale);
}

template <typename matrix_accessor, typename input_accessor,
          typename output_accessor, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    using output_type = typename output_accessor::storage_type;
    spmv_kernel(nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, c,
                [](const arithmetic_type& x) {
                    return static_cast<output_type>(x);
                });
}

__global__ void compute_srow_kernel(const int32 num_rows, const int32 nwarps,
                                     const int32* __restrict__ row_ptrs,
                                     int32* __restrict__ srow)
{
    const int32 warp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (warp_idx >= nwarps) {
        return;
    }

    const int32 nnz = row_ptrs[num_rows];
    const int32 start_idx = get_warp_start_idx(nwarps, nnz, warp_idx);

    int32 left = 0;
    int32 right = num_rows;
    while (left < right) {
        int32 mid = (left + right) / 2;
        if (row_ptrs[mid] <= start_idx) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    srow[warp_idx] = left > 0 ? left - 1 : 0;
}

// ============================================================================
// Test code
// ============================================================================

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main()
{
    printf("Testing standalone abstract_spmv implementation\n");
    printf("================================================\n\n");

    // Create a simple 4x4 CSR matrix:
    // [2  0  1  0]
    // [0  3  0  2]
    // [1  0  4  0]
    // [0  2  0  5]

    const int32 num_rows = 4;
    const int32 num_cols = 4;
    const int32 nnz = 8;

    // Host data
    double h_values[] = {2.0, 1.0, 3.0, 2.0, 1.0, 4.0, 2.0, 5.0};
    int32 h_col_idxs[] = {0, 2, 1, 3, 0, 2, 1, 3};
    int32 h_row_ptrs[] = {0, 2, 4, 6, 8};

    // Input vector: [1, 2, 3, 4]^T
    double h_b[] = {1.0, 2.0, 3.0, 4.0};

    // Output vector (will be computed)
    double h_c[4] = {0.0};

    // Expected result: A * b
    // Row 0: 2*1 + 1*3 = 5
    // Row 1: 3*2 + 2*4 = 14
    // Row 2: 1*1 + 4*3 = 13
    // Row 3: 2*2 + 5*4 = 24
    double expected[] = {5.0, 14.0, 13.0, 24.0};

    // Device pointers
    double *d_values, *d_b, *d_c;
    int32 *d_col_idxs, *d_row_ptrs, *d_srow;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_col_idxs, nnz * sizeof(int32)));
    CHECK_CUDA(cudaMalloc(&d_row_ptrs, (num_rows + 1) * sizeof(int32)));
    CHECK_CUDA(cudaMalloc(&d_b, num_cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_c, num_rows * sizeof(double)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_idxs, h_col_idxs, nnz * sizeof(int32), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_row_ptrs, h_row_ptrs, (num_rows + 1) * sizeof(int32), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, num_cols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_c, 0, num_rows * sizeof(double)));

    // Compute number of warps needed
    int32 nwarps = ceildivT<int32>(nnz, 32);
    if (nwarps == 0) nwarps = 1;

    // Allocate and compute srow
    CHECK_CUDA(cudaMalloc(&d_srow, nwarps * sizeof(int32)));

    int srow_threads = 256;
    int srow_blocks = ceildivT<int32>(nwarps, srow_threads);
    compute_srow_kernel<<<srow_blocks, srow_threads>>>(num_rows, nwarps, d_row_ptrs, d_srow);
    CHECK_CUDA(cudaGetLastError());

    // Launch abstract_spmv kernel
    dim3 block_dim(config::warp_size, warps_in_block);
    dim3 grid_dim(ceildivT<int32>(nwarps, warps_in_block), 1);

    printf("Launching kernel with:\n");
    printf("  Grid: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("  Block: (%d, %d, %d)\n", block_dim.x, block_dim.y, block_dim.z);
    printf("  Number of warps: %d\n", nwarps);
    printf("  NNZ: %d\n\n", nnz);

    // Create accessors
    acc::simple_1d<double, int32> val_acc(d_values);
    acc::simple_row_major_2d<double, int32> b_acc(num_cols, 1, d_b, 1);
    acc::simple_row_major_2d<double, int32> c_acc(num_rows, 1, d_c, 1);

    // Create ranges
    acc::range<acc::simple_1d<double, int32>> val_range(val_acc);
    acc::range<acc::simple_row_major_2d<double, int32>> b_range(b_acc);
    acc::range<acc::simple_row_major_2d<double, int32>> c_range(c_acc);

    abstract_spmv<<<grid_dim, block_dim>>>(
        nwarps, num_rows, val_range, d_col_idxs, d_row_ptrs, d_srow,
        b_range, c_range);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_c, d_c, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

    // Verify results
    printf("Results:\n");
    bool all_correct = true;
    for (int i = 0; i < num_rows; i++) {
        double error = fabs(h_c[i] - expected[i]);
        bool correct = error < 1e-10;
        all_correct = all_correct && correct;
        printf("  c[%d] = %.6f (expected: %.6f) %s\n",
               i, h_c[i], expected[i], correct ? "✓" : "✗");
    }

    printf("\n");
    if (all_correct) {
        printf("SUCCESS: All results match expected values!\n");
    } else {
        printf("FAILURE: Some results do not match!\n");
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_col_idxs));
    CHECK_CUDA(cudaFree(d_row_ptrs));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_srow));

    return all_correct ? 0 : 1;
}
