// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Standalone implementation of abstract_spmv for IndexType=int32, ValueType=double
// Extracted from common/cuda_hip/matrix/csr_kernels.template.cpp

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>

// ============================================================================
// Basic type definitions
// ============================================================================

using size_type = std::int64_t;
using int32 = std::int32_t;
using uint32 = std::uint32_t;

// ============================================================================
// Configuration constants
// ============================================================================

namespace config {
    constexpr uint32 warp_size = 32;
}  // namespace config

constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;

// ============================================================================
// Mathematical utilities
// ============================================================================

template <typename T>
__host__ __device__ __forceinline__ constexpr T zero()
{
    return T{};
}

template <typename T>
__host__ __device__ __forceinline__ constexpr T one()
{
    return T(1);
}

template <typename T>
__host__ __device__ __forceinline__ T ceildivT(T nom, T denom)
{
    return (nom + denom - 1ll) / denom;
}

template <typename T>
__host__ __device__ __forceinline__ T min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
__host__ __device__ __forceinline__ T max(T a, T b)
{
    return a > b ? a : b;
}

// ============================================================================
// Atomic operations
// ============================================================================

__forceinline__ __device__ double atomic_add(double* __restrict__ addr, double val)
{
    return atomicAdd(addr, val);
}

// ============================================================================
// Cooperative groups wrapper
// ============================================================================

namespace group {
    using cooperative_groups::thread_block;
    using cooperative_groups::thread_block_tile;
    using cooperative_groups::this_thread_block;
    using cooperative_groups::tiled_partition;
}  // namespace group

// ============================================================================
// Accessor implementation - Simplified row_major for 2D double arrays
// ============================================================================

namespace acc {

// Simple 2D accessor for row-major layout with double values and int32 indices
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
          data_(data), stride_(stride)
    {}

    __host__ __device__ ValueType& operator()(IndexType row, IndexType col) const
    {
        return data_[row * stride_ + col];
    }

    __host__ __device__ ValueType* get_storage_address(IndexType row, IndexType col) const
    {
        return &data_[row * stride_ + col];
    }

    __host__ __device__ IndexType length(int dimension) const
    {
        return dimension == 0 ? num_rows_ : num_cols_;
    }

private:
    IndexType num_rows_;
    IndexType num_cols_;
    ValueType* data_;
    IndexType stride_;
};

// Simple 1D accessor for CSR matrix values
template <typename ValueType, typename IndexType>
class simple_1d {
public:
    using value_type = ValueType;
    using storage_type = ValueType;
    using arithmetic_type = ValueType;

    __host__ __device__ simple_1d(ValueType* data)
        : data_(data)
    {}

    __host__ __device__ ValueType operator()(IndexType idx) const
    {
        return data_[idx];
    }

private:
    ValueType* data_;
};

// Range wrapper
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
        -> decltype(accessor_(args...))
    {
        return accessor_(args...);
    }

    __host__ __device__ const Accessor* operator->() const
    {
        return &accessor_;
    }

    __host__ __device__ auto length(int dimension) const
        -> decltype(accessor_.length(dimension))
    {
        return accessor_.length(dimension);
    }

private:
    Accessor accessor_;
};

}  // namespace acc

// ============================================================================
// Segment scan operation
// ============================================================================

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

// ============================================================================
// SpMV kernel helper functions
// ============================================================================

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
    // do a local scan to avoid atomic collisions
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
    // segmented scan
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

// ============================================================================
// Main abstract_spmv kernels
// ============================================================================

// Version without alpha scaling
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

// Version with alpha scaling
template <typename matrix_accessor, typename input_accessor,
          typename output_accessor, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    const typename matrix_accessor::storage_type* __restrict__ alpha,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    using output_type = typename output_accessor::storage_type;
    const auto scale_factor = static_cast<arithmetic_type>(alpha[0]);
    spmv_kernel(nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, c,
                [&scale_factor](const arithmetic_type& x) {
                    return static_cast<output_type>(scale_factor * x);
                });
}

// ============================================================================
// Specialized instantiation for int32/double
// ============================================================================

// Explicit instantiation for the common case: IndexType=int32, ValueType=double
using DoubleAccessor1D = acc::simple_1d<double, int32>;
using DoubleAccessor2D = acc::simple_row_major_2d<double, int32>;

// Version without alpha
template __global__ void abstract_spmv<DoubleAccessor1D, DoubleAccessor2D,
                                        DoubleAccessor2D, int32>(
    const int32 nwarps, const int32 num_rows,
    acc::range<DoubleAccessor1D> val, const int32* __restrict__ col_idxs,
    const int32* __restrict__ row_ptrs, const int32* __restrict__ srow,
    acc::range<DoubleAccessor2D> b, acc::range<DoubleAccessor2D> c);

// Version with alpha
template __global__ void abstract_spmv<DoubleAccessor1D, DoubleAccessor2D,
                                        DoubleAccessor2D, int32>(
    const int32 nwarps, const int32 num_rows,
    const double* __restrict__ alpha,
    acc::range<DoubleAccessor1D> val, const int32* __restrict__ col_idxs,
    const int32* __restrict__ row_ptrs, const int32* __restrict__ srow,
    acc::range<DoubleAccessor2D> b, acc::range<DoubleAccessor2D> c);

// ============================================================================
// Helper function to prepare srow array
// ============================================================================

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

    // Binary search to find the row that contains start_idx
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
// Example usage / test function
// ============================================================================

void example_spmv_usage()
{
    // This is a placeholder to demonstrate how to use the kernel
    // In a real application, you would:
    // 1. Allocate and initialize CSR matrix data (values, col_idxs, row_ptrs)
    // 2. Allocate input vector b and output vector c
    // 3. Compute srow array
    // 4. Launch the abstract_spmv kernel

    printf("Example usage of abstract_spmv kernel\n");
    printf("This standalone implementation is ready to be integrated into your code\n");
}

int main()
{
    example_spmv_usage();
    return 0;
}
