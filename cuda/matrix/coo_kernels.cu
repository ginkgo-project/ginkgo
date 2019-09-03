/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/coo_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/format_conversion.cuh"
#include "cuda/components/segment_scan.cuh"
#include "cuda/components/zero_array.hpp"


namespace gko {
namespace kernels {
/**
 * @brief The CUDA namespace.
 *
 * @ingroup cuda
 */
namespace cuda {
/**
 * @brief The Coordinate matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * cuda_config::warp_size;


namespace {


/**
 * The device function of COO spmv
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_line  the maximum round of each warp
 * @param val  the value array of the matrix
 * @param col  the column index array of the matrix
 * @param row  the row index array of the matrix
 * @param b  the input dense vector
 * @param c  the output dense vector
 * @param scale  the function on the added value
 */
template <int subwarp_size = cuda_config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ void spmv_kernel(const size_type nnz, const size_type num_lines,
                            const ValueType *__restrict__ val,
                            const IndexType *__restrict__ col,
                            const IndexType *__restrict__ row,
                            const ValueType *__restrict__ b,
                            const size_type b_stride, ValueType *__restrict__ c,
                            const size_type c_stride, Closure scale)
{
    ValueType temp_val = zero<ValueType>();
    const auto start = static_cast<size_type>(blockDim.x) * blockIdx.x *
                           blockDim.y * num_lines +
                       threadIdx.y * blockDim.x * num_lines;
    const auto column_id = blockIdx.y;
    size_type num = (nnz > start) * ceildiv(nnz - start, subwarp_size);
    num = min(num, num_lines);
    const IndexType ind_start = start + threadIdx.x;
    const IndexType ind_end = ind_start + (num - 1) * subwarp_size;
    IndexType ind = ind_start;
    IndexType curr_row = (ind < nnz) ? row[ind] : 0;
    const auto tile_block =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    for (; ind < ind_end; ind += subwarp_size) {
        temp_val += (ind < nnz) ? val[ind] * b[col[ind] * b_stride + column_id]
                                : zero<ValueType>();
        auto next_row =
            (ind + subwarp_size < nnz) ? row[ind + subwarp_size] : row[nnz - 1];
        // segmented scan
        if (tile_block.any(curr_row != next_row)) {
            bool is_first_in_segment =
                segment_scan<subwarp_size>(tile_block, curr_row, &temp_val);
            if (is_first_in_segment) {
                atomic_add(&(c[curr_row * c_stride + column_id]),
                           scale(temp_val));
            }
            temp_val = zero<ValueType>();
        }
        curr_row = next_row;
    }
    if (num > 0) {
        ind = ind_end;
        temp_val += (ind < nnz) ? val[ind] * b[col[ind] * b_stride + column_id]
                                : zero<ValueType>();
        // segmented scan
        bool is_first_in_segment =
            segment_scan<subwarp_size>(tile_block, curr_row, &temp_val);
        if (is_first_in_segment) {
            atomic_add(&(c[curr_row * c_stride + column_id]), scale(temp_val));
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const size_type nnz, const size_type num_lines,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride)
{
    spmv_kernel(nnz, num_lines, val, col, row, b, b_stride, c, c_stride,
                [](const ValueType &x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const size_type nnz, const size_type num_lines,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col, const IndexType *__restrict__ row,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(
        nnz, num_lines, val, col, row, b, b_stride, c, c_stride,
        [&scale_factor](const ValueType &x) { return scale_factor * x; });
}


template <bool force, bool final, int subwarp_size = cuda_config::warp_size,
          typename Group, typename ValueType, typename IndexType,
          typename Closure>
__device__ void warp_spmm(
    const Group tile_block, const size_type nnz,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride, IndexType *__restrict__ curr_row,
    ValueType *__restrict__ temp, const size_type offset,
    const size_type column_id, const size_type end_col, Closure scale)
{
    const auto coo_val =
        (!final || offset < nnz) ? val[offset] : zero<ValueType>();
    const auto col_id = (column_id < end_col) ? column_id : end_col - 1;
    const auto coo_col = (!final || offset < nnz) ? col[offset] : col[nnz - 1];
    int j = 0;
    auto temp_row = tile_block.shfl(*curr_row, 0);

    if (!final) {
        constexpr int end = subwarp_size - 1;
#pragma unroll
        for (; j < end; j++) {
            const auto temp_next_row = tile_block.shfl(*curr_row, j + 1);
            const auto tval = tile_block.shfl(coo_val, j);
            const auto tcol = tile_block.shfl(coo_col, j);
            *temp += tval * b[tcol * b_stride + col_id];
            if (temp_row != temp_next_row) {
                if (column_id < end_col) {
                    atomic_add(&(c[temp_row * c_stride + col_id]),
                               scale(*temp));
                }
                *temp = zero<ValueType>();
                temp_row = temp_next_row;
            }
        }
    } else {
        const int end = min(nnz - (offset - threadIdx.x),
                            static_cast<size_type>(subwarp_size)) -
                        1;
#pragma unroll
        for (; j < end; j++) {
            const auto temp_next_row = tile_block.shfl(*curr_row, j + 1);
            const auto tval = tile_block.shfl(coo_val, j);
            const auto tcol = tile_block.shfl(coo_col, j);
            *temp += tval * b[tcol * b_stride + col_id];
            if (temp_row != temp_next_row) {
                if (column_id < end_col) {
                    atomic_add(&(c[temp_row * c_stride + col_id]),
                               scale(*temp));
                }
                *temp = zero<ValueType>();
                temp_row = temp_next_row;
            }
        }
    }

    *temp += tile_block.shfl(coo_val, j) *
             b[tile_block.shfl(coo_col, j) * b_stride + col_id];
    if (force) {
        if (column_id < end_col) {
            atomic_add(&(c[temp_row * c_stride + col_id]), scale(*temp));
        }
    } else {
        const auto next_row = (!final || offset + subwarp_size < nnz)
                                  ? row[offset + subwarp_size]
                                  : row[nnz - 1];
        if (tile_block.shfl(next_row, 0) != temp_row) {
            if (column_id < end_col) {
                atomic_add(&(c[temp_row * c_stride + col_id]), scale(*temp));
            }
            *temp = zero<ValueType>();
        }
        *curr_row = next_row;
    }
}

template <int subwarp_size = cuda_config::warp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ void spmm_kernel(const size_type nnz, const size_type num_lines,
                            const ValueType *__restrict__ val,
                            const IndexType *__restrict__ col,
                            const IndexType *__restrict__ row,
                            const size_type end_col,
                            const ValueType *__restrict__ b,
                            const size_type b_stride, ValueType *__restrict__ c,
                            const size_type c_stride, Closure scale)
{
    ValueType temp = zero<ValueType>();
    const auto tidx = threadIdx.x;
    auto coo_idx =
        (static_cast<size_type>(blockDim.y) * blockIdx.x + threadIdx.y) *
            num_lines * subwarp_size +
        tidx;
    const auto tile_block =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const auto column_id = blockIdx.y * subwarp_size + tidx;
    if (blockIdx.x == gridDim.x - 1) {
        // The final block needs to check the index is valid
        if (coo_idx - tidx < nnz) {
            const int lines = min(
                static_cast<int>(ceildiv(nnz - (coo_idx - tidx), subwarp_size)),
                static_cast<int>(num_lines));
            const auto coo_end = coo_idx + (lines - 1) * subwarp_size;
            auto curr_row = (coo_idx < nnz) ? row[coo_idx] : row[nnz - 1];
            for (; coo_idx < coo_end; coo_idx += subwarp_size) {
                warp_spmm<false, true, subwarp_size>(
                    tile_block, nnz, val, col, row, b, b_stride, c, c_stride,
                    &curr_row, &temp, coo_idx, column_id, end_col, scale);
            }
            warp_spmm<true, true, subwarp_size>(
                tile_block, nnz, val, col, row, b, b_stride, c, c_stride,
                &curr_row, &temp, coo_end, column_id, end_col, scale);
        }
    } else {
        auto curr_row = row[coo_idx];
        const auto coo_end = coo_idx + (num_lines - 1) * subwarp_size;
        for (; coo_idx < coo_end; coo_idx += subwarp_size) {
            warp_spmm<false, false, subwarp_size>(
                tile_block, nnz, val, col, row, b, b_stride, c, c_stride,
                &curr_row, &temp, coo_idx, column_id, end_col, scale);
        }
        warp_spmm<true, false, subwarp_size>(
            tile_block, nnz, val, col, row, b, b_stride, c, c_stride, &curr_row,
            &temp, coo_end, column_id, end_col, scale);
    }
}


template <int subwarp_size = cuda_config::warp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmm(
    const size_type nnz, const size_type num_lines,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row, const size_type num_cols,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    spmm_kernel<subwarp_size>(nnz, num_lines, val, col, row, num_cols, b,
                              b_stride, c, c_stride,
                              [](const ValueType &x) { return x; });
}


template <int subwarp_size = cuda_config::warp_size, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmm(
    const size_type nnz, const size_type num_lines,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col, const IndexType *__restrict__ row,
    const size_type num_cols, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride)
{
    ValueType scale_factor = alpha[0];
    spmm_kernel<subwarp_size>(
        nnz, num_lines, val, col, row, num_cols, b, b_stride, c, c_stride,
        [&scale_factor](const ValueType &x) { return scale_factor * x; });
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Coo<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    zero_array(c->get_num_stored_elements(), c->get_values());

    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Coo<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Coo<ValueType, IndexType> *a,
           const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto nnz = a->get_num_stored_elements();

    auto nwarps = host_kernel::calculate_nwarps(exec, nnz);
    if (nwarps > 0) {
        if (b->get_size()[1] == 1) {
            int num_lines = ceildiv(nnz, nwarps * cuda_config::warp_size);
            const dim3 coo_block(cuda_config::warp_size, warps_in_block, 1);
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block));
            abstract_spmv<<<coo_grid, coo_block>>>(
                nnz, num_lines, as_cuda_type(a->get_const_values()),
                a->get_const_col_idxs(), as_cuda_type(a->get_const_row_idxs()),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
        } else {
            int num_lines = ceildiv(nnz, nwarps * cuda_config::warp_size);
            const dim3 coo_block(cuda_config::warp_size, warps_in_block, 1);
            const dim3 coo_grid(
                ceildiv(nnz,
                        num_lines * cuda_config::warp_size * warps_in_block),
                ceildiv(b->get_size()[1], cuda_config::warp_size));
            abstract_spmm<<<coo_grid, coo_block>>>(
                nnz, num_lines, as_cuda_type(a->get_const_values()),
                a->get_const_col_idxs(), as_cuda_type(a->get_const_row_idxs()),
                b->get_size()[1], as_cuda_type(b->get_const_values()),
                b->get_stride(), as_cuda_type(c->get_values()),
                c->get_stride());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Dense<ValueType> *alpha,
                    const matrix::Coo<ValueType, IndexType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c)
{
    auto nnz = a->get_num_stored_elements();

    auto nwarps = host_kernel::calculate_nwarps(exec, nnz);
    if (nwarps > 0) {
        if (b->get_size()[1] == 1) {
            int num_lines = ceildiv(nnz, nwarps * cuda_config::warp_size);
            const dim3 coo_block(cuda_config::warp_size, warps_in_block, 1);
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block));
            abstract_spmv<<<coo_grid, coo_block>>>(
                nnz, num_lines, as_cuda_type(alpha->get_const_values()),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                as_cuda_type(a->get_const_row_idxs()),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
        } else {
            int num_lines = ceildiv(nnz, nwarps * cuda_config::warp_size);
            const dim3 coo_block(cuda_config::warp_size, warps_in_block, 1);
            const dim3 coo_grid(
                ceildiv(nnz,
                        num_lines * cuda_config::warp_size * warps_in_block),
                ceildiv(b->get_size()[1], cuda_config::warp_size));
            abstract_spmm<<<coo_grid, coo_block>>>(
                nnz, num_lines, as_cuda_type(alpha->get_const_values()),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                as_cuda_type(a->get_const_row_idxs()), b->get_size()[1],
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);

namespace kernel {

template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void convert_row_idxs_to_ptrs(
    const IndexType *__restrict__ idxs, size_type num_nonzeros,
    IndexType *__restrict__ ptrs, size_type length)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx == 0) {
        ptrs[0] = 0;
        ptrs[length - 1] = num_nonzeros;
    }

    if (0 < tidx && tidx < num_nonzeros) {
        if (idxs[tidx - 1] < idxs[tidx]) {
            for (auto i = idxs[tidx - 1] + 1; i <= idxs[tidx]; i++) {
                ptrs[i] = tidx;
            }
        }
    }
}

}  // namespace kernel


template <typename IndexType>
void convert_row_idxs_to_ptrs(std::shared_ptr<const CudaExecutor> exec,
                              const IndexType *idxs, size_type num_nonzeros,
                              IndexType *ptrs, size_type length)
{
    const auto grid_dim = ceildiv(num_nonzeros, default_block_size);

    kernel::convert_row_idxs_to_ptrs<<<grid_dim, default_block_size>>>(
        as_cuda_type(idxs), num_nonzeros, as_cuda_type(ptrs), length);
}


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const CudaExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Coo<ValueType, IndexType> *source)
{
    auto num_rows = result->get_size()[0];

    auto row_ptrs = result->get_row_ptrs();
    const auto nnz = result->get_num_stored_elements();

    const auto source_row_idxs = source->get_const_row_idxs();

    convert_row_idxs_to_ptrs(exec, source_row_idxs, nnz, row_ptrs,
                             num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_CSR_KERNEL);


namespace kernel {


template <typename ValueType>
__global__
    __launch_bounds__(cuda_config::max_block_size) void initialize_zero_dense(
        size_type num_rows, size_type num_cols, size_type stride,
        ValueType *__restrict__ result)
{
    const auto tidx_x = threadIdx.x + blockDim.x * blockIdx.x;
    const auto tidx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type nnz, const IndexType *__restrict__ row_idxs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, size_type stride,
    ValueType *__restrict__ result)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < nnz) {
        result[stride * row_idxs[tidx] + col_idxs[tidx]] = values[tidx];
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      matrix::Dense<ValueType> *result,
                      const matrix::Coo<ValueType, IndexType> *source)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = result->get_stride();

    const auto nnz = source->get_num_stored_elements();

    const dim3 block_size(cuda_config::warp_size,
                          cuda_config::max_block_size / cuda_config::warp_size,
                          1);
    const dim3 init_grid_dim(ceildiv(stride, block_size.x),
                             ceildiv(num_rows, block_size.y), 1);
    kernel::initialize_zero_dense<<<init_grid_dim, block_size>>>(
        num_rows, num_cols, stride, as_cuda_type(result->get_values()));

    const auto grid_dim = ceildiv(nnz, default_block_size);
    kernel::fill_in_dense<<<grid_dim, default_block_size>>>(
        nnz, as_cuda_type(source->get_const_row_idxs()),
        as_cuda_type(source->get_const_col_idxs()),
        as_cuda_type(source->get_const_values()), stride,
        as_cuda_type(result->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_DENSE_KERNEL);


}  // namespace coo
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
