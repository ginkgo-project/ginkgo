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


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/format_conversion.hip.hpp"
#include "hip/components/segment_scan.hip.hpp"
#include "hip/components/zero_array.hip.hpp"


namespace gko {
namespace kernels {
/**
 * @brief The HIP namespace.
 *
 * @ingroup hip
 */
namespace hip {
/**
 * @brief The Coordinate matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * hip_config::warp_size;


namespace {


/**
 * The device function of COO spmv
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_lines  the maximum round of each warp
 * @param val  the value array of the matrix
 * @param col  the column index array of the matrix
 * @param row  the row index array of the matrix
 * @param b  the input dense vector
 * @param b_stride  the stride of the input dense vector
 * @param c  the output dense vector
 * @param c_stride  the stride of the output dense vector
 * @param scale  the function on the added value
 */
template <int subwarp_size = hip_config::warp_size, typename ValueType,
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


/**
 * The device function of COO spmm
 *
 * @param nnz  the number of nonzeros in the matrix
 * @param num_elems  the maximum number of nonzeros in each warp
 * @param val  the value array of the matrix
 * @param col  the column index array of the matrix
 * @param row  the row index array of the matrix
 * @param num_cols the number of columns of the matrix
 * @param b  the input dense vector
 * @param b_stride  the stride of the input dense vector
 * @param c  the output dense vector
 * @param c_stride  the stride of the output dense vector
 * @param scale  the function on the added value
 */
template <typename ValueType, typename IndexType, typename Closure>
__device__ void spmm_kernel(const size_type nnz, const size_type num_elems,
                            const ValueType *__restrict__ val,
                            const IndexType *__restrict__ col,
                            const IndexType *__restrict__ row,
                            const size_type num_cols,
                            const ValueType *__restrict__ b,
                            const size_type b_stride, ValueType *__restrict__ c,
                            const size_type c_stride, Closure scale)
{
    ValueType temp = zero<ValueType>();
    const auto coo_idx =
        (static_cast<size_type>(blockDim.y) * blockIdx.x + threadIdx.y) *
        num_elems;
    const auto column_id = blockIdx.y * blockDim.x + threadIdx.x;
    const auto coo_end =
        (coo_idx + num_elems > nnz) ? nnz : coo_idx + num_elems;
    if (column_id < num_cols && coo_idx < nnz) {
        auto curr_row = row[coo_idx];
        auto idx = coo_idx;
        for (; idx < coo_end - 1; idx++) {
            temp += val[idx] * b[col[idx] * b_stride + column_id];
            const auto next_row = row[idx + 1];
            if (next_row != curr_row) {
                atomic_add(&(c[curr_row * c_stride + column_id]), scale(temp));
                curr_row = next_row;
                temp = zero<ValueType>();
            }
        }
        temp += val[idx] * b[col[idx] * b_stride + column_id];
        atomic_add(&(c[curr_row * c_stride + column_id]), scale(temp));
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmm(
    const size_type nnz, const size_type num_elems,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row, const size_type num_cols,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    spmm_kernel(nnz, num_elems, val, col, row, num_cols, b, b_stride, c,
                c_stride, [](const ValueType &x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmm(
    const size_type nnz, const size_type num_elems,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col, const IndexType *__restrict__ row,
    const size_type num_cols, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride)
{
    ValueType scale_factor = alpha[0];
    spmm_kernel(
        nnz, num_elems, val, col, row, num_cols, b, b_stride, c, c_stride,
        [&scale_factor](const ValueType &x) { return scale_factor * x; });
}


}  // namespace


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const HipExecutor> exec,
          const matrix::Coo<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    zero_array(c->get_num_stored_elements(), c->get_values());

    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const HipExecutor> exec,
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
void spmv2(std::shared_ptr<const HipExecutor> exec,
           const matrix::Coo<ValueType, IndexType> *a,
           const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    const auto nnz = a->get_num_stored_elements();
    const auto b_ncols = b->get_size()[1];
    const dim3 coo_block(hip_config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        if (b_ncols < 4) {
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b_ncols);
            int num_lines = ceildiv(nnz, nwarps * hip_config::warp_size);
            hipLaunchKernelGGL(
                (abstract_spmv), dim3(coo_grid), dim3(coo_block), 0, 0, nnz,
                num_lines, as_hip_type(a->get_const_values()),
                a->get_const_col_idxs(), as_hip_type(a->get_const_row_idxs()),
                as_hip_type(b->get_const_values()), b->get_stride(),
                as_hip_type(c->get_values()), c->get_stride());
        } else {
            int num_elems = ceildiv(nnz, nwarps * hip_config::warp_size) *
                            hip_config::warp_size;
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block),
                                ceildiv(b_ncols, hip_config::warp_size));
            hipLaunchKernelGGL(
                (abstract_spmm), dim3(coo_grid), dim3(coo_block), 0, 0, nnz,
                num_elems, as_hip_type(a->get_const_values()),
                a->get_const_col_idxs(), as_hip_type(a->get_const_row_idxs()),
                b_ncols, as_hip_type(b->get_const_values()), b->get_stride(),
                as_hip_type(c->get_values()), c->get_stride());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Dense<ValueType> *alpha,
                    const matrix::Coo<ValueType, IndexType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c)
{
    const auto nnz = a->get_num_stored_elements();
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);
    const dim3 coo_block(hip_config::warp_size, warps_in_block, 1);
    const auto b_ncols = b->get_size()[1];

    if (nwarps > 0) {
        if (b_ncols < 4) {
            int num_lines = ceildiv(nnz, nwarps * hip_config::warp_size);
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b_ncols);
            hipLaunchKernelGGL(
                (abstract_spmv), dim3(coo_grid), dim3(coo_block), 0, 0, nnz,
                num_lines, as_hip_type(alpha->get_const_values()),
                as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
                as_hip_type(a->get_const_row_idxs()),
                as_hip_type(b->get_const_values()), b->get_stride(),
                as_hip_type(c->get_values()), c->get_stride());
        } else {
            int num_elems = ceildiv(nnz, nwarps * hip_config::warp_size) *
                            hip_config::warp_size;
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block),
                                ceildiv(b_ncols, hip_config::warp_size));
            hipLaunchKernelGGL(
                (abstract_spmm), dim3(coo_grid), dim3(coo_block), 0, 0, nnz,
                num_elems, as_hip_type(alpha->get_const_values()),
                as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
                as_hip_type(a->get_const_row_idxs()), b_ncols,
                as_hip_type(b->get_const_values()), b->get_stride(),
                as_hip_type(c->get_values()), c->get_stride());
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
void convert_row_idxs_to_ptrs(std::shared_ptr<const HipExecutor> exec,
                              const IndexType *idxs, size_type num_nonzeros,
                              IndexType *ptrs, size_type length)
{
    const auto grid_dim = ceildiv(num_nonzeros, default_block_size);

    hipLaunchKernelGGL((kernel::convert_row_idxs_to_ptrs), dim3(grid_dim),
                       dim3(default_block_size), 0, 0, as_hip_type(idxs),
                       num_nonzeros, as_hip_type(ptrs), length);
}


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const HipExecutor> exec,
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
__global__ __launch_bounds__(default_block_size) void initialize_zero_dense(
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
void convert_to_dense(std::shared_ptr<const HipExecutor> exec,
                      matrix::Dense<ValueType> *result,
                      const matrix::Coo<ValueType, IndexType> *source)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = result->get_stride();

    const auto nnz = source->get_num_stored_elements();

    const dim3 block_size(hip_config::warp_size,
                          hip_config::max_block_size / hip_config::warp_size,
                          1);
    const dim3 init_grid_dim(ceildiv(stride, block_size.x),
                             ceildiv(num_rows, block_size.y), 1);
    hipLaunchKernelGGL((kernel::initialize_zero_dense), dim3(init_grid_dim),
                       dim3(block_size), 0, 0, num_rows, num_cols, stride,
                       as_hip_type(result->get_values()));

    const auto grid_dim = ceildiv(nnz, default_block_size);
    hipLaunchKernelGGL((kernel::fill_in_dense), dim3(grid_dim),
                       dim3(default_block_size), 0, 0, nnz,
                       as_hip_type(source->get_const_row_idxs()),
                       as_hip_type(source->get_const_col_idxs()),
                       as_hip_type(source->get_const_values()), stride,
                       as_hip_type(result->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_DENSE_KERNEL);


}  // namespace coo
}  // namespace hip
}  // namespace kernels
}  // namespace gko
