/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/matrix/sellp_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/prefix_sum.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The SELL-P matrix format namespace.
 *
 * @ingroup sellp
 */
namespace sellp {


constexpr auto default_block_size = 256;


namespace {


template <typename ValueType, typename IndexType>
void spmv_kernel(size_type num_rows, size_type num_right_hand_sides,
                 size_type b_stride, size_type c_stride,
                 const size_type *__restrict__ slice_lengths,
                 const size_type *__restrict__ slice_sets,
                 const ValueType *__restrict__ a,
                 const IndexType *__restrict__ col,
                 const ValueType *__restrict__ b, ValueType *__restrict__ c,
                 sycl::nd_item<3> item_ct1)
{
    const auto slice_id = item_ct1.get_group(2);
    const auto slice_size = item_ct1.get_local_range().get(2);
    const auto row_in_slice = item_ct1.get_local_id(2);
    const auto global_row =
        static_cast<size_type>(slice_size) * slice_id + row_in_slice;
    const auto column_id = item_ct1.get_group(1);
    ValueType val = 0;
    IndexType ind = 0;
    if (global_row < num_rows && column_id < num_right_hand_sides) {
        for (size_type i = 0; i < slice_lengths[slice_id]; i++) {
            ind = row_in_slice + (slice_sets[slice_id] + i) * slice_size;
            val += a[ind] * b[col[ind] * b_stride + column_id];
        }
        c[global_row * c_stride + column_id] = val;
    }
}

template <typename ValueType, typename IndexType>
void spmv_kernel(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                 sycl::queue *stream, size_type num_rows,
                 size_type num_right_hand_sides, size_type b_stride,
                 size_type c_stride, const size_type *slice_lengths,
                 const size_type *slice_sets, const ValueType *a,
                 const IndexType *col, const ValueType *b, ValueType *c)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                spmv_kernel(num_rows, num_right_hand_sides, b_stride, c_stride,
                            slice_lengths, slice_sets, a, col, b, c, item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void advanced_spmv_kernel(size_type num_rows, size_type num_right_hand_sides,
                          size_type b_stride, size_type c_stride,
                          const size_type *__restrict__ slice_lengths,
                          const size_type *__restrict__ slice_sets,
                          const ValueType *__restrict__ alpha,
                          const ValueType *__restrict__ a,
                          const IndexType *__restrict__ col,
                          const ValueType *__restrict__ b,
                          const ValueType *__restrict__ beta,
                          ValueType *__restrict__ c, sycl::nd_item<3> item_ct1)
{
    const auto slice_id = item_ct1.get_group(2);
    const auto slice_size = item_ct1.get_local_range().get(2);
    const auto row_in_slice = item_ct1.get_local_id(2);
    const auto global_row =
        static_cast<size_type>(slice_size) * slice_id + row_in_slice;
    const auto column_id = item_ct1.get_group(1);
    ValueType val = 0;
    IndexType ind = 0;
    if (global_row < num_rows && column_id < num_right_hand_sides) {
        for (size_type i = 0; i < slice_lengths[slice_id]; i++) {
            ind = row_in_slice + (slice_sets[slice_id] + i) * slice_size;
            val += alpha[0] * a[ind] * b[col[ind] * b_stride + column_id];
        }
        c[global_row * c_stride + column_id] =
            beta[0] * c[global_row * c_stride + column_id] + val;
    }
}

template <typename ValueType, typename IndexType>
void advanced_spmv_kernel(dim3 grid, dim3 block,
                          gko::size_type dynamic_shared_memory,
                          sycl::queue *stream, size_type num_rows,
                          size_type num_right_hand_sides, size_type b_stride,
                          size_type c_stride, const size_type *slice_lengths,
                          const size_type *slice_sets, const ValueType *alpha,
                          const ValueType *a, const IndexType *col,
                          const ValueType *b, const ValueType *beta,
                          ValueType *c)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                advanced_spmv_kernel(num_rows, num_right_hand_sides, b_stride,
                                     c_stride, slice_lengths, slice_sets, alpha,
                                     a, col, b, beta, c, item_ct1);
            });
    });
}


}  // namespace


namespace kernel {


template <typename ValueType>
void initialize_zero_dense(size_type num_rows, size_type num_cols,
                           size_type stride, ValueType *__restrict__ result,
                           sycl::nd_item<3> item_ct1)
{
    const auto tidx_x =
        item_ct1.get_local_id(2) +
        item_ct1.get_local_range().get(2) * item_ct1.get_group(2);
    const auto tidx_y =
        item_ct1.get_local_id(1) +
        item_ct1.get_local_range().get(1) * item_ct1.get_group(1);
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}

template <typename ValueType>
void initialize_zero_dense(dim3 grid, dim3 block,
                           gko::size_type dynamic_shared_memory,
                           sycl::queue *stream, size_type num_rows,
                           size_type num_cols, size_type stride,
                           ValueType *result)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             initialize_zero_dense(num_rows, num_cols, stride,
                                                   result, item_ct1);
                         });
    });
}


template <unsigned int threads_per_row, typename ValueType, typename IndexType>
void fill_in_dense(size_type num_rows, size_type num_cols, size_type stride,
                   size_type slice_size,
                   const size_type *__restrict__ slice_lengths,
                   const size_type *__restrict__ slice_sets,
                   const IndexType *__restrict__ col_idxs,
                   const ValueType *__restrict__ values,
                   ValueType *__restrict__ result, sycl::nd_item<3> item_ct1)
{
    const auto global_row =
        thread::get_subwarp_id_flat<threads_per_row>(item_ct1);
    const auto row = global_row % slice_size;
    const auto slice = global_row / slice_size;
    const auto start_index = item_ct1.get_local_id(2) % threads_per_row;

    if (global_row < num_rows) {
        for (auto i = start_index; i < slice_lengths[slice];
             i += threads_per_row) {
            if (values[(slice_sets[slice] + i) * slice_size + row] !=
                zero<ValueType>()) {
                result[global_row * stride +
                       col_idxs[(slice_sets[slice] + i) * slice_size + row]] =
                    values[(slice_sets[slice] + i) * slice_size + row];
            }
        }
    }
}

template <unsigned int threads_per_row, typename ValueType, typename IndexType>
void fill_in_dense(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                   sycl::queue *stream, size_type num_rows, size_type num_cols,
                   size_type stride, size_type slice_size,
                   const size_type *slice_lengths, const size_type *slice_sets,
                   const IndexType *col_idxs, const ValueType *values,
                   ValueType *result)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                fill_in_dense<threads_per_row>(
                    num_rows, num_cols, stride, slice_size, slice_lengths,
                    slice_sets, col_idxs, values, result, item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void count_nnz_per_row(size_type num_rows, size_type slice_size,
                       const size_type *__restrict__ slice_sets,
                       const ValueType *__restrict__ values,
                       IndexType *__restrict__ result,
                       sycl::nd_item<3> item_ct1)
{
    constexpr auto warp_size = config::warp_size;
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block(item_ct1));
    const auto row_idx = thread::get_subwarp_id_flat<warp_size>(item_ct1);
    const auto slice_id = row_idx / slice_size;
    const auto tid_in_warp = warp_tile.thread_rank();
    const auto row_in_slice = row_idx % slice_size;

    if (row_idx < num_rows) {
        IndexType part_result{};
        for (size_type sellp_ind =
                 (slice_sets[slice_id] + tid_in_warp) * slice_size +
                 row_in_slice;
             sellp_ind < slice_sets[slice_id + 1] * slice_size;
             sellp_ind += warp_size * slice_size) {
            if (values[sellp_ind] != zero<ValueType>()) {
                part_result += 1;
            }
        }
        result[row_idx] = ::gko::kernels::dpcpp::reduce(
            warp_tile, part_result,
            [](const size_type &a, const size_type &b) { return a + b; });
    }
}

template <typename ValueType, typename IndexType>
void count_nnz_per_row(dim3 grid, dim3 block,
                       gko::size_type dynamic_shared_memory,
                       sycl::queue *stream, size_type num_rows,
                       size_type slice_size, const size_type *slice_sets,
                       const ValueType *values, IndexType *result)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             count_nnz_per_row(num_rows, slice_size, slice_sets,
                                               values, result, item_ct1);
                         });
    });
}


template <typename ValueType, typename IndexType>
void fill_in_csr(size_type num_rows, size_type slice_size,
                 const size_type *__restrict__ source_slice_sets,
                 const IndexType *__restrict__ source_col_idxs,
                 const ValueType *__restrict__ source_values,
                 IndexType *__restrict__ result_row_ptrs,
                 IndexType *__restrict__ result_col_idxs,
                 ValueType *__restrict__ result_values,
                 sycl::nd_item<3> item_ct1)
{
    const auto row = thread::get_thread_id_flat(item_ct1);
    const auto slice_id = row / slice_size;
    const auto row_in_slice = row % slice_size;

    if (row < num_rows) {
        size_type csr_ind = result_row_ptrs[row];
        for (size_type sellp_ind =
                 source_slice_sets[slice_id] * slice_size + row_in_slice;
             sellp_ind < source_slice_sets[slice_id + 1] * slice_size;
             sellp_ind += slice_size) {
            if (source_values[sellp_ind] != zero<ValueType>()) {
                result_values[csr_ind] = source_values[sellp_ind];
                result_col_idxs[csr_ind] = source_col_idxs[sellp_ind];
                csr_ind++;
            }
        }
    }
}

template <typename ValueType, typename IndexType>
void fill_in_csr(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                 sycl::queue *stream, size_type num_rows, size_type slice_size,
                 const size_type *source_slice_sets,
                 const IndexType *source_col_idxs,
                 const ValueType *source_values, IndexType *result_row_ptrs,
                 IndexType *result_col_idxs, ValueType *result_values)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                fill_in_csr(num_rows, slice_size, source_slice_sets,
                            source_col_idxs, source_values, result_row_ptrs,
                            result_col_idxs, result_values, item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void extract_diagonal(size_type diag_size, size_type slice_size,
                      const size_type *__restrict__ orig_slice_sets,
                      const ValueType *__restrict__ orig_values,
                      const IndexType *__restrict__ orig_col_idxs,
                      ValueType *__restrict__ diag, sycl::nd_item<3> item_ct1)
{
    constexpr auto warp_size = config::warp_size;
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block(item_ct1));
    const auto slice_id = thread::get_subwarp_id_flat<warp_size>(item_ct1);
    const auto tid_in_warp = warp_tile.thread_rank();
    const auto slice_num = ceildiv(diag_size, slice_size);

    if (slice_id >= slice_num) {
        return;
    }

    const auto start_ind = orig_slice_sets[slice_id] * slice_size + tid_in_warp;
    const auto end_ind = orig_slice_sets[slice_id + 1] * slice_size;

    for (auto sellp_ind = start_ind; sellp_ind < end_ind;
         sellp_ind += warp_size) {
        auto global_row = slice_id * slice_size + sellp_ind % slice_size;
        if (global_row < diag_size) {
            if (orig_col_idxs[sellp_ind] == global_row &&
                orig_values[sellp_ind] != zero<ValueType>()) {
                diag[global_row] = orig_values[sellp_ind];
            }
        }
    }
}

template <typename ValueType, typename IndexType>
void extract_diagonal(dim3 grid, dim3 block,
                      gko::size_type dynamic_shared_memory, sycl::queue *stream,
                      size_type diag_size, size_type slice_size,
                      const size_type *orig_slice_sets,
                      const ValueType *orig_values,
                      const IndexType *orig_col_idxs, ValueType *diag)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                extract_diagonal(diag_size, slice_size, orig_slice_sets,
                                 orig_values, orig_col_idxs, diag, item_ct1);
            });
    });
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::Sellp<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    const dim3 blockSize(matrix::default_slice_size);
    const dim3 gridSize(ceildiv(a->get_size()[0], matrix::default_slice_size),
                        b->get_size()[1]);

    spmv_kernel(gridSize, blockSize, 0, exec->get_queue(), a->get_size()[0],
                b->get_size()[1], b->get_stride(), c->get_stride(),
                a->get_const_slice_lengths(), a->get_const_slice_sets(),
                a->get_const_values(), a->get_const_col_idxs(),
                b->get_const_values(), c->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Sellp<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    const dim3 blockSize(matrix::default_slice_size);
    const dim3 gridSize(ceildiv(a->get_size()[0], matrix::default_slice_size),
                        b->get_size()[1]);

    advanced_spmv_kernel(gridSize, blockSize, 0, exec->get_queue(),
                         a->get_size()[0], b->get_size()[1], b->get_stride(),
                         c->get_stride(), a->get_const_slice_lengths(),
                         a->get_const_slice_sets(), alpha->get_const_values(),
                         a->get_const_values(), a->get_const_col_idxs(),
                         b->get_const_values(), beta->get_const_values(),
                         c->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Sellp<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result)
{
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];
    const auto vals = source->get_const_values();
    const auto col_idxs = source->get_const_col_idxs();
    const auto slice_lengths = source->get_const_slice_lengths();
    const auto slice_sets = source->get_const_slice_sets();
    const auto slice_size = source->get_slice_size();

    const auto slice_num = ceildiv(num_rows, slice_size);

    const dim3 block_size(config::warp_size,
                          config::max_block_size / config::warp_size, 1);
    const dim3 init_grid_dim(ceildiv(num_cols, block_size.x),
                             ceildiv(num_rows, block_size.y), 1);

    if (num_rows > 0 && result->get_stride() > 0) {
        kernel::initialize_zero_dense(
            init_grid_dim, block_size, 0, exec->get_queue(), num_rows, num_cols,
            result->get_stride(), result->get_values());
    }

    constexpr auto threads_per_row = config::warp_size;
    const auto grid_dim =
        ceildiv(slice_size * slice_num * threads_per_row, default_block_size);

    if (grid_dim > 0) {
        kernel::fill_in_dense<threads_per_row>(
            grid_dim, default_block_size, 0, exec->get_queue(), num_rows,
            num_cols, result->get_stride(), slice_size, slice_lengths,
            slice_sets, col_idxs, vals, result->get_values());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Sellp<ValueType, IndexType> *source,
                    matrix::Csr<ValueType, IndexType> *result)
{
    const auto num_rows = source->get_size()[0];
    const auto slice_size = source->get_slice_size();
    const auto slice_num = ceildiv(num_rows, slice_size);

    const auto source_values = source->get_const_values();
    const auto source_slice_lengths = source->get_const_slice_lengths();
    const auto source_slice_sets = source->get_const_slice_sets();
    const auto source_col_idxs = source->get_const_col_idxs();

    auto result_values = result->get_values();
    auto result_col_idxs = result->get_col_idxs();
    auto result_row_ptrs = result->get_row_ptrs();

    auto grid_dim = ceildiv(num_rows * config::warp_size, default_block_size);

    if (grid_dim > 0) {
        kernel::count_nnz_per_row(
            grid_dim, default_block_size, 0, exec->get_queue(), num_rows,
            slice_size, source_slice_sets, source_values, result_row_ptrs);
    }

    grid_dim = ceildiv(num_rows + 1, default_block_size);
    auto add_values = Array<IndexType>(exec, grid_dim);

    components::prefix_sum(exec, result_row_ptrs, num_rows + 1);

    grid_dim = ceildiv(num_rows, default_block_size);

    if (grid_dim > 0) {
        kernel::fill_in_csr(grid_dim, default_block_size, 0, exec->get_queue(),
                            num_rows, slice_size, source_slice_sets,
                            source_col_idxs, source_values, result_row_ptrs,
                            result_col_idxs, result_values);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Sellp<ValueType, IndexType> *source,
                    size_type *result)
{
    const auto num_rows = source->get_size()[0];

    if (num_rows <= 0) {
        *result = 0;
        return;
    }

    const auto slice_size = source->get_slice_size();
    const auto slice_sets = source->get_const_slice_sets();
    const auto values = source->get_const_values();

    auto nnz_per_row = Array<size_type>(exec, num_rows);

    auto grid_dim = ceildiv(num_rows * config::warp_size, default_block_size);

    kernel::count_nnz_per_row(grid_dim, default_block_size, 0,
                              exec->get_queue(), num_rows, slice_size,
                              slice_sets, values, nnz_per_row.get_data());

    *result = reduce_add_array(exec, num_rows, nnz_per_row.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_COUNT_NONZEROS_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Sellp<ValueType, IndexType> *orig,
                      matrix::Diagonal<ValueType> *diag)
{
    const auto diag_size = diag->get_size()[0];
    const auto slice_size = orig->get_slice_size();
    const auto slice_num = ceildiv(diag_size, slice_size);
    const auto num_blocks =
        ceildiv(slice_num * config::warp_size, default_block_size);

    const auto orig_slice_sets = orig->get_const_slice_sets();
    const auto orig_values = orig->get_const_values();
    const auto orig_col_idxs = orig->get_const_col_idxs();
    auto diag_values = diag->get_values();

    kernel::extract_diagonal(
        num_blocks, default_block_size, 0, exec->get_queue(), diag_size,
        slice_size, orig_slice_sets, orig_values, orig_col_idxs, diag_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_EXTRACT_DIAGONAL_KERNEL);


}  // namespace sellp
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
