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

#include "core/matrix/diagonal_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Diagonal matrix format namespace.
 *
 * @ingroup diagonal
 */
namespace diagonal {


constexpr auto default_block_size = 256;


namespace kernel {


template <typename ValueType>
void apply_to_dense(size_type num_rows, size_type num_cols,
                    const ValueType *__restrict__ diag, size_type source_stride,
                    const ValueType *__restrict__ source_values,
                    size_type result_stride,
                    ValueType *__restrict__ result_values,
                    sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    const auto row = tidx / num_cols;
    const auto col = tidx % num_cols;

    if (row < num_rows) {
        result_values[row * result_stride + col] =
            source_values[row * source_stride + col] * diag[row];
    }
}

template <typename ValueType>
void apply_to_dense(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                    sycl::queue *stream, size_type num_rows, size_type num_cols,
                    const ValueType *diag, size_type source_stride,
                    const ValueType *source_values, size_type result_stride,
                    ValueType *result_values)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                apply_to_dense(num_rows, num_cols, diag, source_stride,
                               source_values, result_stride, result_values,
                               item_ct1);
            });
    });
}


template <typename ValueType>
void right_apply_to_dense(size_type num_rows, size_type num_cols,
                          const ValueType *__restrict__ diag,
                          size_type source_stride,
                          const ValueType *__restrict__ source_values,
                          size_type result_stride,
                          ValueType *__restrict__ result_values,
                          sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    const auto row = tidx / num_cols;
    const auto col = tidx % num_cols;

    if (row < num_rows) {
        result_values[row * result_stride + col] =
            source_values[row * source_stride + col] * diag[col];
    }
}

template <typename ValueType>
void right_apply_to_dense(dim3 grid, dim3 block,
                          gko::size_type dynamic_shared_memory,
                          sycl::queue *stream, size_type num_rows,
                          size_type num_cols, const ValueType *diag,
                          size_type source_stride,
                          const ValueType *source_values,
                          size_type result_stride, ValueType *result_values)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                right_apply_to_dense(num_rows, num_cols, diag, source_stride,
                                     source_values, result_stride,
                                     result_values, item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void apply_to_csr(size_type num_rows, const ValueType *__restrict__ diag,
                  const IndexType *__restrict__ row_ptrs,
                  ValueType *__restrict__ result_values,
                  sycl::nd_item<3> item_ct1)
{
    constexpr auto warp_size = config::warp_size;
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block(item_ct1));
    const auto row = thread::get_subwarp_id_flat<warp_size>(item_ct1);
    const auto tid_in_warp = warp_tile.thread_rank();

    if (row >= num_rows) {
        return;
    }

    const auto diag_val = diag[row];

    for (size_type idx = row_ptrs[row] + tid_in_warp; idx < row_ptrs[row + 1];
         idx += warp_size) {
        result_values[idx] *= diag_val;
    }
}

template <typename ValueType, typename IndexType>
void apply_to_csr(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                  sycl::queue *stream, size_type num_rows,
                  const ValueType *diag, const IndexType *row_ptrs,
                  ValueType *result_values)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                apply_to_csr(num_rows, diag, row_ptrs, result_values, item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void right_apply_to_csr(size_type num_nnz, const ValueType *__restrict__ diag,
                        const IndexType *__restrict__ col_idxs,
                        ValueType *__restrict__ result_values,
                        sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx >= num_nnz) {
        return;
    }

    result_values[tidx] *= diag[col_idxs[tidx]];
}

template <typename ValueType, typename IndexType>
void right_apply_to_csr(dim3 grid, dim3 block,
                        gko::size_type dynamic_shared_memory,
                        sycl::queue *stream, size_type num_nnz,
                        const ValueType *diag, const IndexType *col_idxs,
                        ValueType *result_values)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             right_apply_to_csr(num_nnz, diag, col_idxs,
                                                result_values, item_ct1);
                         });
    });
}


template <typename ValueType, typename IndexType>
void convert_to_csr(size_type size, const ValueType *__restrict__ diag_values,
                    IndexType *__restrict__ row_ptrs,
                    IndexType *__restrict__ col_idxs,
                    ValueType *__restrict__ csr_values,
                    sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx >= size) {
        return;
    }
    if (tidx == 0) {
        row_ptrs[size] = size;
    }

    row_ptrs[tidx] = tidx;
    col_idxs[tidx] = tidx;
    csr_values[tidx] = diag_values[tidx];
}

template <typename ValueType, typename IndexType>
void convert_to_csr(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                    sycl::queue *stream, size_type size,
                    const ValueType *diag_values, IndexType *row_ptrs,
                    IndexType *col_idxs, ValueType *csr_values)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             convert_to_csr(size, diag_values, row_ptrs,
                                            col_idxs, csr_values, item_ct1);
                         });
    });
}


template <typename ValueType>
void conj_transpose(size_type size, const ValueType *__restrict__ orig_values,
                    ValueType *__restrict__ trans_values,
                    sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx >= size) {
        return;
    }

    trans_values[tidx] = conj(orig_values[tidx]);
}

template <typename ValueType>
void conj_transpose(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                    sycl::queue *stream, size_type size,
                    const ValueType *orig_values, ValueType *trans_values)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                conj_transpose(size, orig_values, trans_values, item_ct1);
            });
    });
}


}  // namespace kernel


template <typename ValueType>
void apply_to_dense(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Diagonal<ValueType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c)
{
    const auto b_size = b->get_size();
    const auto num_rows = b_size[0];
    const auto num_cols = b_size[1];
    const auto b_stride = b->get_stride();
    const auto c_stride = c->get_stride();
    const auto grid_dim = ceildiv(num_rows * num_cols, default_block_size);

    const auto diag_values = a->get_const_values();
    const auto b_values = b->get_const_values();
    auto c_values = c->get_values();

    kernel::apply_to_dense(grid_dim, default_block_size, 0, exec->get_queue(),
                           num_rows, num_cols, diag_values, b_stride, b_values,
                           c_stride, c_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_APPLY_TO_DENSE_KERNEL);


template <typename ValueType>
void right_apply_to_dense(std::shared_ptr<const DpcppExecutor> exec,
                          const matrix::Diagonal<ValueType> *a,
                          const matrix::Dense<ValueType> *b,
                          matrix::Dense<ValueType> *c)
{
    const auto b_size = b->get_size();
    const auto num_rows = b_size[0];
    const auto num_cols = b_size[1];
    const auto b_stride = b->get_stride();
    const auto c_stride = c->get_stride();
    const auto grid_dim = ceildiv(num_rows * num_cols, default_block_size);

    const auto diag_values = a->get_const_values();
    const auto b_values = b->get_const_values();
    auto c_values = c->get_values();

    kernel::right_apply_to_dense(
        grid_dim, default_block_size, 0, exec->get_queue(), num_rows, num_cols,
        diag_values, b_stride, b_values, c_stride, c_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void apply_to_csr(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Diagonal<ValueType> *a,
                  const matrix::Csr<ValueType, IndexType> *b,
                  matrix::Csr<ValueType, IndexType> *c)
{
    const auto num_rows = b->get_size()[0];
    const auto diag_values = a->get_const_values();
    c->copy_from(b);
    auto csr_values = c->get_values();
    const auto csr_row_ptrs = c->get_const_row_ptrs();

    const auto grid_dim =
        ceildiv(num_rows * config::warp_size, default_block_size);
    kernel::apply_to_csr(grid_dim, default_block_size, 0, exec->get_queue(),
                         num_rows, diag_values, csr_row_ptrs, csr_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_APPLY_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void right_apply_to_csr(std::shared_ptr<const DpcppExecutor> exec,
                        const matrix::Diagonal<ValueType> *a,
                        const matrix::Csr<ValueType, IndexType> *b,
                        matrix::Csr<ValueType, IndexType> *c)
{
    const auto num_nnz = b->get_num_stored_elements();
    const auto diag_values = a->get_const_values();
    c->copy_from(b);
    auto csr_values = c->get_values();
    const auto csr_col_idxs = c->get_const_col_idxs();

    const auto grid_dim = ceildiv(num_nnz, default_block_size);
    kernel::right_apply_to_csr(grid_dim, default_block_size, 0,
                               exec->get_queue(), num_nnz, diag_values,
                               csr_col_idxs, csr_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Diagonal<ValueType> *source,
                    matrix::Csr<ValueType, IndexType> *result)
{
    const auto size = source->get_size()[0];
    const auto grid_dim = ceildiv(size, default_block_size);

    const auto diag_values = source->get_const_values();
    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto csr_values = result->get_values();

    kernel::convert_to_csr(grid_dim, default_block_size, 0, exec->get_queue(),
                           size, diag_values, row_ptrs, col_idxs, csr_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_CONVERT_TO_CSR_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Diagonal<ValueType> *orig,
                    matrix::Diagonal<ValueType> *trans)
{
    const auto size = orig->get_size()[0];
    const auto grid_dim = ceildiv(size, default_block_size);
    const auto orig_values = orig->get_const_values();
    auto trans_values = trans->get_values();

    kernel::conj_transpose(grid_dim, default_block_size, 0, exec->get_queue(),
                           size, orig_values, trans_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_CONJ_TRANSPOSE_KERNEL);


}  // namespace diagonal
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
