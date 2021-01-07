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


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Diagonal matrix format namespace.
 *
 * @ingroup diagonal
 */
namespace diagonal {


constexpr auto default_block_size = 512;


#include "common/matrix/diagonal_kernels.hpp.inc"


template <typename ValueType>
void apply_to_dense(std::shared_ptr<const CudaExecutor> exec,
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

    kernel::apply_to_dense<<<grid_dim, default_block_size>>>(
        num_rows, num_cols, as_cuda_type(diag_values), b_stride,
        as_cuda_type(b_values), c_stride, as_cuda_type(c_values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_APPLY_TO_DENSE_KERNEL);


template <typename ValueType>
void right_apply_to_dense(std::shared_ptr<const CudaExecutor> exec,
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

    kernel::right_apply_to_dense<<<grid_dim, default_block_size>>>(
        num_rows, num_cols, as_cuda_type(diag_values), b_stride,
        as_cuda_type(b_values), c_stride, as_cuda_type(c_values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void apply_to_csr(std::shared_ptr<const CudaExecutor> exec,
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
    kernel::apply_to_csr<<<grid_dim, default_block_size>>>(
        num_rows, as_cuda_type(diag_values), as_cuda_type(csr_row_ptrs),
        as_cuda_type(csr_values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_APPLY_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void right_apply_to_csr(std::shared_ptr<const CudaExecutor> exec,
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
    kernel::right_apply_to_csr<<<grid_dim, default_block_size>>>(
        num_nnz, as_cuda_type(diag_values), as_cuda_type(csr_col_idxs),
        as_cuda_type(csr_values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Diagonal<ValueType> *source,
                    matrix::Csr<ValueType, IndexType> *result)
{
    const auto size = source->get_size()[0];
    const auto grid_dim = ceildiv(size, default_block_size);

    const auto diag_values = source->get_const_values();
    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto csr_values = result->get_values();

    kernel::convert_to_csr<<<grid_dim, default_block_size>>>(
        size, as_cuda_type(diag_values), as_cuda_type(row_ptrs),
        as_cuda_type(col_idxs), as_cuda_type(csr_values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_CONVERT_TO_CSR_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Diagonal<ValueType> *orig,
                    matrix::Diagonal<ValueType> *trans)
{
    const auto size = orig->get_size()[0];
    const auto grid_dim = ceildiv(size, default_block_size);
    const auto orig_values = orig->get_const_values();
    auto trans_values = trans->get_values();

    kernel::conj_transpose<<<grid_dim, default_block_size>>>(
        size, as_cuda_type(orig_values), as_cuda_type(trans_values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_CONJ_TRANSPOSE_KERNEL);


}  // namespace diagonal
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
