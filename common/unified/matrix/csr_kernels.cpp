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

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Csr matrix format namespace.
 *
 * @ingroup csr
 */
namespace csr {


template <typename IndexType>
void invert_permutation(std::shared_ptr<const DefaultExecutor> exec,
                        size_type size, const IndexType* permutation_indices,
                        IndexType* inv_permutation)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tid, auto permutation, auto inv_permutation) {
            inv_permutation[permutation[tid]] = tid;
        },
        size, permutation_indices, inv_permutation);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INVERT_PERMUTATION_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_column_permute(std::shared_ptr<const DefaultExecutor> exec,
                            const IndexType* perm,
                            const matrix::Csr<ValueType, IndexType>* orig,
                            matrix::Csr<ValueType, IndexType>* column_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto nnz = orig->get_num_stored_elements();
    auto size = std::max(num_rows, nnz);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tid, auto num_rows, auto num_nonzeros,
                      auto permutation, auto in_row_ptrs, auto in_col_idxs,
                      auto in_vals, auto out_row_ptrs, auto out_col_idxs,
                      auto out_vals) {
            if (tid < num_nonzeros) {
                out_col_idxs[tid] = permutation[in_col_idxs[tid]];
                out_vals[tid] = in_vals[tid];
            }
            if (tid <= num_rows) {
                out_row_ptrs[tid] = in_row_ptrs[tid];
            }
        },
        size, num_rows, nnz, perm, orig->get_const_row_ptrs(),
        orig->get_const_col_idxs(), orig->get_const_values(),
        column_permuted->get_row_ptrs(), column_permuted->get_col_idxs(),
        column_permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::Dense<ValueType>* alpha,
           matrix::Csr<ValueType, IndexType>* x)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto nnz, auto alpha, auto x) { x[nnz] *= alpha[0]; },
        x->get_num_stored_elements(), alpha->get_const_values(),
        x->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SCALE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_scale(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Dense<ValueType>* alpha,
               matrix::Csr<ValueType, IndexType>* x)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto nnz, auto alpha, auto x) { x[nnz] /= alpha[0]; },
        x->get_num_stored_elements(), alpha->get_const_values(),
        x->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_INV_SCALE_KERNEL);


}  // namespace csr
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
