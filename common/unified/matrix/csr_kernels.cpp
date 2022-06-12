/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* matrix,
                      matrix::Sellp<ValueType, IndexType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto in_cols, auto in_values, auto row_ptrs,
                      auto slice_size, auto slice_sets, auto cols,
                      auto values) {
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            const auto slice = row / slice_size;
            const auto local_row = row % slice_size;
            const auto slice_begin = slice_sets[slice];
            const auto slice_end = slice_sets[slice + 1];
            const auto slice_length = slice_end - slice_begin;
            auto out_idx = slice_begin * slice_size + local_row;
            for (auto i = row_begin; i < row_begin + slice_length; i++) {
                cols[out_idx] =
                    i < row_end ? in_cols[i] : invalid_index<IndexType>();
                values[out_idx] = i < row_end ? unpack_member(in_values[i])
                                              : zero(values[out_idx]);
                out_idx += slice_size;
            }
        },
        output->get_size()[0], matrix->get_const_col_idxs(),
        matrix->get_const_values(), matrix->get_const_row_ptrs(),
        output->get_slice_size(), output->get_const_slice_sets(),
        output->get_col_idxs(), output->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* matrix,
                    matrix::Ell<ValueType, IndexType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto in_cols, auto in_values, auto row_ptrs,
                      auto ell_length, auto ell_stride, auto cols,
                      auto values) {
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            auto out_idx = row;
            for (auto i = row_begin; i < row_begin + ell_length; i++) {
                cols[out_idx] =
                    i < row_end ? in_cols[i] : invalid_index<IndexType>();
                values[out_idx] = i < row_end ? unpack_member(in_values[i])
                                              : zero(values[out_idx]);
                out_idx += ell_stride;
            }
        },
        output->get_size()[0], matrix->get_const_col_idxs(),
        matrix->get_const_values(), matrix->get_const_row_ptrs(),
        output->get_num_stored_elements_per_row(), output->get_stride(),
        output->get_col_idxs(), output->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Csr<ValueType, IndexType>* source,
                       const int64* coo_row_ptrs,
                       matrix::Hybrid<ValueType, IndexType>* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto cols, auto vals,
                      auto ell_stride, auto ell_max_nnz, auto ell_cols,
                      auto ell_vals, auto coo_row_ptrs, auto coo_row_idxs,
                      auto coo_col_idxs, auto coo_vals) {
            const auto row_begin = row_ptrs[row];
            const auto row_size = row_ptrs[row + 1] - row_begin;
            for (int64 i = 0; i < ell_max_nnz; i++) {
                const auto out_idx = row + ell_stride * i;
                const auto in_idx = i + row_begin;
                const bool use = i < row_size;
                ell_cols[out_idx] =
                    use ? cols[in_idx] : invalid_index<IndexType>();
                ell_vals[out_idx] = use ? vals[in_idx] : zero(vals[in_idx]);
            }
            const auto coo_begin = coo_row_ptrs[row];
            for (int64 i = ell_max_nnz; i < row_size; i++) {
                const auto in_idx = i + row_begin;
                const auto out_idx =
                    coo_begin + i - static_cast<int64>(ell_max_nnz);
                coo_row_idxs[out_idx] = row;
                coo_col_idxs[out_idx] = cols[in_idx];
                coo_vals[out_idx] = vals[in_idx];
            }
        },
        source->get_size()[0], source->get_const_row_ptrs(),
        source->get_const_col_idxs(), source->get_const_values(),
        result->get_ell_stride(), result->get_ell_num_stored_elements_per_row(),
        result->get_ell_col_idxs(), result->get_ell_values(), coo_row_ptrs,
        result->get_coo_row_idxs(), result->get_coo_col_idxs(),
        result->get_coo_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL);


}  // namespace csr
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
