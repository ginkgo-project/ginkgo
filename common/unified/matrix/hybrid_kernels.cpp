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

#include "core/matrix/hybrid_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Hybrid matrix format namespace.
 *
 * @ingroup hybrid
 */
namespace hybrid {


void compute_row_nnz(std::shared_ptr<const DefaultExecutor> exec,
                     const Array<int64>& row_ptrs, size_type* row_nnzs)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto row_ptrs, auto row_nnzs) {
            row_nnzs[i] = row_ptrs[i + 1] - row_ptrs[i];
        },
        row_ptrs.get_num_elems() - 1, row_ptrs, row_nnzs);
}


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Hybrid<ValueType, IndexType>* source,
                    const IndexType* ell_row_ptrs,
                    const IndexType* coo_row_ptrs,
                    matrix::Csr<ValueType, IndexType>* result)
{
    const auto ell = source->get_ell();
    const auto coo = source->get_coo();
    // ELL is stored in column-major, so we swap row and column parameters
    run_kernel(
        exec,
        [] GKO_KERNEL(auto ell_col, auto row, auto ell_stride, auto in_cols,
                      auto in_vals, auto ell_row_ptrs, auto coo_row_ptrs,
                      auto out_cols, auto out_vals) {
            const auto ell_idx = ell_col * ell_stride + row;
            const auto out_row_begin = ell_row_ptrs[row] + coo_row_ptrs[row];
            const auto ell_row_size = ell_row_ptrs[row + 1] - ell_row_ptrs[row];
            if (ell_col < ell_row_size) {
                const auto out_idx = out_row_begin + ell_col;
                out_cols[out_idx] = in_cols[ell_idx];
                out_vals[out_idx] = in_vals[ell_idx];
            }
        },
        dim<2>{ell->get_num_stored_elements_per_row(), ell->get_size()[0]},
        static_cast<int64>(ell->get_stride()), ell->get_const_col_idxs(),
        ell->get_const_values(), ell_row_ptrs, coo_row_ptrs,
        result->get_col_idxs(), result->get_values());
    run_kernel(
        exec,
        [] GKO_KERNEL(auto idx, auto ell_row_ptrs, auto coo_row_ptrs,
                      auto out_row_ptrs) {
            out_row_ptrs[idx] = ell_row_ptrs[idx] + coo_row_ptrs[idx];
        },
        source->get_size()[0] + 1, ell_row_ptrs, coo_row_ptrs,
        result->get_row_ptrs());
    run_kernel(
        exec,
        [] GKO_KERNEL(auto idx, auto in_rows, auto in_cols, auto in_vals,
                      auto ell_row_ptrs, auto coo_row_ptrs, auto out_cols,
                      auto out_vals) {
            const auto row = in_rows[idx];
            const auto col = in_cols[idx];
            const auto val = in_vals[idx];
            const auto coo_row_begin = coo_row_ptrs[row];
            const auto coo_local_pos = idx - coo_row_begin;
            // compute row_ptrs[row] + ell_row_size[row]
            const auto out_row_begin = ell_row_ptrs[row + 1] + coo_row_begin;
            const auto out_idx = out_row_begin + coo_local_pos;
            out_cols[out_idx] = col;
            out_vals[out_idx] = val;
        },
        coo->get_num_stored_elements(), coo->get_const_row_idxs(),
        coo->get_const_col_idxs(), coo->get_const_values(), ell_row_ptrs,
        coo_row_ptrs, result->get_col_idxs(), result->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_CONVERT_TO_CSR_KERNEL);


}  // namespace hybrid
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
