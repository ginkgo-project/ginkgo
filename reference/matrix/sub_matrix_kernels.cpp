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

#include "core/matrix/sub_matrix_kernels.hpp"


#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Compressed sparse row matrix format namespace.
 * @ref Sub_Matrix
 * @ingroup sub_matrix
 */
namespace sub_matrix {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::SubMatrix<matrix::Csr<ValueType, IndexType>> *sub_mat,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c,
          const OverlapMask &write_mask)
{
    // auto a = sub_mat->get_sub_matrix();
    // auto overlaps = sub_mat->get_overlap_mtxs();
    // auto num_overlaps = overlaps.size();
    // auto row_offset = sub_mat->get_overlap_sizes().get_data();
    // auto overlap_sizes = std::vector<int>(num_overlaps, 0);
    // auto left_ov_bound = sub_mat->get_left_overlap_bound();
    // bool fl = true;
    // for (int i = 1; i < num_overlaps; ++i) {
    //     overlap_sizes[i] = overlap_sizes[i - 1] + overlaps[i]->get_size()[1];
    //     if (i > left_ov_bound && fl) {
    //         overlap_sizes[i] += a->get_size()[1];
    //         fl = false;
    //     }
    // }
    // auto row_ptrs = a->get_const_row_ptrs();
    // auto col_idxs = a->get_const_col_idxs();
    // auto vals = a->get_const_values();

    // int s_row_offset = sub_mat->get_left_overlap_size();
    // for (size_type row = 0; row < sub_mat->get_size()[0]; ++row) {
    //     for (size_type j = 0; j < c->get_size()[1]; ++j) {
    //         c->at(row, j) = zero<ValueType>();
    //     }
    //     for (size_type k = row_ptrs[row];
    //          k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
    //         auto val = vals[k];
    //         auto col = col_idxs[k];
    //         for (size_type j = 0; j < c->get_size()[1]; ++j) {
    //             c->at(row, j) += val * b->at(s_row_offset + col, j);
    //         }
    //     }
    //     for (size_type n = 0; n < num_overlaps; ++n) {
    //         auto omat_row_ptrs = overlaps[n]->get_const_row_ptrs();
    //         auto omat_col_idxs = overlaps[n]->get_const_col_idxs();
    //         auto omat_vals = overlaps[n]->get_const_values();
    //         for (size_type k = omat_row_ptrs[row];
    //              k < static_cast<size_type>(omat_row_ptrs[row + 1]); ++k) {
    //             auto val = omat_vals[k];
    //             auto col = omat_col_idxs[k];
    //             for (size_type j = 0; j < c->get_size()[1]; ++j) {
    //                 c->at(row, j) += val * b->at(col + overlap_sizes[n], j);
    //             }
    //         }
    //     }
    // }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SUB_MATRIX_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Dense<ValueType> *alpha,
    const matrix::SubMatrix<matrix::Csr<ValueType, IndexType>> *sub_mat,
    const matrix::Dense<ValueType> *b, const matrix::Dense<ValueType> *beta,
    matrix::Dense<ValueType> *c, const OverlapMask &write_mask)
{
    // auto a = sub_mat->get_sub_matrix();
    // auto overlaps = sub_mat->get_overlap_mtxs();
    // auto num_overlaps = overlaps.size();
    // auto overlap_sizes = sub_mat->get_overlap_sizes();
    // auto row_ptrs = a->get_const_row_ptrs();
    // auto col_idxs = a->get_const_col_idxs();
    // auto vals = a->get_const_values();
    // auto valpha = alpha->at(0, 0);
    // auto vbeta = beta->at(0, 0);

    // int s_row_offset = sub_mat->get_left_overlap_size();
    // for (size_type row = 0; row < sub_mat->get_size()[0]; ++row) {
    //     for (size_type j = 0; j < c->get_size()[1]; ++j) {
    //         c->at(row, j) *= vbeta;
    //     }
    //     for (size_type k = row_ptrs[row];
    //          k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
    //         auto val = vals[k];
    //         auto col = col_idxs[k];
    //         for (size_type j = 0; j < c->get_size()[1]; ++j) {
    //             c->at(row, j) += valpha * val * b->at(s_row_offset + col, j);
    //         }
    //     }
    //     for (size_type n = 0; n < num_overlaps; ++n) {
    //         auto omat_row_ptrs = overlaps[n]->get_const_row_ptrs();
    //         auto omat_col_idxs = overlaps[n]->get_const_col_idxs();
    //         auto omat_vals = overlaps[n]->get_const_values();
    //         for (size_type k = omat_row_ptrs[row];
    //              k < static_cast<size_type>(omat_row_ptrs[row + 1]); ++k) {
    //             auto val = omat_vals[k];
    //             auto col = omat_col_idxs[k];
    //             for (size_type j = 0; j < c->get_size()[1]; ++j) {
    //                 c->at(row, j) +=
    //                     valpha * val *
    //                     b->at(col + overlap_sizes.get_data()[n], j);
    //             }
    //         }
    //     }
    // }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SUB_MATRIX_ADVANCED_SPMV_KERNEL);


}  // namespace sub_matrix
}  // namespace reference
}  // namespace kernels
}  // namespace gko
