/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The SELL-P matrix format namespace.
 * @ref Sellp
 * @ingroup sellp
 */
namespace sellp {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Sellp<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto col_idxs = a->get_const_col_idxs();
    auto slice_lengths = a->get_const_slice_lengths();
    auto slice_sets = a->get_const_slice_sets();
    auto slice_size = a->get_slice_size();
    auto slice_num = ceildiv(a->get_size()[0] + slice_size - 1, slice_size);
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= a->get_size()[0]) {
                break;
            }
            for (size_type j = 0; j < c->get_size()[1]; j++) {
                c->at(global_row, j) = zero<ValueType>();
            }
            for (size_type i = 0; i < slice_lengths[slice]; i++) {
                auto val = a->val_at(row, slice_sets[slice], i);
                auto col = a->col_at(row, slice_sets[slice], i);
                for (size_type j = 0; j < c->get_size()[1]; j++) {
                    c->at(global_row, j) += val * b->at(col, j);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Sellp<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    auto vals = a->get_const_values();
    auto col_idxs = a->get_const_col_idxs();
    auto slice_lengths = a->get_const_slice_lengths();
    auto slice_sets = a->get_const_slice_sets();
    auto slice_size = a->get_slice_size();
    auto slice_num = ceildiv(a->get_size()[0] + slice_size - 1, slice_size);
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= a->get_size()[0]) {
                break;
            }
            for (size_type j = 0; j < c->get_size()[1]; j++) {
                c->at(global_row, j) *= vbeta;
            }
            for (size_type i = 0; i < slice_lengths[slice]; i++) {
                auto val = a->val_at(row, slice_sets[slice], i);
                auto col = a->col_at(row, slice_sets[slice], i);
                for (size_type j = 0; j < c->get_size()[1]; j++) {
                    c->at(global_row, j) += valpha * val * b->at(col, j);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Sellp<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto vals = source->get_const_values();
    auto col_idxs = source->get_const_col_idxs();
    auto slice_lengths = source->get_const_slice_lengths();
    auto slice_sets = source->get_const_slice_sets();
    auto slice_size = source->get_slice_size();
    auto slice_num =
        ceildiv(source->get_size()[0] + slice_size - 1, slice_size);
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            for (size_type col = 0; col < num_cols; col++) {
                result->at(global_row, col) = zero<ValueType>();
            }
            for (size_type i = slice_sets[slice]; i < slice_sets[slice + 1];
                 i++) {
                result->at(global_row, col_idxs[row + i * slice_size]) +=
                    vals[row + i * slice_size];
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Sellp<ValueType, IndexType> *source,
                    matrix::Csr<ValueType, IndexType> *result)
{
    auto num_rows = source->get_size()[0];
    auto slice_size = source->get_slice_size();
    auto slice_num = ceildiv(num_rows, slice_size);

    const auto source_vals = source->get_const_values();
    const auto source_slice_lengths = source->get_const_slice_lengths();
    const auto source_slice_sets = source->get_const_slice_sets();
    const auto source_col_idxs = source->get_const_col_idxs();

    auto result_vals = result->get_values();
    auto result_row_ptrs = result->get_row_ptrs();
    auto result_col_idxs = result->get_col_idxs();

    size_type cur_ptr = 0;

    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            auto global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            result_row_ptrs[global_row] = cur_ptr;
            for (size_type sellp_ind =
                     source_slice_sets[slice] * slice_size + row;
                 sellp_ind < source_slice_sets[slice + 1] * slice_size + row;
                 sellp_ind += slice_size) {
                if (source_vals[sellp_ind] != zero<ValueType>()) {
                    result_vals[cur_ptr] = source_vals[sellp_ind];
                    result_col_idxs[cur_ptr] = source_col_idxs[sellp_ind];
                    cur_ptr++;
                }
            }
        }
    }
    result_row_ptrs[num_rows] = cur_ptr;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Sellp<ValueType, IndexType> *source,
                    size_type *result)
{
    auto num_rows = source->get_size()[0];
    auto slice_size = source->get_slice_size();
    auto slice_num = ceildiv(num_rows, slice_size);

    const auto vals = source->get_const_values();
    const auto slice_sets = source->get_const_slice_sets();

    auto num_nonzeros = 0;

    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0;
             row < slice_size && slice_size * slice + row < num_rows; row++) {
            for (size_type sellp_ind = slice_sets[slice] * slice_size + row;
                 sellp_ind < slice_sets[slice + 1] * slice_size + row;
                 sellp_ind += slice_size) {
                if (vals[sellp_ind] != zero<ValueType>()) {
                    num_nonzeros++;
                }
            }
        }
    }

    *result = num_nonzeros;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_COUNT_NONZEROS_KERNEL);


}  // namespace sellp
}  // namespace reference
}  // namespace kernels
}  // namespace gko
