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

#include "core/matrix/diagonal_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Diagonal matrix format namespace.
 *
 * @ingroup diagonal
 */
namespace diagonal {


template <typename ValueType>
void apply_to_dense(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Diagonal<ValueType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c)
{
    const auto diag_values = a->get_const_values();
#pragma omp parallel for
    for (size_type row = 0; row < a->get_size()[0]; row++) {
        const auto scal = diag_values[row];
        for (size_type col = 0; col < b->get_size()[1]; col++) {
            c->at(row, col) = b->at(row, col) * scal;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_APPLY_TO_DENSE_KERNEL);


template <typename ValueType>
void right_apply_to_dense(std::shared_ptr<const OmpExecutor> exec,
                          const matrix::Diagonal<ValueType> *a,
                          const matrix::Dense<ValueType> *b,
                          matrix::Dense<ValueType> *c)
{
    const auto diag_values = a->get_const_values();
#pragma omp parallel for
    for (size_type row = 0; row < b->get_size()[0]; row++) {
        for (size_type col = 0; col < a->get_size()[1]; col++) {
            c->at(row, col) = b->at(row, col) * diag_values[col];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void apply_to_csr(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Diagonal<ValueType> *a,
                  const matrix::Csr<ValueType, IndexType> *b,
                  matrix::Csr<ValueType, IndexType> *c)
{
    const auto diag_values = a->get_const_values();
    c->copy_from(b);
    auto csr_values = c->get_values();
    const auto csr_row_ptrs = c->get_const_row_ptrs();

#pragma omp parallel for
    for (size_type row = 0; row < c->get_size()[0]; row++) {
        const auto scal = diag_values[row];
        for (size_type idx = csr_row_ptrs[row]; idx < csr_row_ptrs[row + 1];
             idx++) {
            csr_values[idx] *= scal;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_APPLY_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void right_apply_to_csr(std::shared_ptr<const OmpExecutor> exec,
                        const matrix::Diagonal<ValueType> *a,
                        const matrix::Csr<ValueType, IndexType> *b,
                        matrix::Csr<ValueType, IndexType> *c)
{
    const auto diag_values = a->get_const_values();
    c->copy_from(b);
    auto csr_values = c->get_values();
    const auto csr_row_ptrs = c->get_const_row_ptrs();
    const auto csr_col_idxs = c->get_const_col_idxs();

#pragma omp parallel for
    for (size_type row = 0; row < c->get_size()[0]; row++) {
        for (size_type idx = csr_row_ptrs[row]; idx < csr_row_ptrs[row + 1];
             idx++) {
            csr_values[idx] *= diag_values[csr_col_idxs[idx]];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Diagonal<ValueType> *source,
                    matrix::Csr<ValueType, IndexType> *result)
{
    const auto size = source->get_size()[0];
    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto csr_values = result->get_values();
    const auto diag_values = source->get_const_values();

#pragma omp parallel for
    for (size_type i = 0; i < size; i++) {
        row_ptrs[i] = i;
        col_idxs[i] = i;
        csr_values[i] = diag_values[i];
    }
    row_ptrs[size] = size;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_CONVERT_TO_CSR_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Diagonal<ValueType> *orig,
                    matrix::Diagonal<ValueType> *trans)
{
    const auto size = orig->get_size()[0];
    const auto orig_values = orig->get_const_values();
    auto trans_values = trans->get_values();

#pragma omp parallel for
    for (size_type i = 0; i < size; i++) {
        trans_values[i] = conj(orig_values[i]);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_CONJ_TRANSPOSE_KERNEL);


}  // namespace diagonal
}  // namespace omp
}  // namespace kernels
}  // namespace gko
