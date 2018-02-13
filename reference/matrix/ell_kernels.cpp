/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/ell_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace ell {


template <typename ValueType, typename IndexType>
void spmv(const matrix::Ell<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    auto max_nnz_row = a->get_const_max_nnz_row();
    auto arows = a->get_num_rows();

    for (size_type row = 0; row < arows; row++) {
        for (size_type j = 0; j < c->get_num_cols(); j++) {
            c->at(row, j) = zero<ValueType>();
        }
        for (size_type i = 0; i < max_nnz_row; i++) {
            auto val = vals[row + i*arows];
            auto col = col_idxs[row + i*arows];
            for (size_type j = 0; j < c->get_num_cols(); j++) {
                c->at(row, j) += val*b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(const matrix::Dense<ValueType> *alpha,
                   const matrix::Ell<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
	auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    auto max_nnz_row = a->get_const_max_nnz_row();
    auto arows = a->get_num_rows();
    auto valpha = alpha->at(0,0);
    auto vbeta = beta->at(0,0);

    for (size_type row = 0; row < arows; row++) {
        for (size_type j = 0; j < c->get_num_cols(); j++) {
            c->at(row, j) *= vbeta;
        }
        for (size_type i = 0; i < max_nnz_row; i++) {
            auto val = vals[row + i*arows];
            auto col = col_idxs[row + i*arows];
            for (size_type j = 0; j < c->get_num_cols(); j++) {
                c->at(row, j) += valpha*val*b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(matrix::Dense<ValueType> *result,
                      const matrix::Ell<ValueType, IndexType> *source)
{
	auto exec = result->get_executor();
    if(exec != exec->get_master()) {
        NOT_SUPPORTED(exec);
    }

    auto num_rows = source->get_num_rows();
    auto num_cols = source->get_num_cols();
    auto num_nonzeros = source->get_num_stored_elements();
    auto vals = source->get_const_values();
    auto col_idxs = source->get_const_col_idxs();
    auto max_nnz_row = source->get_const_max_nnz_row();

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type col = 0; col < num_cols; col++) {
            result->at(row, col) = zero<ValueType>();
        }
        for (size_type i = 0; i < max_nnz_row; i++) {
            result->at(row, col_idxs[row+i*num_rows]) += vals[row+i*num_rows];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_dense(matrix::Dense<ValueType> *result,
                   matrix::Ell<ValueType, IndexType> *source)
{
    reference::ell::convert_to_dense(result, source);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_MOVE_TO_DENSE_KERNEL);


}  // namespace ell
}  // namespace reference
}  // namespace kernels
}  // namespace gko
