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
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Ell<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto num_stored_elements_per_row = a->get_num_stored_elements_per_row();

    for (size_type row = 0; row < a->get_size().num_rows; row++) {
        for (size_type j = 0; j < c->get_size().num_cols; j++) {
            c->at(row, j) = zero<ValueType>();
        }
        for (size_type i = 0; i < num_stored_elements_per_row; i++) {
            auto val = a->val_at(row, i);
            auto col = a->col_at(row, i);
            for (size_type j = 0; j < c->get_size().num_cols; j++) {
                c->at(row, j) += val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Ell<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    auto num_stored_elements_per_row = a->get_num_stored_elements_per_row();
    auto alpha_val = alpha->at(0, 0);
    auto beta_val = beta->at(0, 0);

    for (size_type row = 0; row < a->get_size().num_rows; row++) {
        for (size_type j = 0; j < c->get_size().num_cols; j++) {
            c->at(row, j) *= beta_val;
        }
        for (size_type i = 0; i < num_stored_elements_per_row; i++) {
            auto val = a->val_at(row, i);
            auto col = a->col_at(row, i);
            for (size_type j = 0; j < c->get_size().num_cols; j++) {
                c->at(row, j) += alpha_val * val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const ReferenceExecutor> exec,
                      matrix::Dense<ValueType> *result,
                      const matrix::Ell<ValueType, IndexType> *source)
{
    auto num_rows = source->get_size().num_rows;
    auto num_cols = source->get_size().num_cols;
    auto num_stored_elements_per_row =
        source->get_num_stored_elements_per_row();

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type col = 0; col < num_cols; col++) {
            result->at(row, col) = zero<ValueType>();
        }
        for (size_type i = 0; i < num_stored_elements_per_row; i++) {
            result->at(row, source->col_at(row, i)) += source->val_at(row, i);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_DENSE_KERNEL);


}  // namespace ell
}  // namespace reference
}  // namespace kernels
}  // namespace gko
