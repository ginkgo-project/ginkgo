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

#include "core/matrix/dense_kernels.hpp"


#include "core/base/exception_helpers.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Dense<ValueType> *a,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *c)
{
    for (size_type row = 0; row < c->get_size().num_rows; ++row) {
#pragma omp parallel for
        for (size_type col = 0; col < c->get_size().num_cols; ++col) {
            c->at(row, col) = zero<ValueType>();
        }
    }

    for (size_type row = 0; row < c->get_size().num_rows; ++row) {
        for (size_type inner = 0; inner < a->get_size().num_cols; ++inner) {
#pragma omp parallel for reduction(+ : c->at(row, col))
            for (size_type col = 0; col < c->get_size().num_cols; ++col) {
                c->at(row, col) += a->at(row, inner) * b->at(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *a, const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *c)
{
    if (beta->at(0, 0) != zero<ValueType>()) {
        for (size_type row = 0; row < c->get_size().num_rows; ++row) {
#pragma omp parallel for
            for (size_type col = 0; col < c->get_size().num_cols; ++col) {
                c->at(row, col) *= beta->at(0, 0);
            }
        }
    } else {
        for (size_type row = 0; row < c->get_size().num_rows; ++row) {
#pragma omp parallel for
            for (size_type col = 0; col < c->get_size().num_cols; ++col) {
                c->at(row, col) *= zero<ValueType>();
            }
        }
    }

    for (size_type row = 0; row < c->get_size().num_rows; ++row) {
        for (size_type inner = 0; inner < a->get_size().num_cols; ++inner) {
#pragma omp parallel for reduction(+ : c->at(row, col))
            for (size_type col = 0; col < c->get_size().num_cols; ++col) {
                c->at(row, col) +=
                    alpha->at(0, 0) * a->at(row, inner) * b->at(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *x)
{
    if (alpha->get_size().num_cols == 1) {
        for (size_type i = 0; i < x->get_size().num_rows; ++i) {
#pragma omp parallel for
            for (size_type j = 0; j < x->get_size().num_cols; ++j) {
                x->at(i, j) *= alpha->at(0, 0);
            }
        }
    } else {
        for (size_type i = 0; i < x->get_size().num_rows; ++i) {
#pragma omp parallel for
            for (size_type j = 0; j < x->get_size().num_cols; ++j) {
                x->at(i, j) *= alpha->at(0, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const OmpExecutor> exec,
                const matrix::Dense<ValueType> *alpha,
                const matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *y)
{
    if (alpha->get_size().num_cols == 1) {
        for (size_type i = 0; i < x->get_size().num_rows; ++i) {
#pragma omp parallel for
            for (size_type j = 0; j < x->get_size().num_cols; ++j) {
                y->at(i, j) += alpha->at(0, 0) * x->at(i, j);
            }
        }
    } else {
        for (size_type i = 0; i < x->get_size().num_rows; ++i) {
#pragma omp parallel for
            for (size_type j = 0; j < x->get_size().num_cols; ++j) {
                y->at(i, j) += alpha->at(0, j) * x->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const OmpExecutor> exec,
                 const matrix::Dense<ValueType> *x,
                 const matrix::Dense<ValueType> *y,
                 matrix::Dense<ValueType> *result)
{
#pragma omp parallel for
    for (size_type j = 0; j < x->get_size().num_cols; ++j) {
        result->at(0, j) = zero<ValueType>();
    }
    for (size_type i = 0; i < x->get_size().num_rows; ++i) {
#pragma omp parallel for reduction(+ : result->at(0, j))
        for (size_type j = 0; j < x->get_size().num_cols; ++j) {
            result->at(0, j) += conj(x->at(i, j)) * y->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const OmpExecutor> exec,
                    matrix::Coo<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const OmpExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_csr(std::shared_ptr<const OmpExecutor> exec,
                 matrix::Csr<ValueType, IndexType> *result,
                 const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const OmpExecutor> exec,
                    matrix::Ell<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_ell(std::shared_ptr<const OmpExecutor> exec,
                 matrix::Ell<ValueType, IndexType> *result,
                 const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_ELL_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType> *source, size_type *result)
{
    auto num_rows = source->get_size().num_rows;
    auto num_cols = source->get_size().num_cols;
    auto num_nonzeros = 0;

    for (size_type row = 0; row < num_rows; ++row) {
#pragma omp parallel for reduction(+ : num_nonzeros)
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += (source->at(row, col) != zero<ValueType>());
        }
    }

    *result = num_nonzeros;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nonzeros_per_row(std::shared_ptr<const OmpExecutor> exec,
                                    const matrix::Dense<ValueType> *source,
                                    size_type *result)
{
    auto num_rows = source->get_size().num_rows;
    auto num_cols = source->get_size().num_cols;
    size_type max_nonzeros_per_row = 0;
    size_type num_nonzeros = 0;
    for (size_type row = 0; row < num_rows; ++row) {
        num_nonzeros = 0;
#pragma omp parallel for reduction(+ : num_nonzeros)
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += (source->at(row, col) != zero<ValueType>());
        }
        max_nonzeros_per_row = std::max(num_nonzeros, max_nonzeros_per_row);
    }

    *result = max_nonzeros_per_row;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_MAX_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const OmpExecutor> exec,
               matrix::Dense<ValueType> *trans,
               const matrix::Dense<ValueType> *orig)
{
    for (size_type i = 0; i < orig->get_size().num_rows; ++i) {
#pragma omp parallel for
        for (size_type j = 0; j < orig->get_size().num_cols; ++j) {
            trans->at(j, i) = orig->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const OmpExecutor> exec,
                    matrix::Dense<ValueType> *trans,
                    const matrix::Dense<ValueType> *orig)
{
    for (size_type i = 0; i < orig->get_size().num_rows; ++i) {
#pragma omp parallel for
        for (size_type j = 0; j < orig->get_size().num_cols; ++j) {
            trans->at(j, i) = conj(orig->at(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONJ_TRANSPOSE_KERNEL);


}  // namespace dense
}  // namespace omp
}  // namespace kernels
}  // namespace gko
