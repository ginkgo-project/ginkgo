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


#include "core/base/math.hpp"
#include "core/matrix/csr.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace dense {


template <typename ValueType>
void simple_apply(const matrix::Dense<ValueType> *a,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *c)
{
    for (size_type row = 0; row < c->get_num_rows(); ++row) {
        for (size_type col = 0; col < c->get_num_cols(); ++col) {
            c->at(row, col) = zero<ValueType>();
        }
    }

    for (size_type row = 0; row < c->get_num_rows(); ++row) {
        for (size_type inner = 0; inner < a->get_num_cols(); ++inner) {
            for (size_type col = 0; col < c->get_num_cols(); ++col) {
                c->at(row, col) += a->at(row, inner) * b->at(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *a, const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *c)
{
    for (size_type row = 0; row < c->get_num_rows(); ++row) {
        for (size_type col = 0; col < c->get_num_cols(); ++col) {
            c->at(row, col) *= beta->at(0, 0);
        }
    }

    for (size_type row = 0; row < c->get_num_rows(); ++row) {
        for (size_type inner = 0; inner < a->get_num_cols(); ++inner) {
            for (size_type col = 0; col < c->get_num_cols(); ++col) {
                c->at(row, col) +=
                    alpha->at(0, 0) * a->at(row, inner) * b->at(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(const matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *x)
{
    if (alpha->get_num_cols() == 1) {
        for (size_type i = 0; i < x->get_num_rows(); ++i) {
            for (size_type j = 0; j < x->get_num_cols(); ++j) {
                x->at(i, j) *= alpha->at(0, 0);
            }
        }
    } else {
        for (size_type i = 0; i < x->get_num_rows(); ++i) {
            for (size_type j = 0; j < x->get_num_cols(); ++j) {
                x->at(i, j) *= alpha->at(0, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(const matrix::Dense<ValueType> *alpha,
                const matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *y)
{
    if (alpha->get_num_cols() == 1) {
        for (size_type i = 0; i < x->get_num_rows(); ++i) {
            for (size_type j = 0; j < x->get_num_cols(); ++j) {
                y->at(i, j) += alpha->at(0, 0) * x->at(i, j);
            }
        }
    } else {
        for (size_type i = 0; i < x->get_num_rows(); ++i) {
            for (size_type j = 0; j < x->get_num_cols(); ++j) {
                y->at(i, j) += alpha->at(0, j) * x->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void compute_dot(const matrix::Dense<ValueType> *x,
                 const matrix::Dense<ValueType> *y,
                 matrix::Dense<ValueType> *result)
{
    for (size_type j = 0; j < x->get_num_cols(); ++j) {
        result->at(0, j) = zero<ValueType>();
    }
    for (size_type i = 0; i < x->get_num_rows(); ++i) {
        for (size_type j = 0; j < x->get_num_cols(); ++j) {
            result->at(0, j) += gko::conj(x->at(i, j)) * y->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source)
{
    auto num_rows = result->get_num_rows();
    auto num_cols = result->get_num_cols();
    auto num_nonzeros = result->get_num_stored_elements();

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    size_type cur_ptr = 0;
    row_ptrs[0] = cur_ptr;
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            if (val != zero<ValueType>()) {
                col_idxs[cur_ptr] = col;
                values[cur_ptr] = val;
                ++cur_ptr;
            }
        }
        row_ptrs[row + 1] = cur_ptr;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_csr(matrix::Csr<ValueType, IndexType> *result,
                 const matrix::Dense<ValueType> *source)
{
    reference::dense::convert_to_csr(result, source);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(const matrix::Dense<ValueType> *source, size_type *result)
{
    auto num_rows = source->get_num_rows();
    auto num_cols = source->get_num_cols();
    auto num_nonzeros = 0;

    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += (source->at(row, col) != zero<ValueType>());
        }
    }

    *result = num_nonzeros;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL);

template <typename ValueType>
void transpose(matrix::Dense<ValueType> *trans,
               const matrix::Dense<ValueType> *orig)
{
    using std::swap;
    // ASSERT_EQUAL_DIMENSIONS(trans, orig);
    for (size_type i = 0; i < orig->get_num_rows(); ++i) {
        for (size_type j = 0; j < orig->get_num_cols(); ++j) {
            trans->at(i, j) = orig->at(i, j);
        }
    }
    for (size_type i = 0; i < orig->get_num_rows(); ++i) {
        for (size_type j = i; j < orig->get_num_cols(); ++j) {
            // auto tmp = trans->at(j, i);
            swap(trans->at(j, i), trans->at(i, j));
            // trans->at(i, j) = tmp;
        }
    }
};
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(matrix::Dense<ValueType> *trans,
                    const matrix::Dense<ValueType> *orig)
{
    using std::swap;
    // ASSERT_EQUAL_DIMENSIONS(trans, orig);
    for (size_type i = 0; i < orig->get_num_rows(); ++i) {
        for (size_type j = 0; j < orig->get_num_cols(); ++j) {
            auto tmp = orig->at(i, j);
            trans->at(i, j) = std::conj(tmp);
        }
    }
    for (size_type i = 0; i < orig->get_num_rows(); ++i) {
        for (size_type j = i; j < orig->get_num_cols(); ++j) {
            // auto tmp = trans->at(j, i);
            swap(trans->at(j, i), trans->at(i, j));
            // trans->at(i, j) = tmp;
        }
    }
};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONJ_TRANSPOSE_KERNEL);


}  // namespace dense
}  // namespace reference
}  // namespace kernels
}  // namespace gko
