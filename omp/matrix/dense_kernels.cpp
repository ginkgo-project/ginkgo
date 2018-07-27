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


#include <omp.h>


#include "core/base/exception_helpers.hpp"
#include "core/matrix/sellp.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace dense {


template <typename ValueType, typename AccessorType>
void simple_apply(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Dense<ValueType> *a, AccessorType b,
                  matrix::Dense<ValueType> *c)
{
    NOT_IMPLEMENTED;
}


template <typename ValueType, typename AccessorType>
void apply(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *a, AccessorType b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *c)
{
    NOT_IMPLEMENTED;
}


template <typename ValueType>
void simple_apply(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Dense<ValueType> *a,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *c)
{
#pragma omp parallel for
    for (size_type row = 0; row < c->get_size().num_rows; ++row) {
        for (size_type col = 0; col < c->get_size().num_cols; ++col) {
            c->at(row, col) = zero<ValueType>();
        }
    }

#pragma omp parallel for
    for (size_type row = 0; row < c->get_size().num_rows; ++row) {
        for (size_type inner = 0; inner < a->get_size().num_cols; ++inner) {
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
#pragma omp parallel for
        for (size_type row = 0; row < c->get_size().num_rows; ++row) {
            for (size_type col = 0; col < c->get_size().num_cols; ++col) {
                c->at(row, col) *= beta->at(0, 0);
            }
        }
    } else {
#pragma omp parallel for
        for (size_type row = 0; row < c->get_size().num_rows; ++row) {
            for (size_type col = 0; col < c->get_size().num_cols; ++col) {
                c->at(row, col) *= zero<ValueType>();
            }
        }
    }

#pragma omp parallel for
    for (size_type row = 0; row < c->get_size().num_rows; ++row) {
        for (size_type inner = 0; inner < a->get_size().num_cols; ++inner) {
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
#pragma omp parallel for
        for (size_type i = 0; i < x->get_size().num_rows; ++i) {
            for (size_type j = 0; j < x->get_size().num_cols; ++j) {
                x->at(i, j) *= alpha->at(0, 0);
            }
        }
    } else {
#pragma omp parallel for
        for (size_type i = 0; i < x->get_size().num_rows; ++i) {
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
#pragma omp parallel for
        for (size_type i = 0; i < x->get_size().num_rows; ++i) {
            for (size_type j = 0; j < x->get_size().num_cols; ++j) {
                y->at(i, j) += alpha->at(0, 0) * x->at(i, j);
            }
        }
    } else {
#pragma omp parallel for
        for (size_type i = 0; i < x->get_size().num_rows; ++i) {
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
#pragma omp parallel for
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


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const OmpExecutor> exec,
                       matrix::Hybrid<ValueType, IndexType> *result,
                       const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_hybrid(std::shared_ptr<const OmpExecutor> exec,
                    matrix::Hybrid<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const OmpExecutor> exec,
                      matrix::Sellp<ValueType, IndexType> *result,
                      const matrix::Dense<ValueType> *source)
{
    auto num_rows = result->get_size().num_rows;
    auto num_cols = result->get_size().num_cols;
    auto vals = result->get_values();
    auto col_idxs = result->get_col_idxs();
    auto slice_lengths = result->get_slice_lengths();
    auto slice_sets = result->get_slice_sets();
    auto slice_size = (result->get_slice_size() == 0)
                          ? matrix::default_slice_size
                          : result->get_slice_size();
    auto stride_factor = (result->get_stride_factor() == 0)
                             ? matrix::default_stride_factor
                             : result->get_stride_factor();
    int slice_num = ceildiv(num_rows, slice_size);
    slice_sets[0] = 0;

    for (size_type slice = 0; slice < slice_num; slice++) {
        if (slice > 0) {
            slice_sets[slice] = slice_lengths[slice - 1];
        }
        size_type current_slice_length = 0;
#pragma omp parallel for reduction(max : current_slice_length)
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row < num_rows) {
                size_type max_col = 0;
                for (size_type col = 0; col < num_cols; col++) {
                    if (source->at(global_row, col) != zero<ValueType>()) {
                        max_col += 1;
                    }
                }
                current_slice_length = std::max(current_slice_length, max_col);
            }
        }
        slice_lengths[slice] =
            stride_factor * ceildiv(current_slice_length, stride_factor);
#pragma omp parallel for
        for (size_type row = 0; row < slice_size; row++) {
            const size_type global_row = slice * slice_size + row;
            if (global_row < num_rows) {
                size_type sellp_ind = slice_sets[slice] * slice_size + row;
                for (size_type col = 0; col < num_cols; col++) {
                    auto val = source->at(global_row, col);
                    if (val != zero<ValueType>()) {
                        col_idxs[sellp_ind] = col;
                        vals[sellp_ind] = val;
                        sellp_ind += slice_size;
                    }
                }
                for (size_type i = sellp_ind;
                     i <
                     (slice_sets[slice] + slice_lengths[slice]) * slice_size +
                         row;
                     i += slice_size) {
                    col_idxs[i] = 0;
                    vals[i] = 0;
                }
            }
        }
    }

    slice_sets[slice_num] = slice_lengths[slice_num - 1];
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_sellp(std::shared_ptr<const OmpExecutor> exec,
                   matrix::Sellp<ValueType, IndexType> *result,
                   const matrix::Dense<ValueType> *source)
{
    omp::dense::convert_to_sellp(exec, result, source);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_SELLP_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType> *source, size_type *result)
{
    auto num_rows = source->get_size().num_rows;
    auto num_cols = source->get_size().num_cols;
    auto num_nonzeros = 0;

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += (source->at(row, col) != zero<ValueType>());
        }
    }

    *result = num_nonzeros;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const OmpExecutor> exec,
                               const matrix::Dense<ValueType> *source,
                               size_type *result)
{
    const auto num_rows = source->get_size().num_rows;
    const auto num_cols = source->get_size().num_cols;
    size_type max_nonzeros_per_row = 0;
#pragma omp parallel for reduction(max : max_nonzeros_per_row)
    for (size_type row = 0; row < num_rows; ++row) {
        size_type num_nonzeros = 0;
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += (source->at(row, col) != zero<ValueType>());
        }
        max_nonzeros_per_row = std::max(num_nonzeros, max_nonzeros_per_row);
    }
    *result = max_nonzeros_per_row;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const OmpExecutor> exec,
                                const matrix::Dense<ValueType> *source,
                                Array<size_type> *result) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const OmpExecutor> exec,
                          const matrix::Dense<ValueType> *source,
                          size_type *result, size_type stride_factor)
{
    auto num_rows = source->get_size().num_rows;
    auto num_cols = source->get_size().num_cols;
    auto slice_size = matrix::default_slice_size;
    auto slice_num = ceildiv(num_rows, slice_size);
    size_type total_cols = 0;
#pragma omp parallel for reduction(+ : total_cols)
    for (size_type slice = 0; slice < slice_num; slice++) {
        size_type slice_temp = 0;
        for (size_type row = 0;
             row < slice_size && row + slice * slice_size < num_rows; row++) {
            size_type temp = 0;
            for (size_type col = 0; col < num_cols; col++) {
                temp += (source->at(row + slice * slice_size, col) !=
                         zero<ValueType>());
            }
            slice_temp = (slice_temp < temp) ? temp : slice_temp;
        }
        slice_temp = ceildiv(slice_temp, stride_factor) * stride_factor;
        total_cols += slice_temp;
    }

    *result = total_cols;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const OmpExecutor> exec,
               matrix::Dense<ValueType> *trans,
               const matrix::Dense<ValueType> *orig)
{
#pragma omp parallel for
    for (size_type i = 0; i < orig->get_size().num_rows; ++i) {
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
#pragma omp parallel for
    for (size_type i = 0; i < orig->get_size().num_rows; ++i) {
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
