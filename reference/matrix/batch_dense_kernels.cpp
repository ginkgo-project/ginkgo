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

#include "core/matrix/batch_dense_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The BatchDense matrix format namespace.
 * @ref BatchDense
 * @ingroup batch_dense
 */
namespace batch_dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::BatchDense<ValueType> *a,
                  const matrix::BatchDense<ValueType> *b,
                  matrix::BatchDense<ValueType> *c)
{
    for (size_type batch = 0; batch < c->get_num_batches(); ++batch) {
        for (size_type row = 0; row < c->get_batch_sizes()[batch][0]; ++row) {
            for (size_type col = 0; col < c->get_batch_sizes()[batch][1];
                 ++col) {
                c->at(batch, row, col) = zero<ValueType>();
            }
        }

        for (size_type row = 0; row < c->get_batch_sizes()[batch][0]; ++row) {
            for (size_type inner = 0; inner < a->get_batch_sizes()[batch][1];
                 ++inner) {
                for (size_type col = 0; col < c->get_batch_sizes()[batch][1];
                     ++col) {
                    c->at(batch, row, col) +=
                        a->at(batch, row, inner) * b->at(batch, inner, col);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::BatchDense<ValueType> *alpha,
           const matrix::BatchDense<ValueType> *a,
           const matrix::BatchDense<ValueType> *b,
           const matrix::BatchDense<ValueType> *beta,
           matrix::BatchDense<ValueType> *c)
{
    for (size_type batch = 0; batch < c->get_num_batches(); ++batch) {
        if (beta->at(batch, 0, 0) != zero<ValueType>()) {
            for (size_type row = 0; row < c->get_batch_sizes()[batch][0];
                 ++row) {
                for (size_type col = 0; col < c->get_batch_sizes()[batch][1];
                     ++col) {
                    c->at(batch, row, col) *= beta->at(batch, 0, 0);
                }
            }
        } else {
            for (size_type row = 0; row < c->get_batch_sizes()[batch][0];
                 ++row) {
                for (size_type col = 0; col < c->get_batch_sizes()[batch][1];
                     ++col) {
                    c->at(batch, row, col) *= zero<ValueType>();
                }
            }
        }

        for (size_type row = 0; row < c->get_batch_sizes()[batch][0]; ++row) {
            for (size_type inner = 0; inner < a->get_batch_sizes()[batch][1];
                 ++inner) {
                for (size_type col = 0; col < c->get_batch_sizes()[batch][1];
                     ++col) {
                    c->at(batch, row, col) += alpha->at(batch, 0, 0) *
                                              a->at(batch, row, inner) *
                                              b->at(batch, inner, col);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::BatchDense<ValueType> *alpha,
           matrix::BatchDense<ValueType> *x)
{
    for (size_type batch = 0; batch < x->get_num_batches(); ++batch) {
        if (alpha->get_batch_sizes()[batch][1] == 1) {
            for (size_type i = 0; i < x->get_batch_sizes()[batch][0]; ++i) {
                for (size_type j = 0; j < x->get_batch_sizes()[batch][1]; ++j) {
                    x->at(batch, i, j) *= alpha->at(batch, 0, 0);
                }
            }
        } else {
            for (size_type i = 0; i < x->get_batch_sizes()[batch][0]; ++i) {
                for (size_type j = 0; j < x->get_batch_sizes()[batch][1]; ++j) {
                    x->at(batch, i, j) *= alpha->at(batch, 0, j);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const ReferenceExecutor> exec,
                const matrix::BatchDense<ValueType> *alpha,
                const matrix::BatchDense<ValueType> *x,
                matrix::BatchDense<ValueType> *y)
{
    for (size_type batch = 0; batch < y->get_num_batches(); ++batch) {
        if (alpha->get_batch_sizes()[batch][1] == 1) {
            for (size_type i = 0; i < x->get_batch_sizes()[batch][0]; ++i) {
                for (size_type j = 0; j < x->get_batch_sizes()[batch][1]; ++j) {
                    y->at(batch, i, j) +=
                        alpha->at(batch, 0, 0) * x->at(batch, i, j);
                }
            }
        } else {
            for (size_type i = 0; i < x->get_batch_sizes()[batch][0]; ++i) {
                for (size_type j = 0; j < x->get_batch_sizes()[batch][1]; ++j) {
                    y->at(batch, i, j) +=
                        alpha->at(batch, 0, j) * x->at(batch, i, j);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const ReferenceExecutor> exec,
                     const matrix::BatchDense<ValueType> *alpha,
                     const matrix::Diagonal<ValueType> *x,
                     matrix::BatchDense<ValueType> *y) GKO_NOT_IMPLEMENTED;
// {
// for (size_type batch = 0; batch < y->get_num_batches(); ++batch) {
//     const auto diag_values = x->get_const_values();
//     for (size_type i = 0; i < x->get_batch_sizes()[batch][0]; i++) {
//         y->at(batch,i, i) += alpha->at(batch,0, 0) * diag_values[i];
//     }
// }
// }

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const ReferenceExecutor> exec,
                 const matrix::BatchDense<ValueType> *x,
                 const matrix::BatchDense<ValueType> *y,
                 matrix::BatchDense<ValueType> *result)
{
    for (size_type batch = 0; batch < result->get_num_batches(); ++batch) {
        for (size_type j = 0; j < x->get_batch_sizes()[batch][1]; ++j) {
            result->at(batch, 0, j) = zero<ValueType>();
        }
        for (size_type i = 0; i < x->get_batch_sizes()[batch][0]; ++i) {
            for (size_type j = 0; j < x->get_batch_sizes()[batch][1]; ++j) {
                result->at(batch, 0, j) +=
                    conj(x->at(batch, i, j)) * y->at(batch, i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::BatchDense<ValueType> *x,
                   matrix::BatchDense<remove_complex<ValueType>> *result)
{
    for (size_type batch = 0; batch < result->get_num_batches(); ++batch) {
        for (size_type j = 0; j < x->get_batch_sizes()[batch][1]; ++j) {
            result->at(batch, 0, j) = zero<remove_complex<ValueType>>();
        }
        for (size_type i = 0; i < x->get_batch_sizes()[batch][0]; ++i) {
            for (size_type j = 0; j < x->get_batch_sizes()[batch][1]; ++j) {
                result->at(batch, 0, j) += squared_norm(x->at(batch, i, j));
            }
        }
        for (size_type j = 0; j < x->get_batch_sizes()[batch][1]; ++j) {
            result->at(batch, 0, j) = sqrt(result->at(batch, 0, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::BatchDense<ValueType> *source,
                    size_type *result)
{
    for (size_type batch = 0; batch < source->get_num_batches(); ++batch) {
        auto num_rows = source->get_batch_sizes()[batch][0];
        auto num_cols = source->get_batch_sizes()[batch][1];
        auto num_nonzeros = 0;

        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                num_nonzeros +=
                    (source->at(batch, row, col) != zero<ValueType>());
            }
        }
        result[batch] = num_nonzeros;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                               const matrix::BatchDense<ValueType> *source,
                               size_type *result)
{
    for (size_type batch = 0; batch < source->get_num_batches(); ++batch) {
        auto num_rows = source->get_batch_sizes()[batch][0];
        auto num_cols = source->get_batch_sizes()[batch][1];
        size_type num_stored_elements_per_row = 0;
        size_type num_nonzeros = 0;
        for (size_type row = 0; row < num_rows; ++row) {
            num_nonzeros = 0;
            for (size_type col = 0; col < num_cols; ++col) {
                num_nonzeros +=
                    (source->at(batch, row, col) != zero<ValueType>());
            }
            num_stored_elements_per_row =
                std::max(num_nonzeros, num_stored_elements_per_row);
        }
        result[batch] = num_stored_elements_per_row;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                                const matrix::BatchDense<ValueType> *source,
                                Array<size_type> *result)
{
    for (size_type batch = 0; batch < source->get_num_batches(); ++batch) {
        auto num_rows = source->get_batch_sizes()[batch][0];
        auto num_cols = source->get_batch_sizes()[batch][1];
        auto row_nnz_val = result->get_data();
        size_type offset = 0;
        for (size_type row = 0; row < num_rows; ++row) {
            size_type num_nonzeros = 0;
            for (size_type col = 0; col < num_cols; ++col) {
                num_nonzeros +=
                    (source->at(batch, row, col) != zero<ValueType>());
            }
            row_nnz_val[offset + row] = num_nonzeros;
            ++offset;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const ReferenceExecutor> exec,
                          const matrix::BatchDense<ValueType> *source,
                          size_type *result, size_type *stride_factor,
                          size_type *slice_size)
{
    for (size_type batch = 0; batch < source->get_num_batches(); ++batch) {
        auto num_rows = source->get_batch_sizes()[batch][0];
        auto num_cols = source->get_batch_sizes()[batch][1];
        auto slice_num = ceildiv(num_rows, slice_size[batch]);
        auto total_cols = 0;
        auto temp = 0, slice_temp = 0;
        for (size_type slice = 0; slice < slice_num; slice++) {
            slice_temp = 0;
            for (size_type row = 0; row < slice_size[batch] &&
                                    row + slice * slice_size[batch] < num_rows;
                 row++) {
                temp = 0;
                for (size_type col = 0; col < num_cols; col++) {
                    temp += (source->at(batch, row + slice * slice_size[batch],
                                        col) != zero<ValueType>());
                }
                slice_temp = (slice_temp < temp) ? temp : slice_temp;
            }
            slice_temp = ceildiv(slice_temp, stride_factor[batch]) *
                         stride_factor[batch];
            total_cols += slice_temp;
        }
        result[batch] = total_cols;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               const matrix::BatchDense<ValueType> *orig,
               matrix::BatchDense<ValueType> *trans)
{
    for (size_type batch = 0; batch < orig->get_num_batches(); ++batch) {
        for (size_type i = 0; i < orig->get_batch_sizes()[batch][0]; ++i) {
            for (size_type j = 0; j < orig->get_batch_sizes()[batch][1]; ++j) {
                trans->at(batch, j, i) = orig->at(batch, i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::BatchDense<ValueType> *orig,
                    matrix::BatchDense<ValueType> *trans)
{
    for (size_type batch = 0; batch < orig->get_num_batches(); ++batch) {
        for (size_type i = 0; i < orig->get_batch_sizes()[batch][0]; ++i) {
            for (size_type j = 0; j < orig->get_batch_sizes()[batch][1]; ++j) {
                trans->at(batch, j, i) = conj(orig->at(batch, i, j));
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL);


}  // namespace batch_dense
}  // namespace reference
}  // namespace kernels
}  // namespace gko
