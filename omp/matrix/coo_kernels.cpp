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

#include "core/matrix/coo_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "omp/components/atomic.hpp"
#include "omp/components/format_conversion.hpp"


namespace gko {
namespace kernels {
/**
 * @brief OpenMP namespace.
 *
 * @ingroup omp
 */
namespace omp {
/**
 * @brief The Coordinate matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::Coo<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    dense::fill(exec, c, zero<ValueType>());
    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Coo<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Coo<ValueType, IndexType> *a,
           const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    const auto coo_val = a->get_const_values();
    const auto coo_col = a->get_const_col_idxs();
    const auto coo_row = a->get_const_row_idxs();
    const auto num_rhs = b->get_size()[1];
    const auto sentinel_row = a->get_size()[0] + 1;
    const auto nnz = a->get_num_stored_elements();

#pragma omp parallel
    {
        const auto num_threads = omp_get_num_threads();
        const auto work_per_thread =
            static_cast<size_type>(ceildiv(nnz, num_threads));
        const auto thread_id = static_cast<size_type>(omp_get_thread_num());
        const auto begin = work_per_thread * thread_id;
        const auto end = std::min(begin + work_per_thread, nnz);
        if (begin < end) {
            const auto first = begin > 0 ? coo_row[begin - 1] : sentinel_row;
            const auto last = end < nnz ? coo_row[end] : sentinel_row;
            auto nz = begin;
            for (; nz < end && coo_row[nz] == first; nz++) {
                const auto row = first;
                const auto col = coo_col[nz];
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    atomic_add(c->at(row, rhs), coo_val[nz] * b->at(col, rhs));
                }
            }
            for (; nz < end && coo_row[nz] != last; nz++) {
                const auto row = coo_row[nz];
                const auto col = coo_col[nz];
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    c->at(row, rhs) += coo_val[nz] * b->at(col, rhs);
                }
            }
            for (; nz < end; nz++) {
                const auto row = last;
                const auto col = coo_col[nz];
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    atomic_add(c->at(row, rhs), coo_val[nz] * b->at(col, rhs));
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType> *alpha,
                    const matrix::Coo<ValueType, IndexType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c)
{
    const auto coo_val = a->get_const_values();
    const auto coo_col = a->get_const_col_idxs();
    const auto coo_row = a->get_const_row_idxs();
    const auto num_rhs = b->get_size()[1];
    const auto sentinel_row = a->get_size()[0] + 1;
    const auto nnz = a->get_num_stored_elements();
    const auto scale = alpha->at(0, 0);

#pragma omp parallel
    {
        const auto num_threads = omp_get_num_threads();
        const auto work_per_thread =
            static_cast<size_type>(ceildiv(nnz, num_threads));
        const auto thread_id = static_cast<size_type>(omp_get_thread_num());
        const auto begin = work_per_thread * thread_id;
        const auto end = std::min(begin + work_per_thread, nnz);
        if (begin < end) {
            const auto first = begin > 0 ? coo_row[begin - 1] : sentinel_row;
            const auto last = end < nnz ? coo_row[end] : sentinel_row;
            auto nz = begin;
            for (; nz < end && coo_row[nz] == first; nz++) {
                const auto row = first;
                const auto col = coo_col[nz];
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    atomic_add(c->at(row, rhs),
                               scale * coo_val[nz] * b->at(col, rhs));
                }
            }
            for (; nz < end && coo_row[nz] != last; nz++) {
                const auto row = coo_row[nz];
                const auto col = coo_col[nz];
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    c->at(row, rhs) += scale * coo_val[nz] * b->at(col, rhs);
                }
            }
            for (; nz < end; nz++) {
                const auto row = last;
                const auto col = coo_col[nz];
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    atomic_add(c->at(row, rhs),
                               scale * coo_val[nz] * b->at(col, rhs));
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);


template <typename IndexType>
void convert_row_idxs_to_ptrs(std::shared_ptr<const OmpExecutor> exec,
                              const IndexType *idxs, size_type num_nonzeros,
                              IndexType *ptrs, size_type num_rows)
{
    convert_sorted_idxs_to_ptrs(idxs, num_nonzeros, ptrs, num_rows);
}


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Coo<ValueType, IndexType> *source,
                    matrix::Csr<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];

    auto row_ptrs = result->get_row_ptrs();
    const auto nnz = result->get_num_stored_elements();

    const auto source_row_idxs = source->get_const_row_idxs();

    convert_row_idxs_to_ptrs(exec, source_row_idxs, nnz, row_ptrs, num_rows);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Coo<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result)
{
    auto coo_val = source->get_const_values();
    auto coo_col = source->get_const_col_idxs();
    auto coo_row = source->get_const_row_idxs();
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; row++) {
        for (size_type col = 0; col < num_cols; col++) {
            result->at(row, col) = zero<ValueType>();
        }
    }
#pragma omp parallel for
    for (size_type i = 0; i < source->get_num_stored_elements(); i++) {
        result->at(coo_row[i], coo_col[i]) += coo_val[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_DENSE_KERNEL);


}  // namespace coo
}  // namespace omp
}  // namespace kernels
}  // namespace gko
