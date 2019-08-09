/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/matrix/hybrid_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/ell_kernels.hpp"
#include "omp/components/format_conversion.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Hybrid matrix format namespace.
 *
 * @ingroup hybrid
 */
namespace hybrid {


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const OmpExecutor> exec,
                      matrix::Dense<ValueType> *result,
                      const matrix::Hybrid<ValueType, IndexType> *source)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto ell_val = source->get_const_ell_values();
    auto ell_col = source->get_const_ell_col_idxs();

    auto ell_num_stored_elements_per_row =
        source->get_ell_num_stored_elements_per_row();

    for (size_type row = 0; row < num_rows; row++) {
#pragma omp parallel for
        for (size_type col = 0; col < num_cols; col++) {
            result->at(row, col) = zero<ValueType>();
        }
#pragma omp parallel for
        for (size_type i = 0; i < ell_num_stored_elements_per_row; i++) {
            result->at(row, source->ell_col_at(row, i)) +=
                source->ell_val_at(row, i);
        }
    }

    auto coo_val = source->get_const_coo_values();
    auto coo_col = source->get_const_coo_col_idxs();
    auto coo_row = source->get_const_coo_row_idxs();
// The following parallelization is dangerous and can fail if the
// COO matrix contains several elements assigned to the same location.
#pragma omp parallel for
    for (size_type i = 0; i < source->get_coo_num_stored_elements(); i++) {
        result->at(coo_row[i], coo_col[i]) += coo_val[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const OmpExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Hybrid<ValueType, IndexType> *source)
{
    auto csr_val = result->get_values();
    auto csr_col_idxs = result->get_col_idxs();
    auto csr_row_ptrs = result->get_row_ptrs();
    const auto ell = source->get_ell();
    const auto max_nnz_per_row = ell->get_num_stored_elements_per_row();
    const auto coo_val = source->get_const_coo_values();
    const auto coo_col = source->get_const_coo_col_idxs();
    const auto coo_row = source->get_const_coo_row_idxs();
    const auto coo_nnz = source->get_coo_num_stored_elements();
    const auto num_rows = source->get_size()[0];
    auto coo_row_ptrs = Array<IndexType>(exec, num_rows + 1);
    auto coo_row_ptrs_val = coo_row_ptrs.get_data();
    convert_sorted_idxs_to_ptrs(coo_row, coo_nnz, coo_row_ptrs_val,
                                num_rows + 1);

    // Compute the row offset of Coo without zeros
    auto coo_offset = Array<IndexType>(exec, num_rows);
    auto coo_offset_val = coo_offset.get_data();
    for (size_type i = 0; i < num_rows; i++) {
        IndexType nonzeros = 0;
#pragma omp parallel for reduction(+ : nonzeros)
        for (size_type j = coo_row_ptrs_val[i]; j < coo_row_ptrs_val[i + 1];
             j++) {
            nonzeros += coo_val[j] != zero<ValueType>();
        }
        coo_offset_val[i] = nonzeros;
    }

    // Compute row pointer of Csr
    csr_row_ptrs[0] = 0;
#pragma omp parallel for
    for (size_type i = 0; i < num_rows; i++) {
        csr_row_ptrs[i + 1] = coo_offset_val[i];
    }

    for (size_type col = 0; col < max_nnz_per_row; col++) {
#pragma omp parallel for
        for (size_type row = 0; row < num_rows; row++) {
            csr_row_ptrs[row + 1] +=
                (ell->val_at(row, col) != zero<ValueType>());
        }
    }

    auto workspace = Array<IndexType>(exec, num_rows + 1);
    auto workspace_val = workspace.get_data();
    for (size_type i = 1; i < num_rows + 1; i <<= 1) {
#pragma omp parallel for
        for (size_type j = i; j < num_rows + 1; j++) {
            workspace_val[j] = csr_row_ptrs[j] + csr_row_ptrs[j - i];
        }
#pragma omp parallel for
        for (size_type j = i; j < num_rows + 1; j++) {
            csr_row_ptrs[j] = workspace_val[j];
        }
    }

// Fill in Csr
#pragma omp parallel for
    for (IndexType row = 0; row < num_rows; row++) {
        // Ell part
        size_type csr_idx = csr_row_ptrs[row];
        size_type coo_idx = coo_offset_val[row];
        for (IndexType col = 0; col < max_nnz_per_row; col++) {
            const auto val = ell->val_at(row, col);
            if (val != zero<ValueType>()) {
                csr_val[csr_idx] = val;
                csr_col_idxs[csr_idx] = ell->col_at(row, col);
                csr_idx++;
            }
        }
        // Coo part
        for (auto coo_idx = coo_row_ptrs_val[row];
             coo_idx < coo_row_ptrs_val[row + 1]; coo_idx++) {
            if (coo_val[coo_idx] != zero<ValueType>()) {
                csr_val[csr_idx] = coo_val[coo_idx];
                csr_col_idxs[csr_idx] = coo_col[coo_idx];
                csr_idx++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Hybrid<ValueType, IndexType> *source,
                    size_type *result)
{
    size_type ell_nnz = 0;
    size_type coo_nnz = 0;
    gko::kernels::omp::ell::count_nonzeros(exec, source->get_ell(), &ell_nnz);
    const auto coo_val = source->get_const_coo_values();
    const auto coo_max_nnz = source->get_coo_num_stored_elements();

#pragma omp parallel for reduction(+ : coo_nnz)
    for (size_type ind = 0; ind < coo_max_nnz; ind++) {
        coo_nnz += (coo_val[ind] != zero<ValueType>());
    }
    *result = ell_nnz + coo_nnz;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_COUNT_NONZEROS_KERNEL);


}  // namespace hybrid
}  // namespace omp
}  // namespace kernels
}  // namespace gko
