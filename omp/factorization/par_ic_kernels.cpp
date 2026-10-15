// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ic_kernels.hpp"

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/utils.hpp"
#include "omp/components/atomic.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The parallel ic factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ic_factorization {


template <typename ValueType, typename IndexType>
void init_factor(std::shared_ptr<const DefaultExecutor> exec,
                 matrix::view::csr<ValueType, IndexType> l)
{
    auto num_rows = l.size[0];
    auto l_row_ptrs = l.row_ptrs;
    auto l_vals = l.values;

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto l_nz = l_row_ptrs[row + 1] - 1;
        auto diag = sqrt(l_vals[l_nz]);
        if (is_finite(diag)) {
            l_vals[l_nz] = diag;
        } else {
            l_vals[l_nz] = one<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_INIT_FACTOR_KERNEL);


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    size_type iterations,
                    matrix::view::coo<const ValueType, const IndexType> a_lower,
                    matrix::view::csr<ValueType, IndexType> l)
{
    auto num_rows = a_lower.size[0];
    auto l_row_ptrs = l.row_ptrs;
    auto l_col_idxs = l.col_idxs;
    auto l_vals = l.values;
    auto a_vals = a_lower.values;

    for (size_type i = 0; i < iterations; ++i) {
#pragma omp parallel for
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type l_nz = l_row_ptrs[row]; l_nz < l_row_ptrs[row + 1];
                 ++l_nz) {
                auto col = l_col_idxs[l_nz];
                auto a_val = a_vals[l_nz];
                // accumulate l(row,:) * l(col,:) without the last entry l(col,
                // col)
                ValueType sum{};
                auto l_begin = l_row_ptrs[row];
                auto l_end = l_row_ptrs[row + 1];
                auto lh_begin = l_row_ptrs[col];
                auto lh_end = l_row_ptrs[col + 1];
                while (l_begin < l_end && lh_begin < lh_end) {
                    auto l_col = l_col_idxs[l_begin];
                    auto lh_row = l_col_idxs[lh_begin];
                    if (l_col == lh_row && l_col < col) {
                        sum += load(l_vals + l_begin) *
                               conj(load(l_vals + lh_begin));
                    }
                    l_begin += (l_col <= lh_row);
                    lh_begin += (lh_row <= l_col);
                }
                auto new_val = a_val - sum;
                if (row == col) {
                    new_val = sqrt(new_val);
                } else {
                    auto diag = load(l_vals + l_row_ptrs[col + 1] - 1);
                    new_val = new_val / diag;
                }
                if (is_finite(new_val)) {
                    store(l_vals + l_nz, new_val);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_COMPUTE_FACTOR_KERNEL);


}  // namespace par_ic_factorization
}  // namespace omp
}  // namespace kernels
}  // namespace gko
