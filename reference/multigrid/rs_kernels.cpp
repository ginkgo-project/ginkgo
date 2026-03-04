// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/rs_kernels.hpp"

#include <algorithm>
#include <memory>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The RS solver namespace.
 *
 * @ingroup rs
 */
namespace rs {


template <typename ValueType, typename IndexType>
void compute_soc_row_ptrs(std::shared_ptr<const ReferenceExecutor> exec,
                          const matrix::Csr<ValueType, IndexType>* A,
                          remove_complex<ValueType> theta, IndexType* row_ptrs)
{
    using real_type = remove_complex<ValueType>;

    const auto n = A->get_size()[0];
    const auto* a_row_ptrs = A->get_const_row_ptrs();
    const auto* a_col_idxs = A->get_const_col_idxs();
    const auto* a_vals = A->get_const_values();

    row_ptrs[0] = 0;

    for (IndexType i = 0; i < n; ++i) {
        real_type max_offdiag = zero<real_type>();
        IndexType row_nnz = 0;

        for (IndexType jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
            const auto j = a_col_idxs[jj];
            if (j != i) {
                // assuming an M-matrix
                max_offdiag = std::max(max_offdiag, -real(a_vals[jj]));
            }
        }

        for (IndexType jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
            const auto j = a_col_idxs[jj];
            if (j != i && -real(a_vals[jj]) >= theta * max_offdiag) {
                row_nnz++;
            }
        }

        row_ptrs[i + 1] = row_ptrs[i] + row_nnz;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_COMPUTE_SOC_ROW_PTRS_KERNEL);


template <typename ValueType, typename IndexType>
void fill_soc(std::shared_ptr<const ReferenceExecutor> exec,
              const matrix::Csr<ValueType, IndexType>* A,
              remove_complex<ValueType> theta,
              matrix::Csr<ValueType, IndexType>* S)
{
    using real_type = remove_complex<ValueType>;

    const auto n = A->get_size()[0];
    const auto* a_row_ptrs = A->get_const_row_ptrs();
    const auto* a_col_idxs = A->get_const_col_idxs();
    const auto* a_vals = A->get_const_values();

    const auto* s_row_ptrs = S->get_const_row_ptrs();
    auto* s_col_idxs = S->get_col_idxs();
    auto* s_vals = S->get_values();

    for (IndexType i = 0; i < n; ++i) {
        real_type max_offdiag = zero<real_type>();

        for (IndexType jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
            const auto j = a_col_idxs[jj];
            if (j != i) {
                max_offdiag = std::max(max_offdiag, -real(a_vals[jj]));
            }
        }

        IndexType write_pos = s_row_ptrs[i];

        for (IndexType jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
            const auto j = a_col_idxs[jj];
            if (j != i && -real(a_vals[jj]) >= theta * max_offdiag) {
                s_col_idxs[write_pos] = j;
                s_vals[write_pos] = one<ValueType>();
                write_pos++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RS_FILL_SOC_KERNEL);


// Compute lambda_i = number of strong neighbors
template <typename ValueType, typename IndexType>
void compute_lambda(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* S,
                    IndexType* lambda)
{
    const auto n = S->get_size()[0];
    const auto* row_ptrs = S->get_const_row_ptrs();

    for (IndexType i = 0; i < n; ++i) {
        lambda[i] = row_ptrs[i + 1] - row_ptrs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_COMPUTE_LAMBDA_KERNEL);


// Init all nodes as undecided (0)
template <typename IndexType>
void init_cf(std::shared_ptr<const ReferenceExecutor> exec,
             array<IndexType>& cf_marker)
{
    auto* cf = cf_marker.get_data();
    for (size_type i = 0; i < cf_marker.get_size(); ++i) {
        cf[i] = 0;  // 0 = undecided
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RS_INIT_CF_KERNEL);


// Classical RS greedy selection
template <typename ValueType, typename IndexType>
void rs_coarsening(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Csr<ValueType, IndexType>* S,
                   IndexType* lambda, array<IndexType>& cf_marker)
{
    const auto n = S->get_size()[0];
    const auto* row_ptrs = S->get_const_row_ptrs();
    const auto* col_idxs = S->get_const_col_idxs();

    auto* cf = cf_marker.get_data();

    while (true) {
        // Find max lambda among undecided (cf == 0)
        IndexType max_idx = -1;
        IndexType max_val = -1;

        for (IndexType i = 0; i < n; ++i) {
            if (cf[i] == 0 && lambda[i] > max_val) {
                max_val = lambda[i];
                max_idx = i;
            }
        }

        if (max_idx == -1) {
            break;  // done
        }

        // Mark as C-point
        cf[max_idx] = 1;

        // Strong neighbors become F-points
        for (IndexType jj = row_ptrs[max_idx]; jj < row_ptrs[max_idx + 1];
             ++jj) {
            const auto j = col_idxs[jj];
            if (cf[j] == 0) {
                cf[j] = -1;  // F-point

                // Reduce lambda of its strong neighbors
                for (IndexType kk = row_ptrs[j]; kk < row_ptrs[j + 1]; ++kk) {
                    const auto neigh = col_idxs[kk];
                    if (cf[neigh] == 0) {
                        lambda[neigh]--;
                    }
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RS_COARSENING_KERNEL);


// Cleanup: ensure no undecided remain (make them F)
template <typename IndexType>
void rs_cleanup(std::shared_ptr<const ReferenceExecutor> exec,
                array<IndexType>& cf_marker)
{
    auto* cf = cf_marker.get_data();
    for (size_type i = 0; i < cf_marker.get_size(); ++i) {
        if (cf[i] == 0) {  // undecided
            cf[i] = -1;    // make F
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RS_CLEANUP_KERNEL);


// Count C-points
template <typename IndexType>
void count_coarse(std::shared_ptr<const ReferenceExecutor> exec,
                  const array<IndexType>& cf_marker, IndexType* coarse_size)
{
    IndexType count = 0;
    const auto* cf = cf_marker.get_const_data();

    for (size_type i = 0; i < cf_marker.get_size(); ++i) {
        if (cf[i] == 1) {
            count++;
        }
    }

    *coarse_size = count;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RS_COUNT_COARSE_KERNEL);


// Fill coarse row index array
template <typename IndexType>
void fill_coarse_rows(std::shared_ptr<const ReferenceExecutor> exec,
                      const array<IndexType>& cf_marker, IndexType* coarse_rows)
{
    const auto* cf = cf_marker.get_const_data();

    IndexType idx = 0;
    for (size_type i = 0; i < cf_marker.get_size(); ++i) {
        if (cf[i] == 1) {
            coarse_rows[idx++] = static_cast<IndexType>(i);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RS_FILL_COARSE_ROWS_KERNEL);


}  // namespace rs
}  // namespace reference
}  // namespace kernels
}  // namespace gko
