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

#include "core/multigrid/rs_helpers.hpp"

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
void check_m_matrix(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::view::csr<const ValueType, const IndexType> matrix,
                    array<bool>& is_m_matrix_array)
{
    const auto num_rows = matrix.size[0];
    const auto row_ptrs = matrix.row_ptrs;
    const auto col_idxs = matrix.col_idxs;
    const auto values = matrix.values;

    auto is_m_matrix = is_m_matrix_array.get_data();
    *is_m_matrix = true;

    for (size_type row = 0; row < num_rows; ++row) {
        bool has_diag = false;

        for (auto nz = row_ptrs[row]; nz < row_ptrs[row + 1]; ++nz) {
            const auto col = col_idxs[nz];
            const auto val = real(values[nz]);

            if (row == col) {
                has_diag = true;
                if (val <= 0.0) {
                    *is_m_matrix = false;
                    return;
                }
            } else {
                if (val > 0.0) {
                    *is_m_matrix = false;
                    return;
                }
            }
        }

        if (!has_diag) {
            *is_m_matrix = false;
            return;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_CHECK_M_MATRIX_KERNEL);


template <typename ValueType, typename IndexType>
void compute_soc_and_run_rs(
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> A, double theta,
    array<bool>& is_strong, array<IndexType>& lambda,
    array<IndexType>& cf_marker, IndexType& coarse_size)
{
    using real_type = remove_complex<ValueType>;
    const auto n = A.size[0];
    const auto* a_row_ptrs = A.row_ptrs;
    const auto* a_col_idxs = A.col_idxs;
    const auto* a_vals = A.values;
    bool* is_strong_vals = is_strong.get_data();
    auto* lambda_vals = lambda.get_data();
    auto* cf = cf_marker.get_data();

    /// 1. COMPUTE SOC MASK
    // assuming an M-matrix
    for (IndexType i = 0; i < n; ++i) {
        real_type max_offdiag = zero<real_type>();

        // pass 1: find max off-diagonal
        for (IndexType jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
            if (A.col_idxs[jj] != i) {
                max_offdiag = std::max(max_offdiag, -real(a_vals[jj]));
            }
        }

        // pass 2: set mask
        for (IndexType jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
            const auto j = A.col_idxs[jj];
            is_strong_vals[jj] =
                (j != i && -real(a_vals[jj]) >= theta * max_offdiag);
        }
    }

    /// 2. COMPUTE lambda_i = number of strong nbrs
    for (IndexType i = 0; i < n; ++i) {
        IndexType count = 0;
        for (IndexType jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
            if (is_strong_vals[jj]) count++;
        }
        lambda_vals[i] = count;
    }

    /// 3. RS-COARSENING (0 = undecided, 1 = C-point, -1 = F-point)
    gko::multigrid::rs::greedy_cf_splitting(exec, n, a_row_ptrs, a_col_idxs,
                                            is_strong_vals, lambda_vals, cf);

    /// 4. CLEANUP, MAKE SURE NO UNDECIDED REMAIN
    for (size_type i = 0; i < cf_marker.get_size(); ++i) {
        if (cf[i] == 0) {  // undecided
            cf[i] = -1;    // make F
        }
    }

    /// 5. COUNT C-POINTS
    IndexType count = 0;
    for (size_type i = 0; i < cf_marker.get_size(); ++i) {
        if (cf[i] == 1) {
            count++;
        }
    }
    coarse_size = count;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_COMPUTE_SOC_AND_RUN_RS_KERNEL);


template <typename ValueType, typename IndexType>
void fill_coarse_and_compute_prolong_row_ptrs(
    std::shared_ptr<const ReferenceExecutor> exec,
    const array<IndexType>& cf_marker, array<IndexType>& coarse_rows,
    array<IndexType>& fine_to_coarse,
    matrix::view::csr<const ValueType, const IndexType> A,
    const array<bool>& is_strong, array<IndexType>& row_ptrs)
{
    const auto* cf = cf_marker.get_const_data();
    auto* coarse_rows_vals = coarse_rows.get_data();
    auto* fine_to_coarse_vals = fine_to_coarse.get_data();
    auto* row_ptrs_vals = row_ptrs.get_data();
    const bool* is_strong_vals = is_strong.get_const_data();
    const auto n = A.size[0];
    const auto* a_row_ptrs = A.row_ptrs;
    const auto* a_col_idxs = A.col_idxs;

    /// 1. FILL COARSE ROW INDEX ARRAY
    IndexType idx = 0;
    for (size_type i = 0; i < cf_marker.get_size(); ++i) {
        if (cf[i] == 1) {
            coarse_rows_vals[idx++] = static_cast<IndexType>(i);
        }
    }

    /// 2. FILL IN THE fine_to_coarse
    IndexType coarse_id = 0;
    for (size_type i = 0; i < cf_marker.get_size(); ++i) {
        if (cf[i] == 1) {
            fine_to_coarse_vals[i] = coarse_id++;
        } else {
            fine_to_coarse_vals[i] = -1;
        }
    }

    /// 3. COMPUTE INTERPOLATION ROW PTRS
    row_ptrs_vals[0] = 0;
    for (IndexType i = 0; i < n; ++i) {
        IndexType row_nnz = 0;
        if (cf[i] == 1) {
            row_nnz = 1;  // identity for C-points
        } else {
            // count strong C-neighbors
            for (auto jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
                if (is_strong_vals[jj] && cf[a_col_idxs[jj]] == 1) {
                    row_nnz++;
                }
            }
        }
        row_ptrs_vals[i + 1] = row_ptrs_vals[i] + row_nnz;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_FILL_COARSE_AND_COMPUTE_PROLONG_ROW_PTRS_KERNEL);


template <typename ValueType, typename IndexType>
void compute_interpolation(
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> A,
    const bool* is_strong, const array<IndexType>& cf_marker,
    const IndexType* fine_to_coarse, matrix::view::csr<ValueType, IndexType> P)
{
    const auto n = A.size[0];
    const auto* a_row_ptrs = A.row_ptrs;
    const auto* a_col_idxs = A.col_idxs;
    const auto* a_vals = A.values;
    const auto* cf = cf_marker.get_const_data();
    auto* p_row_ptrs = P.row_ptrs;
    auto* p_col_idxs = P.col_idxs;
    auto* p_vals = P.values;

    for (IndexType i = 0; i < n; ++i) {
        auto p_idx = p_row_ptrs[i];
        if (cf[i] == 1) {
            p_col_idxs[p_idx] = fine_to_coarse[i];
            p_vals[p_idx] = one<ValueType>();
        } else {
            // full classical RS interpolation formula:
            // w_ij = -(a_ij + sum_{k in F_strong} (a_ik * a_kj / sum_{m in
            // C_strong} a_im))
            //        / (a_ii + sum_{k in weak} a_ik)

            ValueType diag = zero<ValueType>();
            ValueType sum_weak = zero<ValueType>();
            ValueType sum_strong_c_val = zero<ValueType>();

            // accumulate sums
            for (auto jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
                auto j = a_col_idxs[jj];
                if (i == j)
                    diag = a_vals[jj];
                else if (!is_strong[jj])
                    sum_weak += a_vals[jj];
                else if (cf[j] == 1)
                    sum_strong_c_val += a_vals[jj];
            }

            ValueType denominator = diag + sum_weak;

            // compute weights for each strong C-neighbor
            for (auto jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
                if (is_strong[jj] && cf[a_col_idxs[jj]] == 1) {
                    auto j = a_col_idxs[jj];
                    ValueType numerator = a_vals[jj];  // a_ij is right here!

                    // contribution from strong F-neighbors k
                    for (auto kk = a_row_ptrs[i]; kk < a_row_ptrs[i + 1];
                         ++kk) {
                        if (is_strong[kk] && cf[a_col_idxs[kk]] == -1) {
                            auto k = a_col_idxs[kk];
                            ValueType a_ik = a_vals[kk];
                            ValueType a_kj = zero<ValueType>();
                            // only search for a_kj in row k
                            for (auto n_kj = a_row_ptrs[k];
                                 n_kj < a_row_ptrs[k + 1]; ++n_kj) {
                                if (a_col_idxs[n_kj] == j) {
                                    a_kj = a_vals[n_kj];
                                    break;
                                }
                            }
                            numerator += (a_ik * a_kj) / sum_strong_c_val;
                        }
                    }
                    p_col_idxs[p_idx] = fine_to_coarse[j];
                    p_vals[p_idx] = -numerator / denominator;
                    p_idx++;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_COMPUTE_INTERPOLATION_KERNEL);


}  // namespace rs
}  // namespace reference
}  // namespace kernels
}  // namespace gko
