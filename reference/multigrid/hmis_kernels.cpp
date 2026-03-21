// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/hmis_kernels.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/allocator.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace hmis {


template <typename ValueType, typename IndexType>
void compute_strength(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* system_matrix,
                      remove_complex<ValueType> theta,
                      IndexType* strength_row_ptrs,
                      array<IndexType>& strength_col_idxs)
{
    const auto num_rows = system_matrix->get_size()[0];
    const auto row_ptrs = system_matrix->get_const_row_ptrs();
    const auto col_idxs = system_matrix->get_const_col_idxs();
    const auto vals = system_matrix->get_const_values();

    // Pass 1: count strong connections per row
    for (size_type row = 0; row < num_rows; ++row) {
        // Find max negative off-diagonal: max_{k!=i}(-a_ik)
        auto max_neg = zero<remove_complex<ValueType>>();
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            if (col_idxs[idx] != static_cast<IndexType>(row)) {
                auto neg_val = -real(vals[idx]);
                max_neg = max(max_neg, neg_val);
            }
        }
        auto threshold = theta * max_neg;
        IndexType count = 0;
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            if (col_idxs[idx] != static_cast<IndexType>(row)) {
                auto neg_val = -real(vals[idx]);
                if (neg_val >= threshold) {
                    count++;
                }
            }
        }
        strength_row_ptrs[row] = count;
    }

    // Prefix sum to get row pointers
    components::prefix_sum_nonnegative(exec, strength_row_ptrs, num_rows + 1);

    // Pass 2: fill column indices
    const auto total_nnz = strength_row_ptrs[num_rows];
    strength_col_idxs.resize_and_reset(total_nnz);
    auto col_data = strength_col_idxs.get_data();
    for (size_type row = 0; row < num_rows; ++row) {
        auto max_neg = zero<remove_complex<ValueType>>();
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            if (col_idxs[idx] != static_cast<IndexType>(row)) {
                auto neg_val = -real(vals[idx]);
                max_neg = max(max_neg, neg_val);
            }
        }
        auto threshold = theta * max_neg;
        auto write_pos = strength_row_ptrs[row];
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            if (col_idxs[idx] != static_cast<IndexType>(row)) {
                auto neg_val = -real(vals[idx]);
                if (neg_val >= threshold) {
                    col_data[write_pos] = col_idxs[idx];
                    write_pos++;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HMIS_COMPUTE_STRENGTH_KERNEL);


template <typename ValueType, typename IndexType>
void ruge_stueben_first_pass(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* strength,
    const matrix::Csr<ValueType, IndexType>* strength_transpose,
    array<IndexType>& cf_marker)
{
    // CF marker values: -1 = unassigned, 1 = C-point, -2 = Z_PT (provisional
    // F-point from first pass), 0 = F-point
    const auto num_rows = cf_marker.get_size();
    const auto s_row_ptrs = strength->get_const_row_ptrs();
    const auto s_col_idxs = strength->get_const_col_idxs();
    const auto st_row_ptrs = strength_transpose->get_const_row_ptrs();
    const auto st_col_idxs = strength_transpose->get_const_col_idxs();
    auto marker = cf_marker.get_data();

    // Initialize all to unassigned
    for (size_type i = 0; i < num_rows; ++i) {
        marker[i] = -1;
    }

    // Compute initial measures: w_i = |S_i^T| (number of points strongly
    // depending on i)
    std::vector<IndexType> measures(num_rows);
    for (size_type i = 0; i < num_rows; ++i) {
        measures[i] = st_row_ptrs[i + 1] - st_row_ptrs[i];
    }

    // Points with no strong connections at all become C-points immediately
    for (size_type i = 0; i < num_rows; ++i) {
        if (s_row_ptrs[i + 1] == s_row_ptrs[i] && measures[i] == 0) {
            marker[i] = 1;  // C-point (isolated)
        }
    }

    // Main Ruge-Stueben first pass loop
    IndexType num_unassigned = 0;
    for (size_type i = 0; i < num_rows; ++i) {
        if (marker[i] == -1) {
            num_unassigned++;
        }
    }

    while (num_unassigned > 0) {
        // Find the unassigned point with maximum measure
        IndexType max_measure = -1;
        IndexType max_point = -1;
        for (size_type i = 0; i < num_rows; ++i) {
            if (marker[i] == -1 && measures[i] > max_measure) {
                max_measure = measures[i];
                max_point = static_cast<IndexType>(i);
            }
        }
        if (max_point == -1) {
            break;
        }

        // Mark as C-point
        marker[max_point] = 1;
        num_unassigned--;

        // S^T-neighbors of new C-point (points that depend on it) become Z_PT
        for (auto idx = st_row_ptrs[max_point];
             idx < st_row_ptrs[max_point + 1]; ++idx) {
            auto dep = st_col_idxs[idx];
            if (marker[dep] == -1) {
                marker[dep] = -2;  // Z_PT (provisional F)
                num_unassigned--;
                // For each new Z_PT: increment measures of its S-neighbors
                // (they lost coverage and need more)
                for (auto jdx = s_row_ptrs[dep]; jdx < s_row_ptrs[dep + 1];
                     ++jdx) {
                    auto neighbor = s_col_idxs[jdx];
                    if (marker[neighbor] == -1) {
                        measures[neighbor]++;
                    }
                }
            }
        }

        // S-neighbors of new C-point: decrement measures
        for (auto idx = s_row_ptrs[max_point]; idx < s_row_ptrs[max_point + 1];
             ++idx) {
            auto neighbor = s_col_idxs[idx];
            if (marker[neighbor] == -1) {
                measures[neighbor]--;
                if (measures[neighbor] <= 0) {
                    marker[neighbor] = -2;  // Z_PT
                    num_unassigned--;
                    // Cascade: increment measures of this new Z_PT's
                    // S-neighbors
                    for (auto jdx = s_row_ptrs[neighbor];
                         jdx < s_row_ptrs[neighbor + 1]; ++jdx) {
                        auto nn = s_col_idxs[jdx];
                        if (marker[nn] == -1) {
                            measures[nn]++;
                        }
                    }
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HMIS_RUGE_STUEBEN_FIRST_PASS_KERNEL);


template <typename ValueType, typename IndexType>
void pmis_coarsen(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* strength,
                  const matrix::Csr<ValueType, IndexType>* strength_transpose,
                  unsigned max_iterations, array<IndexType>& cf_marker)
{
    // PMIS finalization: handle Z_PT points from Ruge-Stueben first pass
    const auto num_rows = cf_marker.get_size();
    const auto s_row_ptrs = strength->get_const_row_ptrs();
    const auto s_col_idxs = strength->get_const_col_idxs();
    const auto st_row_ptrs = strength_transpose->get_const_row_ptrs();
    const auto st_col_idxs = strength_transpose->get_const_col_idxs();
    auto marker = cf_marker.get_data();

    // Check if there are any Z_PT points to process
    bool has_zpt = false;
    for (size_type i = 0; i < num_rows; ++i) {
        if (marker[i] == -2) {
            has_zpt = true;
            break;
        }
    }
    if (!has_zpt) {
        // All points already assigned, just convert any remaining -2 to F
        return;
    }

    // Compute measures for undecided points
    // Z_PT points with measure >= 1 become undecided (-1), else F-point (0)
    std::vector<double> measures(num_rows, 0.0);
    for (size_type i = 0; i < num_rows; ++i) {
        if (marker[i] == -2) {
            // Measure = number of S^T-neighbors that are undecided or Z_PT
            IndexType count = 0;
            for (auto idx = st_row_ptrs[i]; idx < st_row_ptrs[i + 1]; ++idx) {
                auto dep = st_col_idxs[idx];
                if (marker[dep] == -2 || marker[dep] == -1) {
                    count++;
                }
            }
            if (count >= 1) {
                // Deterministic pseudo-random perturbation based on row index
                double rand_val =
                    static_cast<double>((i * 541 + 1013) % 10000) / 10000.0;
                measures[i] = static_cast<double>(count) + rand_val;
                marker[i] = -1;  // undecided
            } else {
                // Check if this point has any C-point in its strong
                // connections
                bool has_c = false;
                for (auto idx = s_row_ptrs[i]; idx < s_row_ptrs[i + 1]; ++idx) {
                    if (marker[s_col_idxs[idx]] == 1) {
                        has_c = true;
                        break;
                    }
                }
                if (has_c) {
                    marker[i] = 0;  // F-point
                } else {
                    marker[i] = 1;  // isolated, make C-point
                }
            }
        }
    }

    // PMIS iterative loop (simplified for serial reference)
    for (unsigned iter = 0; iter < max_iterations; ++iter) {
        bool changed = false;

        // Step 1: Tentatively mark undecided points with measure > 1 as IS
        // candidates Step 2: Conflict resolution among S-neighbors: higher
        // measure wins
        std::vector<bool> is_candidate(num_rows, false);
        for (size_type i = 0; i < num_rows; ++i) {
            if (marker[i] != -1) continue;
            if (measures[i] <= 1.0) continue;
            // Check if i has max measure among all undecided S and S^T
            // neighbors
            bool is_max = true;
            for (auto idx = s_row_ptrs[i]; idx < s_row_ptrs[i + 1]; ++idx) {
                auto j = s_col_idxs[idx];
                if (marker[j] == -1 && measures[j] > measures[i]) {
                    is_max = false;
                    break;
                }
            }
            if (is_max) {
                for (auto idx = st_row_ptrs[i]; idx < st_row_ptrs[i + 1];
                     ++idx) {
                    auto j = st_col_idxs[idx];
                    if (marker[j] == -1 && measures[j] > measures[i]) {
                        is_max = false;
                        break;
                    }
                }
            }
            if (is_max) {
                is_candidate[i] = true;
            }
        }

        // IS survivors become C-points
        for (size_type i = 0; i < num_rows; ++i) {
            if (is_candidate[i]) {
                marker[i] = 1;  // C-point
                changed = true;
            }
        }

        // Undecided with measure < 1 become F-points
        for (size_type i = 0; i < num_rows; ++i) {
            if (marker[i] == -1 && measures[i] < 1.0) {
                marker[i] = 0;  // F-point
                changed = true;
            }
        }

        // Undecided adjacent to new C-point become F-points
        for (size_type i = 0; i < num_rows; ++i) {
            if (marker[i] != -1) continue;
            for (auto idx = s_row_ptrs[i]; idx < s_row_ptrs[i + 1]; ++idx) {
                if (marker[s_col_idxs[idx]] == 1) {
                    marker[i] = 0;  // F-point
                    changed = true;
                    break;
                }
            }
        }

        // Check if all assigned
        bool all_assigned = true;
        for (size_type i = 0; i < num_rows; ++i) {
            if (marker[i] == -1) {
                all_assigned = false;
                break;
            }
        }
        if (all_assigned || !changed) break;
    }

    // Force any remaining undecided to F-points
    for (size_type i = 0; i < num_rows; ++i) {
        if (marker[i] == -1 || marker[i] == -2) {
            marker[i] = 0;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HMIS_PMIS_COARSEN_KERNEL);


template <typename IndexType>
void build_cpoint_map(std::shared_ptr<const ReferenceExecutor> exec,
                      const array<IndexType>& cf_marker,
                      IndexType* num_c_points, array<IndexType>& c_point_map)
{
    const auto num_rows = cf_marker.get_size();
    const auto marker = cf_marker.get_const_data();
    auto map = c_point_map.get_data();

    IndexType count = 0;
    for (size_type i = 0; i < num_rows; ++i) {
        if (marker[i] == 1) {
            map[i] = count;
            count++;
        } else {
            map[i] = -1;
        }
    }
    *num_c_points = count;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_HMIS_BUILD_CPOINT_MAP_KERNEL);


template <typename ValueType, typename IndexType>
void build_prolongation(std::shared_ptr<const ReferenceExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* system_matrix,
                        const matrix::Csr<ValueType, IndexType>* strength,
                        const array<IndexType>& cf_marker,
                        IndexType num_c_points,
                        const array<IndexType>& c_point_map,
                        IndexType* prolongation_row_ptrs,
                        array<IndexType>& prolongation_col_idxs,
                        array<ValueType>& prolongation_vals)
{
    const auto num_rows = system_matrix->get_size()[0];
    const auto a_row_ptrs = system_matrix->get_const_row_ptrs();
    const auto a_col_idxs = system_matrix->get_const_col_idxs();
    const auto a_vals = system_matrix->get_const_values();
    const auto s_row_ptrs = strength->get_const_row_ptrs();
    const auto s_col_idxs = strength->get_const_col_idxs();
    const auto marker = cf_marker.get_const_data();
    const auto c_map = c_point_map.get_const_data();

    // Helper: check if column j is a strong connection of row i
    auto is_strong = [&](IndexType row, IndexType col) -> bool {
        for (auto idx = s_row_ptrs[row]; idx < s_row_ptrs[row + 1]; ++idx) {
            if (s_col_idxs[idx] == col) return true;
        }
        return false;
    };

    // Pass 1: count nnz per row
    for (size_type i = 0; i < num_rows; ++i) {
        if (marker[i] == 1) {
            // C-point: one entry (injection)
            prolongation_row_ptrs[i] = 1;
        } else {
            // F-point: count strong C-neighbors
            IndexType count = 0;
            for (auto idx = s_row_ptrs[i]; idx < s_row_ptrs[i + 1]; ++idx) {
                if (marker[s_col_idxs[idx]] == 1) {
                    count++;
                }
            }
            prolongation_row_ptrs[i] = count;
        }
    }

    // Prefix sum
    components::prefix_sum_nonnegative(exec, prolongation_row_ptrs,
                                       num_rows + 1);
    const auto total_nnz = prolongation_row_ptrs[num_rows];
    prolongation_col_idxs.resize_and_reset(total_nnz);
    prolongation_vals.resize_and_reset(total_nnz);
    auto p_col = prolongation_col_idxs.get_data();
    auto p_val = prolongation_vals.get_data();

    // Pass 2: fill prolongation entries
    for (size_type i = 0; i < num_rows; ++i) {
        auto write_pos = prolongation_row_ptrs[i];

        if (marker[i] == 1) {
            // C-point: identity injection
            p_col[write_pos] = c_map[i];
            p_val[write_pos] = one<ValueType>();
            continue;
        }

        // F-point: classical interpolation
        // Collect strong C-neighbors of i (C_i)
        std::vector<IndexType> strong_c_neighbors;
        for (auto idx = s_row_ptrs[i]; idx < s_row_ptrs[i + 1]; ++idx) {
            if (marker[s_col_idxs[idx]] == 1) {
                strong_c_neighbors.push_back(s_col_idxs[idx]);
            }
        }

        if (strong_c_neighbors.empty()) {
            // No strong C-neighbors: this shouldn't normally happen after HMIS,
            // but handle gracefully by setting zero weights
            continue;
        }

        // Initialize weights for each strong C-neighbor
        std::vector<ValueType> weights(strong_c_neighbors.size(),
                                       zero<ValueType>());

        // Initialize modified diagonal
        auto diagonal = zero<ValueType>();

        // Process each neighbor j of i in the system matrix
        for (auto idx = a_row_ptrs[i]; idx < a_row_ptrs[i + 1]; ++idx) {
            auto j = a_col_idxs[idx];
            auto a_ij = a_vals[idx];

            if (j == static_cast<IndexType>(i)) {
                // Diagonal entry
                diagonal += a_ij;
                continue;
            }

            if (marker[j] == 1 && is_strong(static_cast<IndexType>(i), j)) {
                // Case 1: j is a strong C-neighbor of i
                // Find index in strong_c_neighbors
                for (size_type k = 0; k < strong_c_neighbors.size(); ++k) {
                    if (strong_c_neighbors[k] == j) {
                        weights[k] += a_ij;
                        break;
                    }
                }
            } else if (marker[j] == 0 &&
                       is_strong(static_cast<IndexType>(i), j)) {
                // Case 2: j is a strong F-neighbor of i
                // Distribute a_ij to strong C-neighbors of i proportionally
                // to j's connections to those C-neighbors
                // Find diagonal of j for sign determination
                auto diag_j = zero<ValueType>();
                for (auto jdx = a_row_ptrs[j]; jdx < a_row_ptrs[j + 1]; ++jdx) {
                    if (a_col_idxs[jdx] == j) {
                        diag_j = a_vals[jdx];
                        break;
                    }
                }
                auto sgn_positive =
                    (real(diag_j) >= zero<remove_complex<ValueType>>());

                // Compute sum of a_jk for k in C_i where sgn*a_jk < 0
                auto sum = zero<ValueType>();
                std::vector<ValueType> a_jk_vals(strong_c_neighbors.size(),
                                                 zero<ValueType>());
                for (size_type k = 0; k < strong_c_neighbors.size(); ++k) {
                    auto ck = strong_c_neighbors[k];
                    // Find a_jk in the system matrix
                    for (auto jdx = a_row_ptrs[j]; jdx < a_row_ptrs[j + 1];
                         ++jdx) {
                        if (a_col_idxs[jdx] == ck) {
                            a_jk_vals[k] = a_vals[jdx];
                            bool neg_product =
                                sgn_positive
                                    ? (real(a_vals[jdx]) <
                                       zero<remove_complex<ValueType>>())
                                    : (real(a_vals[jdx]) >
                                       zero<remove_complex<ValueType>>());
                            if (neg_product) {
                                sum += a_vals[jdx];
                            }
                            break;
                        }
                    }
                }

                if (sum != zero<ValueType>()) {
                    auto distribute = a_ij / sum;
                    for (size_type k = 0; k < strong_c_neighbors.size(); ++k) {
                        bool neg_product =
                            sgn_positive ? (real(a_jk_vals[k]) <
                                            zero<remove_complex<ValueType>>())
                                         : (real(a_jk_vals[k]) >
                                            zero<remove_complex<ValueType>>());
                        if (neg_product) {
                            weights[k] += distribute * a_jk_vals[k];
                        }
                    }
                } else {
                    // No valid distribution path: lump into diagonal
                    diagonal += a_ij;
                }
            } else {
                // Case 3: weak connection - lump into diagonal
                diagonal += a_ij;
            }
        }

        // Normalize: P[i,j] = -w[j] / diagonal
        if (diagonal != zero<ValueType>()) {
            for (size_type k = 0; k < strong_c_neighbors.size(); ++k) {
                p_col[write_pos + k] = c_map[strong_c_neighbors[k]];
                p_val[write_pos + k] = -weights[k] / diagonal;
            }
        } else {
            // Diagonal is zero: set all weights to zero
            for (size_type k = 0; k < strong_c_neighbors.size(); ++k) {
                p_col[write_pos + k] = c_map[strong_c_neighbors[k]];
                p_val[write_pos + k] = zero<ValueType>();
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HMIS_BUILD_PROLONGATION_KERNEL);


}  // namespace hmis
}  // namespace reference
}  // namespace kernels
}  // namespace gko
