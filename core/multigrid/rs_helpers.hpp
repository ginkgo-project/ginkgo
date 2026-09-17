// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MULTIGRID_RS_HELPERS_HPP_
#define GKO_CORE_MULTIGRID_RS_HELPERS_HPP_


#include <algorithm>
#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/device_views.hpp>


namespace gko {
namespace multigrid {
namespace rs {


/**
 * Returns the empty CSR view that stands in for a missing off-diagonal block.
 *
 * The RS kernels take the off-diagonal block of a distributed matrix as a
 * device view, which - unlike a pointer - cannot be null. A non-distributed
 * matrix has no such block and passes this view instead: its null row pointers
 * tell the kernels that the local rows have no remote couplings.
 */
template <typename ValueType, typename IndexType>
constexpr matrix::view::csr<const ValueType, const IndexType> no_off_diag_view()
{
    return {dim<2>{}, 0, nullptr, nullptr, nullptr};
}


/**
 * Runs the greedy Ruge-Stueben C/F splitting on host-resident data.
 *
 * Repeatedly picks the undecided row with the largest number of strong
 * neighbors (`lambda`) and turns it into a C-point, turns that row's undecided
 * strong neighbors into F-points, and counts `lambda` down for the still
 * undecided strong neighbors of those new F-points. Ties between rows with
 * equal `lambda` are broken towards the smaller row index.
 *
 * All the pointers must be accessible on the host.
 *
 * @param host_exec  executor owning the host memory used for scratch space
 * @param num_rows  the number of rows of the system matrix
 * @param row_ptrs  the row pointers of the system matrix
 * @param col_idxs  the column indices of the system matrix
 * @param is_strong  the strength-of-connection mask, one entry per stored
 *                   element of the system matrix
 * @param lambda  the number of strong neighbors per row, counted down in place
 * @param cf_marker  the resulting C/F marker, overwritten with 0 = undecided,
 * 1 = C-point, -1 = F-point
 */
template <typename IndexType>
void greedy_cf_splitting(std::shared_ptr<const Executor> host_exec,
                         size_type num_rows, const IndexType* row_ptrs,
                         const IndexType* col_idxs, const bool* is_strong,
                         IndexType* lambda, IndexType* cf_marker)
{
    constexpr IndexType invalid = -1;
    const auto n = static_cast<IndexType>(num_rows);

    std::fill_n(cf_marker, num_rows, IndexType{0});
    if (n == 0) {
        return;
    }

    // lambda only ever decreases, so its initial maximum bounds every value it
    // will take and thus the number of buckets
    const auto max_lambda = *std::max_element(lambda, lambda + n);
    array<IndexType> bucket_heads(host_exec,
                                  static_cast<size_type>(max_lambda) + 1);
    array<IndexType> next_rows(host_exec, num_rows);
    array<IndexType> prev_rows(host_exec, num_rows);
    array<IndexType> ordered_rows(host_exec, num_rows);
    auto heads = bucket_heads.get_data();
    auto next = next_rows.get_data();
    auto prev = prev_rows.get_data();
    auto ordered = ordered_rows.get_data();
    std::fill_n(heads, bucket_heads.get_size(), invalid);

    // a row is linked into bucket lambda[row] exactly while it is undecided
    // and its lambda is non-negative
    const auto link = [&](IndexType row, IndexType bucket) {
        prev[row] = invalid;
        next[row] = heads[bucket];
        if (heads[bucket] != invalid) {
            prev[heads[bucket]] = row;
        }
        heads[bucket] = row;
    };
    const auto unlink = [&](IndexType row, IndexType bucket) {
        if (prev[row] != invalid)
            next[prev[row]] = next[row];
        else
            heads[bucket] = next[row];
        if (next[row] != invalid) {
            prev[next[row]] = prev[row];
        }
    };

    for (IndexType row = 0; row < n; ++row) {
        link(row, lambda[row]);
    }

    for (auto bucket = max_lambda; bucket >= 0; --bucket) {
        // Put the bucket into index order, so that taking its head reproduces
        // the "smallest index wins" tie-break of the plain argmax scan. Rows
        // counted down into this bucket arrived in arbitrary order, but no row
        // is ever added to it again: every row still undecided at this point
        // has a lambda of at most `bucket`, so counting down can only move
        // rows to strictly lower buckets. One sort here is therefore enough.
        IndexType num_candidates{};
        for (auto row = heads[bucket]; row != invalid; row = next[row]) {
            ordered[num_candidates++] = row;
        }
        std::sort(ordered, ordered + num_candidates);
        heads[bucket] = invalid;
        for (auto i = num_candidates; i > 0; --i) {
            link(ordered[i - 1], bucket);
        }

        while (heads[bucket] != invalid) {
            const auto c_point = heads[bucket];
            unlink(c_point, bucket);
            cf_marker[c_point] = 1;

            for (auto nz = row_ptrs[c_point]; nz < row_ptrs[c_point + 1];
                 ++nz) {
                const auto f_point = col_idxs[nz];
                if (!is_strong[nz] || cf_marker[f_point] != 0) {
                    continue;
                }
                if (lambda[f_point] >= 0) {
                    unlink(f_point, lambda[f_point]);
                }
                cf_marker[f_point] = -1;

                // the new F-point no longer counts towards its own undecided
                // strong neighbors
                for (auto f_nz = row_ptrs[f_point];
                     f_nz < row_ptrs[f_point + 1]; ++f_nz) {
                    const auto neighbor = col_idxs[f_nz];
                    if (!is_strong[f_nz] || cf_marker[neighbor] != 0) {
                        continue;
                    }
                    if (lambda[neighbor] >= 0) {
                        unlink(neighbor, lambda[neighbor]);
                    }
                    if (--lambda[neighbor] >= 0) {
                        link(neighbor, lambda[neighbor]);
                    }
                }
            }
        }
    }
}


}  // namespace rs
}  // namespace multigrid
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_RS_HELPERS_HPP_
