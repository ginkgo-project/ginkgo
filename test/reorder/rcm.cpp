// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <deque>
#include <fstream>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/reorder/rcm.hpp>


#include "core/components/disjoint_sets.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


class Rcm : public CommonTestFixture {
protected:
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;
    using reorder_type = gko::reorder::Rcm<value_type, index_type>;
    using new_reorder_type = gko::experimental::reorder::Rcm<index_type>;
    using perm_type = gko::matrix::Permutation<index_type>;

    Rcm()
        : rng{63749},
          o_1138_bus_mtx(gko::read<CsrMtx>(
              std::ifstream(gko::matrices::location_1138_bus_mtx, std::ios::in),
              ref)),
          d_1138_bus_mtx(gko::read<CsrMtx>(
              std::ifstream(gko::matrices::location_1138_bus_mtx, std::ios::in),
              exec))
    {}

    static void ubfs_reference(
        std::shared_ptr<CsrMtx> mtx,
        index_type* const
            levels,  // Must be inf/max in all nodes connected to source
        const index_type start)
    {
        const auto row_ptrs = mtx->get_const_row_ptrs();
        const auto col_idxs = mtx->get_const_col_idxs();

        std::deque<index_type> q(0);
        q.push_back(start);
        levels[start] = 0;

        while (!q.empty()) {
            const auto node = q.front();
            q.pop_front();

            const auto level = levels[node];
            const auto neighbours_level = level + 1;
            const auto row_start = row_ptrs[node];
            const auto row_end = row_ptrs[node + 1];

            for (auto neighbour_i = row_start; neighbour_i < row_end;
                 ++neighbour_i) {
                const auto neighbour = col_idxs[neighbour_i];
                if (neighbours_level < levels[neighbour]) {
                    levels[neighbour] = neighbours_level;
                    q.push_back(neighbour);
                }
            }
        }
    }

    static gko::disjoint_sets<index_type> connected_components_reference(
        std::shared_ptr<CsrMtx> mtx)
    {
        auto num_rows = static_cast<index_type>(mtx->get_size()[0]);
        const auto row_ptrs = mtx->get_const_row_ptrs();
        const auto cols = mtx->get_const_col_idxs();
        gko::disjoint_sets<index_type> sets{mtx->get_executor(), num_rows};
        for (index_type row = 0; row < num_rows; row++) {
            for (auto nz = row_ptrs[row]; nz < row_ptrs[row + 1]; nz++) {
                const auto col = cols[nz];
                sets.join(row, col);
            }
        }
        return sets;
    }

    static void check_valid_start_node(std::shared_ptr<CsrMtx> mtx,
                                       index_type start,
                                       const gko::disjoint_sets<index_type>& cc,
                                       gko::reorder::starting_strategy strategy)
    {
        SCOPED_TRACE(start);
        const auto start_rep = cc.const_find(start);

        const auto n = mtx->get_size()[0];
        auto degrees = std::vector<index_type>(n);
        for (gko::size_type i = 0; i < n; ++i) {
            degrees[i] =
                mtx->get_const_row_ptrs()[i + 1] - mtx->get_const_row_ptrs()[i];
        }

        if (strategy == gko::reorder::starting_strategy::minimum_degree) {
            auto min_degree = std::numeric_limits<index_type>::max();
            for (gko::size_type i = 0; i < n; ++i) {
                if (cc.const_find(i) == start_rep && degrees[i] < min_degree) {
                    min_degree = degrees[i];
                }
            }
            ASSERT_EQ(min_degree, degrees[start]) << start;
            return;
        }

        // Check if any valid contender has a lowereq height than the
        // selected start node.

        std::vector<index_type> reference_current_levels(n);
        std::fill(reference_current_levels.begin(),
                  reference_current_levels.end(),
                  std::numeric_limits<index_type>::max());
        ubfs_reference(mtx, &reference_current_levels[0], start);

        // First find all contender nodes in the last UBFS level
        std::vector<index_type> reference_contenders;
        auto current_height = std::numeric_limits<index_type>::min();
        for (gko::size_type i = 0; i < n; ++i) {
            if (cc.const_find(i) == start_rep &&
                reference_current_levels[i] >= current_height) {
                if (reference_current_levels[i] > current_height) {
                    reference_contenders.clear();
                }
                reference_contenders.push_back(static_cast<index_type>(i));
                current_height = reference_current_levels[i];
            }
        }
        // remove all contenders of non-minimal degree
        auto contender_min_degree = *std::min_element(
            reference_contenders.begin(), reference_contenders.end(),
            [&](index_type u, index_type v) {
                return degrees[u] < degrees[v];
            });
        reference_contenders.erase(
            std::remove_if(reference_contenders.begin(),
                           reference_contenders.end(),
                           [&](index_type u) {
                               return degrees[u] > contender_min_degree;
                           }),
            reference_contenders.end());

        // then compute a level array for each of the contenders
        std::vector<std::vector<index_type>> reference_contenders_levels(
            reference_contenders.size());
        for (gko::size_type i = 0; i < reference_contenders.size(); ++i) {
            std::vector<index_type> reference_contender_levels(n);
            std::fill(reference_contender_levels.begin(),
                      reference_contender_levels.end(),
                      std::numeric_limits<index_type>::max());
            ubfs_reference(mtx, &reference_contender_levels[0],
                           reference_contenders[i]);
            reference_contenders_levels[i] = reference_contender_levels;
        }

        // and check if there is at least one minimum-degree contender with
        // lower or equal height
        std::vector<index_type> lower_contenders;
        for (gko::size_type i = 0; i < reference_contenders.size(); ++i) {
            auto contender_height = std::numeric_limits<index_type>::min();
            for (gko::size_type j = 0; j < n; ++j) {
                if (cc.const_find(j) == start_rep &&
                    reference_contenders_levels[i][j] > contender_height) {
                    contender_height = reference_contenders_levels[i][j];
                }
            }
            if (contender_height <= current_height) {
                lower_contenders.push_back(reference_contenders[i]);
            }
        }
        ASSERT_FALSE(lower_contenders.empty());
    }

    static void check_rcm_ordered(std::shared_ptr<CsrMtx> mtx,
                                  const perm_type* d_permutation,
                                  gko::reorder::starting_strategy strategy)
    {
        const auto host_permutation = d_permutation->clone(mtx->get_executor());
        const auto permutation = host_permutation->get_const_permutation();
        const auto n = mtx->get_size()[0];
        const auto row_ptrs = mtx->get_const_row_ptrs();
        const auto col_idxs = mtx->get_const_col_idxs();
        auto degrees = std::vector<index_type>(n);
        for (gko::size_type i = 0; i < n; ++i) {
            degrees[i] =
                mtx->get_const_row_ptrs()[i + 1] - mtx->get_const_row_ptrs()[i];
        }
        const auto cc = connected_components_reference(mtx);

        // Following checks for cm ordering, therefore create a reversed perm.
        std::vector<index_type> perm(permutation, permutation + n);
        std::reverse(perm.begin(), perm.end());
        std::vector<index_type> inv_perm(n, gko::invalid_index<index_type>());
        for (gko::size_type i = 0; i < n; i++) {
            ASSERT_GE(perm[i], 0) << i;
            ASSERT_LT(perm[i], n) << i;
            ASSERT_EQ(inv_perm[perm[i]], gko::invalid_index<index_type>()) << i;
            inv_perm[perm[i]] = static_cast<index_type>(i);
        }

        // Now check for cm ordering.

        gko::size_type base_offset = 0;
        std::vector<bool> already_visited(n);
        while (base_offset != n) {
            // Assert valid start node.
            check_valid_start_node(mtx, perm[base_offset], cc, strategy);

            // Assert valid level structure.
            // Also update base_offset and mark as visited while at it.
            std::vector<index_type> levels(n);
            std::fill(levels.begin(), levels.end(),
                      std::numeric_limits<index_type>::max());
            ubfs_reference(mtx, &levels[0], perm[base_offset]);

            index_type current_level = 0;
            const auto previous_base_offset = base_offset;
            for (gko::size_type i = 0; i < n; ++i) {
                const auto node = perm[i];
                if (levels[node] != std::numeric_limits<index_type>::max() &&
                    !already_visited[node]) {
                    already_visited[node] = true;
                    ++base_offset;

                    if (levels[node] == current_level) {
                        continue;
                    }
                    if (levels[node] == current_level + 1) {
                        ++current_level;
                        continue;
                    }
                    GTEST_FAIL() << "Level structure invalid at node " << node
                                 << ", level " << current_level;
                }
            }

            // Assert cm order within levels.
            for (auto i = previous_base_offset + 1 /* Skip start node */;
                 i < base_offset - 1; ++i) {
                const auto x = perm[i];
                const auto y = perm[i + 1];
                if (levels[x] != levels[y]) {
                    continue;  // Skip if on level border
                }
                const auto level = levels[x];

                // Get first neighbour of x in the previous level.
                auto x_first_neighbour =
                    perm[n - 1];  // There is always a neighbour, this is valid.
                const auto x_row_start = row_ptrs[x];
                const auto x_row_end = row_ptrs[x + 1];
                for (auto x_neighbour_idx = x_row_start;
                     x_neighbour_idx < x_row_end; ++x_neighbour_idx) {
                    const auto x_neighbour = col_idxs[x_neighbour_idx];
                    if (levels[x_neighbour] == level - 1) {
                        if (inv_perm[x_neighbour] <
                            inv_perm[x_first_neighbour]) {
                            x_first_neighbour = x_neighbour;
                        }
                    }
                }
                // Same again, for y.
                auto y_first_neighbour = perm[n - 1];
                const auto y_row_start = row_ptrs[y];
                const auto y_row_end = row_ptrs[y + 1];
                for (auto y_neighbour_idx = y_row_start;
                     y_neighbour_idx < y_row_end; ++y_neighbour_idx) {
                    const auto y_neighbour = col_idxs[y_neighbour_idx];
                    if (levels[y_neighbour] == level - 1) {
                        if (inv_perm[y_neighbour] <
                            inv_perm[y_first_neighbour]) {
                            y_first_neighbour = y_neighbour;
                        }
                    }
                }

                // Assert the ... is not after the ... in the previous level.
                ASSERT_GE(inv_perm[y_first_neighbour],
                          inv_perm[x_first_neighbour])
                    << "First neighbor ordering violated between nodes " << x
                    << " and " << y << ", first neighbors were "
                    << x_first_neighbour << " and " << y_first_neighbour;

                if (y_first_neighbour == x_first_neighbour) {
                    ASSERT_GE(degrees[y], degrees[x])
                        << "Degree ordering violated between nodes " << x
                        << " and " << y << ", degrees were " << degrees[x]
                        << " and " << degrees[y];
                }
            }
        }
    }

    void build_multiple_connected_components()
    {
        gko::matrix_data<value_type, index_type> data;
        d_1138_bus_mtx->write(data);
        const auto num_rows = data.size[0];
        const auto nnz = data.nonzeros.size();
        const int num_copies = 5;
        data.size[0] =
            num_rows * num_copies + 10;  // add a handful of isolated vertices
        data.size[1] = data.size[0];
        for (gko::size_type i = 0; i < nnz; i++) {
            const auto entry = data.nonzeros[i];
            // create copies of the matrix
            for (int copy = 1; copy < num_copies; copy++) {
                data.nonzeros.emplace_back(entry.row + copy * num_rows,
                                           entry.column + copy * num_rows,
                                           entry.value);
            }
        }
        std::vector<index_type> permutation(data.size[0]);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(), rng);
        for (auto& entry : data.nonzeros) {
            entry.row = permutation[entry.row];
            entry.column = permutation[entry.column];
        }
        data.sort_row_major();
        d_1138_bus_mtx->read(data);
        o_1138_bus_mtx->read(data);
    }

    std::default_random_engine rng;
    std::shared_ptr<CsrMtx> o_1138_bus_mtx;
    std::shared_ptr<CsrMtx> d_1138_bus_mtx;
    // Can't std::move parameter when using ASSERT_PREDN, no perfect forwarding.
    // Therefore, use shared pointer
    std::shared_ptr<reorder_type> d_reorder_op;
};


TEST_F(Rcm, PermutationIsRcmOrdered)
{
    d_reorder_op = reorder_type::build()
                       .with_construct_inverse_permutation(true)
                       .on(exec)
                       ->generate(d_1138_bus_mtx);

    auto perm = d_reorder_op->get_permutation();

    check_rcm_ordered(o_1138_bus_mtx, perm.get(),
                      d_reorder_op->get_parameters().strategy);
    GKO_ASSERT_MTX_EQ_SPARSITY(perm->compute_inverse(),
                               d_reorder_op->get_inverse_permutation());
}


TEST_F(Rcm, PermutationIsRcmOrderedMinDegree)
{
    d_reorder_op =
        reorder_type::build()
            .with_strategy(gko::reorder::starting_strategy::minimum_degree)
            .on(exec)
            ->generate(d_1138_bus_mtx);

    auto perm = d_reorder_op->get_permutation();

    check_rcm_ordered(o_1138_bus_mtx, perm.get(),
                      d_reorder_op->get_parameters().strategy);
    ASSERT_EQ(d_reorder_op->get_inverse_permutation(), nullptr);
}


TEST_F(Rcm, PermutationIsRcmOrderedNewInterface)
{
    auto perm = new_reorder_type::build().on(exec)->generate(d_1138_bus_mtx);

    check_rcm_ordered(o_1138_bus_mtx, perm.get(),
                      gko::reorder::starting_strategy::pseudo_peripheral);
}


TEST_F(Rcm, PermutationIsRcmOrderedMultipleConnectedComponents)
{
    this->build_multiple_connected_components();

    d_reorder_op = reorder_type::build().on(exec)->generate(d_1138_bus_mtx);

    auto perm = d_reorder_op->get_permutation();
    check_rcm_ordered(o_1138_bus_mtx, perm.get(),
                      d_reorder_op->get_parameters().strategy);
}


TEST_F(Rcm, PermutationIsRcmOrderedShuffledFromFile)
{
    o_1138_bus_mtx = gko::read<CsrMtx>(
        std::ifstream{gko::matrices::location_1138_bus_shuffled_mtx}, ref);
    d_1138_bus_mtx = gko::clone(exec, o_1138_bus_mtx);

    d_reorder_op = reorder_type::build().on(exec)->generate(d_1138_bus_mtx);

    auto perm = d_reorder_op->get_permutation();
    check_rcm_ordered(o_1138_bus_mtx, perm.get(),
                      d_reorder_op->get_parameters().strategy);
}


TEST_F(Rcm, PermutationIsRcmOrderedMinDegreeMultipleConnectedComponents)
{
    this->build_multiple_connected_components();

    d_reorder_op =
        reorder_type::build()
            .with_strategy(gko::reorder::starting_strategy::minimum_degree)
            .on(exec)
            ->generate(d_1138_bus_mtx);

    auto perm = d_reorder_op->get_permutation();
    check_rcm_ordered(o_1138_bus_mtx, perm.get(),
                      d_reorder_op->get_parameters().strategy);
}
