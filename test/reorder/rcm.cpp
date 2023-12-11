// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <deque>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/reorder/rcm.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


namespace {


class Rcm : public CommonTestFixture {
protected:
    using v_type = double;
    using i_type = int;
    using Mtx = gko::matrix::Dense<v_type>;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::Rcm<v_type, i_type>;
    using new_reorder_type = gko::experimental::reorder::Rcm<i_type>;
    using perm_type = gko::matrix::Permutation<i_type>;

    Rcm()
        : o_1138_bus_mtx(gko::read<CsrMtx>(
              std::ifstream(gko::matrices::location_1138_bus_mtx, std::ios::in),
              ref)),
          d_1138_bus_mtx(gko::read<CsrMtx>(
              std::ifstream(gko::matrices::location_1138_bus_mtx, std::ios::in),
              exec))
    {}

    static void ubfs_reference(
        std::shared_ptr<CsrMtx> mtx,
        i_type* const
            levels,  // Must be inf/max in all nodes connected to source
        const i_type start)
    {
        const auto row_ptrs = mtx->get_const_row_ptrs();
        const auto col_idxs = mtx->get_const_col_idxs();

        std::deque<i_type> q(0);
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

    static void check_valid_start_node(std::shared_ptr<CsrMtx> mtx,
                                       const i_type* permutation, i_type start,
                                       std::vector<bool>& already_visited,
                                       gko::reorder::starting_strategy strategy)
    {
        ASSERT_FALSE(already_visited[start]) << start;

        const auto n = mtx->get_size()[0];
        auto degrees = std::vector<i_type>(n);
        for (gko::size_type i = 0; i < n; ++i) {
            degrees[i] =
                mtx->get_const_row_ptrs()[i + 1] - mtx->get_const_row_ptrs()[i];
        }

        switch (strategy) {
        case gko::reorder::starting_strategy::minimum_degree: {
            auto min_degree = std::numeric_limits<i_type>::max();
            for (gko::size_type i = 0; i < n; ++i) {
                if (!already_visited[i] && degrees[i] < min_degree) {
                    min_degree = degrees[i];
                }
            }
            ASSERT_EQ(min_degree, degrees[start]) << start;
            break;
        }

        case gko::reorder::starting_strategy::pseudo_peripheral: {
            // Check if any valid contender has a lowereq height than the
            // selected start node.

            std::vector<i_type> reference_current_levels(n);
            std::fill(reference_current_levels.begin(),
                      reference_current_levels.end(),
                      std::numeric_limits<i_type>::max());
            ubfs_reference(mtx, &reference_current_levels[0], start);

            std::vector<i_type> reference_contenders(0);
            auto current_height = std::numeric_limits<i_type>::min();
            for (gko::size_type i = 0; i < n; ++i) {
                if (reference_current_levels[i] !=
                        std::numeric_limits<i_type>::max() &&
                    reference_current_levels[i] >= current_height) {
                    if (reference_current_levels[i] > current_height) {
                        reference_contenders.clear();
                    }
                    reference_contenders.push_back(i);
                    current_height = reference_current_levels[i];
                }
            }

            std::vector<std::vector<i_type>> reference_contenders_levels(
                reference_contenders.size());
            for (gko::size_type i = 0; i < reference_contenders.size(); ++i) {
                std::vector<i_type> reference_contender_levels(n);
                std::fill(reference_contender_levels.begin(),
                          reference_contender_levels.end(),
                          std::numeric_limits<i_type>::max());
                ubfs_reference(mtx, &reference_contender_levels[0],
                               reference_contenders[i]);
                reference_contenders_levels[i] = reference_contender_levels;
            }

            for (gko::size_type i = 0; i < reference_contenders.size(); ++i) {
                auto contender_height = std::numeric_limits<i_type>::min();
                for (gko::size_type j = 0; j < n; ++j) {
                    if (reference_contenders_levels[i][j] !=
                            std::numeric_limits<i_type>::max() &&
                        reference_contenders_levels[i][j] > contender_height) {
                        contender_height = reference_contenders_levels[i][j];
                    }
                }
                if (contender_height <= current_height) {
                    return;
                }
            }
            GTEST_FAIL() << "there is a contender with larger height";
        }
        }
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
        auto degrees = std::vector<i_type>(n);
        for (gko::size_type i = 0; i < n; ++i) {
            degrees[i] =
                mtx->get_const_row_ptrs()[i + 1] - mtx->get_const_row_ptrs()[i];
        }

        // Following checks for cm ordering, therefore create a reversed perm.
        std::vector<i_type> perm(permutation, permutation + n);
        std::reverse(perm.begin(), perm.end());

        // Now check for cm ordering.

        gko::size_type base_offset = 0;
        std::vector<bool> already_visited(n);
        while (base_offset != n) {
            // Assert valid start node.
            check_valid_start_node(mtx, permutation, perm[base_offset],
                                   already_visited, strategy);

            // Assert valid level structure.
            // Also update base_offset and mark as visited while at it.
            std::vector<i_type> levels(n);
            std::fill(levels.begin(), levels.end(),
                      std::numeric_limits<i_type>::max());
            ubfs_reference(mtx, &levels[0], perm[base_offset]);

            i_type current_level = 0;
            const auto previous_base_offset = base_offset;
            for (gko::size_type i = 0; i < n; ++i) {
                const auto node = perm[i];
                if (levels[node] != std::numeric_limits<i_type>::max() &&
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
                        if (std::find(perm.begin(), perm.end(), x_neighbour) <
                            std::find(perm.begin(), perm.end(),
                                      x_first_neighbour)) {
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
                        if (std::find(perm.begin(), perm.end(), y_neighbour) <
                            std::find(perm.begin(), perm.end(),
                                      y_first_neighbour)) {
                            y_first_neighbour = y_neighbour;
                        }
                    }
                }

                // Assert the ... is not after the ... in the previous level.
                if (std::find(perm.begin(), perm.end(), y_first_neighbour) <
                    std::find(perm.begin(), perm.end(), x_first_neighbour)) {
                    GTEST_FAIL()
                        << "First neighbor ordering violated between nodes "
                        << x << " and " << y << ", first neighbors were "
                        << x_first_neighbour << " and " << y_first_neighbour;
                }

                if (y_first_neighbour == x_first_neighbour) {
                    if (degrees[y] < degrees[x]) {
                        GTEST_FAIL()
                            << "Degree ordering violated between nodes " << x
                            << " and " << y << ", degrees were " << degrees[x]
                            << " and " << degrees[y];
                    }
                }
            }
        }
    }

    std::shared_ptr<CsrMtx> o_1138_bus_mtx;
    std::shared_ptr<CsrMtx> d_1138_bus_mtx;
    // Can't std::move parameter when using ASSERT_PREDN, no perfect forwarding.
    // Therefore, use shared pointer
    std::shared_ptr<reorder_type> d_reorder_op;
};

TEST_F(Rcm, PermutationIsRcmOrdered)
{
    d_reorder_op = reorder_type::build().on(exec)->generate(d_1138_bus_mtx);

    auto perm = d_reorder_op->get_permutation();

    check_rcm_ordered(o_1138_bus_mtx, perm.get(),
                      d_reorder_op->get_parameters().strategy);
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
}

TEST_F(Rcm, PermutationIsRcmOrderedNewInterface)
{
    auto perm = new_reorder_type::build().on(exec)->generate(d_1138_bus_mtx);

    check_rcm_ordered(o_1138_bus_mtx, perm.get(),
                      gko::reorder::starting_strategy::pseudo_peripheral);
}

}  // namespace
