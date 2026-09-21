// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_UTILS_REORDERING_HPP_
#define GKO_CORE_TEST_UTILS_REORDERING_HPP_


#include <array>
#include <vector>

#include <ginkgo/core/base/dim.hpp>


namespace gko {
namespace test {


template <typename T>
inline T get_natural_flat_index_from_3d(const std::array<T, 3>& dims,
                                        const std::array<T, 3>& idx)
{
    return idx[2] * dims[1] * dims[0] + idx[1] * dims[0] + idx[0];
}

template <typename itype>
struct MulticolorOrdering {
    std::vector<itype> new_to_old;
    std::vector<itype> old_to_new;
    std::vector<itype> color_ptrs;
};

/**
 * Compute 8-color independent-set ordering for a 3d box (27-pt) stencil.
 */
template <typename itype>
MulticolorOrdering<itype> compute_multicolor_ordering_regular_box(
    const gko::dim<3>& local_grid_dims)
{
    const std::array<int, 3> ldims{static_cast<int>(local_grid_dims[0]),
                                   static_cast<int>(local_grid_dims[1]),
                                   static_cast<int>(local_grid_dims[2])};
    const int ln = ldims[0] * ldims[1] * ldims[2];
    std::vector<itype> old_to_new(ln);
    std::vector<itype> new_to_old(ln);
    std::vector<itype> cnt(8, 0);

    for (itype k = 0; k < ldims[2]; ++k)
        for (itype j = 0; j < ldims[1]; ++j)
            for (itype i = 0; i < ldims[0]; ++i)
                ++cnt[(i % 2) + 2 * (j % 2) + 4 * (k % 2)];

    std::vector<int> color_ptrs(9);
    color_ptrs[0] = 0;
    for (int c = 0; c < 8; ++c) {
        color_ptrs[c + 1] = color_ptrs[c] + cnt[c];
    }

    std::vector<itype> fill(8, 0);
    for (itype k = 0; k < ldims[2]; ++k) {
        for (itype j = 0; j < ldims[1]; ++j) {
            for (itype i = 0; i < ldims[0]; ++i) {
                const std::array<itype, 3> idx{i, j, k};
                const int old_idx = get_natural_flat_index_from_3d(ldims, idx);
                const int color = (i % 2) + 2 * (j % 2) + 4 * (k % 2);
                const int new_idx = color_ptrs[color] + fill[color]++;
                old_to_new[old_idx] = new_idx;
                new_to_old[new_idx] = old_idx;
            }
        }
    }
    return MulticolorOrdering<itype>{new_to_old, old_to_new, color_ptrs};
}

/**
 * Compute 2-color independent-set ordering for a 2d star (5-pt) stencil.
 */
template <typename itype>
MulticolorOrdering<itype> compute_multicolor_ordering_regular_star(
    const gko::dim<2>& local_grid_dims)
{
    const int ln = local_grid_dims[0] * local_grid_dims[1];
    std::vector<itype> old_to_new(ln);
    std::vector<itype> new_to_old(ln);
    std::vector<itype> cnt(2, 0);

    for (itype j = 0; j < local_grid_dims[1]; ++j) {
        for (itype i = 0; i < local_grid_dims[0]; ++i) {
            ++cnt[(i + j) % 2];
        }
    }

    std::vector<itype> color_ptrs(3);
    color_ptrs[0] = 0;
    for (int c = 0; c < 2; ++c) {
        color_ptrs[c + 1] = color_ptrs[c] + cnt[c];
    }

    std::vector<itype> fill(2, 0);
    for (itype j = 0; j < local_grid_dims[1]; ++j) {
        for (itype i = 0; i < local_grid_dims[0]; ++i) {
            const int old_idx = local_grid_dims[0] * j + i;
            const int color = (i + j) % 2;
            const int new_idx = color_ptrs[color] + fill[color]++;
            old_to_new[old_idx] = new_idx;
            new_to_old[new_idx] = old_idx;
        }
    }
    return MulticolorOrdering<itype>{new_to_old, old_to_new, color_ptrs};
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_REORDERING_HPP_
