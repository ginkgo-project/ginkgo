// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/reorder/multicolor_kernels.hpp"

#include <algorithm>
#include <vector>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace multicolor {


template <typename IndexType>
void compute_permutation_csr(std::shared_ptr<const ReferenceExecutor> exec,
                             const IndexType num_vertices,
                             const IndexType* const row_ptrs,
                             const IndexType* const col_idxs,
                             gko::array<IndexType>& color_ptrs,
                             IndexType* const permutation,
                             IndexType* const inv_permutation)
{
    const auto local_nrows = num_vertices;

    std::vector<int> color(local_nrows, -1);

    for (int i = 0; i < local_nrows; i++) {
        int mycolor = -1;
        bool overlap = true;
        while (overlap) {
            mycolor++;
            overlap = false;
            for (int jz = row_ptrs[i]; jz < row_ptrs[i + 1]; jz++) {
                const auto j = col_idxs[jz];
                if (i != j && mycolor == color[j]) {
                    overlap = true;
                }
            }
        }
        color[i] = mycolor;
    }

    std::map<int, std::vector<IndexType>> color_points;

    // color_points stores old indices of points in each color group.
    for (IndexType i = 0; i < local_nrows; i++) {
        assert(color[i] >= 0 && color[i] < row_ptrs[i + 1] - row_ptrs[i]);
        color_points[color[i]].push_back(i);
    }
    const auto num_colors = static_cast<int>(color_points.size());
    color_ptrs.resize_and_reset(num_colors + 1);
    auto* const cp = color_ptrs.get_data();
    cp[0] = 0;

    for (int ic = 0; ic < num_colors; ic++) {
        // map is sorted, so this should be stable.
        const auto color_size = static_cast<int>(color_points[ic].size());
        for (int i = 0; i < color_size; i++) {
            permutation[color_points[ic][i]] = cp[ic] + i;
            inv_permutation[cp[ic] + i] = color_points[ic][i];
        }
        cp[ic + 1] = cp[ic] + color_size;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_MULTICOLOR_COMPUTE_PERMUTATION_CSR_KERNEL);


}  // namespace multicolor
}  // namespace reference
}  // namespace kernels
}  // namespace gko
