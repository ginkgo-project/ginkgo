/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/reorder/rcm_kernels.hpp"


#include <algorithm>
#include <iterator>
#include <memory>
#include <queue>
#include <utility>
#include <vector>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace rcm {


template <typename IndexType>
void get_degree_of_nodes(std::shared_ptr<const OmpExecutor> exec,
                         const size_type num_vertices,
                         const IndexType *const row_ptrs,
                         IndexType *const degrees)
{
#pragma omp parallel for
    for (auto i = 0; i < num_vertices; ++i) {
        degrees[i] = row_ptrs[i + 1] - row_ptrs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RCM_GET_DEGREE_OF_NODES_KERNEL);


template <typename IndexType>
IndexType find_index(std::vector<std::pair<IndexType, IndexType>> &a,
                     IndexType x)
{
    for (auto i = 0; i < a.size(); i++)
        if (a[i].first == x) return i;
    return -1;
}


template <typename IndexType>
void get_permutation(std::shared_ptr<const OmpExecutor> exec,
                     const size_type num_vertices,
                     const IndexType *const row_ptrs,
                     const IndexType *const col_idxs,
                     const IndexType *const degrees,
                     IndexType *const permutation,
                     IndexType *const inv_permutation,
                     const gko::reorder::starting_strategy strategy)
{
    // Phase 1:
    //     Compute the level of each node using UBFS.
    //     Find all starting vertices.
    // Phase 2 (for each connected component,
    // starting once respective starting vertex is confirmed):
    //     Get the primary offset into the perm.
    //     Compute the level borders as prefix sum of the counts.
    // Phase 3 (for each connected component)
    //     Start by writing the starting node.
    //     Threads watch their level:
    //         If the thread to the left writes a new node to your level:
    //         Write those neighbours of the node which are in the next level to
    //         the next level, sorted by degree.
    //     (Can either by implemented by spinning or tasks)
    // Once the last node in the last level is written for each component,
    // the algorithm is finished.
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RCM_GET_PERMUTATION_KERNEL);


}  // namespace rcm
}  // namespace omp
}  // namespace kernels
}  // namespace gko
