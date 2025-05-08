// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <oneapi/dpl/algorithm>

#include "core/factorization/elimination_forest.hpp"

#include <algorithm>
#include <memory>

#include <sycl/sycl.hpp>

#include <ginkgo/core/matrix/csr.hpp>

#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"
#include "dpcpp/base/onedpl.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace elimination_forest {


template <typename IndexType>
void compute_skeleton_tree(std::shared_ptr<const DefaultExecutor> exec,
                           const IndexType* row_ptrs, const IndexType* cols,
                           size_type size, IndexType* out_row_ptrs,
                           IndexType* out_cols) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SKELETON_TREE);


template <typename IndexType>
void build_children_from_parents(
    std::shared_ptr<const DefaultExecutor> exec,
    gko::factorization::elimination_forest<IndexType>& forest)
{
    const auto num_rows = forest.parents.get_size();
    // build COO representation of the tree
    array<IndexType> col_idx_array{exec, num_rows};
    const auto col_idxs = col_idx_array.get_data();
    const auto parents = forest.parents.get_const_data();
    const auto children = forest.children.get_data();
    const auto child_ptrs = forest.child_ptrs.get_data();
    exec->copy(num_rows, parents, col_idxs);
    components::fill_seq_array(exec, children, num_rows);
    // group by parent
    oneapi::dpl::sort_by_key(onedpl_policy(exec), col_idxs, col_idxs + num_rows,
                             children);
    // create child pointers for groups of children
    components::convert_idxs_to_ptrs(exec, col_idxs, num_rows,
                                     num_rows + 1,  // rows plus sentinel
                                     child_ptrs);
}


template <typename ValueType, typename IndexType>
void from_factor(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::Csr<ValueType, IndexType>* factors,
                 gko::factorization::elimination_forest<IndexType>& forest)
{
    const auto num_rows = factors->get_size()[0];
    const auto it = oneapi::dpl::counting_iterator<IndexType>(IndexType{});
    oneapi::dpl::transform(
        onedpl_policy(exec), it, it + num_rows, forest.parents.get_data(),
        [row_ptrs = factors->get_const_row_ptrs(),
         col_idxs = factors->get_const_col_idxs(), num_rows](IndexType l_col) {
            const auto llt_row_begin = row_ptrs[l_col];
            const auto llt_row_end = row_ptrs[l_col + 1];
            for (auto nz = llt_row_begin; nz < llt_row_end; nz++) {
                const auto l_row = col_idxs[nz];
                // parent[j] = min(i | i > j and l_ij =/= 0)
                // we read from L^T stored above the diagonal in factors
                // assuming a sorted order of the columns
                if (l_row > l_col) {
                    return l_row;
                }
            }
            // sentinel pseudo-root
            return static_cast<IndexType>(num_rows);
        });
    build_children_from_parents(exec, forest);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_FROM_FACTOR);


}  // namespace elimination_forest
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
