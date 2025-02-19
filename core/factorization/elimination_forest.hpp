// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_ELIMINATION_FOREST_HPP_
#define GKO_CORE_FACTORIZATION_ELIMINATION_FOREST_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/components/disjoint_sets.hpp"


namespace gko {
namespace factorization {


enum class elimination_forest_algorithm : int {
    /** compute everything on the host. */
    host = 0,
    /**
     * compute parents, child_ptrs, children, postorder, inv_postorder on host,
     * postorder_parents, postorder_child_ptrs, postorder_children on the device
     */
    device_map_postorder = 1,
    /**
     * compute parents, child_ptrs, children on host,
     * postorder, inv_postorder and postorder-mapped tree on the device.
     */
    device_postorder = 2,
    /**
     * compute parents on host,
     * child_ptrs, children and postorder + mapped tree on the device.
     */
    device_children = 3,
};


template <typename IndexType>
struct elimination_forest {
    elimination_forest(std::shared_ptr<const Executor> exec, IndexType size,
                       elimination_forest_algorithm algorithm =
                           elimination_forest_algorithm::host)
        : parents{exec->get_master(), static_cast<size_type>(size)},
          child_ptrs{exec->get_master()},
          children{exec->get_master()},
          postorder{exec->get_master()},
          inv_postorder{exec->get_master()},
          postorder_parents{exec->get_master()},
          postorder_child_ptrs{exec->get_master()},
          postorder_children{exec->get_master()}
    {
        switch (algorithm) {
        case elimination_forest_algorithm::device_children:
            children.set_executor(exec);
            child_ptrs.set_executor(exec);
            [[fallthrough]];
        case elimination_forest_algorithm::device_postorder:
            postorder.set_executor(exec);
            inv_postorder.set_executor(exec);
            [[fallthrough]];
        case elimination_forest_algorithm::device_map_postorder:
            postorder_parents.set_executor(exec);
            postorder_child_ptrs.set_executor(exec);
            postorder_children.set_executor(exec);
            [[fallthrough]];
        case elimination_forest_algorithm::host:
        default:
            break;
        }
        const auto usize = static_cast<size_type>(size);
        child_ptrs.resize_and_reset(usize + 2);
        children.resize_and_reset(usize);
        postorder.resize_and_reset(usize);
        inv_postorder.resize_and_reset(usize);
        postorder_parents.resize_and_reset(usize);
        postorder_child_ptrs.resize_and_reset(usize + 2);
        postorder_children.resize_and_reset(usize);
    }

    void set_executor(std::shared_ptr<const Executor> exec);

    array<IndexType> parents;
    array<IndexType> child_ptrs;
    array<IndexType> children;
    array<IndexType> postorder;
    array<IndexType> inv_postorder;
    array<IndexType> postorder_parents;
    array<IndexType> postorder_child_ptrs;
    array<IndexType> postorder_children;
};


template <typename ValueType, typename IndexType>
void compute_elimination_forest(
    const matrix::Csr<ValueType, IndexType>* mtx,
    std::unique_ptr<elimination_forest<IndexType>>& forest,
    elimination_forest_algorithm algorithm =
        elimination_forest_algorithm::host);


template <typename ValueType, typename IndexType>
void elimination_forest_from_factors(
    const matrix::Csr<ValueType, IndexType>* factors,
    std::unique_ptr<elimination_forest<IndexType>>& forest,
    elimination_forest_algorithm algorithm =
        elimination_forest_algorithm::host);


}  // namespace factorization
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_ELIMINATION_FOREST_HPP_
