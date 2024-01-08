// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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


template <typename IndexType>
struct elimination_forest {
    elimination_forest(std::shared_ptr<const Executor> host_exec,
                       IndexType size)
        : parents{host_exec, static_cast<size_type>(size)},
          child_ptrs{host_exec, static_cast<size_type>(size + 2)},
          children{host_exec, static_cast<size_type>(size)},
          postorder{host_exec, static_cast<size_type>(size)},
          inv_postorder{host_exec, static_cast<size_type>(size)},
          postorder_parents{host_exec, static_cast<size_type>(size)}
    {}

    void set_executor(std::shared_ptr<const Executor> exec);

    array<IndexType> parents;
    array<IndexType> child_ptrs;
    array<IndexType> children;
    array<IndexType> postorder;
    array<IndexType> inv_postorder;
    array<IndexType> postorder_parents;
};


template <typename ValueType, typename IndexType>
void compute_elim_forest(
    const matrix::Csr<ValueType, IndexType>* mtx,
    std::unique_ptr<elimination_forest<IndexType>>& forest);


}  // namespace factorization
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_ELIMINATION_FOREST_HPP_
