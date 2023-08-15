// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/rcm.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/csr_kernels.hpp"
#include "core/reorder/rcm_kernels.hpp"


namespace gko {
namespace reorder {
namespace rcm {
namespace {


GKO_REGISTER_OPERATION(get_permutation, rcm::get_permutation);
GKO_REGISTER_OPERATION(get_degree_of_nodes, rcm::get_degree_of_nodes);


}  // anonymous namespace
}  // namespace rcm


template <typename ValueType, typename IndexType>
void Rcm<ValueType, IndexType>::generate(
    std::shared_ptr<const Executor>& exec,
    std::unique_ptr<SparsityMatrix> adjacency_matrix) const
{
    const IndexType num_rows = adjacency_matrix->get_size()[0];
    const auto mtx = adjacency_matrix.get();
    auto degrees = array<IndexType>(exec, num_rows);
    // RCM is only valid for symmetric matrices. Need to add an expensive check
    // for symmetricity here ?
    exec->run(rcm::make_get_degree_of_nodes(num_rows, mtx->get_const_row_ptrs(),
                                            degrees.get_data()));
    exec->run(rcm::make_get_permutation(
        num_rows, mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
        degrees.get_const_data(), permutation_->get_permutation(),
        inv_permutation_.get() ? inv_permutation_->get_permutation() : nullptr,
        parameters_.strategy));
}


#define GKO_DECLARE_RCM(ValueType, IndexType) class Rcm<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RCM);


}  // namespace reorder
}  // namespace gko
