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


GKO_REGISTER_OPERATION(get_permutation, rcm::get_permutation);
GKO_REGISTER_OPERATION(get_degree_of_nodes, rcm::get_degree_of_nodes);


}  // namespace rcm


template <typename ValueType, typename IndexType>
void Rcm<ValueType, IndexType>::generate(
    std::shared_ptr<const Executor> &exec,
    std::unique_ptr<SparsityMatrix> adjacency_matrix) const
{
    const IndexType num_rows = adjacency_matrix->get_size()[0];
    const auto mtx = adjacency_matrix.get();
    auto degrees = Array<IndexType>(exec, num_rows);
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
