/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/reorder/mc64.hpp>


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
#include "core/reorder/mc64_kernels.hpp"


namespace gko {
namespace reorder {
namespace mc64 {
namespace {


GKO_REGISTER_OPERATION(initialize_weights, mc64::initialize_weights);
GKO_REGISTER_OPERATION(initial_matching, mc64::initial_matching);
GKO_REGISTER_OPERATION(shortest_augmenting_path,
                       mc64::shortest_augmenting_path);


}  // anonymous namespace
}  // namespace mc64


template <typename ValueType, typename IndexType>
void Mc64<ValueType, IndexType>::generate(
    std::shared_ptr<const Executor>& exec,
    std::shared_ptr<LinOp> system_matrix) const
{
    auto mtx = as<matrix_type>(system_matrix);
    size_type num_rows = mtx->get_size()[0];

    Array<remove_complex<ValueType>> workspace{exec};
    Array<IndexType> permutation{exec, num_rows};
    Array<IndexType> inv_permutation{exec, num_rows};
    permutation.fill(-one<IndexType>());
    inv_permutation.fill(-one<IndexType>());
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();

    exec->run(mc64::make_initialize_weights(mtx.get(), workspace));

    std::list<IndexType> unmatched_rows{};
    exec->run(mc64::make_initial_matching(num_rows, row_ptrs, col_idxs,
                                          workspace, permutation,
                                          inv_permutation, unmatched_rows));

    Array<IndexType> parents{exec, num_rows};
    for (auto root : unmatched_rows) {
        exec->run(mc64::make_shortest_augmenting_path(
            num_rows, row_ptrs, col_idxs, workspace, permutation,
            inv_permutation, root, parents));
    }

    permutation_->copy_from(
        PermutationMatrix::create(exec, system_matrix->get_size(), permutation)
            .get());
    inv_permutation_->copy_from(
        share(PermutationMatrix::create(exec, system_matrix->get_size(),
                                        inv_permutation,
                                        matrix::column_permute))
            .get());
}


#define GKO_DECLARE_MC64(ValueType, IndexType) class Mc64<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MC64);


}  // namespace reorder
}  // namespace gko
