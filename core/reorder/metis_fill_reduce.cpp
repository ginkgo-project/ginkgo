/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/reorder/metis_fill_reduce.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/matrix/csr_kernels.hpp"
#include "core/reorder/metis_fill_reduce_kernels.hpp"


namespace gko {
namespace reorder {
namespace metis_fill_reduce {


GKO_REGISTER_OPERATION(get_permutation, metis_fill_reduce::get_permutation);
GKO_REGISTER_OPERATION(construct_inverse_permutation_matrix,
                       metis_fill_reduce::construct_inverse_permutation_matrix);
GKO_REGISTER_OPERATION(construct_permutation_matrix,
                       metis_fill_reduce::construct_permutation_matrix);
GKO_REGISTER_OPERATION(permute, metis_fill_reduce::permute);


}  // namespace metis_fill_reduce


template <typename ValueType, typename IndexType>
void MetisFillReduce<ValueType, IndexType>::generate() const
{
    const gko::size_type num_rows = system_matrix_->get_size()[0];
    const auto exec = this->get_executor();
    exec->run(metis_fill_reduce::make_get_permutation(
        num_rows, system_matrix_->get_const_row_ptrs(),
        system_matrix_->get_const_col_idxs(), vertex_weights_->get_const_data(),
        permutation_->get_data(), inv_permutation_->get_data()));

    exec->run(metis_fill_reduce::make_construct_permutation_matrix(
        permutation_->get_const_data(), gko::lend(permutation_mat_)));

    if (parameters_.construct_inverse_permutation) {
        exec->run(metis_fill_reduce::make_construct_inverse_permutation_matrix(
            inv_permutation_->get_const_data(),
            gko::lend(inv_permutation_mat_)));
    }
}


template <typename ValueType, typename IndexType>
void MetisFillReduce<ValueType, IndexType>::permute(LinOp *to_permute) const
{
    const auto exec = this->get_executor();

    exec->run(metis_fill_reduce::make_permute(gko::lend(permutation_mat_),
                                              to_permute));
}


template <typename ValueType, typename IndexType>
void MetisFillReduce<ValueType, IndexType>::inverse_permute(
    LinOp *to_permute) const
{
    const auto exec = this->get_executor();

    if (parameters_.construct_inverse_permutation) {
        exec->run(metis_fill_reduce::make_permute(
            gko::lend(inv_permutation_mat_), to_permute));
    }
}

#define GKO_DECLARE_METIS_FILL_REDUCE(ValueType, IndexType) \
    class MetisFillReduce<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_METIS_FILL_REDUCE);


}  // namespace reorder
}  // namespace gko
