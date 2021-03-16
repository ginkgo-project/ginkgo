/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/ras.hpp>


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/block_approx.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/base/utils.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/preconditioner/ras_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace ras {


GKO_REGISTER_OPERATION(simple_apply, ras::simple_apply);
GKO_REGISTER_OPERATION(apply, ras::apply);
GKO_REGISTER_OPERATION(find_blocks, ras::find_blocks);
GKO_REGISTER_OPERATION(generate, ras::generate);
GKO_REGISTER_OPERATION(transpose_ras, ras::transpose_ras);
GKO_REGISTER_OPERATION(conj_transpose_ras, ras::conj_transpose_ras);
GKO_REGISTER_OPERATION(convert_to_dense, ras::convert_to_dense);
GKO_REGISTER_OPERATION(initialize_precisions, ras::initialize_precisions);


}  // namespace ras


template <typename ValueType, typename IndexType>
void Ras<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using Dense = matrix::Dense<ValueType>;
    auto dense_b = const_cast<Dense *>(as<Dense>(b));
    auto dense_x = as<Dense>(x);
    const auto num_subdomains = this->inner_solvers_.size();
    size_type offset = 0;
    for (size_type i = 0; i < num_subdomains; ++i) {
        auto loc_size = this->inner_solvers_[i]->get_size();
        const auto loc_b = dense_b->create_submatrix(
            span{offset, offset + loc_size[0]}, span{0, b->get_size()[1]});
        auto loc_x = dense_x->create_submatrix(
            span{offset, offset + loc_size[0]}, span{0, x->get_size()[1]});
        this->inner_solvers_[i]->apply(loc_b.get(), loc_x.get());
        offset += loc_size[0];
    }
}


template <typename ValueType, typename IndexType>
void Ras<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta, LinOp *x) const
{
    using Dense = matrix::Dense<ValueType>;
    auto dense_b = const_cast<Dense *>(as<Dense>(b));
    auto dense_x = as<Dense>(x);
    const auto num_subdomains = this->inner_solvers_.size();
    size_type offset = 0;
    for (size_type i = 0; i < num_subdomains; ++i) {
        auto loc_size = this->inner_solvers_[i]->get_size();
        const auto loc_b = dense_b->create_submatrix(
            span{offset, offset + loc_size[0]}, span{0, b->get_size()[1]});
        auto loc_x = dense_x->create_submatrix(
            span{offset, offset + loc_size[0]}, span{0, x->get_size()[1]});
        this->inner_solvers_[i]->apply(alpha, loc_b.get(), beta, loc_x.get());
        offset += loc_size[0];
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Ras<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Ras<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void Ras<ValueType, IndexType>::generate(const LinOp *system_matrix)
{
    using block_t = matrix::BlockApprox<matrix::Csr<ValueType, IndexType>>;
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    auto block_mtxs = as<block_t>(system_matrix)->get_block_mtxs();
    const auto num_subdomains = block_mtxs.size();
    for (size_type i = 0; i < num_subdomains; ++i) {
        this->inner_solvers_.emplace_back(
            parameters_.solver->generate(block_mtxs[i]));
    }
}


#define GKO_DECLARE_RAS(ValueType, IndexType) class Ras<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RAS);


}  // namespace preconditioner
}  // namespace gko
