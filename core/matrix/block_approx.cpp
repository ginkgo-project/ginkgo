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

#include <ginkgo/core/matrix/block_approx.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace matrix {


template <typename ConcreteType>
void BlockApprox<ConcreteType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using value_type = typename ConcreteType::value_type;
    using index_type = typename ConcreteType::index_type;
    using Dense = Dense<value_type>;

    auto dense_b = const_cast<Dense *>(as<Dense>(b));
    auto dense_x = as<Dense>(x);
    size_type offset = 0;
    for (size_type i = 0; i < this->get_num_blocks(); ++i) {
        auto loc_size = this->get_block_dimensions()[i];
        auto loc_mtx = this->get_block_mtxs()[i];
        const auto loc_b = dense_b->create_submatrix(
            span{offset, offset + loc_size[0]}, span{0, b->get_size()[1]});
        auto loc_x = dense_x->create_submatrix(
            span{offset, offset + loc_size[0]}, span{0, x->get_size()[1]});
        loc_mtx->apply(loc_b.get(), loc_x.get());
        offset += loc_size[0];
    }
}


template <typename ConcreteType>
void BlockApprox<ConcreteType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta, LinOp *x) const
{
    using value_type = typename ConcreteType::value_type;
    using index_type = typename ConcreteType::index_type;
    using Dense = Dense<value_type>;

    auto dense_b = const_cast<Dense *>(as<Dense>(b));
    auto dense_x = as<Dense>(x);
    size_type offset = 0;
    for (size_type i = 0; i < this->get_num_blocks(); ++i) {
        auto loc_size = this->get_block_dimensions()[i];
        auto loc_mtx = this->get_block_mtxs()[i];
        const auto loc_b = dense_b->create_submatrix(
            span{offset, offset + loc_size[0]}, span{0, b->get_size()[1]});
        auto loc_x = dense_x->create_submatrix(
            span{offset, offset + loc_size[0]}, span{0, x->get_size()[1]});
        loc_mtx->apply(alpha, loc_b.get(), beta, loc_x.get());
        offset += loc_size[0];
    }
}


#define GKO_DECLARE_BLOCK_APPROX_CSR_APPLY(ValueType, IndexType)            \
    void BlockApprox<Csr<ValueType, IndexType>>::apply_impl(const LinOp *b, \
                                                            LinOp *x) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_APPROX_CSR_APPLY);


#define GKO_DECLARE_BLOCK_APPROX_CSR_APPLY2(ValueType, IndexType) \
    void BlockApprox<Csr<ValueType, IndexType>>::apply_impl(      \
        const LinOp *alpha, const LinOp *b, const LinOp *beta, LinOp *x) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_APPROX_CSR_APPLY2);


}  // namespace matrix
}  // namespace gko
