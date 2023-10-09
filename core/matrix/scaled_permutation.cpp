/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/matrix/scaled_permutation.hpp>
#include "core/matrix/scaled_permutation_kernels.hpp"
#include "ginkgo/core/base/executor.hpp"
#include "ginkgo/core/base/precision_dispatch.hpp"


namespace gko {
namespace matrix {
namespace scaled_permutation {
namespace {


GKO_REGISTER_OPERATION(invert, scaled_permutation::invert);


}  // namespace
}  // namespace scaled_permutation


template <typename ValueType, typename IndexType>
ScaledPermutation<ValueType, IndexType>::ScaledPermutation(
    std::shared_ptr<const Executor> exec, size_type size)
    : ScaledPermutation{exec, array<ValueType>{exec, size},
                        array<IndexType>{exec, size}}
{}


template <typename ValueType, typename IndexType>
ScaledPermutation<ValueType, IndexType>::ScaledPermutation(
    std::shared_ptr<const Executor> exec, array<value_type> scaling_factors,
    array<index_type> permutation_indices)
    : EnableLinOp<ScaledPermutation>(exec,
                                     dim<2>{scaling_factors.get_num_elems(),
                                            scaling_factors.get_num_elems()}),
      scale_{exec, std::move(scaling_factors)},
      permutation_{exec, std::move(permutation_indices)}
{
    GKO_ASSERT_EQ(scale_.get_num_elems(), permutation_.get_num_elems());
}


template <typename ValueType, typename IndexType>
std::unique_ptr<ScaledPermutation<ValueType, IndexType>>
ScaledPermutation<ValueType, IndexType>::invert() const
{
    const auto exec = this->get_executor();
    const auto size = this->get_size()[0];
    array<index_type> inv_permutation{exec, size};
    array<value_type> inv_scale{exec, size};
    exec->run(scaled_permutation::make_invert(
        this->get_const_permutation(), this->get_const_scale(), size,
        inv_permutation.get_data(), inv_scale.get_data()));
    return ScaledPermutation::create(exec, std::move(inv_scale),
                                     std::move(inv_permutation));
}


template <typename ValueType, typename IndexType>
void ScaledPermutation<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                         LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            dense_b->scale_permute(this, dense_x, permute_mode::rows);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void ScaledPermutation<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                         const LinOp* b,
                                                         const LinOp* beta,
                                                         LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto tmp = dense_b->scale_permute(this, permute_mode::rows);
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, tmp);
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
void ScaledPermutation<ValueType, IndexType>::write(
    gko::matrix_data<value_type, index_type>& data) const
{
    const auto host_this =
        make_temporary_clone(this->get_executor()->get_master(), this);
    data.size = this->get_size();
    data.nonzeros.clear();
    data.nonzeros.reserve(data.size[0]);
    for (IndexType row = 0; row < this->get_size()[0]; row++) {
        data.nonzeros.emplace_back(row, host_this->get_const_permutation()[row],
                                   host_this->get_const_scale()[row]);
    }
}


#define GKO_DECLARE_SCALED_PERMUTATION_MATRIX(ValueType, IndexType) \
    class ScaledPermutation<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SCALED_PERMUTATION_MATRIX);


}  // namespace matrix
}  // namespace gko
