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

#include <ginkgo/core/base/combination.hpp>


#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace {


template <typename ValueType>
inline void initialize_scalars(std::shared_ptr<const Executor> exec,
                               std::unique_ptr<LinOp> &zero,
                               std::unique_ptr<LinOp> &one)
{
    if (zero == nullptr) {
        zero = initialize<matrix::Dense<ValueType>>({gko::zero<ValueType>()},
                                                    exec);
    }
    if (one == nullptr) {
        one =
            initialize<matrix::Dense<ValueType>>({gko::one<ValueType>()}, exec);
    }
}


}  // namespace


template <typename ValueType>
std::unique_ptr<LinOp> Combination<ValueType>::transpose() const
{
    auto transposed = Combination<ValueType>::create(this->get_executor());
    transposed->set_size(gko::transpose(this->get_size()));
    // copy coefficients
    for (auto &coef : get_coefficients()) {
        transposed->coefficients_.push_back(share(as<LinOp>(coef->clone())));
    }
    // transpose operators
    for (auto &op : get_operators()) {
        transposed->operators_.push_back(
            share(as<Transposable>(op)->transpose()));
    }

    return std::move(transposed);
}


template <typename ValueType>
std::unique_ptr<LinOp> Combination<ValueType>::conj_transpose() const
{
    auto transposed = Combination<ValueType>::create(this->get_executor());
    transposed->set_size(gko::transpose(this->get_size()));
    // conjugate coefficients!
    for (auto &coef : get_coefficients()) {
        transposed->coefficients_.push_back(
            share(as<Transposable>(coef)->conj_transpose()));
    }
    // conjugate-transpose operators
    for (auto &op : get_operators()) {
        transposed->operators_.push_back(
            share(as<Transposable>(op)->conj_transpose()));
    }

    return std::move(transposed);
}


template <typename ValueType>
void Combination<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    initialize_scalars<ValueType>(this->get_executor(), cache_.zero,
                                  cache_.one);
    operators_[0]->apply(lend(coefficients_[0]), b, lend(cache_.zero), x);
    for (size_type i = 1; i < operators_.size(); ++i) {
        operators_[i]->apply(lend(coefficients_[i]), b, lend(cache_.one), x);
    }
}


template <typename ValueType>
void Combination<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                        const LinOp *beta, LinOp *x) const
{
    if (cache_.intermediate_x == nullptr ||
        cache_.intermediate_x->get_size() != x->get_size()) {
        cache_.intermediate_x = x->clone();
    }
    this->apply_impl(b, lend(cache_.intermediate_x));
    auto dense_x = as<matrix::Dense<ValueType>>(x);
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, lend(cache_.intermediate_x));
}


#define GKO_DECLARE_COMBINATION(_type) class Combination<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_COMBINATION);


}  // namespace gko
