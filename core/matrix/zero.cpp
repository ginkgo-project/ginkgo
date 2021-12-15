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


#include <ginkgo/core/matrix/zero.hpp>


#include <ginkgo/core/base/precision_dispatch.hpp>


#include "core/distributed/helpers.hpp"


namespace gko {
namespace matrix {


template <typename ValueType>
std::unique_ptr<LinOp> Zero<ValueType>::transpose() const
{
    return Zero::create(this->get_executor(),
                        dim<2>{this->get_size()[1], this->get_size()[0]});
}


template <typename ValueType>
std::unique_ptr<LinOp> Zero<ValueType>::conj_transpose() const
{
    return Zero::create(this->get_executor(),
                        dim<2>{this->get_size()[1], this->get_size()[0]});
}


template <typename ValueType>
void Zero<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [](const auto b, auto x) {
            gko::distributed::detail::get_local(x)->fill(zero<ValueType>());
        },
        b, x);
}


template <typename ValueType>
void Zero<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                 const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [](const auto alpha, const auto b, const auto beta, auto x) {
            gko::distributed::detail::get_local(x)->scale(beta);
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_ZERO_OPERATOR(ValueType) class Zero<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_ZERO_OPERATOR);
#undef GKO_DECLARE_ZERO_OPERATOR


}  // namespace matrix
}  // namespace gko
