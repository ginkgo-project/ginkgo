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

#include <ginkgo/core/base/composition.hpp>


#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace {


template <typename ValueType, typename OpIterator, typename VecIterator>
inline void allocate_vectors(OpIterator begin, OpIterator end, VecIterator res)
{
    for (auto it = begin; it != end; ++it, ++res) {
        if (*res != nullptr && (*res)->get_size()[0] == (*it)->get_size()[0]) {
            continue;
        }
        *res = matrix::Dense<ValueType>::create(
            (*it)->get_executor(), gko::dim<2>{(*it)->get_size()[0], 1});
    }
}


inline const LinOp *apply_inner_operators(
    const std::vector<std::shared_ptr<const LinOp>> &operators,
    const std::vector<std::unique_ptr<LinOp>> &intermediate, const LinOp *rhs)
{
    for (auto i = operators.size() - 1; i > 0u; --i) {
        auto solution = lend(intermediate[i - 1]);
        operators[i]->apply(rhs, solution);
        rhs = solution;
    }
    return rhs;
}


}  // namespace


template <typename ValueType>
void Composition<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    cache_.intermediate.resize(operators_.size() - 1);
    allocate_vectors<ValueType>(begin(operators_) + 1, end(operators_),
                                begin(cache_.intermediate));
    operators_[0]->apply(
        apply_inner_operators(operators_, cache_.intermediate, b), x);
}


template <typename ValueType>
void Composition<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                        const LinOp *beta, LinOp *x) const
{
    cache_.intermediate.resize(operators_.size() - 1);
    allocate_vectors<ValueType>(begin(operators_) + 1, end(operators_),
                                begin(cache_.intermediate));
    operators_[0]->apply(
        alpha, apply_inner_operators(operators_, cache_.intermediate, b), beta,
        x);
}


#define GKO_DECLARE_COMPOSITION(_type) class Composition<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_COMPOSITION);


}  // namespace gko
