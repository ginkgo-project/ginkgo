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

#include <ginkgo/core/solver/lower_trs_isai.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/solver/lower_trs_isai_kernels.hpp"


namespace gko {
namespace solver {
namespace lower_trs_isai {

GKO_REGISTER_OPERATION(build_isai, lower_trs_isai::build_isai);

}  // namespace lower_trs_isai

template <typename ValueType, typename IndexType>
void LowerTrsIsai<ValueType, IndexType>::build_isai()
{
    this->get_executor()->run(lower_trs_isai::make_build_isai(
        gko::lend(this->system_matrix_), gko::lend(this->isai_)));
}


template <typename ValueType, typename IndexType>
void LowerTrsIsai<ValueType, IndexType>::apply_impl(
    const LinOp *b,  // right hand side
    LinOp *x         // system solution after relaxation steps
    ) const
{
    using Vec = gko::matrix::Dense<ValueType>;

    auto exec = this->get_executor();  // Current executor

    auto one_op = gko::initialize<Vec>({gko::one<ValueType>()}, exec);
    auto neg_one_op = gko::initialize<Vec>({-gko::one<ValueType>()}, exec);

    auto dense_b = as<Vec>(b);
    auto dense_x = as<Vec>(x);

    // TODO use caches for reusing intermediate vectors?
    auto d = Vec::create_with_config_of(dense_b);
    auto w1 = Vec::create_with_config_of(dense_b);
    auto w2 = Vec::create_with_config_of(dense_b);

    if (parameters_.niter <= 0) {
        // No relaxation steps

        // x = M * b
        isai_->apply(b, x);
    } else {
        // Perform niter relaxation steps

        // Init x
        // x = M * b
        // m->apply(b, x);

        // d = M * b
        isai_->apply(b, d.get());
        // d->copy_from(x);

        for (int i = 0; i < parameters_.niter; ++i) {
            // w1 = A * x
            system_matrix_->apply(x, w1.get());
            // w2 = M * w1 = M * A * x
            isai_->apply(w1.get(), w2.get());
            // x = x - w2 = x - M * A * x
            dense_x->add_scaled(neg_one_op.get(), w2.get());
            // x = x + d = x - M * A * x + M * b
            dense_x->add_scaled(one_op.get(), d.get());
        }
    }
}


template <typename ValueType, typename IndexType>
void LowerTrsIsai<ValueType, IndexType>::apply_impl(const LinOp *alpha,
                                                    const LinOp *b,
                                                    const LinOp *beta,
                                                    LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, gko::lend(x_clone));
}

#define GKO_DECLARE_LOWER_TRS_ISAI(_vtype, _itype) \
    class LowerTrsIsai<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWER_TRS_ISAI);


}  // namespace solver
}  // namespace gko
