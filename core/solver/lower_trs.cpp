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

#include <ginkgo/core/solver/lower_trs.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/solver/lower_trs_kernels.hpp"


namespace gko {
namespace solver {
namespace lower_trs {


GKO_REGISTER_OPERATION(generate, lower_trs::generate);
GKO_REGISTER_OPERATION(init_struct, lower_trs::init_struct);
GKO_REGISTER_OPERATION(should_perform_transpose,
                       lower_trs::should_perform_transpose);
GKO_REGISTER_OPERATION(solve, lower_trs::solve);


}  // namespace lower_trs


template <typename ValueType, typename IndexType>
void LowerTrs<ValueType, IndexType>::init_trs_solve_struct()
{
    this->get_executor()->run(lower_trs::make_init_struct(this->solve_struct_));
}


template <typename ValueType, typename IndexType>
void LowerTrs<ValueType, IndexType>::generate()
{
    this->get_executor()->run(lower_trs::make_generate(
        gko::lend(system_matrix_), gko::lend(this->solve_struct_),
        parameters_.num_rhs));
}


template <typename ValueType, typename IndexType>
void LowerTrs<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using Vector = matrix::Dense<ValueType>;
    const auto exec = this->get_executor();

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);

    // This kernel checks if a transpose is needed for the multiple rhs case.
    // Currently only the algorithm for CUDA version <=9.1 needs this
    // transposition due to the limitation in the cusparse algorithm. The other
    // executors (omp and reference) do not use the transpose (trans_x and
    // trans_b) and hence are passed in empty pointers.
    bool do_transpose = false;
    std::shared_ptr<Vector> trans_b;
    std::shared_ptr<Vector> trans_x;
    this->get_executor()->run(
        lower_trs::make_should_perform_transpose(do_transpose));
    if (do_transpose) {
        trans_b = Vector::create(exec, gko::transpose(dense_b->get_size()));
        trans_x = Vector::create(exec, gko::transpose(dense_x->get_size()));
    } else {
        trans_b = Vector::create(exec);
        trans_x = Vector::create(exec);
    }
    exec->run(lower_trs::make_solve(
        gko::lend(system_matrix_), gko::lend(this->solve_struct_),
        gko::lend(trans_b), gko::lend(trans_x), dense_b, dense_x));
}


template <typename ValueType, typename IndexType>
void LowerTrs<ValueType, IndexType>::apply_impl(const LinOp *alpha,
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


#define GKO_DECLARE_LOWER_TRS(_vtype, _itype) class LowerTrs<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWER_TRS);


}  // namespace solver
}  // namespace gko
