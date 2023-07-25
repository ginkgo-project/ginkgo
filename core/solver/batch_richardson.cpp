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

#include <ginkgo/core/solver/batch_richardson.hpp>


#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/preconditioner/batch_preconditioner_strings.hpp>


#include "core/solver/batch_richardson_kernels.hpp"


namespace gko {
namespace solver {
namespace batch_rich {


GKO_REGISTER_OPERATION(apply, batch_rich::apply);


}  // namespace batch_rich


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchRichardson<ValueType>::transpose() const
{
    return build()
        .with_preconditioner(parameters_.preconditioner)
        .with_max_iterations(parameters_.max_iterations)
        .with_rel_residual_tol(parameters_.rel_residual_tol)
        .with_relaxation_factor(parameters_.relaxation_factor)
        .on(this->get_executor())
        ->generate(share(
            as<BatchTransposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchRichardson<ValueType>::conj_transpose() const
{
    return build()
        .with_preconditioner(parameters_.preconditioner)
        .with_max_iterations(parameters_.max_iterations)
        .with_rel_residual_tol(parameters_.rel_residual_tol)
        .with_relaxation_factor(conj(parameters_.relaxation_factor))
        .on(this->get_executor())
        ->generate(share(as<BatchTransposable>(this->get_system_matrix())
                             ->conj_transpose()));
}


template <typename ValueType>
void BatchRichardson<ValueType>::apply_impl(const BatchLinOp *const b,
                                            BatchLinOp *const x) const
{
    using Vector = matrix::BatchDense<ValueType>;
    using real_type = remove_complex<ValueType>;
    auto exec = this->get_executor();
    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    const kernels::batch_rich::BatchRichardsonOptions<real_type> opts{
        parameters_.preconditioner, parameters_.max_iterations,
        parameters_.rel_residual_tol, parameters_.relaxation_factor};

    log::BatchLogData<ValueType> logdata;
    // allocate logging arrays assuming uniform size batch
    const size_type num_rhs = dense_b->get_size().at(0)[1];
    const size_type num_batches = dense_b->get_num_batches();
    batch_dim sizes(num_batches, dim<2>{1, num_rhs});
    logdata.res_norms =
        matrix::BatchDense<real_type>::create(this->get_executor(), sizes);
    logdata.iter_counts.set_executor(this->get_executor());
    logdata.iter_counts.resize_and_reset(num_rhs * num_batches);

    exec->run(batch_rich::make_apply(opts, system_matrix_.get(), dense_b,
                                     dense_x, logdata));

    this->template log<log::Logger::batch_solver_completed>(
        logdata.iter_counts, logdata.res_norms.get());
}


template <typename ValueType>
void BatchRichardson<ValueType>::apply_impl(const BatchLinOp *alpha,
                                            const BatchLinOp *b,
                                            const BatchLinOp *beta,
                                            BatchLinOp *x) const
{
    auto dense_x = as<matrix::BatchDense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_BATCH_RICHARDSON(_type) class BatchRichardson<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_RICHARDSON);


}  // namespace solver
}  // namespace gko
