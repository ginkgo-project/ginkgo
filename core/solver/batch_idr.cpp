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

#include <ginkgo/core/solver/batch_idr.hpp>


#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/preconditioner/batch_preconditioner_strings.hpp>


#include "core/solver/batch_idr_kernels.hpp"


namespace gko {
namespace solver {
namespace batch_idr {


GKO_REGISTER_OPERATION(apply, batch_idr::apply);


}  // namespace batch_idr


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchIdr<ValueType>::transpose() const
{
    return build()
        .with_preconditioner(parameters_.preconditioner)
        .with_max_iterations(parameters_.max_iterations)
        .with_rel_residual_tol(parameters_.rel_residual_tol)
        .with_abs_residual_tol(parameters_.abs_residual_tol)
        .with_subspace_dim(parameters_.subspace_dim)
        .with_complex_subspace(parameters_.complex_subspace)
        .with_kappa(parameters_.kappa)
        .with_tolerance_type(parameters_.tolerance_type)
        .on(this->get_executor())
        ->generate(share(
            as<BatchTransposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchIdr<ValueType>::conj_transpose() const
{
    return build()
        .with_preconditioner(parameters_.preconditioner)
        .with_max_iterations(parameters_.max_iterations)
        .with_rel_residual_tol(parameters_.rel_residual_tol)
        .with_abs_residual_tol(parameters_.abs_residual_tol)
        .with_subspace_dim(parameters_.subspace_dim)
        .with_complex_subspace(parameters_.complex_subspace)
        .with_kappa(parameters_.kappa)
        .with_tolerance_type(parameters_.tolerance_type)
        .on(this->get_executor())
        ->generate(share(as<BatchTransposable>(this->get_system_matrix())
                             ->conj_transpose()));
}


template <typename ValueType>
void BatchIdr<ValueType>::apply_impl(const BatchLinOp *b, BatchLinOp *x) const
{
    using Vector = matrix::BatchDense<ValueType>;
    using real_type = remove_complex<ValueType>;

    auto exec = this->get_executor();
    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    const kernels::batch_idr::BatchIdrOptions<remove_complex<ValueType>> opts{
        parameters_.preconditioner,
        parameters_.max_iterations,
        parameters_.rel_residual_tol,
        parameters_.abs_residual_tol,
        parameters_.subspace_dim,
        parameters_.complex_subspace,
        parameters_.kappa,
        parameters_.tolerance_type};

    log::BatchLogData<ValueType> logdata;

    // allocate logging arrays assuming uniform size batch
    // GKO_ASSERT(dense_b->stores_equal_sizes());

    const size_type num_rhs = dense_b->get_size().at(0)[1];
    const size_type num_batches = dense_b->get_num_batches();
    batch_dim sizes(num_batches, dim<2>{1, num_rhs});

    logdata.res_norms =
        matrix::BatchDense<real_type>::create(this->get_executor(), sizes);
    logdata.iter_counts.set_executor(this->get_executor());
    logdata.iter_counts.resize_and_reset(num_rhs * num_batches);

    exec->run(batch_idr::make_apply(opts, system_matrix_.get(),
                                    this->left_scale_, this->right_scale_,
                                    dense_b, dense_x, logdata));

    this->template log<log::Logger::batch_solver_completed>(
        logdata.iter_counts, logdata.res_norms.get());
}


template <typename ValueType>
void BatchIdr<ValueType>::apply_impl(const BatchLinOp *alpha,
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


#define GKO_DECLARE_BATCH_IDR(_type) class BatchIdr<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_IDR);


}  // namespace solver
}  // namespace gko
