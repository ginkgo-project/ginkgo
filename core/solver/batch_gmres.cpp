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

#include <ginkgo/core/solver/batch_gmres.hpp>


#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_gmres_kernels.hpp"
#include "core/solver/batch_solver.ipp"


namespace gko {
namespace solver {
namespace batch_gmres {


GKO_REGISTER_OPERATION(apply, batch_gmres::apply);


}  // namespace batch_gmres


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchGmres<ValueType>::transpose() const
{
    auto tsolver =
        build()
            .with_preconditioner(parameters_.preconditioner)
            .with_generated_preconditioner(share(
                as<BatchTransposable>(this->get_preconditioner())->transpose()))
            .with_left_scaling_op(
                share(as<BatchTransposable>(this->get_left_scaling_op())
                          ->transpose()))
            .with_right_scaling_op(
                share(as<BatchTransposable>(this->get_right_scaling_op())
                          ->transpose()))
            .with_default_max_iterations(parameters_.default_max_iterations)
            .with_default_residual_tol(
                static_cast<real_type>(parameters_.default_residual_tol))
            .with_restart(parameters_.restart)
            .with_tolerance_type(parameters_.tolerance_type)
            .on(this->get_executor())
            ->generate(share(
                as<BatchTransposable>(this->get_system_matrix())->transpose()));
    tsolver->set_residual_tolerance(this->residual_tol_);
    tsolver->set_max_iterations(this->max_iterations_);
    return tsolver;
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchGmres<ValueType>::conj_transpose() const
{
    auto ctsolver =
        build()
            .with_preconditioner(parameters_.preconditioner)
            .with_generated_preconditioner(
                share(as<BatchTransposable>(this->get_preconditioner())
                          ->conj_transpose()))
            .with_left_scaling_op(
                share(as<BatchTransposable>(this->get_left_scaling_op())
                          ->conj_transpose()))
            .with_right_scaling_op(
                share(as<BatchTransposable>(this->get_right_scaling_op())
                          ->conj_transpose()))
            .with_default_max_iterations(parameters_.default_max_iterations)
            .with_default_residual_tol(
                static_cast<real_type>(parameters_.default_residual_tol))
            .with_restart(parameters_.restart)
            .with_tolerance_type(parameters_.tolerance_type)
            .on(this->get_executor())
            ->generate(share(as<BatchTransposable>(this->get_system_matrix())
                                 ->conj_transpose()));
    ctsolver->set_residual_tolerance(this->residual_tol_);
    ctsolver->set_max_iterations(this->max_iterations_);
    return ctsolver;
}


template <typename ValueType>
void BatchGmres<ValueType>::solver_apply(const BatchLinOp* const b,
                                         BatchLinOp* const x,
                                         BatchInfo* const info) const
{
    using Dense = matrix::BatchDense<ValueType>;
    const kernels::batch_gmres::BatchGmresOptions<remove_complex<ValueType>>
        opts{this->max_iterations_, static_cast<real_type>(this->residual_tol_),
             parameters_.restart, parameters_.tolerance_type};
    auto exec = this->get_executor();
    exec->run(batch_gmres::make_apply(
        opts, this->system_matrix_.get(), this->preconditioner_.get(),
        as<const Dense>(b), as<Dense>(x),
        *as<log::BatchLogData<ValueType>>(info->logdata.get())));
}


#define GKO_DECLARE_BATCH_GMRES(_type) class BatchGmres<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_GMRES);


#define GKO_DECLARE_BATCH_GMRES_APPLY_FUNCTION(_type)                         \
    EnableBatchSolver<BatchGmres<_type>, BatchLinOp>::EnableBatchSolver(      \
        std::shared_ptr<const Executor> exec,                                 \
        std::shared_ptr<const BatchLinOp> system_matrix,                      \
        detail::common_batch_params common_params);                           \
    template void                                                             \
    EnableBatchSolver<BatchGmres<_type>, BatchLinOp>::apply_impl(             \
        const BatchLinOp* b, BatchLinOp* x) const;                            \
    template void                                                             \
    EnableBatchSolver<BatchGmres<_type>, BatchLinOp>::apply_impl(             \
        const BatchLinOp* alpha, const BatchLinOp* b, const BatchLinOp* beta, \
        BatchLinOp* x) const
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_GMRES_APPLY_FUNCTION);


}  // namespace solver
}  // namespace gko
