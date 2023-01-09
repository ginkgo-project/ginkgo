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

#ifndef GKO_CORE_SOLVER_BATCH_SOLVER_IPP_
#define GKO_CORE_SOLVER_BATCH_SOLVER_IPP_


#include <ginkgo/core/log/batch_convergence.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>
#include <ginkgo/core/solver/batch_solver.hpp>


#include "core/log/batch_logging.hpp"
#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"


namespace gko {
namespace solver {


struct BatchInfo {
    std::unique_ptr<gko::log::BatchLogDataBase> logdata;
};


template <typename ConcreteSolver, typename PolymorphicBase>
EnableBatchSolver<ConcreteSolver, PolymorphicBase>::EnableBatchSolver(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const BatchLinOp> system_matrix,
    detail::common_batch_params common_params)
    : BatchSolver(system_matrix, nullptr,
                  common_params.left_scaling_op, common_params.right_scaling_op,
                  common_params.residual_tolerance,
                  common_params.max_iterations),
      EnableBatchLinOp<ConcreteSolver, PolymorphicBase>(
          exec, gko::transpose(system_matrix->get_size()))
{
    GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(system_matrix_);

    using value_type = typename ConcreteSolver::value_type;
    using Csr = matrix::BatchCsr<value_type>;
    using Diag = matrix::BatchDiagonal<value_type>;
    using Identity = matrix::BatchIdentity<value_type>;
    using real_type = remove_complex<value_type>;

    const bool to_scale = std::dynamic_pointer_cast<const Diag>(left_scaling_)
        && std::dynamic_pointer_cast<const Diag>(right_scaling_);
    if (to_scale) {
        auto a_scaled_smart = gko::share(gko::clone(system_matrix_.get()));
        matrix::two_sided_batch_transform(exec,
            as<const Diag>(this->left_scaling_.get()),
            as<const Diag>(this->right_scaling_.get()), a_scaled_smart.get());
        system_matrix_ = a_scaled_smart;
    }

    if(!to_scale && left_scaling_ && right_scaling_) {
        GKO_NOT_SUPPORTED(left_scaling_);
    }
    if(!to_scale && (left_scaling_ || right_scaling_)) {
        throw std::runtime_error("One-sided scaling is not supported!");
    }

    if(!to_scale) {
        // this enables transpose for non-scaled solvers
        left_scaling_ = gko::share(Identity::create(exec, system_matrix->get_size()));
        right_scaling_ = gko::share(Identity::create(exec, system_matrix->get_size()));
    }

    if (common_params.generated_prec) {
        GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(common_params.generated_prec, this);
        preconditioner_ = std::move(common_params.generated_prec);
    } else if (common_params.prec_factory) {
        preconditioner_ = common_params.prec_factory->generate(system_matrix_);
    } else {
        auto id = gko::matrix::BatchIdentity<value_type>::create(exec,
            system_matrix->get_size());
        preconditioner_ = std::move(id);
    }
}


template <typename ConcreteSolver, typename PolymorphicBase>
void EnableBatchSolver<ConcreteSolver, PolymorphicBase>::apply_impl(const BatchLinOp* b,
                                          BatchLinOp* x) const
{
    using value_type = typename ConcreteSolver::value_type;
    using Diag = matrix::BatchDiagonal<value_type>;
    using Vector = matrix::BatchDense<value_type>;
    using real_type = remove_complex<value_type>;

    auto exec = this->get_executor();
    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    const bool to_scale = std::dynamic_pointer_cast<const Diag>(left_scaling_)
        && std::dynamic_pointer_cast<const Diag>(right_scaling_);
    auto b_scaled = Vector::create(exec);
    const Vector* b_scaled_ptr{};

    // copies to scale
    if (to_scale) {
        b_scaled->copy_from(dense_b);
        as<const Diag>(this->left_scaling_)->apply(b_scaled.get(), b_scaled.get());
        b_scaled_ptr = b_scaled.get();
    } else {
        b_scaled_ptr = dense_b;
    }

    const size_type num_rhs = dense_b->get_size().at(0)[1];
    const size_type num_batches = dense_b->get_num_batch_entries();
    batch_dim<> sizes(num_batches, dim<2>{1, num_rhs});

    BatchInfo info;
    info.logdata = std::move(std::make_unique<log::BatchLogData<value_type>>());
    auto concrete_logdata = static_cast<log::BatchLogData<value_type>*>(info.logdata.get());
    concrete_logdata->res_norms =
        matrix::BatchDense<real_type>::create(this->get_executor(), sizes);
    concrete_logdata->iter_counts.set_executor(this->get_executor());
    concrete_logdata->iter_counts.resize_and_reset(num_rhs * num_batches);

    this->solver_apply(b_scaled_ptr, dense_x, &info);

    this->template log<log::Logger::batch_solver_completed>(
        concrete_logdata->iter_counts, concrete_logdata->res_norms.get());

    if (to_scale) {
        as<const Diag>(this->right_scaling_)->apply(dense_x, dense_x);
    }
}


template <typename ConcreteSolver, typename PolymorphicBase>
void EnableBatchSolver<ConcreteSolver, PolymorphicBase>::apply_impl(
    const BatchLinOp* alpha, const BatchLinOp* b,
    const BatchLinOp* beta, BatchLinOp* x) const
{
    using value_type = typename ConcreteSolver::value_type;
    auto dense_x = as<matrix::BatchDense<value_type>>(x);
    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


}
}

#endif
