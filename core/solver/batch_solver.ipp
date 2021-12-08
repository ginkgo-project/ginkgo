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

#ifndef GKO_CORE_SOLVER_BATCH_SOLVER_IPP_
#define GKO_CORE_SOLVER_BATCH_SOLVER_IPP_


#include <ginkgo/core/log/batch_convergence.hpp>
#include <ginkgo/core/solver/batch_solver.hpp>


#include "core/log/batch_logging.hpp"
#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"


namespace gko {
namespace solver {
namespace batch {


GKO_REGISTER_OPERATION(pre_diag_scale_system, batch_csr::pre_diag_scale_system);
GKO_REGISTER_OPERATION(vec_scale, batch_dense::batch_scale);


}


struct BatchInfo {
    //gko::log::BatchLogData<ValueType> logdata;
    void* logdata;
};


template <typename ConcreteSolver, typename PolymorphicBase>
void EnableBatchSolver<ConcreteSolver, PolymorphicBase>::apply_impl(const BatchLinOp* b,
                                          BatchLinOp* x) const
{
    using value_type = typename ConcreteSolver::value_type;
    using Csr = matrix::BatchCsr<value_type>;
    using Vector = matrix::BatchDense<value_type>;
    using real_type = remove_complex<value_type>;

    auto exec = this->get_executor();
    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    const bool to_scale =
        this->get_left_scaling_vector() && this->get_right_scaling_vector();
    const auto acsr = dynamic_cast<const Csr*>(system_matrix_.get());
    const BatchLinOp* a_scaled{};
    const Vector* b_scaled{};
    auto a_scaled_smart = Csr::create(exec);
    auto b_scaled_smart = Vector::create(exec);
    if (to_scale && !acsr) {
        GKO_NOT_SUPPORTED(system_matrix_);
    }

    // copies to scale
    if (to_scale) {
        a_scaled_smart->copy_from(acsr);
        b_scaled_smart->copy_from(dense_b);
        exec->run(batch::make_pre_diag_scale_system(
            as<const Vector>(this->get_left_scaling_vector()),
            as<const Vector>(this->get_right_scaling_vector()),
            a_scaled_smart.get(), b_scaled_smart.get()));
        a_scaled = a_scaled_smart.get();
        b_scaled = b_scaled_smart.get();
    } else {
        a_scaled = system_matrix_.get();
        b_scaled = dense_b;
    }

    // allocate logging arrays assuming uniform size batch
    // GKO_ASSERT(dense_b->stores_equal_sizes());

    const size_type num_rhs = dense_b->get_size().at(0)[1];
    const size_type num_batches = dense_b->get_num_batch_entries();
    batch_dim<> sizes(num_batches, dim<2>{1, num_rhs});

    log::BatchLogData<value_type> concrete_logdata;
    concrete_logdata.res_norms =
        matrix::BatchDense<real_type>::create(this->get_executor(), sizes);
    concrete_logdata.iter_counts.set_executor(this->get_executor());
    concrete_logdata.iter_counts.resize_and_reset(num_rhs * num_batches);
    BatchInfo info;
    info.logdata = &concrete_logdata;

    this->solver_apply(a_scaled, b_scaled, dense_x, info);

    this->template log<log::Logger::batch_solver_completed>(
        concrete_logdata.iter_counts, concrete_logdata.res_norms.get());

    if (to_scale) {
        exec->run(batch::make_vec_scale(
            as<Vector>(this->get_right_scaling_vector()), dense_x));
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
