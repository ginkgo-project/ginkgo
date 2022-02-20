/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/stop/residual_norm.hpp>


#include <ginkgo/core/distributed/vector.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/stop/residual_norm_kernels.hpp"


namespace gko {
namespace stop {
namespace residual_norm {
namespace {


GKO_REGISTER_OPERATION(residual_norm, residual_norm::residual_norm);


}  // anonymous namespace
}  // namespace residual_norm


namespace implicit_residual_norm {
namespace {


GKO_REGISTER_OPERATION(implicit_residual_norm,
                       implicit_residual_norm::implicit_residual_norm);


}  // anonymous namespace
}  // namespace implicit_residual_norm


template <typename ValueType>
bool ResidualNormBase<ValueType>::check_impl(
    uint8 stopping_id, bool set_finalized, Array<stopping_status>* stop_status,
    bool* one_changed, const Criterion::Updater& updater)
{
#if GINKGO_BUILD_MPI
    using DistributedComplex = distributed::Vector<gko::to_complex<ValueType>>;
    using DistributedVector = distributed::Vector<ValueType>;
#endif
    const NormVector* dense_tau;
    if (updater.residual_norm_ != nullptr) {
        dense_tau = as<NormVector>(updater.residual_norm_);
    } else if (updater.residual_ != nullptr) {
#if GINKGO_BUILD_MPI
        if (dynamic_cast<const distributed::DistributedBase*>(
                updater.residual_)) {
            // the vector is distributed
            if (dynamic_cast<const DistributedComplex*>(updater.residual_)) {
                // handle solvers that use complex vectors even for real systems
                auto dense_r = as<DistributedComplex>(updater.residual_);
                dense_r->compute_norm2(u_dense_tau_.get());
            } else {
                auto dense_r = as<DistributedVector>(updater.residual_);
                dense_r->compute_norm2(u_dense_tau_.get());
            }
#else
        bool is_distributed = false;
        if (is_distributed) {
#endif
        } else {
            // the vector is non-distributed
            if (dynamic_cast<const ComplexVector*>(updater.residual_)) {
                // handle solvers that use complex vectors even for real systems
                auto dense_r = as<ComplexVector>(updater.residual_);
                dense_r->compute_norm2(u_dense_tau_.get());
            } else {
                auto dense_r = as<Vector>(updater.residual_);
                dense_r->compute_norm2(u_dense_tau_.get());
            }
        }
        dense_tau = u_dense_tau_.get();
    } else if (updater.solution_ != nullptr && system_matrix_ != nullptr &&
               b_ != nullptr) {
        auto exec = this->get_executor();
        // when LinOp is real but rhs is complex, we use real view on complex,
        // so it still uses the same type of scalar in apply.
        if (auto vec_b = std::dynamic_pointer_cast<const Vector>(b_)) {
            auto dense_r = vec_b->clone();
            system_matrix_->apply(neg_one_.get(), updater.solution_, one_.get(),
                                  dense_r.get());
            dense_r->compute_norm2(u_dense_tau_.get());
        } else if (auto vec_b =
                       std::dynamic_pointer_cast<const ComplexVector>(b_)) {
            auto dense_r = vec_b->clone();
            system_matrix_->apply(neg_one_.get(), updater.solution_, one_.get(),
                                  dense_r.get());
            dense_r->compute_norm2(u_dense_tau_.get());
        } else {
            GKO_NOT_SUPPORTED(nullptr);
        }
        dense_tau = u_dense_tau_.get();
    } else {
        GKO_NOT_SUPPORTED(nullptr);
    }
    bool all_converged = true;

    this->get_executor()->run(residual_norm::make_residual_norm(
        dense_tau, starting_tau_.get(), reduction_factor_, stopping_id,
        set_finalized, stop_status, &device_storage_, &all_converged,
        one_changed));

    return all_converged;
}


template <typename ValueType>
bool ImplicitResidualNorm<ValueType>::check_impl(
    uint8 stopping_id, bool set_finalized, Array<stopping_status>* stop_status,
    bool* one_changed, const Criterion::Updater& updater)
{
    const Vector* dense_tau;
    if (updater.implicit_sq_residual_norm_ != nullptr) {
        dense_tau = as<Vector>(updater.implicit_sq_residual_norm_);
    } else {
        GKO_NOT_SUPPORTED(nullptr);
    }
    bool all_converged = true;

    this->get_executor()->run(
        implicit_residual_norm::make_implicit_residual_norm(
            dense_tau, this->starting_tau_.get(), this->reduction_factor_,
            stopping_id, set_finalized, stop_status, &this->device_storage_,
            &all_converged, one_changed));

    return all_converged;
}


#define GKO_DECLARE_RESIDUAL_NORM(_type) class ResidualNormBase<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM);


#define GKO_DECLARE_IMPLICIT_RESIDUAL_NORM(_type) \
    class ImplicitResidualNorm<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IMPLICIT_RESIDUAL_NORM);


}  // namespace stop
}  // namespace gko
