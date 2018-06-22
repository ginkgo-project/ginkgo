/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


#include "core/stop/residual_norm_reduction.hpp"
#include "core/stop/residual_norm_reduction_kernels.hpp"


namespace gko {
namespace stop {
namespace {
template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(
        residual_norm_reduction,
        residual_norm_reduction::residual_norm_reduction<ValueType>);
};


}  // namespace


template <typename ValueType>
bool ResidualNormReduction<ValueType>::check(
    uint8 stoppingId, bool setFinalized, Array<stopping_status> *stop_status,
    bool *one_changed, const Criterion::Updater &updater)
{
    if (!initialized_tau_) {
        if (updater.residual_norm_ != nullptr) {
            starting_tau_->copy_from(updater.residual_norm_);
        } else if (updater.residual_ != nullptr) {
            auto dense_r = as<Vector>(updater.residual_);
            dense_r->compute_dot(dense_r, starting_tau_.get());
        } else {
            NOT_IMPLEMENTED;
        }
        initialized_tau_ = true;
        return false;
    }

    std::unique_ptr<Vector> u_dense_tau;
    const Vector *dense_tau;
    if (updater.residual_norm_ != nullptr) {
        dense_tau = as<Vector>(updater.residual_norm_);
    } else if (updater.residual_ != nullptr) {
        u_dense_tau = Vector::create_with_config_of(starting_tau_.get());
        auto dense_r = as<Vector>(updater.residual_);
        dense_r->compute_dot(dense_r, u_dense_tau.get());
        dense_tau = u_dense_tau.get();
    } else {
        NOT_IMPLEMENTED;
    }

    bool all_converged{};
    this->get_executor()->run(
        TemplatedOperation<ValueType>::make_residual_norm_reduction_operation(
            dense_tau, starting_tau_.get(), parameters_.reduction_factor,
            stoppingId, setFinalized, stop_status, &all_converged,
            one_changed));
    return all_converged;
}


#define GKO_DECLARE_RESIDUAL_NORM_REDUCTION(_type) \
    class ResidualNormReduction<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM_REDUCTION);
#undef GKO_DECLARE_RESIDUAL_NORM_REDUCTION


}  // namespace stop
}  // namespace gko
