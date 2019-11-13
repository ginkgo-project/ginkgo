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


#include <ginkgo/core/stop/residual_norm_reduction.hpp>


#include "core/stop/residual_norm_reduction_kernels.hpp"


namespace gko {
namespace stop {
namespace residual_norm_reduction {


GKO_REGISTER_OPERATION(residual_norm_reduction,
                       residual_norm_reduction::residual_norm_reduction);


}  // namespace residual_norm_reduction


template <typename ValueType>
bool ResidualNormReduction<ValueType>::check_impl(
    uint8 stoppingId, bool setFinalized, Array<stopping_status> *stop_status,
    bool *one_changed, const Criterion::Updater &updater)
{
    std::unique_ptr<Vector> u_dense_tau;
    const Vector *dense_tau;
    if (updater.residual_norm_ != nullptr) {
        dense_tau = as<Vector>(updater.residual_norm_);
    } else if (updater.residual_ != nullptr) {
        auto *dense_r = as<Vector>(updater.residual_);
        dense_r->compute_norm2(u_dense_tau_.get());
        dense_tau = u_dense_tau_.get();
    } else {
        throw GKO_NOT_SUPPORTED(nullptr);
    }
    bool all_converged = true;

    this->get_executor()->run(
        residual_norm_reduction::make_residual_norm_reduction(
            dense_tau, starting_tau_.get(), parameters_.reduction_factor,
            stoppingId, setFinalized, stop_status, &this->device_storage_,
            &all_converged, one_changed));
    return all_converged;
}


#define GKO_DECLARE_RESIDUAL_NORM_REDUCTION(_type) \
    class ResidualNormReduction<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM_REDUCTION);


}  // namespace stop
}  // namespace gko
