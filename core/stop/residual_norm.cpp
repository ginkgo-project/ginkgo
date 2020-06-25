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

#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/components/fill_array.hpp"
#include "core/stop/residual_norm_kernels.hpp"


namespace gko {
namespace stop {
namespace residual_norm {


GKO_REGISTER_OPERATION(residual_norm, residual_norm::residual_norm);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);


}  // namespace residual_norm


template <typename ValueType>
bool ResidualNorm<ValueType>::check_impl(uint8 stoppingId, bool setFinalized,
                                         Array<stopping_status> *stop_status,
                                         bool *one_changed,
                                         const Criterion::Updater &updater)
{
    const NormVector *dense_tau;
    if (updater.residual_norm_ != nullptr) {
        dense_tau = as<NormVector>(updater.residual_norm_);
    } else if (updater.residual_ != nullptr) {
        auto *dense_r = as<Vector>(updater.residual_);
        dense_r->compute_norm2(this->u_dense_tau_.get());
        dense_tau = this->u_dense_tau_.get();
    } else {
        GKO_NOT_SUPPORTED(nullptr);
    }
    bool all_converged = true;

    this->get_executor()->run(residual_norm::make_residual_norm(
        dense_tau, this->starting_tau_.get(), this->tolerance_, stoppingId,
        setFinalized, stop_status, &this->device_storage_, &all_converged,
        one_changed));

    return all_converged;
}

template <typename ValueType>
void AbsoluteResidualNorm<ValueType>::initialize_starting_tau()
{
    this->get_executor()->run(residual_norm::make_fill_array(
        this->starting_tau_->get_values(), this->starting_tau_->get_size()[1],
        gko::one<remove_complex<ValueType>>()));
}


#define GKO_DECLARE_RESIDUAL_NORM(_type) class ResidualNorm<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM);


#define GKO_DECLARE_ABSOLUTE_RESIDUAL_NORM(_type) \
    class AbsoluteResidualNorm<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_ABSOLUTE_RESIDUAL_NORM);


}  // namespace stop
}  // namespace gko
