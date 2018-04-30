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


#include "core/stop/relative_residual_norm.hpp"
#include "core/stop/relative_residual_norm_kernels.hpp"


namespace gko {
namespace stop {
namespace {


template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(
        relative_residual_norm,
        relative_residual_norm::relative_residual_norm<ValueType>);
};


}  // namespace


template <typename ValueType>
std::unique_ptr<Criterion>
RelResidualNorm<ValueType>::Factory::create_criterion(
    std::shared_ptr<const LinOp> system_matrix, std::shared_ptr<const LinOp> b,
    const LinOp *x) const
{
    return std::unique_ptr<RelResidualNorm>(new RelResidualNorm<ValueType>(
        v_, std::move(exec_), b->get_num_cols()));
}


template <typename ValueType>
bool RelResidualNorm<ValueType>::check(Array<bool> &converged,
                                       const Updater &updater)
{
    if (!initialized_tau_) {
        starting_tau_->copy_from(updater.residual_norm_);
        return false;
    }

    bool all_converged = false;
    exec_->run(
        TemplatedOperation<ValueType>::make_relative_residual_norm_operation(
            updater.residual_norm_, starting_tau_.get(), rel_residual_goal_,
            &converged, &all_converged));
    return all_converged;
}


}  // namespace stop
}  // namespace gko
