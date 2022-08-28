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

#include <ginkgo/core/stop/criterion.hpp>


#include "core/stop/criterion_kernels.hpp"


namespace gko {
namespace stop {
namespace criterion {
namespace {


GKO_REGISTER_OPERATION(set_all_statuses, set_all_statuses::set_all_statuses);


}  // anonymous namespace
}  // namespace criterion


Criterion::Criterion(std::shared_ptr<const gko::Executor> exec)
    : EnableAbstractPolymorphicObject<Criterion>(exec)
{}


bool Criterion::check(uint8 stopping_id, bool set_finalized,
                      array<stopping_status>* stop_status, bool* one_changed,
                      const Updater& updater)
{
    this->template log<log::Logger::criterion_check_started>(
        this, updater.num_iterations_, updater.residual_,
        updater.residual_norm_, updater.solution_, stopping_id, set_finalized);
    auto all_converged = this->check_impl(stopping_id, set_finalized,
                                          stop_status, one_changed, updater);
    this->template log<log::Logger::criterion_check_completed>(
        this, updater.num_iterations_, updater.residual_,
        updater.residual_norm_, updater.implicit_sq_residual_norm_,
        updater.solution_, stopping_id, set_finalized, stop_status,
        *one_changed, all_converged);
    return all_converged;
}


void Criterion::set_all_statuses(uint8 stoppingId, bool setFinalized,
                                 array<stopping_status>* stop_status)
{
    this->get_executor()->run(criterion::make_set_all_statuses(
        stoppingId, setFinalized, stop_status));
}


CriterionArgs::CriterionArgs(std::shared_ptr<const LinOp> system_matrix,
                             std::shared_ptr<const LinOp> b, const LinOp* x,
                             const LinOp* initial_residual)
    : system_matrix{system_matrix},
      b{b},
      x{x},
      initial_residual{initial_residual}
{}


}  // namespace stop
}  // namespace gko
