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

#include "core/stop/residual_norm_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Residual norm stopping criterion namespace.
 * @ref resnorm
 * @ingroup resnorm
 */
namespace residual_norm {


template <typename ValueType>
void residual_norm(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Dense<ValueType> *tau,
                   const matrix::Dense<ValueType> *orig_tau,
                   ValueType rel_residual_goal, uint8 stoppingId,
                   bool setFinalized, Array<stopping_status> *stop_status,
                   Array<bool> *device_storage, bool *all_converged,
                   bool *one_changed)
{
    static_assert(is_complex_s<ValueType>::value == false,
                  "ValueType must not be complex in this function!");
    bool local_one_changed = false;
#pragma omp parallel for reduction(|| : local_one_changed)
    for (size_type i = 0; i < tau->get_size()[1]; ++i) {
        if (tau->at(i) < rel_residual_goal * orig_tau->at(i)) {
            stop_status->get_data()[i].converge(stoppingId, setFinalized);
            local_one_changed = true;
        }
    }
    *one_changed = local_one_changed;
    // No early stopping here because one cannot use break with parallel for
    // But it's parallel so does it matter?
    bool local_all_converged = true;
#pragma omp parallel for reduction(&& : local_all_converged)
    for (size_type i = 0; i < stop_status->get_num_elems(); ++i) {
        if (!stop_status->get_const_data()[i].has_stopped()) {
            local_all_converged = false;
        }
    }
    *all_converged = local_all_converged;
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_RESIDUAL_NORM_KERNEL);


}  // namespace residual_norm
}  // namespace omp
}  // namespace kernels
}  // namespace gko
