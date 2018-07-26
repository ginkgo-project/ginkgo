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

#include "core/stop/residual_norm_reduction_kernels.hpp"


#include "core/base/array.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"


#include <algorithm>


namespace gko {
namespace kernels {
namespace reference {
namespace residual_norm_reduction {


template <typename ValueType>
void residual_norm_reduction(std::shared_ptr<const ReferenceExecutor> exec,
                             const matrix::Dense<ValueType> *tau,
                             const matrix::Dense<ValueType> *orig_tau,
                             remove_complex<ValueType> rel_residual_goal,
                             uint8 stoppingId, bool setFinalized,
                             Array<stopping_status> *stop_status,
                             bool *all_converged, bool *one_changed)
{
    *all_converged = true;
    for (size_type i = 0; i < tau->get_size()[1]; ++i) {
        if (abs(tau->at(i)) < rel_residual_goal * abs(orig_tau->at(i))) {
            stop_status->get_data()[i].converge(stoppingId, setFinalized);
            *one_changed = true;
        }
    }
    for (size_type i = 0; i < stop_status->get_num_elems(); ++i) {
        if (!stop_status->get_const_data()[i].has_stopped()) {
            *all_converged = false;
            break;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM_REDUCTION_KERNEL);


}  // namespace residual_norm_reduction
}  // namespace reference
}  // namespace kernels
}  // namespace gko
