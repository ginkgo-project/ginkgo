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

#include "core/stop/relative_residual_norm_kernels.hpp"


#include "core/base/exception_helpers.hpp"


namespace gko {
namespace kernels {
namespace cpu {
namespace relative_residual_norm {


template <typename ValueType>
void relative_residual_norm(std::shared_ptr<const CpuExecutor> exec,
                            const matrix::Dense<ValueType> *tau,
                            const matrix::Dense<ValueType> *orig_tau,
                            remove_complex<ValueType> rel_residual_goal,
                            uint8 stoppingId, bool setFinalized,
                            Array<stopping_status> *stop_status,
                            bool *all_converged,
                            bool *one_changed) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RELATIVE_RESIDUAL_NORM_KERNEL);


}  // namespace relative_residual_norm
}  // namespace cpu
}  // namespace kernels
}  // namespace gko
