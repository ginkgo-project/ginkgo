/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/solver/gcr_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The GCR solver namespace.
 *
 * @ingroup grc
 */
namespace gcr {


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ValueType>* b,
                matrix::Dense<ValueType>* residual,
                stopping_status* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto b, auto residual, auto stop) {
            if (row == 0) {
                stop[col].reset();
            }
            // TODO: possibly set other vals to zero here?
            residual(row, col) = b(row, col);
        },
        // Note: default_stride only applied to objects created using
        // creat_with_config_of as this guarantees identical stride.
        b->get_size(), b->get_stride(), b, residual, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_INITIALIZE_KERNEL);


template <typename ValueType>
void restart(std::shared_ptr<const DefaultExecutor> exec,
             const matrix::Dense<ValueType>* residual,
             matrix::Dense<ValueType>* A_residual,
             matrix::Dense<ValueType>* p_bases,
             matrix::Dense<ValueType>* Ap_bases, size_type* final_iter_nums)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto residual, auto A_residual,
                      auto p_bases, auto Ap_bases, auto final_iter_nums) {
            if (row == 0) {
                final_iter_nums[col] = 0;
            }
            p_bases(row, col) = residual(row, col);
            Ap_bases(row, col) = A_residual(row, col);
        },
        residual->get_size(), residual->get_stride(), residual, A_residual,
        p_bases, Ap_bases, final_iter_nums);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_RESTART_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* residual,
            const matrix::Dense<ValueType>* p,
            const matrix::Dense<ValueType>* Ap,
            const matrix::Dense<remove_complex<ValueType>>* Ap_norm,
            const matrix::Dense<ValueType>* alpha,
            const stopping_status* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto residual, auto p,
                      auto Ap, auto Ap_norm, auto alpha, auto stop) {
            if (!stop[col].has_stopped()) {
                auto tmp = alpha[col] / Ap_norm[col];
                x(row, col) += tmp * p(row, col);
                residual(row, col) -= tmp * Ap(row, col);
            }
        },
        x->get_size(), p->get_stride(), x, residual, p, Ap, Ap_norm, alpha,
        stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_STEP_1_KERNEL);

}  // namespace gcr
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
