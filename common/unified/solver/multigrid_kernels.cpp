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

#include "core/solver/multigrid_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch_solver.hpp"
#include "core/components/fill_array_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The MULTIGRID solver namespace.
 *
 * @ingroup multigrid
 */
namespace multigrid {


template <typename ValueType>
void kcycle_step_1(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Dense<ValueType>* rho,
                   const matrix::Dense<ValueType>* v,
                   matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* d,
                   matrix::Dense<ValueType>* e)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto e, auto d, auto g, auto v,
                      auto alpha, auto rho) {
            const auto temp = alpha[col] / rho[col];
            const bool update = is_finite(temp);
            auto store_e = e(row, col);
            if (update) {
                g(row, col) -= temp * v(row, col);
                store_e *= temp;
                e(row, col) = store_e;
            }
            d(row, col) = store_e;
        },
        e->get_size(), e->get_stride(), e, default_stride(d), default_stride(g),
        default_stride(v), alpha->get_const_values(), rho->get_const_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL);


template <typename ValueType>
void kcycle_step_2(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Dense<ValueType>* rho,
                   const matrix::Dense<ValueType>* gamma,
                   const matrix::Dense<ValueType>* beta,
                   const matrix::Dense<ValueType>* zeta,
                   const matrix::Dense<ValueType>* d,
                   matrix::Dense<ValueType>* e)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto e, auto d, auto alpha, auto rho,
                      auto gamma, auto beta, auto zeta) {
            const auto scalar_d =
                zeta[col] / (beta[col] - gamma[col] * gamma[col] / rho[col]);
            const auto scalar_e =
                one(gamma[col]) - gamma[col] / alpha[col] * scalar_d;
            if (is_finite(scalar_d) && is_finite(scalar_e)) {
                e(row, col) = scalar_e * e(row, col) + scalar_d * d(row, col);
            }
        },
        e->get_size(), e->get_stride(), e, default_stride(d),
        alpha->get_const_values(), rho->get_const_values(),
        gamma->get_const_values(), beta->get_const_values(),
        zeta->get_const_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL);


template <typename ValueType>
void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Dense<ValueType>* old_norm,
                       const matrix::Dense<ValueType>* new_norm,
                       const ValueType rel_tol, bool& is_stop)
{
    gko::Array<bool> dis_stop(exec, 1);
    components::fill_array(exec, dis_stop.get_data(), dis_stop.get_num_elems(),
                           true);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, auto old_norm, auto new_norm, auto rel_tol,
                      auto is_stop) {
            if (new_norm[tidx] > rel_tol * old_norm[tidx]) {
                *is_stop = false;
            }
        },
        new_norm->get_size()[1], old_norm->get_const_values(),
        new_norm->get_const_values(), rel_tol, dis_stop.get_data());
    is_stop = exec->copy_val_to_host(dis_stop.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL);


}  // namespace multigrid
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
