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

#include "core/solver/cg_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The CG solver namespace.
 *
 * @ingroup cg
 */
namespace cg {


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* r,
                matrix::Dense<ValueType>* z, matrix::Dense<ValueType>* p,
                matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* prev_rho,
                matrix::Dense<ValueType>* rho,
                array<stopping_status>* stop_status)
{
    if (b->get_size()) {
        run_kernel_solver(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto b, auto r, auto z, auto p,
                          auto q, auto prev_rho, auto rho, auto stop) {
                if (row == 0) {
                    rho[col] = zero(rho[col]);
                    prev_rho[col] = one(prev_rho[col]);
                    stop[col].reset();
                }
                r(row, col) = b(row, col);
                z(row, col) = p(row, col) = q(row, col) = zero(z(row, col));
            },
            b->get_size(), b->get_stride(), b, default_stride(r),
            default_stride(z), default_stride(p), default_stride(q),
            row_vector(prev_rho), row_vector(rho), *stop_status);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto col, auto prev_rho, auto rho, auto stop) {
                rho[col] = zero(rho[col]);
                prev_rho[col] = one(prev_rho[col]);
                stop[col].reset();
            },
            b->get_size()[1], row_vector(prev_rho), row_vector(rho),
            *stop_status);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* p, const matrix::Dense<ValueType>* z,
            const matrix::Dense<ValueType>* rho,
            const matrix::Dense<ValueType>* prev_rho,
            const array<stopping_status>* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto p, auto z, auto rho,
                      auto prev_rho, auto stop) {
            if (!stop[col].has_stopped()) {
                auto tmp = safe_divide(rho[col], prev_rho[col]);
                p(row, col) = z(row, col) + tmp * p(row, col);
            }
        },
        p->get_size(), p->get_stride(), default_stride(p), default_stride(z),
        row_vector(rho), row_vector(prev_rho), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* r,
            const matrix::Dense<ValueType>* p,
            const matrix::Dense<ValueType>* q,
            const matrix::Dense<ValueType>* beta,
            const matrix::Dense<ValueType>* rho,
            const array<stopping_status>* stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x, auto r, auto p, auto q,
                      auto beta, auto rho, auto stop) {
            if (!stop[col].has_stopped()) {
                auto tmp = safe_divide(rho[col], beta[col]);
                x(row, col) += tmp * p(row, col);
                r(row, col) -= tmp * q(row, col);
            }
        },
        x->get_size(), r->get_stride(), x, default_stride(r), default_stride(p),
        default_stride(q), row_vector(beta), row_vector(rho), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_STEP_2_KERNEL);


}  // namespace cg
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
