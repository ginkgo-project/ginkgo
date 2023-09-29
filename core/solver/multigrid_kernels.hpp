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

#ifndef GKO_CORE_SOLVER_MULTIGRID_KERNELS_HPP_
#define GKO_CORE_SOLVER_MULTIGRID_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace multigrid {


#define GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL(_type)                      \
    void kcycle_step_1(std::shared_ptr<const DefaultExecutor> exec,            \
                       const matrix::Dense<_type>* alpha,                      \
                       const matrix::Dense<_type>* rho,                        \
                       const matrix::Dense<_type>* v, matrix::Dense<_type>* g, \
                       matrix::Dense<_type>* d, matrix::Dense<_type>* e)

#define GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL(_type)                    \
    void kcycle_step_2(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const matrix::Dense<_type>* alpha, const matrix::Dense<_type>* rho,  \
        const matrix::Dense<_type>* gamma, const matrix::Dense<_type>* beta, \
        const matrix::Dense<_type>* zeta, const matrix::Dense<_type>* d,     \
        matrix::Dense<_type>* e)

#define GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL(_type)           \
    void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec, \
                           const matrix::Dense<_type>* old_norm,        \
                           const matrix::Dense<_type>* new_norm,        \
                           const _type rel_tol, bool& is_stop)


#define GKO_DECLARE_ALL_AS_TEMPLATES                       \
    template <typename ValueType>                          \
    GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL(ValueType); \
    template <typename ValueType>                          \
    GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL(ValueType); \
    template <typename ValueType>                          \
    GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL(ValueType)


}  // namespace multigrid


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(multigrid,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_MULTIGRID_KERNELS_HPP_
