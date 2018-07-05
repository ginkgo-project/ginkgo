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

#ifndef GKO_CORE_SOLVER_CG_KERNELS_HPP_
#define GKO_CORE_SOLVER_CG_KERNELS_HPP_


#include "core/base/array.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "core/matrix/dense.hpp"

namespace gko {
namespace kernels {
namespace cg {


#define GKO_DECLARE_CG_INITIALIZE_KERNEL(_type)                              \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,             \
                    const matrix::Dense<_type> *b, matrix::Dense<_type> *r,  \
                    matrix::Dense<_type> *z, matrix::Dense<_type> *p,        \
                    matrix::Dense<_type> *q, matrix::Dense<_type> *prev_rho, \
                    matrix::Dense<_type> *rho, Array<bool> *converged)

#define GKO_DECLARE_CG_TEST_CONVERGENCE_KERNEL(_type)                  \
    void test_convergence(std::shared_ptr<const DefaultExecutor> exec, \
                          const matrix::Dense<_type> *tau,             \
                          const matrix::Dense<_type> *orig_tau,        \
                          remove_complex<_type> rel_residual_goal,     \
                          Array<bool> *converged, bool *all_converged)

#define GKO_DECLARE_CG_STEP_1_KERNEL(_type)                                   \
    void step_1(                                                              \
        std::shared_ptr<const DefaultExecutor> exec, matrix::Dense<_type> *p, \
        const matrix::Dense<_type> *z, const matrix::Dense<_type> *rho,       \
        const matrix::Dense<_type> *prev_rho, const Array<bool> &converged)


#define GKO_DECLARE_CG_STEP_2_KERNEL(_type)                                   \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,                  \
                matrix::Dense<_type> *x, matrix::Dense<_type> *r,             \
                const matrix::Dense<_type> *p, const matrix::Dense<_type> *q, \
                const matrix::Dense<_type> *beta,                             \
                const matrix::Dense<_type> *rho, const Array<bool> &converged)


#define DECLARE_ALL_AS_TEMPLATES                       \
    template <typename ValueType>                      \
    GKO_DECLARE_CG_INITIALIZE_KERNEL(ValueType);       \
    template <typename ValueType>                      \
    GKO_DECLARE_CG_TEST_CONVERGENCE_KERNEL(ValueType); \
    template <typename ValueType>                      \
    GKO_DECLARE_CG_STEP_1_KERNEL(ValueType);           \
    template <typename ValueType>                      \
    GKO_DECLARE_CG_STEP_2_KERNEL(ValueType)


}  // namespace cg


namespace omp {
namespace cg {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace cg
}  // namespace omp


namespace cuda {
namespace cg {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace cg
}  // namespace cuda


namespace reference {
namespace cg {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace cg
}  // namespace reference


#undef DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_CG_KERNELS_HPP
