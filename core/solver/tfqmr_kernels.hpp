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

#ifndef GKO_CORE_SOLVER_TFQMR_KERNELS_HPP_
#define GKO_CORE_SOLVER_TFQMR_KERNELS_HPP_


#include "core/base/types.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {
namespace tfqmr {


#define GKO_DECLARE_TFQMR_INITIALIZE_KERNEL(_type)                            \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,              \
                    const matrix::Dense<_type> *b, matrix::Dense<_type> *r,   \
                    matrix::Dense<_type> *r0, matrix::Dense<_type> *u_m,      \
                    matrix::Dense<_type> *u_mp1, matrix::Dense<_type> *pu_m,  \
                    matrix::Dense<_type> *Au, matrix::Dense<_type> *Ad,       \
                    matrix::Dense<_type> *w, matrix::Dense<_type> *v,         \
                    matrix::Dense<_type> *d, matrix::Dense<_type> *taut,      \
                    matrix::Dense<_type> *rho_old, matrix::Dense<_type> *rho, \
                    matrix::Dense<_type> *alpha, matrix::Dense<_type> *beta,  \
                    matrix::Dense<_type> *tau, matrix::Dense<_type> *sigma,   \
                    matrix::Dense<_type> *rov, matrix::Dense<_type> *eta,     \
                    matrix::Dense<_type> *nomw, matrix::Dense<_type> *theta)


#define GKO_DECLARE_TFQMR_STEP_1_KERNEL(_type)                                \
    void step_1(std::shared_ptr<const DefaultExecutor> exec,                  \
                matrix::Dense<_type> *alpha, const matrix::Dense<_type> *rov, \
                const matrix::Dense<_type> *rho,                              \
                const matrix::Dense<_type> *v,                                \
                const matrix::Dense<_type> *u_m, matrix::Dense<_type> *u_mp1)


#define GKO_DECLARE_TFQMR_STEP_2_KERNEL(_type)                                \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,                  \
                const matrix::Dense<_type> *theta,                            \
                const matrix::Dense<_type> *alpha,                            \
                const matrix::Dense<_type> *eta, matrix::Dense<_type> *sigma, \
                const matrix::Dense<_type> *Au,                               \
                const matrix::Dense<_type> *pu_m, matrix::Dense<_type> *w,    \
                matrix::Dense<_type> *d, matrix::Dense<_type> *Ad)


#define GKO_DECLARE_TFQMR_STEP_3_KERNEL(_type)                                 \
    void step_3(std::shared_ptr<const DefaultExecutor> exec,                   \
                matrix::Dense<_type> *theta, const matrix::Dense<_type> *nomw, \
                matrix::Dense<_type> *taut, matrix::Dense<_type> *eta,         \
                const matrix::Dense<_type> *alpha)


#define GKO_DECLARE_TFQMR_STEP_4_KERNEL(_type)                                 \
    void step_4(std::shared_ptr<const DefaultExecutor> exec,                   \
                const matrix::Dense<_type> *eta,                               \
                const matrix::Dense<_type> *d, const matrix::Dense<_type> *Ad, \
                matrix::Dense<_type> *x, matrix::Dense<_type> *r)


#define GKO_DECLARE_TFQMR_STEP_5_KERNEL(_type)                           \
    void step_5(                                                         \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        matrix::Dense<_type> *beta, const matrix::Dense<_type> *rho_old, \
        const matrix::Dense<_type> *rho, const matrix::Dense<_type> *w,  \
        const matrix::Dense<_type> *u_m, matrix::Dense<_type> *u_mp1)


#define GKO_DECLARE_TFQMR_STEP_6_KERNEL(_type)               \
    void step_6(std::shared_ptr<const DefaultExecutor> exec, \
                const matrix::Dense<_type> *beta,            \
                const matrix::Dense<_type> *Au_new,          \
                const matrix::Dense<_type> *Au, matrix::Dense<_type> *v)

#define GKO_DECLARE_TFQMR_STEP_7_KERNEL(_type)                               \
    void step_7(std::shared_ptr<const DefaultExecutor> exec,                 \
                const matrix::Dense<_type> *Au_new,                          \
                const matrix::Dense<_type> *u_mp1, matrix::Dense<_type> *Au, \
                matrix::Dense<_type> *u_m)

#define DECLARE_ALL_AS_TEMPLATES                    \
    template <typename ValueType>                   \
    GKO_DECLARE_TFQMR_INITIALIZE_KERNEL(ValueType); \
    template <typename ValueType>                   \
    GKO_DECLARE_TFQMR_STEP_1_KERNEL(ValueType);     \
    template <typename ValueType>                   \
    GKO_DECLARE_TFQMR_STEP_2_KERNEL(ValueType);     \
    template <typename ValueType>                   \
    GKO_DECLARE_TFQMR_STEP_3_KERNEL(ValueType);     \
    template <typename ValueType>                   \
    GKO_DECLARE_TFQMR_STEP_4_KERNEL(ValueType);     \
    template <typename ValueType>                   \
    GKO_DECLARE_TFQMR_STEP_5_KERNEL(ValueType);     \
    template <typename ValueType>                   \
    GKO_DECLARE_TFQMR_STEP_6_KERNEL(ValueType);     \
    template <typename ValueType>                   \
    GKO_DECLARE_TFQMR_STEP_7_KERNEL(ValueType)


}  // namespace tfqmr


namespace cpu {
namespace tfqmr {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace tfqmr
}  // namespace cpu


namespace gpu {
namespace tfqmr {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace tfqmr
}  // namespace gpu


namespace reference {
namespace tfqmr {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace tfqmr
}  // namespace reference


#undef DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_TFQMR_KERNELS_HPP_
