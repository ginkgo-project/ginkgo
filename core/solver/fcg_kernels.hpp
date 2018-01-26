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

#ifndef GKO_CORE_SOLVER_FCG_KERNELS_HPP_
#define GKO_CORE_SOLVER_FCG_KERNELS_HPP_


#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {
namespace fcg {


#define GKO_DECLARE_FCG_INITIALIZE_KERNEL(_type)                             \
    void initialize(const matrix::Dense<_type> *b, matrix::Dense<_type> *r,  \
                    matrix::Dense<_type> *z, matrix::Dense<_type> *p,        \
                    matrix::Dense<_type> *q, matrix::Dense<_type> *prev_r,   \
                    matrix::Dense<_type> *t, matrix::Dense<_type> *prev_rho, \
                    matrix::Dense<_type> *rho, matrix::Dense<_type> *rho_t)


#define GKO_DECLARE_FCG_STEP_1_KERNEL(_type)                            \
    void step_1(matrix::Dense<_type> *p, const matrix::Dense<_type> *z, \
                const matrix::Dense<_type> *rho_t,                      \
                const matrix::Dense<_type> *prev_rho)


#define GKO_DECLARE_FCG_STEP_2_KERNEL(_type)                                  \
    void step_2(matrix::Dense<_type> *x, matrix::Dense<_type> *r,             \
                matrix::Dense<_type> *prev_r, matrix::Dense<_type> *t,        \
                const matrix::Dense<_type> *p, const matrix::Dense<_type> *q, \
                const matrix::Dense<_type> *beta,                             \
                const matrix::Dense<_type> *rho)


#define DECLARE_ALL_AS_TEMPLATES                  \
    template <typename ValueType>                 \
    GKO_DECLARE_FCG_INITIALIZE_KERNEL(ValueType); \
    template <typename ValueType>                 \
    GKO_DECLARE_FCG_STEP_1_KERNEL(ValueType);     \
    template <typename ValueType>                 \
    GKO_DECLARE_FCG_STEP_2_KERNEL(ValueType)


}  // namespace fcg


namespace cpu {
namespace fcg {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace fcg
}  // namespace cpu


namespace gpu {
namespace fcg {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace fcg
}  // namespace gpu


namespace reference {
namespace fcg {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace fcg
}  // namespace reference


#undef DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_FCG_KERNELS_HPP_
