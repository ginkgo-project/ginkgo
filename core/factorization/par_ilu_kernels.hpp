/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_FACTORIZATION_PAR_ILU_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_PAR_ILU_KERNELS_HPP_


#include <ginkgo/core/factorization/par_ilu.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {

#define GKO_DECLARE_PAR_ILU_COMPUTE_L_U_FACTORS_KERNEL(ValueType, IndexType) \
    void compute_l_u_factors(                                                \
        std::shared_ptr<const DefaultExecutor> exec, size_type iterations,   \
        const matrix::Coo<ValueType, IndexType> *system_matrix,              \
        matrix::Csr<ValueType, IndexType> *l_factor,                         \
        matrix::Csr<ValueType, IndexType> *u_factor)


#define GKO_DECLARE_ALL_AS_TEMPLATES                  \
    template <typename ValueType, typename IndexType> \
    GKO_DECLARE_PAR_ILU_COMPUTE_L_U_FACTORS_KERNEL(ValueType, IndexType)


namespace omp {
namespace par_ilu_factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace par_ilu_factorization
}  // namespace omp


namespace cuda {
namespace par_ilu_factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace par_ilu_factorization
}  // namespace cuda


namespace reference {
namespace par_ilu_factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace par_ilu_factorization
}  // namespace reference


namespace hip {
namespace par_ilu_factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace par_ilu_factorization
}  // namespace hip


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_PAR_ILU_KERNELS_HPP_
