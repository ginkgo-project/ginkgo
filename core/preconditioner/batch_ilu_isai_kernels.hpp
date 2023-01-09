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

#ifndef GKO_CORE_PRECONDITIONER_BATCH_ILU_ISAI_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_BATCH_ILU_ISAI_KERNELS_HPP_


#include <ginkgo/core/preconditioner/batch_ilu_isai.hpp>


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_ILU_ISAI_APPLY_KERNEL(ValueType, IndexType)       \
    void apply_ilu_isai(                                                    \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::BatchCsr<ValueType, IndexType>* sys_mat,              \
        const matrix::BatchCsr<ValueType, IndexType>* l,                    \
        const matrix::BatchCsr<ValueType, IndexType>* u,                    \
        const matrix::BatchCsr<ValueType, IndexType>* l_inv,                \
        const matrix::BatchCsr<ValueType, IndexType>* u_inv,                \
        const matrix::BatchCsr<ValueType, IndexType>* mult_invs,            \
        const matrix::BatchCsr<ValueType, IndexType>* iter_mat_lower_solve, \
        const matrix::BatchCsr<ValueType, IndexType>* iter_mat_upper_solve, \
        const preconditioner::batch_ilu_isai_apply apply_type,              \
        const int num_relaxation_steps,                                     \
        const matrix::BatchDense<ValueType>* r,                             \
        matrix::BatchDense<ValueType>* z)


#define GKO_DECLARE_ALL_AS_TEMPLATES                  \
    template <typename ValueType, typename IndexType> \
    GKO_DECLARE_BATCH_ILU_ISAI_APPLY_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_ilu_isai,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BATCH_ILU_ISAI_KERNELS_HPP_
