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

#ifndef GKO_CORE_PRECONDITIONER_BATCH_ILU_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_BATCH_ILU_KERNELS_HPP_


#include <ginkgo/core/preconditioner/batch_ilu.hpp>


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_EXACT_ILU_COMPUTE_FACTORIZATION_KERNEL(ValueType, \
                                                                 IndexType) \
    void compute_ilu0_factorization(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const IndexType* const diag_locs,                                   \
        matrix::BatchCsr<ValueType, IndexType>* mat_fact)

#define GKO_DECLARE_BATCH_PARILU_COMPUTE_FACTORIZATION_KERNEL(ValueType, \
                                                              IndexType) \
    void compute_parilu0_factorization(                                  \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::BatchCsr<ValueType, IndexType>* sys_mat,           \
        matrix::BatchCsr<ValueType, IndexType>* mat_fact,                \
        const int parilu_num_sweeps, const IndexType* dependencies,      \
        const IndexType* nz_ptrs)

#define GKO_DECLARE_BATCH_ILU_APPLY_KERNEL(ValueType, IndexType)            \
    void apply_ilu(                                                         \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::BatchCsr<ValueType, IndexType>* sys_mat,              \
        const matrix::BatchCsr<ValueType, IndexType>* factored_matrix,      \
        const IndexType* diag_locs, const matrix::BatchDense<ValueType>* r, \
        matrix::BatchDense<ValueType>* z)

#define GKO_DECLARE_BATCH_ILU_GENERATE_COMMON_PATTERN_KERNEL(ValueType, \
                                                             IndexType) \
    void generate_common_pattern_to_fill_l_and_u(                       \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::Csr<ValueType, IndexType>* first_sys_mat,         \
        const IndexType* l_row_ptrs, const IndexType* u_row_ptrs,       \
        IndexType* l_col_holders, IndexType* u_col_holders)


#define GKO_DECLARE_BATCH_ILU_INITIALIZE_BATCH_L_AND_BATCH_U(ValueType, \
                                                             IndexType) \
    void initialize_batch_l_and_batch_u(                                \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::BatchCsr<ValueType, IndexType>* sys_mat,          \
        matrix::BatchCsr<ValueType, IndexType>* l_factor,               \
        matrix::BatchCsr<ValueType, IndexType>* u_factor,               \
        const IndexType* l_col_holders, const IndexType* u_col_holders)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                     \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_BATCH_EXACT_ILU_COMPUTE_FACTORIZATION_KERNEL(ValueType,  \
                                                             IndexType); \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_BATCH_PARILU_COMPUTE_FACTORIZATION_KERNEL(ValueType,     \
                                                          IndexType);    \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_BATCH_ILU_APPLY_KERNEL(ValueType, IndexType);            \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_BATCH_ILU_GENERATE_COMMON_PATTERN_KERNEL(ValueType,      \
                                                         IndexType);     \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_BATCH_ILU_INITIALIZE_BATCH_L_AND_BATCH_U(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_ilu,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BATCH_ILU_KERNELS_HPP_
