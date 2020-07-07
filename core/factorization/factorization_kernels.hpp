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

#ifndef GKO_CORE_FACTORIZATION_FACTORIZATION_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_FACTORIZATION_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_ELEMENTS_KERNEL(ValueType,   \
                                                               IndexType)   \
    void add_diagonal_elements(std::shared_ptr<const DefaultExecutor> exec, \
                               matrix::Csr<ValueType, IndexType> *mtx,      \
                               bool is_sorted)

#define GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_U_KERNEL(ValueType, \
                                                                 IndexType) \
    void initialize_row_ptrs_l_u(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::Csr<ValueType, IndexType> *system_matrix,             \
        IndexType *l_row_ptrs, IndexType *u_row_ptrs)

#define GKO_DECLARE_FACTORIZATION_INITIALIZE_L_U_KERNEL(ValueType, IndexType) \
    void initialize_l_u(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::Csr<ValueType, IndexType> *system_matrix,               \
        matrix::Csr<ValueType, IndexType> *l_factor,                          \
        matrix::Csr<ValueType, IndexType> *u_factor)

#define GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_KERNEL(ValueType, \
                                                               IndexType) \
    void initialize_row_ptrs_l(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const matrix::Csr<ValueType, IndexType> *system_matrix,           \
        IndexType *l_row_ptrs)

#define GKO_DECLARE_FACTORIZATION_INITIALIZE_L_KERNEL(ValueType, IndexType)   \
    void initialize_l(std::shared_ptr<const DefaultExecutor> exec,            \
                      const matrix::Csr<ValueType, IndexType> *system_matrix, \
                      matrix::Csr<ValueType, IndexType> *l_factor,            \
                      bool diag_sqrt)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                       \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_ELEMENTS_KERNEL(ValueType,      \
                                                           IndexType);     \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_U_KERNEL(ValueType,    \
                                                             IndexType);   \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_U_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_KERNEL(ValueType,      \
                                                           IndexType);     \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_KERNEL(ValueType, IndexType)


namespace omp {
namespace factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace factorization
}  // namespace omp


namespace cuda {
namespace factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace factorization
}  // namespace cuda


namespace reference {
namespace factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace factorization
}  // namespace reference


namespace hip {
namespace factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace factorization
}  // namespace hip


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_FACTORIZATION_KERNELS_HPP_
