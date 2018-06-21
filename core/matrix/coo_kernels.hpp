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

#ifndef GKO_CORE_MATRIX_COO_KERNELS_HPP_
#define GKO_CORE_MATRIX_COO_KERNELS_HPP_


#include "core/base/types.hpp"
#include "core/matrix/coo.hpp"
#include "core/matrix/csr.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_COO_SPMV_KERNEL(ValueType, IndexType)  \
    void spmv(std::shared_ptr<const DefaultExecutor> exec, \
              const matrix::Coo<ValueType, IndexType> *a,  \
              const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)

#define GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL(ValueType, IndexType)  \
    void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec, \
                       const matrix::Dense<ValueType> *alpha,       \
                       const matrix::Coo<ValueType, IndexType> *a,  \
                       const matrix::Dense<ValueType> *b,           \
                       const matrix::Dense<ValueType> *beta,        \
                       matrix::Dense<ValueType> *c)

#define GKO_DECLARE_COO_CONVERT_ROW_IDXS_TO_PTRS_KERNEL(IndexType)          \
    void convert_row_idxs_to_ptrs(                                          \
        std::shared_ptr<const DefaultExecutor> exec, const IndexType *idxs, \
        size_type num_nonzeros, IndexType *ptrs, size_type length)

#define GKO_DECLARE_COO_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType)  \
    void convert_to_dense(std::shared_ptr<const DefaultExecutor> exec, \
                          matrix::Dense<ValueType> *result,            \
                          const matrix::Coo<ValueType, IndexType> *source)

#define GKO_DECLARE_COO_TRANSPOSE_KERNEL(ValueType, IndexType)  \
    void transpose(std::shared_ptr<const DefaultExecutor> exec, \
                   matrix::Coo<ValueType, IndexType> *trans,    \
                   const matrix::Coo<ValueType, IndexType> *orig)

#define GKO_DECLARE_COO_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType)  \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec, \
                        matrix::Coo<ValueType, IndexType> *trans,    \
                        const matrix::Coo<ValueType, IndexType> *orig)

#define DECLARE_ALL_AS_TEMPLATES                                   \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_COO_SPMV_KERNEL(ValueType, IndexType);             \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL(ValueType, IndexType);    \
    template <typename IndexType>                                  \
    GKO_DECLARE_COO_CONVERT_ROW_IDXS_TO_PTRS_KERNEL(IndexType);    \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_COO_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_COO_TRANSPOSE_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_COO_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType)


namespace omp {
namespace coo {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace coo
}  // namespace omp


namespace gpu {
namespace coo {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace coo
}  // namespace gpu


namespace reference {
namespace coo {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace coo
}  // namespace reference


#undef DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_COO_KERNELS_HPP_
