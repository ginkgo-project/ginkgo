/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_MATRIX_CSR_KERNELS_HPP_
#define GKO_CORE_MATRIX_CSR_KERNELS_HPP_


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/ell.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_CSR_SPMV_KERNEL(ValueType, IndexType)  \
    void spmv(std::shared_ptr<const DefaultExecutor> exec, \
              const matrix::Csr<ValueType, IndexType> *a,  \
              const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)

#define GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType)  \
    void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec, \
                       const matrix::Dense<ValueType> *alpha,       \
                       const matrix::Csr<ValueType, IndexType> *a,  \
                       const matrix::Dense<ValueType> *b,           \
                       const matrix::Dense<ValueType> *beta,        \
                       matrix::Dense<ValueType> *c)

#define GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType)  \
    void convert_to_dense(std::shared_ptr<const DefaultExecutor> exec, \
                          matrix::Dense<ValueType> *result,            \
                          const matrix::Csr<ValueType, IndexType> *source)

#define GKO_DECLARE_CSR_MOVE_TO_DENSE_KERNEL(ValueType, IndexType)  \
    void move_to_dense(std::shared_ptr<const DefaultExecutor> exec, \
                       matrix::Dense<ValueType> *result,            \
                       matrix::Csr<ValueType, IndexType> *source)

#define GKO_DECLARE_CSR_CONVERT_TO_COO_KERNEL(ValueType, IndexType)  \
    void convert_to_coo(std::shared_ptr<const DefaultExecutor> exec, \
                        matrix::Coo<ValueType, IndexType> *result,   \
                        const matrix::Csr<ValueType, IndexType> *source)

#define GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL(ValueType, IndexType)  \
    void convert_to_sellp(std::shared_ptr<const DefaultExecutor> exec, \
                          matrix::Sellp<ValueType, IndexType> *result, \
                          const matrix::Csr<ValueType, IndexType> *source)

#define GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL(ValueType, IndexType)  \
    void convert_to_ell(std::shared_ptr<const DefaultExecutor> exec, \
                        matrix::Ell<ValueType, IndexType> *result,   \
                        const matrix::Csr<ValueType, IndexType> *source)

#define GKO_DECLARE_CSR_CALCULATE_TOTAL_COLS_KERNEL(ValueType, IndexType)      \
    void calculate_total_cols(std::shared_ptr<const DefaultExecutor> exec,     \
                              const matrix::Csr<ValueType, IndexType> *source, \
                              size_type *result, size_type stride_factor,      \
                              size_type slice_size)

#define GKO_DECLARE_CSR_TRANSPOSE_KERNEL(ValueType, IndexType)  \
    void transpose(std::shared_ptr<const DefaultExecutor> exec, \
                   matrix::Csr<ValueType, IndexType> *trans,    \
                   const matrix::Csr<ValueType, IndexType> *orig)

#define GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType)  \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec, \
                        matrix::Csr<ValueType, IndexType> *trans,    \
                        const matrix::Csr<ValueType, IndexType> *orig)

#define GKO_DECLARE_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(ValueType, IndexType) \
    void calculate_max_nnz_per_row(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::Csr<ValueType, IndexType> *source, size_type *result)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                   \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_CSR_SPMV_KERNEL(ValueType, IndexType);                 \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_CSR_MOVE_TO_DENSE_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_CSR_CONVERT_TO_COO_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_CSR_CALCULATE_TOTAL_COLS_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_CSR_TRANSPOSE_KERNEL(ValueType, IndexType);            \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(ValueType, IndexType)


namespace omp {
namespace csr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace csr
}  // namespace omp


namespace cuda {
namespace csr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace csr
}  // namespace cuda


namespace reference {
namespace csr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace csr
}  // namespace reference


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CSR_KERNELS_HPP_
