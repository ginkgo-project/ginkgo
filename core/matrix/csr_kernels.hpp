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

#ifndef GKO_CORE_MATRIX_CSR_KERNELS_HPP_
#define GKO_CORE_MATRIX_CSR_KERNELS_HPP_


#include "core/matrix/csr.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_CSR_SPMV_KERNEL(ValueType, IndexType) \
    void spmv(const matrix::Csr<ValueType, IndexType> *a, \
              const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)

#define GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType) \
    void advanced_spmv(const matrix::Dense<ValueType> *alpha,      \
                       const matrix::Csr<ValueType, IndexType> *a, \
                       const matrix::Dense<ValueType> *b,          \
                       const matrix::Dense<ValueType> *beta,       \
                       matrix::Dense<ValueType> *c)

#define GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType) \
    void convert_to_dense(matrix::Dense<ValueType> *result,           \
                          const matrix::Csr<ValueType, IndexType> *source)

#define GKO_DECLARE_CSR_MOVE_TO_DENSE_KERNEL(ValueType, IndexType) \
    void move_to_dense(matrix::Dense<ValueType> *result,           \
                       matrix::Csr<ValueType, IndexType> *source)

#define DECLARE_ALL_AS_TEMPLATES                                   \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_CSR_SPMV_KERNEL(ValueType, IndexType);             \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_CSR_MOVE_TO_DENSE_KERNEL(ValueType, IndexType)


namespace cpu {
namespace csr {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace csr
}  // namespace cpu


namespace gpu {
namespace csr {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace csr
}  // namespace gpu


namespace reference {
namespace csr {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace csr
}  // namespace reference


#undef DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CSR_KERNELS_HPP_
