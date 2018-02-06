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

#ifndef GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
#define GKO_CORE_MATRIX_DENSE_KERNELS_HPP_


#include "core/base/types.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {

#define GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(_type) \
    void simple_apply(const matrix::Dense<_type> *a, \
                      const matrix::Dense<_type> *b, matrix::Dense<_type> *c)

#define GKO_DECLARE_DENSE_APPLY_KERNEL(_type)                                \
    void apply(const matrix::Dense<_type> *alpha,                            \
               const matrix::Dense<_type> *a, const matrix::Dense<_type> *b, \
               const matrix::Dense<_type> *beta, matrix::Dense<_type> *c)


#define GKO_DECLARE_DENSE_SCALE_KERNEL(_type) \
    void scale(const matrix::Dense<_type> *alpha, matrix::Dense<_type> *x)


#define GKO_DECLARE_DENSE_ADD_SCALED_KERNEL(_type)     \
    void add_scaled(const matrix::Dense<_type> *alpha, \
                    const matrix::Dense<_type> *x, matrix::Dense<_type> *y)


#define GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL(_type) \
    void compute_dot(const matrix::Dense<_type> *x, \
                     const matrix::Dense<_type> *y, \
                     matrix::Dense<_type> *result)

#define GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL(_type, _prec) \
    void convert_to_csr(matrix::Csr<_type, _prec> *other,     \
                        const matrix::Dense<_type> *source)

#define GKO_DECLARE_DENSE_MOVE_TO_CSR_KERNEL(_type, _prec) \
    void move_to_csr(matrix::Csr<_type, _prec> *other,     \
                     const matrix::Dense<_type> *source)

#define GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL(_type, _prec) \
    void convert_to_ell(matrix::Ell<_type, _prec> *other,     \
                        const matrix::Dense<_type> *source)

#define GKO_DECLARE_DENSE_MOVE_TO_ELL_KERNEL(_type, _prec) \
    void move_to_ell(matrix::Ell<_type, _prec> *other,     \
                     const matrix::Dense<_type> *source)

#define GKO_DECLARE_DENSE_CONVERT_TO_SLICED_ELL_KERNEL(_type, _prec) \
    void convert_to_sliced_ell(matrix::Sliced_ell<_type, _prec> *other,     \
                        const matrix::Dense<_type> *source)

#define GKO_DECLARE_DENSE_MOVE_TO_SLICED_ELL_KERNEL(_type, _prec) \
    void move_to_sliced_ell(matrix::Sliced_ell<_type, _prec> *other,     \
                     const matrix::Dense<_type> *source)

#define GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL(_type) \
    void count_nonzeros(const matrix::Dense<_type> *source, size_type *result)

#define GKO_DECLARE_DENSE_COUNT_MAX_NNZ_ROW_KERNEL(_type) \
    void count_max_nnz_row(const matrix::Dense<_type> *source, size_type *result)

#define GKO_DECLARE_DENSE_COUNT_TOTAL_COLS_KERNEL(_type) \
    void count_total_cols(const matrix::Dense<_type> *source, size_type *result)

#define GKO_DECLARE_TRANSPOSE_KERNEL(_type)     \
    void transpose(matrix::Dense<_type> *trans, \
                   const matrix::Dense<_type> *orig)

#define GKO_DECLARE_CONJ_TRANSPOSE_KERNEL(_type)     \
    void conj_transpose(matrix::Dense<_type> *trans, \
                        const matrix::Dense<_type> *orig)

#define DECLARE_ALL_AS_TEMPLATES                                   \
    template <typename ValueType>                                  \
    GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL(ValueType);              \
    template <typename ValueType>                                  \
    GKO_DECLARE_DENSE_APPLY_KERNEL(ValueType);                     \
    template <typename ValueType>                                  \
    GKO_DECLARE_DENSE_SCALE_KERNEL(ValueType);                     \
    template <typename ValueType>                                  \
    GKO_DECLARE_DENSE_ADD_SCALED_KERNEL(ValueType);                \
    template <typename ValueType>                                  \
    GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL(ValueType);               \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_DENSE_MOVE_TO_CSR_KERNEL(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_DENSE_MOVE_TO_ELL_KERNEL(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_DENSE_CONVERT_TO_SLICED_ELL_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_DENSE_MOVE_TO_SLICED_ELL_KERNEL(ValueType, IndexType);    \
    template <typename ValueType>                                  \
    GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL(ValueType);            \
    template <typename ValueType>                                  \
    GKO_DECLARE_DENSE_COUNT_MAX_NNZ_ROW_KERNEL(ValueType);         \
    template <typename ValueType>                                  \
    GKO_DECLARE_DENSE_COUNT_TOTAL_COLS_KERNEL(ValueType);          \
    template <typename ValueType>                                  \
    GKO_DECLARE_TRANSPOSE_KERNEL(ValueType);                       \
    template <typename ValueType>                                  \
    GKO_DECLARE_CONJ_TRANSPOSE_KERNEL(ValueType)

namespace cpu {
namespace dense {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace dense
}  // namespace cpu


namespace gpu {
namespace dense {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace dense
}  // namespace gpu


namespace reference {
namespace dense {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace dense
}  // namespace reference


#undef DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DENSE_KERNELS_HPP_
