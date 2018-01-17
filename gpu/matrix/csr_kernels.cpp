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

#include "core/matrix/csr_kernels.hpp"

#include "core/base/exception_helpers.hpp"

#include <cusparse.h>

namespace gko {
namespace kernels {
namespace gpu {
namespace csr {

namespace {

template <typename T>
T convert_to_cusparse_type(T val)
{
    return val;
}

cuComplex *convert_to_cusparse_type(std::complex<float> *ptr)
{
    return reinterpret_cast<cuComplex *>(ptr);
}

const cuComplex *convert_to_cusparse_type(const std::complex<float> *ptr)
{
    return reinterpret_cast<const cuComplex *>(ptr);
}

cuDoubleComplex *convert_to_cusparse_type(std::complex<double> *ptr)
{
    return reinterpret_cast<cuDoubleComplex *>(ptr);
}

const cuDoubleComplex *convert_to_cusparse_type(const std::complex<double> *ptr)
{
    return reinterpret_cast<const cuDoubleComplex *>(ptr);
}

}  // namespace

namespace {

#define BIND_CUSPARSE_SPMV(ValueType, CusparseName)                            \
    inline void cusparse_spmv(                                                 \
        cusparseHandle_t handle, cusparseOperation_t transA, int m, int n,     \
        int nnz, const ValueType *alpha, const cusparseMatDescr_t descrA,      \
        const ValueType *csrValA, const int *csrRowPtrA,                       \
        const int *csrColIndA, const ValueType *x, const ValueType *beta,      \
        ValueType *y)                                                          \
    {                                                                          \
        ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                                \
            handle, transA, m, n, nnz, convert_to_cusparse_type(alpha),        \
            descrA, convert_to_cusparse_type(csrValA), csrRowPtrA, csrColIndA, \
            convert_to_cusparse_type(x), convert_to_cusparse_type(beta),       \
            convert_to_cusparse_type(y)));                                     \
    }

template <typename ValueType, typename IndexType>
inline void cusparse_spmv(cusparseHandle_t handle, cusparseOperation_t transA,
                          int m, int n, int nnz, const ValueType *alpha,
                          const cusparseMatDescr_t descrA,
                          const ValueType *csrValA, const IndexType *csrRowPtrA,
                          const IndexType *csrColIndA, const ValueType *x,
                          const ValueType *beta, ValueType *y) NOT_IMPLEMENTED;

BIND_CUSPARSE_SPMV(float, cusparseScsrmv);
BIND_CUSPARSE_SPMV(double, cusparseDcsrmv);
BIND_CUSPARSE_SPMV(std::complex<float>, cusparseCcsrmv);
BIND_CUSPARSE_SPMV(std::complex<double>, cusparseZcsrmv);

#undef BIND_CUSPARSE_SPMV

}  // namespace

template <typename ValueType, typename IndexType>
void spmv(const matrix::Csr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b,
          matrix::Dense<ValueType> *c) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);

template <typename ValueType, typename IndexType>
void advanced_spmv(const matrix::Dense<ValueType> *alpha,
                   const matrix::Csr<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    cusparseHandle_t handle;
    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseMatDescr_t descrA;
    ASSERT_NO_CUSPARSE_ERRORS(cusparseCreate(&handle));
    ASSERT_NO_CUSPARSE_ERRORS(
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
    ASSERT_CONFORMANT(a, b);
    ASSERT_EQUAL_ROWS(a, c);
    ASSERT_EQUAL_COLS(b, c);
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    // const size scalar{1, 1};
    // auto valpha = alpha->get_values().get_const_data();
    // auto vbeta = beta->get_values().get_const_data();
    for (size_type col = 0; col < c->get_num_cols(); ++col) {
        cusparse_spmv(handle, transA, a->get_num_rows(), c->get_num_cols(),
                      row_ptrs[a->get_num_rows()] - row_ptrs[0],
                      alpha->get_const_values(), descrA,
                      a->get_const_values() + col, row_ptrs, col_idxs,
                      b->get_const_values() + col, beta->get_const_values(),
                      c->get_values() + col);
    }
};

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(matrix::Dense<ValueType> *result,
                      const matrix::Csr<ValueType, IndexType> *source)
    NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_dense(matrix::Dense<ValueType> *result,
                   matrix::Csr<ValueType, IndexType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_MOVE_TO_DENSE_KERNEL);


}  // namespace csr
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
