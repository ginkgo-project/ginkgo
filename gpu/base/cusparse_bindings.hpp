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

#ifndef GPU_BASE_CUSPARSE_BINDINGS_HPP_
#define GPU_BASE_CUSPARSE_BINDINGS_HPP_


#include <cusparse.h>


#include "core/base/exception_helpers.hpp"
#include "gpu/base/types.hpp"


namespace gko {
namespace cusparse {
namespace {

#define BIND_CUSPARSE32_SPMV(ValueType, CusparseName)                         \
    inline void spmv(cusparseHandle_t handle, cusparseOperation_t transA,     \
                     size_type m, size_type n, size_type nnz,                 \
                     const ValueType *alpha, const cusparseMatDescr_t descrA, \
                     const ValueType *csrValA, const int32 *csrRowPtrA,       \
                     const int32 *csrColIndA, const ValueType *x,             \
                     const ValueType *beta, ValueType *y)                     \
    {                                                                         \
        ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                               \
            handle, transA, m, n, nnz, as_culibs_type(alpha), descrA,         \
            as_culibs_type(csrValA), csrRowPtrA, csrColIndA,                  \
            as_culibs_type(x), as_culibs_type(beta), as_culibs_type(y)));     \
    }

#define BIND_CUSPARSE64_SPMV(ValueType, CusparseName)                         \
    inline void spmv(cusparseHandle_t handle, cusparseOperation_t transA,     \
                     size_type m, size_type n, size_type nnz,                 \
                     const ValueType *alpha, const cusparseMatDescr_t descrA, \
                     const ValueType *csrValA, const int64 *csrRowPtrA,       \
                     const int64 *csrColIndA, const ValueType *x,             \
                     const ValueType *beta, ValueType *y) NOT_IMPLEMENTED;

BIND_CUSPARSE32_SPMV(float, cusparseScsrmv);
BIND_CUSPARSE32_SPMV(double, cusparseDcsrmv);
BIND_CUSPARSE32_SPMV(std::complex<float>, cusparseCcsrmv);
BIND_CUSPARSE32_SPMV(std::complex<double>, cusparseZcsrmv);
BIND_CUSPARSE64_SPMV(float, cusparseScsrmv);
BIND_CUSPARSE64_SPMV(double, cusparseDcsrmv);
BIND_CUSPARSE64_SPMV(std::complex<float>, cusparseCcsrmv);
BIND_CUSPARSE64_SPMV(std::complex<double>, cusparseZcsrmv);

#undef BIND_CUSPARSE32_SPMV
#undef BIND_CUSPARSE64_SPMV

inline cusparseHandle_t init()
{
    cusparseHandle_t handle{};
    ASSERT_NO_CUSPARSE_ERRORS(cusparseCreate(&handle));
    ASSERT_NO_CUSPARSE_ERRORS(
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
    return handle;
}


inline void destroy(cusparseHandle_t handle)
{
    ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroy(handle));
}


inline cusparseMatDescr_t create_mat_descr()
{
    cusparseMatDescr_t descr{};
    ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateMatDescr(&descr));
    return descr;
}


inline void destroy(cusparseMatDescr_t descr)
{
    ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyMatDescr(descr));
}


}  // namespace
}  // namespace cusparse
}  // namespace gko


#endif  // GPU_BASE_CUSPARSE_BINDINGS_HPP_
