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

#ifndef GKO_CUDA_BASE_CUSPARSE_BINDINGS_HPP_
#define GKO_CUDA_BASE_CUSPARSE_BINDINGS_HPP_


#include <cuda.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The CUSPARSE namespace.
 *
 * @ingroup cusparse
 */
namespace cusparse {
/**
 * @brief The detail namespace.
 *
 * @ingroup detail
 */
namespace detail {


template <typename... Args>
inline int64 not_implemented(Args...)
{
    return static_cast<int64>(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}


}  // namespace detail


template <typename ValueType, typename IndexType>
struct is_supported : std::false_type {};

template <>
struct is_supported<float, int32> : std::true_type {};

template <>
struct is_supported<double, int32> : std::true_type {};

template <>
struct is_supported<std::complex<float>, int32> : std::true_type {};

template <>
struct is_supported<std::complex<double>, int32> : std::true_type {};


#define GKO_BIND_CUSPARSE32_SPMV(ValueType, CusparseName)                    \
    inline void spmv(cusparseHandle_t handle, cusparseOperation_t transA,    \
                     int32 m, int32 n, int32 nnz, const ValueType *alpha,    \
                     const cusparseMatDescr_t descrA,                        \
                     const ValueType *csrValA, const int32 *csrRowPtrA,      \
                     const int32 *csrColIndA, const ValueType *x,            \
                     const ValueType *beta, ValueType *y)                    \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                          \
            handle, transA, m, n, nnz, as_culibs_type(alpha), descrA,        \
            as_culibs_type(csrValA), csrRowPtrA, csrColIndA,                 \
            as_culibs_type(x), as_culibs_type(beta), as_culibs_type(y)));    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_SPMV(ValueType, CusparseName)                      \
    inline void spmv(cusparseHandle_t handle, cusparseOperation_t transA,      \
                     int64 m, int64 n, int64 nnz, const ValueType *alpha,      \
                     const cusparseMatDescr_t descrA,                          \
                     const ValueType *csrValA, const int64 *csrRowPtrA,        \
                     const int64 *csrColIndA, const ValueType *x,              \
                     const ValueType *beta, ValueType *y) GKO_NOT_IMPLEMENTED; \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_SPMV(float, cusparseScsrmv);
GKO_BIND_CUSPARSE32_SPMV(double, cusparseDcsrmv);
GKO_BIND_CUSPARSE32_SPMV(std::complex<float>, cusparseCcsrmv);
GKO_BIND_CUSPARSE32_SPMV(std::complex<double>, cusparseZcsrmv);
GKO_BIND_CUSPARSE64_SPMV(float, cusparseScsrmv);
GKO_BIND_CUSPARSE64_SPMV(double, cusparseDcsrmv);
GKO_BIND_CUSPARSE64_SPMV(std::complex<float>, cusparseCcsrmv);
GKO_BIND_CUSPARSE64_SPMV(std::complex<double>, cusparseZcsrmv);
template <typename ValueType>
GKO_BIND_CUSPARSE32_SPMV(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_SPMV(ValueType, detail::not_implemented);


#undef GKO_BIND_CUSPARSE32_SPMV
#undef GKO_BIND_CUSPARSE64_SPMV


#define GKO_BIND_CUSPARSE32_SPMV(ValueType, CusparseName)                    \
    inline void spmv_mp(cusparseHandle_t handle, cusparseOperation_t transA, \
                        int32 m, int32 n, int32 nnz, const ValueType *alpha, \
                        const cusparseMatDescr_t descrA,                     \
                        const ValueType *csrValA, const int32 *csrRowPtrA,   \
                        const int32 *csrColIndA, const ValueType *x,         \
                        const ValueType *beta, ValueType *y)                 \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                          \
            handle, transA, m, n, nnz, as_culibs_type(alpha), descrA,        \
            as_culibs_type(csrValA), csrRowPtrA, csrColIndA,                 \
            as_culibs_type(x), as_culibs_type(beta), as_culibs_type(y)));    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_SPMV(ValueType, CusparseName)                      \
    inline void spmv_mp(                                                       \
        cusparseHandle_t handle, cusparseOperation_t transA, int64 m, int64 n, \
        int64 nnz, const ValueType *alpha, const cusparseMatDescr_t descrA,    \
        const ValueType *csrValA, const int64 *csrRowPtrA,                     \
        const int64 *csrColIndA, const ValueType *x, const ValueType *beta,    \
        ValueType *y) GKO_NOT_IMPLEMENTED;                                     \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_SPMV(float, cusparseScsrmv_mp);
GKO_BIND_CUSPARSE32_SPMV(double, cusparseDcsrmv_mp);
GKO_BIND_CUSPARSE32_SPMV(std::complex<float>, cusparseCcsrmv_mp);
GKO_BIND_CUSPARSE32_SPMV(std::complex<double>, cusparseZcsrmv_mp);
GKO_BIND_CUSPARSE64_SPMV(float, cusparseScsrmv_mp);
GKO_BIND_CUSPARSE64_SPMV(double, cusparseDcsrmv_mp);
GKO_BIND_CUSPARSE64_SPMV(std::complex<float>, cusparseCcsrmv_mp);
GKO_BIND_CUSPARSE64_SPMV(std::complex<double>, cusparseZcsrmv_mp);
template <typename ValueType>
GKO_BIND_CUSPARSE32_SPMV(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_SPMV(ValueType, detail::not_implemented);


#undef GKO_BIND_CUSPARSE32_SPMV
#undef GKO_BIND_CUSPARSE64_SPMV


#define GKO_BIND_CUSPARSE32_SPMM(ValueType, CusparseName)                     \
    inline void spmm(cusparseHandle_t handle, cusparseOperation_t transA,     \
                     int32 m, int32 n, int32 k, int32 nnz,                    \
                     const ValueType *alpha, const cusparseMatDescr_t descrA, \
                     const ValueType *csrValA, const int32 *csrRowPtrA,       \
                     const int32 *csrColIndA, const ValueType *B, int32 ldb,  \
                     const ValueType *beta, ValueType *C, int32 ldc)          \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                        \
            CusparseName(handle, transA, m, n, k, nnz, as_culibs_type(alpha), \
                         descrA, as_culibs_type(csrValA), csrRowPtrA,         \
                         csrColIndA, as_culibs_type(B), ldb,                  \
                         as_culibs_type(beta), as_culibs_type(C), ldc));      \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_SPMM(ValueType, CusparseName)                     \
    inline void spmm(cusparseHandle_t handle, cusparseOperation_t transA,     \
                     int64 m, int64 n, int64 k, int64 nnz,                    \
                     const ValueType *alpha, const cusparseMatDescr_t descrA, \
                     const ValueType *csrValA, const int64 *csrRowPtrA,       \
                     const int64 *csrColIndA, const ValueType *B, int64 ldb,  \
                     const ValueType *beta, ValueType *C, int64 ldc)          \
        GKO_NOT_IMPLEMENTED;                                                  \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_SPMM(float, cusparseScsrmm);
GKO_BIND_CUSPARSE32_SPMM(double, cusparseDcsrmm);
GKO_BIND_CUSPARSE32_SPMM(std::complex<float>, cusparseCcsrmm);
GKO_BIND_CUSPARSE32_SPMM(std::complex<double>, cusparseZcsrmm);
GKO_BIND_CUSPARSE64_SPMM(float, cusparseScsrmm);
GKO_BIND_CUSPARSE64_SPMM(double, cusparseDcsrmm);
GKO_BIND_CUSPARSE64_SPMM(std::complex<float>, cusparseCcsrmm);
GKO_BIND_CUSPARSE64_SPMM(std::complex<double>, cusparseZcsrmm);
template <typename ValueType>
GKO_BIND_CUSPARSE32_SPMM(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_SPMM(ValueType, detail::not_implemented);


#undef GKO_BIND_CUSPARSE32_SPMM
#undef GKO_BIND_CUSPARSE64_SPMM


template <typename ValueType, typename IndexType>
inline void spmv(cusparseHandle_t handle, cusparseAlgMode_t alg,
                 cusparseOperation_t transA, IndexType m, IndexType n,
                 IndexType nnz, const ValueType *alpha,
                 const cusparseMatDescr_t descrA, const ValueType *csrValA,
                 const IndexType *csrRowPtrA, const IndexType *csrColIndA,
                 const ValueType *x, const ValueType *beta, ValueType *y,
                 void *buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_SPMV(ValueType)                                      \
    template <>                                                                \
    inline void spmv<ValueType, int32>(                                        \
        cusparseHandle_t handle, cusparseAlgMode_t alg,                        \
        cusparseOperation_t transA, int32 m, int32 n, int32 nnz,               \
        const ValueType *alpha, const cusparseMatDescr_t descrA,               \
        const ValueType *csrValA, const int32 *csrRowPtrA,                     \
        const int32 *csrColIndA, const ValueType *x, const ValueType *beta,    \
        ValueType *y, void *buffer)                                            \
    {                                                                          \
        auto data_type = gko::kernels::cuda::cuda_data_type<ValueType>();      \
        if (data_type == CUDA_C_8U) {                                          \
            GKO_NOT_IMPLEMENTED;                                               \
        }                                                                      \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCsrmvEx(                         \
            handle, alg, transA, m, n, nnz, alpha, data_type, descrA, csrValA, \
            data_type, csrRowPtrA, csrColIndA, x, data_type, beta, data_type,  \
            y, data_type, data_type, buffer));                                 \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_SPMV(float);
GKO_BIND_CUSPARSE_SPMV(double);
GKO_BIND_CUSPARSE_SPMV(std::complex<float>);
GKO_BIND_CUSPARSE_SPMV(std::complex<double>);


#undef GKO_BIND_CUSPARSE_SPMV


template <typename ValueType, typename IndexType>
inline void spmv_buffersize(cusparseHandle_t handle, cusparseAlgMode_t alg,
                            cusparseOperation_t transA, IndexType m,
                            IndexType n, IndexType nnz, const ValueType *alpha,
                            const cusparseMatDescr_t descrA,
                            const ValueType *csrValA,
                            const IndexType *csrRowPtrA,
                            const IndexType *csrColIndA, const ValueType *x,
                            const ValueType *beta, ValueType *y,
                            size_t *bufferSizeInBytes) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(ValueType)                           \
    template <>                                                                \
    inline void spmv_buffersize<ValueType, int32>(                             \
        cusparseHandle_t handle, cusparseAlgMode_t alg,                        \
        cusparseOperation_t transA, int32 m, int32 n, int32 nnz,               \
        const ValueType *alpha, const cusparseMatDescr_t descrA,               \
        const ValueType *csrValA, const int32 *csrRowPtrA,                     \
        const int32 *csrColIndA, const ValueType *x, const ValueType *beta,    \
        ValueType *y, size_t *bufferSizeInBytes)                               \
    {                                                                          \
        auto data_type = gko::kernels::cuda::cuda_data_type<ValueType>();      \
        if (data_type == CUDA_C_8U) {                                          \
            GKO_NOT_IMPLEMENTED;                                               \
        }                                                                      \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCsrmvEx_bufferSize(              \
            handle, alg, transA, m, n, nnz, alpha, data_type, descrA, csrValA, \
            data_type, csrRowPtrA, csrColIndA, x, data_type, beta, data_type,  \
            y, data_type, data_type, bufferSizeInBytes));                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(float);
GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(double);
GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(std::complex<float>);
GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(std::complex<double>);


#undef GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE


#define GKO_BIND_CUSPARSE32_SPMV(ValueType, CusparseName)                     \
    inline void spmv(cusparseHandle_t handle, cusparseOperation_t transA,     \
                     const ValueType *alpha, const cusparseMatDescr_t descrA, \
                     const cusparseHybMat_t hybA, const ValueType *x,         \
                     const ValueType *beta, ValueType *y)                     \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                           \
            handle, transA, as_culibs_type(alpha), descrA, hybA,              \
            as_culibs_type(x), as_culibs_type(beta), as_culibs_type(y)));     \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_SPMV(float, cusparseShybmv);
GKO_BIND_CUSPARSE32_SPMV(double, cusparseDhybmv);
GKO_BIND_CUSPARSE32_SPMV(std::complex<float>, cusparseChybmv);
GKO_BIND_CUSPARSE32_SPMV(std::complex<double>, cusparseZhybmv);
template <typename ValueType>
GKO_BIND_CUSPARSE32_SPMV(ValueType, detail::not_implemented);


#undef GKO_BIND_CUSPARSE32_SPMV


template <typename IndexType, typename ValueType>
void spgemm_buffer_size(
    cusparseHandle_t handle, IndexType m, IndexType n, IndexType k,
    const ValueType *alpha, const cusparseMatDescr_t descrA, IndexType nnzA,
    const IndexType *csrRowPtrA, const IndexType *csrColIndA,
    const cusparseMatDescr_t descrB, IndexType nnzB,
    const IndexType *csrRowPtrB, const IndexType *csrColIndB,
    const ValueType *beta, const cusparseMatDescr_t descrD, IndexType nnzD,
    const IndexType *csrRowPtrD, const IndexType *csrColIndD,
    csrgemm2Info_t info, size_type &result) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_SPGEMM_BUFFER_SIZE(ValueType, CusparseName)          \
    template <>                                                                \
    inline void spgemm_buffer_size<int32, ValueType>(                          \
        cusparseHandle_t handle, int32 m, int32 n, int32 k,                    \
        const ValueType *alpha, const cusparseMatDescr_t descrA, int32 nnzA,   \
        const int32 *csrRowPtrA, const int32 *csrColIndA,                      \
        const cusparseMatDescr_t descrB, int32 nnzB, const int32 *csrRowPtrB,  \
        const int32 *csrColIndB, const ValueType *beta,                        \
        const cusparseMatDescr_t descrD, int32 nnzD, const int32 *csrRowPtrD,  \
        const int32 *csrColIndD, csrgemm2Info_t info, size_type &result)       \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, m, n, k, as_culibs_type(alpha), descrA, nnzA, \
                         csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB,     \
                         csrColIndB, as_culibs_type(beta), descrD, nnzD,       \
                         csrRowPtrD, csrColIndD, info, &result));              \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_SPGEMM_BUFFER_SIZE(float, cusparseScsrgemm2_bufferSizeExt);
GKO_BIND_CUSPARSE_SPGEMM_BUFFER_SIZE(double, cusparseDcsrgemm2_bufferSizeExt);
GKO_BIND_CUSPARSE_SPGEMM_BUFFER_SIZE(std::complex<float>,
                                     cusparseCcsrgemm2_bufferSizeExt);
GKO_BIND_CUSPARSE_SPGEMM_BUFFER_SIZE(std::complex<double>,
                                     cusparseZcsrgemm2_bufferSizeExt);


#undef GKO_BIND_CUSPARSE_SPGEMM_BUFFER_SIZE


template <typename IndexType>
void spgemm_nnz(cusparseHandle_t handle, IndexType m, IndexType n, IndexType k,
                const cusparseMatDescr_t descrA, IndexType nnzA,
                const IndexType *csrRowPtrA, const IndexType *csrColIndA,
                const cusparseMatDescr_t descrB, IndexType nnzB,
                const IndexType *csrRowPtrB, const IndexType *csrColIndB,
                const cusparseMatDescr_t descrD, IndexType nnzD,
                const IndexType *csrRowPtrD, const IndexType *csrColIndD,
                const cusparseMatDescr_t descrC, IndexType *csrRowPtrC,
                IndexType *nnzC, csrgemm2Info_t info,
                void *buffer) GKO_NOT_IMPLEMENTED;

template <>
inline void spgemm_nnz<int32>(
    cusparseHandle_t handle, int32 m, int32 n, int32 k,
    const cusparseMatDescr_t descrA, int32 nnzA, const int32 *csrRowPtrA,
    const int32 *csrColIndA, const cusparseMatDescr_t descrB, int32 nnzB,
    const int32 *csrRowPtrB, const int32 *csrColIndB,
    const cusparseMatDescr_t descrD, int32 nnzD, const int32 *csrRowPtrD,
    const int32 *csrColIndD, const cusparseMatDescr_t descrC, int32 *csrRowPtrC,
    int32 *nnzC, csrgemm2Info_t info, void *buffer)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseXcsrgemm2Nnz(
        handle, m, n, k, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB,
        csrRowPtrB, csrColIndB, descrD, nnzD, csrRowPtrD, csrColIndD, descrC,
        csrRowPtrC, nnzC, info, buffer));
}


template <typename IndexType, typename ValueType>
void spgemm(cusparseHandle_t handle, IndexType m, IndexType n, IndexType k,
            const ValueType *alpha, const cusparseMatDescr_t descrA,
            IndexType nnzA, const ValueType *csrValA,
            const IndexType *csrRowPtrA, const IndexType *csrColIndA,
            const cusparseMatDescr_t descrB, IndexType nnzB,
            const ValueType *csrValB, const IndexType *csrRowPtrB,
            const IndexType *csrColIndB, const ValueType *beta,
            const cusparseMatDescr_t descrD, IndexType nnzD,
            const ValueType *csrValD, const IndexType *csrRowPtrD,
            const IndexType *csrColIndD, const cusparseMatDescr_t descrC,
            ValueType *csrValC, const IndexType *csrRowPtrC,
            IndexType *csrColIndC, csrgemm2Info_t info,
            void *buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_SPGEMM(ValueType, CusparseName)                      \
    template <>                                                                \
    inline void spgemm<int32, ValueType>(                                      \
        cusparseHandle_t handle, int32 m, int32 n, int32 k,                    \
        const ValueType *alpha, const cusparseMatDescr_t descrA, int32 nnzA,   \
        const ValueType *csrValA, const int32 *csrRowPtrA,                     \
        const int32 *csrColIndA, const cusparseMatDescr_t descrB, int32 nnzB,  \
        const ValueType *csrValB, const int32 *csrRowPtrB,                     \
        const int32 *csrColIndB, const ValueType *beta,                        \
        const cusparseMatDescr_t descrD, int32 nnzD, const ValueType *csrValD, \
        const int32 *csrRowPtrD, const int32 *csrColIndD,                      \
        const cusparseMatDescr_t descrC, ValueType *csrValC,                   \
        const int32 *csrRowPtrC, int32 *csrColIndC, csrgemm2Info_t info,       \
        void *buffer)                                                          \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                            \
            handle, m, n, k, as_culibs_type(alpha), descrA, nnzA,              \
            as_culibs_type(csrValA), csrRowPtrA, csrColIndA, descrB, nnzB,     \
            as_culibs_type(csrValB), csrRowPtrB, csrColIndB,                   \
            as_culibs_type(beta), descrD, nnzD, as_culibs_type(csrValD),       \
            csrRowPtrD, csrColIndD, descrC, as_culibs_type(csrValC),           \
            csrRowPtrC, csrColIndC, info, buffer));                            \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_SPGEMM(float, cusparseScsrgemm2);
GKO_BIND_CUSPARSE_SPGEMM(double, cusparseDcsrgemm2);
GKO_BIND_CUSPARSE_SPGEMM(std::complex<float>, cusparseCcsrgemm2);
GKO_BIND_CUSPARSE_SPGEMM(std::complex<double>, cusparseZcsrgemm2);


#undef GKO_BIND_CUSPARSE_SPGEMM


#define GKO_BIND_CUSPARSE32_CSR2HYB(ValueType, CusparseName)                 \
    inline void csr2hyb(cusparseHandle_t handle, int32 m, int32 n,           \
                        const cusparseMatDescr_t descrA,                     \
                        const ValueType *csrValA, const int32 *csrRowPtrA,   \
                        const int32 *csrColIndA, cusparseHybMat_t hybA,      \
                        int32 userEllWidth,                                  \
                        cusparseHybPartition_t partitionType)                \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(CusparseName(                          \
            handle, m, n, descrA, as_culibs_type(csrValA), csrRowPtrA,       \
            csrColIndA, hybA, userEllWidth, partitionType));                 \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_CSR2HYB(ValueType, CusparseName)                 \
    inline void csr2hyb(                                                     \
        cusparseHandle_t handle, int64 m, int64 n,                           \
        const cusparseMatDescr_t descrA, const ValueType *csrValA,           \
        const int64 *csrRowPtrA, const int64 *csrColIndA,                    \
        cusparseHybMat_t hybA, int64 userEllWidth,                           \
        cusparseHybPartition_t partitionType) GKO_NOT_IMPLEMENTED;           \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_CSR2HYB(float, cusparseScsr2hyb);
GKO_BIND_CUSPARSE32_CSR2HYB(double, cusparseDcsr2hyb);
GKO_BIND_CUSPARSE32_CSR2HYB(std::complex<float>, cusparseCcsr2hyb);
GKO_BIND_CUSPARSE32_CSR2HYB(std::complex<double>, cusparseZcsr2hyb);
GKO_BIND_CUSPARSE64_CSR2HYB(float, cusparseScsr2hyb);
GKO_BIND_CUSPARSE64_CSR2HYB(double, cusparseDcsr2hyb);
GKO_BIND_CUSPARSE64_CSR2HYB(std::complex<float>, cusparseCcsr2hyb);
GKO_BIND_CUSPARSE64_CSR2HYB(std::complex<double>, cusparseZcsr2hyb);
template <typename ValueType>
GKO_BIND_CUSPARSE32_CSR2HYB(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_CSR2HYB(ValueType, detail::not_implemented);


#undef GKO_BIND_CUSPARSE32_CSR2HYB
#undef GKO_BIND_CUSPARSE64_CSR2HYB


#define GKO_BIND_CUSPARSE_TRANSPOSE32(ValueType, CusparseName)                \
    inline void transpose(cusparseHandle_t handle, size_type m, size_type n,  \
                          size_type nnz, const ValueType *OrigValA,           \
                          const int32 *OrigRowPtrA, const int32 *OrigColIndA, \
                          ValueType *TransValA, int32 *TransRowPtrA,          \
                          int32 *TransColIndA, cusparseAction_t copyValues,   \
                          cusparseIndexBase_t idxBase)                        \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                        \
            CusparseName(handle, m, n, nnz, as_culibs_type(OrigValA),         \
                         OrigRowPtrA, OrigColIndA, as_culibs_type(TransValA), \
                         TransRowPtrA, TransColIndA, copyValues, idxBase));   \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE_TRANSPOSE64(ValueType, CusparseName)                \
    inline void transpose(cusparseHandle_t handle, size_type m, size_type n,  \
                          size_type nnz, const ValueType *OrigValA,           \
                          const int64 *OrigRowPtrA, const int64 *OrigColIndA, \
                          ValueType *TransValA, int64 *TransRowPtrA,          \
                          int64 *TransColIndA, cusparseAction_t copyValues,   \
                          cusparseIndexBase_t idxBase) GKO_NOT_IMPLEMENTED;   \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_TRANSPOSE32(float, cusparseScsr2csc);
GKO_BIND_CUSPARSE_TRANSPOSE32(double, cusparseDcsr2csc);
GKO_BIND_CUSPARSE_TRANSPOSE64(float, cusparseScsr2csc);
GKO_BIND_CUSPARSE_TRANSPOSE64(double, cusparseDcsr2csc);
GKO_BIND_CUSPARSE_TRANSPOSE32(std::complex<float>, cusparseCcsr2csc);
GKO_BIND_CUSPARSE_TRANSPOSE32(std::complex<double>, cusparseZcsr2csc);
GKO_BIND_CUSPARSE_TRANSPOSE64(std::complex<float>, cusparseCcsr2csc);
GKO_BIND_CUSPARSE_TRANSPOSE64(std::complex<double>, cusparseZcsr2csc);
template <typename ValueType>
GKO_BIND_CUSPARSE_TRANSPOSE32(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE_TRANSPOSE64(ValueType, detail::not_implemented);

#undef GKO_BIND_CUSPARSE_TRANSPOSE

#define GKO_BIND_CUSPARSE_CONJ_TRANSPOSE32(ValueType, CusparseName)          \
    inline void conj_transpose(                                              \
        cusparseHandle_t handle, size_type m, size_type n, size_type nnz,    \
        const ValueType *OrigValA, const int32 *OrigRowPtrA,                 \
        const int32 *OrigColIndA, ValueType *TransValA, int32 *TransRowPtrA, \
        int32 *TransColIndA, cusparseAction_t copyValues,                    \
        cusparseIndexBase_t idxBase) GKO_NOT_IMPLEMENTED;                    \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE_CONJ_TRANSPOSE64(ValueType, CusparseName)          \
    inline void conj_transpose(                                              \
        cusparseHandle_t handle, size_type m, size_type n, size_type nnz,    \
        const ValueType *OrigValA, const int64 *OrigRowPtrA,                 \
        const int64 *OrigColIndA, ValueType *TransValA, int64 *TransRowPtrA, \
        int64 *TransColIndA, cusparseAction_t copyValues,                    \
        cusparseIndexBase_t idxBase) GKO_NOT_IMPLEMENTED;                    \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_CONJ_TRANSPOSE32(float, cusparseScsr2csc);
GKO_BIND_CUSPARSE_CONJ_TRANSPOSE32(double, cusparseDcsr2csc);
GKO_BIND_CUSPARSE_CONJ_TRANSPOSE64(float, cusparseScsr2csc);
GKO_BIND_CUSPARSE_CONJ_TRANSPOSE64(double, cusparseDcsr2csc);
GKO_BIND_CUSPARSE_CONJ_TRANSPOSE32(std::complex<float>, cusparseCcsr2csc);
GKO_BIND_CUSPARSE_CONJ_TRANSPOSE32(std::complex<double>, cusparseZcsr2csc);
GKO_BIND_CUSPARSE_CONJ_TRANSPOSE64(std::complex<float>, cusparseCcsr2csc);
GKO_BIND_CUSPARSE_CONJ_TRANSPOSE64(std::complex<double>, cusparseZcsr2csc);
template <typename ValueType>
GKO_BIND_CUSPARSE_CONJ_TRANSPOSE32(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE_CONJ_TRANSPOSE64(ValueType, detail::not_implemented);

#undef GKO_BIND_CUSPARSE_CONJ_TRANSPOSE


inline cusparseHandle_t init()
{
    cusparseHandle_t handle{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreate(&handle));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
    return handle;
}


inline void destroy(cusparseHandle_t handle)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroy(handle));
}


inline cusparseMatDescr_t create_mat_descr()
{
    cusparseMatDescr_t descr{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateMatDescr(&descr));
    return descr;
}


inline void destroy(cusparseMatDescr_t descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyMatDescr(descr));
}


inline csrgemm2Info_t create_spgemm_info()
{
    csrgemm2Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateCsrgemm2Info(&info));
    return info;
}


inline void destroy(csrgemm2Info_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyCsrgemm2Info(info));
}


// CUDA versions 9.2 and above have csrsm2.
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))


inline csrsm2Info_t create_solve_info()
{
    csrsm2Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateCsrsm2Info(&info));
    return info;
}


inline void destroy(csrsm2Info_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyCsrsm2Info(info));
}


// CUDA_VERSION<=9.1 do not support csrsm2.
#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))


inline cusparseSolveAnalysisInfo_t create_solve_info()
{
    cusparseSolveAnalysisInfo_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateSolveAnalysisInfo(&info));
    return info;
}


inline void destroy(cusparseSolveAnalysisInfo_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroySolveAnalysisInfo(info));
}


#endif


inline csrilu02Info_t create_ilu0_info()
{
    csrilu02Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateCsrilu02Info(&info));
    return info;
}


inline void destroy(csrilu02Info_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyCsrilu02Info(info));
}


// CUDA versions 9.2 and above have csrsm2.
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))


#define GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(ValueType, CusparseName)            \
    inline void buffer_size_ext(                                              \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,        \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz,  \
        const ValueType *one, const cusparseMatDescr_t descr,                 \
        const ValueType *csrVal, const int32 *csrRowPtr,                      \
        const int32 *csrColInd, const ValueType *rhs, int32 sol_size,         \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,               \
        size_t *factor_work_size)                                             \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                        \
            CusparseName(handle, algo, trans1, trans2, m, n, nnz,             \
                         as_culibs_type(one), descr, as_culibs_type(csrVal),  \
                         csrRowPtr, csrColInd, as_culibs_type(rhs), sol_size, \
                         factor_info, policy, factor_work_size));             \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(ValueType, CusparseName)           \
    inline void buffer_size_ext(                                             \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,       \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz, \
        const ValueType *one, const cusparseMatDescr_t descr,                \
        const ValueType *csrVal, const int64 *csrRowPtr,                     \
        const int64 *csrColInd, const ValueType *rhs, int64 sol_size,        \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,              \
        size_t *factor_work_size) GKO_NOT_IMPLEMENTED;                       \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(float, cusparseScsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(double, cusparseDcsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(std::complex<float>,
                                  cusparseCcsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(std::complex<double>,
                                  cusparseZcsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(float, cusparseScsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(double, cusparseDcsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(std::complex<float>,
                                  cusparseCcsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(std::complex<double>,
                                  cusparseZcsrsm2_bufferSizeExt);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_BUFFERSIZEEXT
#undef GKO_BIND_CUSPARSE64_BUFFERSIZEEXT


#define GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(ValueType, CusparseName)          \
    inline void csrsm2_analysis(                                              \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,        \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz,  \
        const ValueType *one, const cusparseMatDescr_t descr,                 \
        const ValueType *csrVal, const int32 *csrRowPtr,                      \
        const int32 *csrColInd, const ValueType *rhs, int32 sol_size,         \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,               \
        void *factor_work_vec)                                                \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                        \
            CusparseName(handle, algo, trans1, trans2, m, n, nnz,             \
                         as_culibs_type(one), descr, as_culibs_type(csrVal),  \
                         csrRowPtr, csrColInd, as_culibs_type(rhs), sol_size, \
                         factor_info, policy, factor_work_vec));              \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(ValueType, CusparseName)         \
    inline void csrsm2_analysis(                                             \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,       \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz, \
        const ValueType *one, const cusparseMatDescr_t descr,                \
        const ValueType *csrVal, const int64 *csrRowPtr,                     \
        const int64 *csrColInd, const ValueType *rhs, int64 sol_size,        \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,              \
        void *factor_work_vec) GKO_NOT_IMPLEMENTED;                          \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(float, cusparseScsrsm2_analysis);
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(double, cusparseDcsrsm2_analysis);
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(std::complex<float>,
                                    cusparseCcsrsm2_analysis);
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(std::complex<double>,
                                    cusparseZcsrsm2_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(float, cusparseScsrsm2_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(double, cusparseDcsrsm2_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(std::complex<float>,
                                    cusparseCcsrsm2_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(std::complex<double>,
                                    cusparseZcsrsm2_analysis);
template <typename ValueType>
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS
#undef GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS


#define GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(ValueType, CusparseName)            \
    inline void csrsm2_solve(                                                \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,       \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz, \
        const ValueType *one, const cusparseMatDescr_t descr,                \
        const ValueType *csrVal, const int32 *csrRowPtr,                     \
        const int32 *csrColInd, ValueType *rhs, int32 sol_stride,            \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,              \
        void *factor_work_vec)                                               \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, algo, trans1, trans2, m, n, nnz,            \
                         as_culibs_type(one), descr, as_culibs_type(csrVal), \
                         csrRowPtr, csrColInd, as_culibs_type(rhs),          \
                         sol_stride, factor_info, policy, factor_work_vec)); \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(ValueType, CusparseName)            \
    inline void csrsm2_solve(                                                \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,       \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz, \
        const ValueType *one, const cusparseMatDescr_t descr,                \
        const ValueType *csrVal, const int64 *csrRowPtr,                     \
        const int64 *csrColInd, ValueType *rhs, int64 sol_stride,            \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,              \
        void *factor_work_vec) GKO_NOT_IMPLEMENTED;                          \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(float, cusparseScsrsm2_solve);
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(double, cusparseDcsrsm2_solve);
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(std::complex<float>, cusparseCcsrsm2_solve);
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(std::complex<double>, cusparseZcsrsm2_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(float, cusparseScsrsm2_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(double, cusparseDcsrsm2_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(std::complex<float>, cusparseCcsrsm2_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(std::complex<double>, cusparseZcsrsm2_solve);
template <typename ValueType>
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_CSRSM2_SOLVE
#undef GKO_BIND_CUSPARSE64_CSRSM2_SOLVE


// CUDA_VERSION<=9.1 do not support csrsm2.
#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))


#define GKO_BIND_CUSPARSE32_CSRSM_ANALYSIS(ValueType, CusparseName)            \
    inline void csrsm_analysis(                                                \
        cusparseHandle_t handle, cusparseOperation_t trans, size_type m,       \
        size_type nnz, const cusparseMatDescr_t descr,                         \
        const ValueType *csrVal, const int32 *csrRowPtr,                       \
        const int32 *csrColInd, cusparseSolveAnalysisInfo_t factor_info)       \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, trans, m, nnz, descr, as_culibs_type(csrVal), \
                         csrRowPtr, csrColInd, factor_info));                  \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_CSRSM_ANALYSIS(ValueType, CusparseName)      \
    inline void csrsm_analysis(                                          \
        cusparseHandle_t handle, cusparseOperation_t trans, size_type m, \
        size_type nnz, const cusparseMatDescr_t descr,                   \
        const ValueType *csrVal, const int64 *csrRowPtr,                 \
        const int64 *csrColInd, cusparseSolveAnalysisInfo_t factor_info) \
        GKO_NOT_IMPLEMENTED;                                             \
    static_assert(true,                                                  \
                  "This assert is used to counter the "                  \
                  "false positive extra "                                \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_CSRSM_ANALYSIS(float, cusparseScsrsm_analysis);
GKO_BIND_CUSPARSE32_CSRSM_ANALYSIS(double, cusparseDcsrsm_analysis);
GKO_BIND_CUSPARSE32_CSRSM_ANALYSIS(std::complex<float>,
                                   cusparseCcsrsm_analysis);
GKO_BIND_CUSPARSE32_CSRSM_ANALYSIS(std::complex<double>,
                                   cusparseZcsrsm_analysis);
GKO_BIND_CUSPARSE64_CSRSM_ANALYSIS(float, cusparseScsrsm_analysis);
GKO_BIND_CUSPARSE64_CSRSM_ANALYSIS(double, cusparseDcsrsm_analysis);
GKO_BIND_CUSPARSE64_CSRSM_ANALYSIS(std::complex<float>,
                                   cusparseCcsrsm_analysis);
GKO_BIND_CUSPARSE64_CSRSM_ANALYSIS(std::complex<double>,
                                   cusparseZcsrsm_analysis);
template <typename ValueType>
GKO_BIND_CUSPARSE32_CSRSM_ANALYSIS(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_CSRSM_ANALYSIS(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_CSRSM_ANALYSIS
#undef GKO_BIND_CUSPARSE64_CSRSM_ANALYSIS

#define GKO_BIND_CUSPARSE32_CSRSM_SOLVE(ValueType, CusparseName)             \
    inline void csrsm_solve(                                                 \
        cusparseHandle_t handle, cusparseOperation_t trans, size_type m,     \
        size_type n, const ValueType *one, const cusparseMatDescr_t descr,   \
        const ValueType *csrVal, const int32 *csrRowPtr,                     \
        const int32 *csrColInd, cusparseSolveAnalysisInfo_t factor_info,     \
        const ValueType *rhs, int32 rhs_stride, ValueType *sol,              \
        int32 sol_stride)                                                    \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, trans, m, n, as_culibs_type(one), descr,    \
                         as_culibs_type(csrVal), csrRowPtr, csrColInd,       \
                         factor_info, as_culibs_type(rhs), rhs_stride,       \
                         as_culibs_type(sol), sol_stride));                  \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_CSRSM_SOLVE(ValueType, CusparseName)             \
    inline void csrsm_solve(                                                 \
        cusparseHandle_t handle, cusparseOperation_t trans1, size_type m,    \
        size_type n, const ValueType *one, const cusparseMatDescr_t descr,   \
        const ValueType *csrVal, const int64 *csrRowPtr,                     \
        const int64 *csrColInd, cusparseSolveAnalysisInfo_t factor_info,     \
        const ValueType *rhs, int64 rhs_stride, ValueType *sol,              \
        int64 sol_stride) GKO_NOT_IMPLEMENTED;                               \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_CSRSM_SOLVE(float, cusparseScsrsm_solve);
GKO_BIND_CUSPARSE32_CSRSM_SOLVE(double, cusparseDcsrsm_solve);
GKO_BIND_CUSPARSE32_CSRSM_SOLVE(std::complex<float>, cusparseCcsrsm_solve);
GKO_BIND_CUSPARSE32_CSRSM_SOLVE(std::complex<double>, cusparseZcsrsm_solve);
GKO_BIND_CUSPARSE64_CSRSM_SOLVE(float, cusparseScsrsm_solve);
GKO_BIND_CUSPARSE64_CSRSM_SOLVE(double, cusparseDcsrsm_solve);
GKO_BIND_CUSPARSE64_CSRSM_SOLVE(std::complex<float>, cusparseCcsrsm_solve);
GKO_BIND_CUSPARSE64_CSRSM_SOLVE(std::complex<double>, cusparseZcsrsm_solve);
template <typename ValueType>
GKO_BIND_CUSPARSE32_CSRSM_SOLVE(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_CSRSM_SOLVE(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_CSRSM_SOLVE
#undef GKO_BIND_CUSPARSE64_CSRSM_SOLVE


#endif


template <typename IndexType>
void create_identity_permutation(cusparseHandle_t handle, IndexType size,
                                 IndexType *permutation) GKO_NOT_IMPLEMENTED;

template <>
inline void create_identity_permutation<int32>(cusparseHandle_t handle,
                                               int32 size, int32 *permutation)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseCreateIdentityPermutation(handle, size, permutation));
}


template <typename IndexType>
void csrsort_buffer_size(cusparseHandle_t handle, IndexType m, IndexType n,
                         IndexType nnz, const IndexType *row_ptrs,
                         const IndexType *col_idxs,
                         size_type &buffer_size) GKO_NOT_IMPLEMENTED;

template <>
inline void csrsort_buffer_size<int32>(cusparseHandle_t handle, int32 m,
                                       int32 n, int32 nnz,
                                       const int32 *row_ptrs,
                                       const int32 *col_idxs,
                                       size_type &buffer_size)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseXcsrsort_bufferSizeExt(
        handle, m, n, nnz, row_ptrs, col_idxs, &buffer_size));
}


template <typename IndexType>
void csrsort(cusparseHandle_t handle, IndexType m, IndexType n, IndexType nnz,
             const cusparseMatDescr_t descr, const IndexType *row_ptrs,
             IndexType *col_idxs, IndexType *permutation,
             void *buffer) GKO_NOT_IMPLEMENTED;

template <>
inline void csrsort<int32>(cusparseHandle_t handle, int32 m, int32 n, int32 nnz,
                           const cusparseMatDescr_t descr,
                           const int32 *row_ptrs, int32 *col_idxs,
                           int32 *permutation, void *buffer)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseXcsrsort(
        handle, m, n, nnz, descr, row_ptrs, col_idxs, permutation, buffer));
}


template <typename IndexType, typename ValueType>
void gather(cusparseHandle_t handle, IndexType nnz, const ValueType *in,
            ValueType *out, const IndexType *permutation) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_GATHER(ValueType, CusparseName)                      \
    template <>                                                                \
    inline void gather<int32, ValueType>(cusparseHandle_t handle, int32 nnz,   \
                                         const ValueType *in, ValueType *out,  \
                                         const int32 *permutation)             \
    {                                                                          \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                         \
            CusparseName(handle, nnz, as_culibs_type(in), as_culibs_type(out), \
                         permutation, CUSPARSE_INDEX_BASE_ZERO));              \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_GATHER(float, cusparseSgthr);
GKO_BIND_CUSPARSE_GATHER(double, cusparseDgthr);
GKO_BIND_CUSPARSE_GATHER(std::complex<float>, cusparseCgthr);
GKO_BIND_CUSPARSE_GATHER(std::complex<double>, cusparseZgthr);

#undef GKO_BIND_CUSPARSE_GATHER


template <typename ValueType, typename IndexType>
void ilu0_buffer_size(cusparseHandle_t handle, IndexType m, IndexType nnz,
                      const cusparseMatDescr_t descr, const ValueType *vals,
                      const IndexType *row_ptrs, const IndexType *col_idxs,
                      csrilu02Info_t info,
                      size_type &buffer_size) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(ValueType, CusparseName)          \
    template <>                                                              \
    inline void ilu0_buffer_size<ValueType, int32>(                          \
        cusparseHandle_t handle, int32 m, int32 nnz,                         \
        const cusparseMatDescr_t descr, const ValueType *vals,               \
        const int32 *row_ptrs, const int32 *col_idxs, csrilu02Info_t info,   \
        size_type &buffer_size)                                              \
    {                                                                        \
        int tmp_buffer_size{};                                               \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr,                              \
                         as_culibs_type(const_cast<ValueType *>(vals)),      \
                         row_ptrs, col_idxs, info, &tmp_buffer_size));       \
        buffer_size = tmp_buffer_size;                                       \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(float, cusparseScsrilu02_bufferSize);
GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(double, cusparseDcsrilu02_bufferSize);
GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(std::complex<float>,
                                   cusparseCcsrilu02_bufferSize);
GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(std::complex<double>,
                                   cusparseZcsrilu02_bufferSize);

#undef GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE


template <typename ValueType, typename IndexType>
void ilu0_analysis(cusparseHandle_t handle, IndexType m, IndexType nnz,
                   const cusparseMatDescr_t descr, const ValueType *vals,
                   const IndexType *row_ptrs, const IndexType *col_idxs,
                   csrilu02Info_t info, cusparseSolvePolicy_t policy,
                   void *buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_ILU0_ANALYSIS(ValueType, CusparseName)             \
    template <>                                                              \
    inline void ilu0_analysis<ValueType, int32>(                             \
        cusparseHandle_t handle, int32 m, int32 nnz,                         \
        const cusparseMatDescr_t descr, const ValueType *vals,               \
        const int32 *row_ptrs, const int32 *col_idxs, csrilu02Info_t info,   \
        cusparseSolvePolicy_t policy, void *buffer)                          \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr, as_culibs_type(vals),        \
                         row_ptrs, col_idxs, info, policy, buffer));         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_ILU0_ANALYSIS(float, cusparseScsrilu02_analysis);
GKO_BIND_CUSPARSE_ILU0_ANALYSIS(double, cusparseDcsrilu02_analysis);
GKO_BIND_CUSPARSE_ILU0_ANALYSIS(std::complex<float>,
                                cusparseCcsrilu02_analysis);
GKO_BIND_CUSPARSE_ILU0_ANALYSIS(std::complex<double>,
                                cusparseZcsrilu02_analysis);

#undef GKO_BIND_CUSPARSE_ILU0_ANALYSIS


template <typename ValueType, typename IndexType>
void ilu0(cusparseHandle_t handle, IndexType m, IndexType nnz,
          const cusparseMatDescr_t descr, ValueType *vals,
          const IndexType *row_ptrs, const IndexType *col_idxs,
          csrilu02Info_t info, cusparseSolvePolicy_t policy,
          void *buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_ILU0(ValueType, CusparseName)                      \
    template <>                                                              \
    inline void ilu0<ValueType, int32>(                                      \
        cusparseHandle_t handle, int32 m, int32 nnz,                         \
        const cusparseMatDescr_t descr, ValueType *vals,                     \
        const int32 *row_ptrs, const int32 *col_idxs, csrilu02Info_t info,   \
        cusparseSolvePolicy_t policy, void *buffer)                          \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr, as_culibs_type(vals),        \
                         row_ptrs, col_idxs, info, policy, buffer));         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_ILU0(float, cusparseScsrilu02);
GKO_BIND_CUSPARSE_ILU0(double, cusparseDcsrilu02);
GKO_BIND_CUSPARSE_ILU0(std::complex<float>, cusparseCcsrilu02);
GKO_BIND_CUSPARSE_ILU0(std::complex<double>, cusparseZcsrilu02);

#undef GKO_BIND_CUSPARSE_ILU0


}  // namespace cusparse
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CUSPARSE_BINDINGS_HPP_
