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

#ifndef GKO_CUDA_BASE_CUSPARSE_BINDINGS_HPP_
#define GKO_CUDA_BASE_CUSPARSE_BINDINGS_HPP_


#include <cuda.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/types.hpp"


namespace gko {
namespace solver {


#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))
struct SolveStruct {
    int algorithm;
    csrsm2Info_t solve_info;
    cusparseSolvePolicy_t policy;
    cusparseMatDescr_t factor_descr;
    size_t factor_work_size;
    void *factor_work_vec;
};

#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))
struct SolveStruct {
    cusparseSolveAnalysisInfo_t solve_info;
    cusparseMatDescr_t factor_descr;
};
#endif


}  // namespace solver


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


inline gko::solver::SolveStruct *init_trs_solve_struct()
{
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))
    gko::solver::SolveStruct *solve_struct = new gko::solver::SolveStruct{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseCreateCsrsm2Info(&solve_struct->solve_info));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseCreateMatDescr(&solve_struct->factor_descr));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatIndexBase(
        solve_struct->factor_descr, CUSPARSE_INDEX_BASE_ZERO));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatType(
        solve_struct->factor_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatDiagType(
        solve_struct->factor_descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
    solve_struct->algorithm = 0;
    solve_struct->policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))
    gko::solver::SolveStruct *solve_struct = new gko::solver::SolveStruct{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseCreateSolveAnalysisInfo(&solve_struct->solve_info));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseCreateMatDescr(&solve_struct->factor_descr));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatIndexBase(
        solve_struct->factor_descr, CUSPARSE_INDEX_BASE_ZERO));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatType(
        solve_struct->factor_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatDiagType(
        solve_struct->factor_descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
#endif
    return solve_struct;
}

inline void clear_trs_solve_struct(gko::solver::SolveStruct *solve_struct)
{
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))
    cusparse::destroy(solve_struct->factor_descr);
    if (solve_struct->solve_info) {
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseDestroyCsrsm2Info(solve_struct->solve_info));
    }
    if (solve_struct->factor_work_vec != nullptr) {
        GKO_ASSERT_NO_CUDA_ERRORS(cudaFree(solve_struct->factor_work_vec));
    }
#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))
    cusparse::destroy(solve_struct->factor_descr);
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseDestroySolveAnalysisInfo(solve_struct->solve_info));
#endif
}


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


#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))
// CUDA versions 9.1 and below do not have csrsm2.
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

#endif


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

#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))
// CUDA versions 9.2 and above have csrsm2.
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

#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))

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

#endif


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

#define GKO_BIND_CUSPARSE32_CSRSM_SOLVE(ValueType, CusparseName)             \
    inline void csrsm_solve(                                                 \
        cusparseHandle_t handle, cusparseOperation_t trans, size_type m,     \
        size_type n, const ValueType *one, const cusparseMatDescr_t descr,   \
        const ValueType *csrVal, const int32 *csrRowPtr,                     \
        const int32 *csrColInd, cusparseSolveAnalysisInfo_t factor_info,     \
        ValueType *rhs, int32 rhs_stride, ValueType *sol, int32 sol_stride)  \
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
        ValueType *rhs, int64 rhs_stride, ValueType *sol, int64 sol_stride)  \
        GKO_NOT_IMPLEMENTED;                                                 \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))
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

#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))

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


}  // namespace cusparse
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CUSPARSE_BINDINGS_HPP_
