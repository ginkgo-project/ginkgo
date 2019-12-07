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

#ifndef GKO_HIP_BASE_HIPSPARSE_BINDINGS_HIP_HPP_
#define GKO_HIP_BASE_HIPSPARSE_BINDINGS_HIP_HPP_


#include <hipsparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The HIPSPARSE namespace.
 *
 * @ingroup hipsparse
 */
namespace hipsparse {
/**
 * @brief The detail namespace.
 *
 * @ingroup detail
 */
namespace detail {


template <typename... Args>
inline int64 not_implemented(Args...)
{
    return static_cast<int64>(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}


}  // namespace detail


template <typename ValueType, typename IndexType>
struct is_supported : std::false_type {};

template <>
struct is_supported<float, int32> : std::true_type {};

template <>
struct is_supported<double, int32> : std::true_type {};


#define GKO_BIND_HIPSPARSE32_SPMV(ValueType, HipsparseName)                  \
    inline void spmv(hipsparseHandle_t handle, hipsparseOperation_t transA,  \
                     int32 m, int32 n, int32 nnz, const ValueType *alpha,    \
                     const hipsparseMatDescr_t descrA,                       \
                     const ValueType *csrValA, const int32 *csrRowPtrA,      \
                     const int32 *csrColIndA, const ValueType *x,            \
                     const ValueType *beta, ValueType *y)                    \
    {                                                                        \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(HipsparseName(                        \
            handle, transA, m, n, nnz, as_hiplibs_type(alpha), descrA,       \
            as_hiplibs_type(csrValA), csrRowPtrA, csrColIndA,                \
            as_hiplibs_type(x), as_hiplibs_type(beta), as_hiplibs_type(y))); \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_HIPSPARSE64_SPMV(ValueType, HipsparseName)                    \
    inline void spmv(hipsparseHandle_t handle, hipsparseOperation_t transA,    \
                     int64 m, int64 n, int64 nnz, const ValueType *alpha,      \
                     const hipsparseMatDescr_t descrA,                         \
                     const ValueType *csrValA, const int64 *csrRowPtrA,        \
                     const int64 *csrColIndA, const ValueType *x,              \
                     const ValueType *beta, ValueType *y) GKO_NOT_IMPLEMENTED; \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE32_SPMV(float, hipsparseScsrmv);
GKO_BIND_HIPSPARSE32_SPMV(double, hipsparseDcsrmv);
GKO_BIND_HIPSPARSE64_SPMV(float, hipsparseScsrmv);
GKO_BIND_HIPSPARSE64_SPMV(double, hipsparseDcsrmv);
template <typename ValueType>
GKO_BIND_HIPSPARSE32_SPMV(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_HIPSPARSE64_SPMV(ValueType, detail::not_implemented);


#undef GKO_BIND_HIPSPARSE32_SPMV
#undef GKO_BIND_HIPSPARSE64_SPMV


#define GKO_BIND_HIPSPARSE32_SPMM(ValueType, HipsparseName)                    \
    inline void spmm(hipsparseHandle_t handle, hipsparseOperation_t transA,    \
                     int32 m, int32 n, int32 k, int32 nnz,                     \
                     const ValueType *alpha, const hipsparseMatDescr_t descrA, \
                     const ValueType *csrValA, const int32 *csrRowPtrA,        \
                     const int32 *csrColIndA, const ValueType *B, int32 ldb,   \
                     const ValueType *beta, ValueType *C, int32 ldc)           \
    {                                                                          \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(HipsparseName(                          \
            handle, transA, m, n, k, nnz, as_hiplibs_type(alpha), descrA,      \
            as_hiplibs_type(csrValA), csrRowPtrA, csrColIndA,                  \
            as_hiplibs_type(B), ldb, as_hiplibs_type(beta),                    \
            as_hiplibs_type(C), ldc));                                         \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_BIND_HIPSPARSE64_SPMM(ValueType, HipsparseName)                    \
    inline void spmm(hipsparseHandle_t handle, hipsparseOperation_t transA,    \
                     int64 m, int64 n, int64 k, int64 nnz,                     \
                     const ValueType *alpha, const hipsparseMatDescr_t descrA, \
                     const ValueType *csrValA, const int64 *csrRowPtrA,        \
                     const int64 *csrColIndA, const ValueType *B, int64 ldb,   \
                     const ValueType *beta, ValueType *C, int64 ldc)           \
        GKO_NOT_IMPLEMENTED;                                                   \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE32_SPMM(float, hipsparseScsrmm);
GKO_BIND_HIPSPARSE32_SPMM(double, hipsparseDcsrmm);
GKO_BIND_HIPSPARSE64_SPMM(float, hipsparseScsrmm);
GKO_BIND_HIPSPARSE64_SPMM(double, hipsparseDcsrmm);
template <typename ValueType>
GKO_BIND_HIPSPARSE32_SPMM(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_HIPSPARSE64_SPMM(ValueType, detail::not_implemented);


#undef GKO_BIND_HIPSPARSE32_SPMM
#undef GKO_BIND_HIPSPARSE64_SPMM


#define GKO_BIND_HIPSPARSE32_SPMV(ValueType, HipsparseName)                    \
    inline void spmv(hipsparseHandle_t handle, hipsparseOperation_t transA,    \
                     const ValueType *alpha, const hipsparseMatDescr_t descrA, \
                     const hipsparseHybMat_t hybA, const ValueType *x,         \
                     const ValueType *beta, ValueType *y)                      \
    {                                                                          \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(HipsparseName(                          \
            handle, transA, as_hiplibs_type(alpha), descrA, hybA,              \
            as_hiplibs_type(x), as_hiplibs_type(beta), as_hiplibs_type(y)));   \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE32_SPMV(float, hipsparseShybmv);
GKO_BIND_HIPSPARSE32_SPMV(double, hipsparseDhybmv);
template <typename ValueType>
GKO_BIND_HIPSPARSE32_SPMV(ValueType, detail::not_implemented);


#undef GKO_BIND_HIPSPARSE32_SPMV


template <typename IndexType, typename ValueType>
void spgemm_buffer_size(
    hipsparseHandle_t handle, IndexType m, IndexType n, IndexType k,
    const ValueType *alpha, const hipsparseMatDescr_t descrA, IndexType nnzA,
    const IndexType *csrRowPtrA, const IndexType *csrColIndA,
    const hipsparseMatDescr_t descrB, IndexType nnzB,
    const IndexType *csrRowPtrB, const IndexType *csrColIndB,
    const ValueType *beta, const hipsparseMatDescr_t descrD, IndexType nnzD,
    const IndexType *csrRowPtrD, const IndexType *csrColIndD,
    csrgemm2Info_t info, size_type &result) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_HIPSPARSE_SPGEMM_BUFFER_SIZE(ValueType, HipsparseName)        \
    template <>                                                                \
    inline void spgemm_buffer_size<int32, ValueType>(                          \
        hipsparseHandle_t handle, int32 m, int32 n, int32 k,                   \
        const ValueType *alpha, const hipsparseMatDescr_t descrA, int32 nnzA,  \
        const int32 *csrRowPtrA, const int32 *csrColIndA,                      \
        const hipsparseMatDescr_t descrB, int32 nnzB, const int32 *csrRowPtrB, \
        const int32 *csrColIndB, const ValueType *beta,                        \
        const hipsparseMatDescr_t descrD, int32 nnzD, const int32 *csrRowPtrD, \
        const int32 *csrColIndD, csrgemm2Info_t info, size_type &result)       \
    {                                                                          \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(HipsparseName(                          \
            handle, m, n, k, as_hiplibs_type(alpha), descrA, nnzA, csrRowPtrA, \
            csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB,                  \
            as_hiplibs_type(beta), descrD, nnzD, csrRowPtrD, csrColIndD, info, \
            &result));                                                         \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE_SPGEMM_BUFFER_SIZE(float, hipsparseScsrgemm2_bufferSizeExt);
GKO_BIND_HIPSPARSE_SPGEMM_BUFFER_SIZE(double, hipsparseDcsrgemm2_bufferSizeExt);
// TODO: uncomment when the shipped hipSparse gets updated
// GKO_BIND_HIPSPARSE_SPGEMM_BUFFER_SIZE(std::complex<float>,
//                                       hipsparseCcsrgemm2_bufferSizeExt);
// GKO_BIND_HIPSPARSE_SPGEMM_BUFFER_SIZE(std::complex<double>,
//                                       hipsparseZcsrgemm2_bufferSizeExt);*/


#undef GKO_BIND_HIPSPARSE_SPGEMM_BUFFER_SIZE


template <typename IndexType>
void spgemm_nnz(hipsparseHandle_t handle, IndexType m, IndexType n, IndexType k,
                const hipsparseMatDescr_t descrA, IndexType nnzA,
                const IndexType *csrRowPtrA, const IndexType *csrColIndA,
                const hipsparseMatDescr_t descrB, IndexType nnzB,
                const IndexType *csrRowPtrB, const IndexType *csrColIndB,
                const hipsparseMatDescr_t descrD, IndexType nnzD,
                const IndexType *csrRowPtrD, const IndexType *csrColIndD,
                const hipsparseMatDescr_t descrC, IndexType *csrRowPtrC,
                IndexType *nnzC, csrgemm2Info_t info,
                void *buffer) GKO_NOT_IMPLEMENTED;

template <>
inline void spgemm_nnz<int32>(
    hipsparseHandle_t handle, int32 m, int32 n, int32 k,
    const hipsparseMatDescr_t descrA, int32 nnzA, const int32 *csrRowPtrA,
    const int32 *csrColIndA, const hipsparseMatDescr_t descrB, int32 nnzB,
    const int32 *csrRowPtrB, const int32 *csrColIndB,
    const hipsparseMatDescr_t descrD, int32 nnzD, const int32 *csrRowPtrD,
    const int32 *csrColIndD, const hipsparseMatDescr_t descrC,
    int32 *csrRowPtrC, int32 *nnzC, csrgemm2Info_t info, void *buffer)
{
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseXcsrgemm2Nnz(
        handle, m, n, k, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB,
        csrRowPtrB, csrColIndB, descrD, nnzD, csrRowPtrD, csrColIndD, descrC,
        csrRowPtrC, nnzC, info, buffer));
}


template <typename IndexType, typename ValueType>
void spgemm(hipsparseHandle_t handle, IndexType m, IndexType n, IndexType k,
            const ValueType *alpha, const hipsparseMatDescr_t descrA,
            IndexType nnzA, const ValueType *csrValA,
            const IndexType *csrRowPtrA, const IndexType *csrColIndA,
            const hipsparseMatDescr_t descrB, IndexType nnzB,
            const ValueType *csrValB, const IndexType *csrRowPtrB,
            const IndexType *csrColIndB, const ValueType *beta,
            const hipsparseMatDescr_t descrD, IndexType nnzD,
            const ValueType *csrValD, const IndexType *csrRowPtrD,
            const IndexType *csrColIndD, const hipsparseMatDescr_t descrC,
            ValueType *csrValC, const IndexType *csrRowPtrC,
            IndexType *csrColIndC, csrgemm2Info_t info,
            void *buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_HIPSPARSE_SPGEMM(ValueType, HipsparseName)                    \
    template <>                                                                \
    inline void spgemm<int32, ValueType>(                                      \
        hipsparseHandle_t handle, int32 m, int32 n, int32 k,                   \
        const ValueType *alpha, const hipsparseMatDescr_t descrA, int32 nnzA,  \
        const ValueType *csrValA, const int32 *csrRowPtrA,                     \
        const int32 *csrColIndA, const hipsparseMatDescr_t descrB, int32 nnzB, \
        const ValueType *csrValB, const int32 *csrRowPtrB,                     \
        const int32 *csrColIndB, const ValueType *beta,                        \
        const hipsparseMatDescr_t descrD, int32 nnzD,                          \
        const ValueType *csrValD, const int32 *csrRowPtrD,                     \
        const int32 *csrColIndD, const hipsparseMatDescr_t descrC,             \
        ValueType *csrValC, const int32 *csrRowPtrC, int32 *csrColIndC,        \
        csrgemm2Info_t info, void *buffer)                                     \
    {                                                                          \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(HipsparseName(                          \
            handle, m, n, k, as_hiplibs_type(alpha), descrA, nnzA,             \
            as_hiplibs_type(csrValA), csrRowPtrA, csrColIndA, descrB, nnzB,    \
            as_hiplibs_type(csrValB), csrRowPtrB, csrColIndB,                  \
            as_hiplibs_type(beta), descrD, nnzD, as_hiplibs_type(csrValD),     \
            csrRowPtrD, csrColIndD, descrC, as_hiplibs_type(csrValC),          \
            csrRowPtrC, csrColIndC, info, buffer));                            \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE_SPGEMM(float, hipsparseScsrgemm2);
GKO_BIND_HIPSPARSE_SPGEMM(double, hipsparseDcsrgemm2);
// TODO: uncomment when the shipped hipSparse gets updated
// GKO_BIND_HIPSPARSE_SPGEMM(std::complex<float>, hipsparseCcsrgemm2);
// GKO_BIND_HIPSPARSE_SPGEMM(std::complex<double>, hipsparseZcsrgemm2);


#undef GKO_BIND_HIPSPARSE_SPGEMM


#define GKO_BIND_HIPSPARSE32_CSR2HYB(ValueType, HipsparseName)               \
    inline void csr2hyb(hipsparseHandle_t handle, int32 m, int32 n,          \
                        const hipsparseMatDescr_t descrA,                    \
                        const ValueType *csrValA, const int32 *csrRowPtrA,   \
                        const int32 *csrColIndA, hipsparseHybMat_t hybA,     \
                        int32 userEllWidth,                                  \
                        hipsparseHybPartition_t partitionType)               \
    {                                                                        \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(HipsparseName(                        \
            handle, m, n, descrA, as_hiplibs_type(csrValA), csrRowPtrA,      \
            csrColIndA, hybA, userEllWidth, partitionType));                 \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_HIPSPARSE64_CSR2HYB(ValueType, HipsparseName)               \
    inline void csr2hyb(                                                     \
        hipsparseHandle_t handle, int64 m, int64 n,                          \
        const hipsparseMatDescr_t descrA, const ValueType *csrValA,          \
        const int64 *csrRowPtrA, const int64 *csrColIndA,                    \
        hipsparseHybMat_t hybA, int64 userEllWidth,                          \
        hipsparseHybPartition_t partitionType) GKO_NOT_IMPLEMENTED;          \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE32_CSR2HYB(float, hipsparseScsr2hyb);
GKO_BIND_HIPSPARSE32_CSR2HYB(double, hipsparseDcsr2hyb);
GKO_BIND_HIPSPARSE64_CSR2HYB(float, hipsparseScsr2hyb);
GKO_BIND_HIPSPARSE64_CSR2HYB(double, hipsparseDcsr2hyb);
template <typename ValueType>
GKO_BIND_HIPSPARSE32_CSR2HYB(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_HIPSPARSE64_CSR2HYB(ValueType, detail::not_implemented);


#undef GKO_BIND_HIPSPARSE32_CSR2HYB
#undef GKO_BIND_HIPSPARSE64_CSR2HYB


#define GKO_BIND_HIPSPARSE_TRANSPOSE32(ValueType, HipsparseName)              \
    inline void transpose(hipsparseHandle_t handle, size_type m, size_type n, \
                          size_type nnz, const ValueType *OrigValA,           \
                          const int32 *OrigRowPtrA, const int32 *OrigColIndA, \
                          ValueType *TransValA, int32 *TransRowPtrA,          \
                          int32 *TransColIndA, hipsparseAction_t copyValues,  \
                          hipsparseIndexBase_t idxBase)                       \
    {                                                                         \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(HipsparseName(                         \
            handle, m, n, nnz, as_hiplibs_type(OrigValA), OrigRowPtrA,        \
            OrigColIndA, as_hiplibs_type(TransValA), TransRowPtrA,            \
            TransColIndA, copyValues, idxBase));                              \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_HIPSPARSE_TRANSPOSE64(ValueType, HipsparseName)              \
    inline void transpose(hipsparseHandle_t handle, size_type m, size_type n, \
                          size_type nnz, const ValueType *OrigValA,           \
                          const int64 *OrigRowPtrA, const int64 *OrigColIndA, \
                          ValueType *TransValA, int64 *TransRowPtrA,          \
                          int64 *TransColIndA, hipsparseAction_t copyValues,  \
                          hipsparseIndexBase_t idxBase) GKO_NOT_IMPLEMENTED;  \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE_TRANSPOSE32(float, hipsparseScsr2csc);
GKO_BIND_HIPSPARSE_TRANSPOSE32(double, hipsparseDcsr2csc);
GKO_BIND_HIPSPARSE_TRANSPOSE64(float, hipsparseScsr2csc);
GKO_BIND_HIPSPARSE_TRANSPOSE64(double, hipsparseDcsr2csc);
template <typename ValueType>
GKO_BIND_HIPSPARSE_TRANSPOSE32(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_HIPSPARSE_TRANSPOSE64(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPSPARSE_TRANSPOSE

#define GKO_BIND_HIPSPARSE_CONJ_TRANSPOSE32(ValueType, HipsparseName)        \
    inline void conj_transpose(                                              \
        hipsparseHandle_t handle, size_type m, size_type n, size_type nnz,   \
        const ValueType *OrigValA, const int32 *OrigRowPtrA,                 \
        const int32 *OrigColIndA, ValueType *TransValA, int32 *TransRowPtrA, \
        int32 *TransColIndA, hipsparseAction_t copyValues,                   \
        hipsparseIndexBase_t idxBase) GKO_NOT_IMPLEMENTED;                   \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_HIPSPARSE_CONJ_TRANSPOSE64(ValueType, HipsparseName)        \
    inline void conj_transpose(                                              \
        hipsparseHandle_t handle, size_type m, size_type n, size_type nnz,   \
        const ValueType *OrigValA, const int64 *OrigRowPtrA,                 \
        const int64 *OrigColIndA, ValueType *TransValA, int64 *TransRowPtrA, \
        int64 *TransColIndA, hipsparseAction_t copyValues,                   \
        hipsparseIndexBase_t idxBase) GKO_NOT_IMPLEMENTED;                   \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE_CONJ_TRANSPOSE32(float, hipsparseScsr2csc);
GKO_BIND_HIPSPARSE_CONJ_TRANSPOSE32(double, hipsparseDcsr2csc);
GKO_BIND_HIPSPARSE_CONJ_TRANSPOSE64(float, hipsparseScsr2csc);
GKO_BIND_HIPSPARSE_CONJ_TRANSPOSE64(double, hipsparseDcsr2csc);
template <typename ValueType>
GKO_BIND_HIPSPARSE_CONJ_TRANSPOSE32(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_HIPSPARSE_CONJ_TRANSPOSE64(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPSPARSE_CONJ_TRANSPOSE


#define GKO_BIND_HIPSPARSE32_CSRSV2_BUFFERSIZE(ValueType, HipsparseName)     \
    inline void csrsv2_buffer_size(                                          \
        hipsparseHandle_t handle, hipsparseOperation_t trans,                \
        const size_type m, size_type nnz, const hipsparseMatDescr_t descr,   \
        const ValueType *csrVal, const int32 *csrRowPtr,                     \
        const int32 *csrColInd, csrsv2Info_t factor_info,                    \
        int *factor_work_size)                                               \
    {                                                                        \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(HipsparseName(                        \
            handle, trans, m, nnz, descr,                                    \
            as_hiplibs_type(const_cast<ValueType *>(csrVal)), csrRowPtr,     \
            csrColInd, factor_info, factor_work_size));                      \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_BIND_HIPSPARSE64_CSRSV2_BUFFERSIZE(ValueType, HipsparseName)   \
    inline void csrsv2_buffer_size(                                        \
        hipsparseHandle_t handle, hipsparseOperation_t trans, size_type m, \
        size_type nnz, const hipsparseMatDescr_t descr,                    \
        const ValueType *csrVal, const int64 *csrRowPtr,                   \
        const int64 *csrColInd, csrsv2Info_t factor_info,                  \
        int *factor_work_size) GKO_NOT_IMPLEMENTED;                        \
    static_assert(true,                                                    \
                  "This assert is used to counter the "                    \
                  "false positive extra "                                  \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE32_CSRSV2_BUFFERSIZE(float, hipsparseScsrsv2_bufferSize);
GKO_BIND_HIPSPARSE32_CSRSV2_BUFFERSIZE(double, hipsparseDcsrsv2_bufferSize);
GKO_BIND_HIPSPARSE64_CSRSV2_BUFFERSIZE(float, hipsparseScsrsv2_bufferSize);
GKO_BIND_HIPSPARSE64_CSRSV2_BUFFERSIZE(double, hipsparseDcsrsv2_bufferSize);
template <typename ValueType>
GKO_BIND_HIPSPARSE32_CSRSV2_BUFFERSIZE(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_HIPSPARSE64_CSRSV2_BUFFERSIZE(ValueType, detail::not_implemented);
#undef GKO_BIND_HIPSPARSE32_CSRSV2_BUFFERSIZE
#undef GKO_BIND_HIPSPARSE64_CSRSV2_BUFFERSIZE

#define GKO_BIND_HIPSPARSE32_CSRSV2_ANALYSIS(ValueType, HipsparseName)        \
    inline void csrsv2_analysis(                                              \
        hipsparseHandle_t handle, hipsparseOperation_t trans, size_type m,    \
        size_type nnz, const hipsparseMatDescr_t descr,                       \
        const ValueType *csrVal, const int32 *csrRowPtr,                      \
        const int32 *csrColInd, csrsv2Info_t factor_info,                     \
        hipsparseSolvePolicy_t policy, void *factor_work_vec)                 \
    {                                                                         \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(HipsparseName(                         \
            handle, trans, m, nnz, descr, as_hiplibs_type(csrVal), csrRowPtr, \
            csrColInd, factor_info, policy, factor_work_vec));                \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_HIPSPARSE64_CSRSV2_ANALYSIS(ValueType, HipsparseName)     \
    inline void csrsv2_analysis(                                           \
        hipsparseHandle_t handle, hipsparseOperation_t trans, size_type m, \
        size_type nnz, const hipsparseMatDescr_t descr,                    \
        const ValueType *csrVal, const int64 *csrRowPtr,                   \
        const int64 *csrColInd, csrsv2Info_t factor_info,                  \
        hipsparseSolvePolicy_t policy, void *factor_work_vec)              \
        GKO_NOT_IMPLEMENTED;                                               \
    static_assert(true,                                                    \
                  "This assert is used to counter the "                    \
                  "false positive extra "                                  \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE32_CSRSV2_ANALYSIS(float, hipsparseScsrsv2_analysis);
GKO_BIND_HIPSPARSE32_CSRSV2_ANALYSIS(double, hipsparseDcsrsv2_analysis);
GKO_BIND_HIPSPARSE64_CSRSV2_ANALYSIS(float, hipsparseScsrsv2_analysis);
GKO_BIND_HIPSPARSE64_CSRSV2_ANALYSIS(double, hipsparseDcsrsv2_analysis);
template <typename ValueType>
GKO_BIND_HIPSPARSE32_CSRSV2_ANALYSIS(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_HIPSPARSE64_CSRSV2_ANALYSIS(ValueType, detail::not_implemented);
#undef GKO_BIND_HIPSPARSE32_CSRSV2_ANALYSIS
#undef GKO_BIND_HIPSPARSE64_CSRSV2_ANALYSIS

#define GKO_BIND_HIPSPARSE32_CSRSV2_SOLVE(ValueType, HipsparseName)           \
    inline void csrsv2_solve(                                                 \
        hipsparseHandle_t handle, hipsparseOperation_t trans, size_type m,    \
        size_type nnz, const ValueType *one, const hipsparseMatDescr_t descr, \
        const ValueType *csrVal, const int32 *csrRowPtr,                      \
        const int32 *csrColInd, csrsv2Info_t factor_info,                     \
        const ValueType *rhs, ValueType *sol, hipsparseSolvePolicy_t policy,  \
        void *factor_work_vec)                                                \
    {                                                                         \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(                                       \
            HipsparseName(handle, trans, m, nnz, as_hiplibs_type(one), descr, \
                          as_hiplibs_type(csrVal), csrRowPtr, csrColInd,      \
                          factor_info, as_hiplibs_type(rhs),                  \
                          as_hiplibs_type(sol), policy, factor_work_vec));    \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_HIPSPARSE64_CSRSV2_SOLVE(ValueType, HipsparseName)           \
    inline void csrsv2_solve(                                                 \
        hipsparseHandle_t handle, hipsparseOperation_t trans, size_type m,    \
        size_type nnz, const ValueType *one, const hipsparseMatDescr_t descr, \
        const ValueType *csrVal, const int64 *csrRowPtr,                      \
        const int64 *csrColInd, csrsv2Info_t factor_info,                     \
        const ValueType *rhs, ValueType *sol, hipsparseSolvePolicy_t policy,  \
        void *factor_work_vec) GKO_NOT_IMPLEMENTED;                           \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE32_CSRSV2_SOLVE(float, hipsparseScsrsv2_solve);
GKO_BIND_HIPSPARSE32_CSRSV2_SOLVE(double, hipsparseDcsrsv2_solve);
GKO_BIND_HIPSPARSE64_CSRSV2_SOLVE(float, hipsparseScsrsv2_solve);
GKO_BIND_HIPSPARSE64_CSRSV2_SOLVE(double, hipsparseDcsrsv2_solve);
template <typename ValueType>
GKO_BIND_HIPSPARSE32_CSRSV2_SOLVE(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_HIPSPARSE64_CSRSV2_SOLVE(ValueType, detail::not_implemented);
#undef GKO_BIND_HIPSPARSE32_CSRSV2_SOLVE
#undef GKO_BIND_HIPSPARSE64_CSRSV2_SOLVE


inline hipsparseContext *init()
{
    hipsparseHandle_t handle{};
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseCreate(&handle));
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(
        hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    return reinterpret_cast<hipsparseContext *>(handle);
}


inline void destroy_hipsparse_handle(hipsparseContext *handle)
{
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(
        hipsparseDestroy(reinterpret_cast<hipsparseHandle_t>(handle)));
}


inline hipsparseMatDescr_t create_mat_descr()
{
    hipsparseMatDescr_t descr{};
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseCreateMatDescr(&descr));
    return descr;
}


inline void destroy(hipsparseMatDescr_t descr)
{
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseDestroyMatDescr(descr));
}


inline csrgemm2Info_t create_spgemm_info()
{
    csrgemm2Info_t info{};
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseCreateCsrgemm2Info(&info));
    return info;
}

// on HIP, hipsparseMatDescr_t and csrgemm2Info_t are both void*, so this would
// be a redefinition
#ifndef __HIP_PLATFORM_HCC__
inline void destroy(csrgemm2Info_t info)
{
    GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseDestroyCsrgemm2Info(info));
}
#endif


}  // namespace hipsparse
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_HIPSPARSE_BINDINGS_HIP_HPP_
