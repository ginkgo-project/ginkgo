// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_HIPSPARSE_BLOCK_BINDINGS_HIP_HPP_
#define GKO_HIP_BASE_HIPSPARSE_BLOCK_BINDINGS_HIP_HPP_


#include <hip/hip_runtime.h>
#if HIP_VERSION >= 50200000
#include <hipsparse/hipsparse.h>
#else
#include <hipsparse.h>
#endif


#include <ginkgo/core/base/exception_helpers.hpp>


#include "hip/base/hipsparse_bindings.hip.hpp"
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


/// Default storage layout within each small dense block
constexpr hipsparseDirection_t blockDir = HIPSPARSE_DIRECTION_COLUMN;


#define GKO_BIND_HIPSPARSE32_BSRMV(ValueType, HipsparseName)                   \
    inline void bsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA,   \
                      int32 mb, int32 nb, int32 nnzb, const ValueType* alpha,  \
                      const hipsparseMatDescr_t descrA, const ValueType* valA, \
                      const int32* rowPtrA, const int32* colIndA,              \
                      int block_size, const ValueType* x,                      \
                      const ValueType* beta, ValueType* y)                     \
    {                                                                          \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(HipsparseName(                          \
            handle, blockDir, transA, mb, nb, nnzb, as_hiplibs_type(alpha),    \
            descrA, as_hiplibs_type(valA), rowPtrA, colIndA, block_size,       \
            as_hiplibs_type(x), as_hiplibs_type(beta), as_hiplibs_type(y)));   \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_BIND_HIPSPARSE64_BSRMV(ValueType, HipsparseName)                   \
    inline void bsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA,   \
                      int64 mb, int64 nb, int64 nnzb, const ValueType* alpha,  \
                      const hipsparseMatDescr_t descrA, const ValueType* valA, \
                      const int64* rowPtrA, const int64* colIndA,              \
                      int block_size, const ValueType* x,                      \
                      const ValueType* beta, ValueType* y)                     \
        GKO_NOT_IMPLEMENTED;                                                   \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE32_BSRMV(float, hipsparseSbsrmv);
GKO_BIND_HIPSPARSE32_BSRMV(double, hipsparseDbsrmv);
GKO_BIND_HIPSPARSE32_BSRMV(std::complex<float>, hipsparseCbsrmv);
GKO_BIND_HIPSPARSE32_BSRMV(std::complex<double>, hipsparseZbsrmv);
GKO_BIND_HIPSPARSE64_BSRMV(float, hipsparseSbsrmv);
GKO_BIND_HIPSPARSE64_BSRMV(double, hipsparseDbsrmv);
GKO_BIND_HIPSPARSE64_BSRMV(std::complex<float>, hipsparseCbsrmv);
GKO_BIND_HIPSPARSE64_BSRMV(std::complex<double>, hipsparseZbsrmv);
template <typename ValueType>
GKO_BIND_HIPSPARSE32_BSRMV(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_HIPSPARSE64_BSRMV(ValueType, detail::not_implemented);


#undef GKO_BIND_HIPSPARSE32_BSRMV
#undef GKO_BIND_HIPSPARSE64_BSRMV


#define GKO_BIND_HIPSPARSE32_BSRMM(ValueType, HipsparseName)                   \
    inline void bsrmm(hipsparseHandle_t handle, hipsparseOperation_t transA,   \
                      hipsparseOperation_t transB, int32 mb, int32 n,          \
                      int32 kb, int32 nnzb, const ValueType* alpha,            \
                      const hipsparseMatDescr_t descrA, const ValueType* valA, \
                      const int32* rowPtrA, const int32* colIndA,              \
                      int block_size, const ValueType* B, int32 ldb,           \
                      const ValueType* beta, ValueType* C, int32 ldc)          \
    {                                                                          \
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(HipsparseName(                          \
            handle, blockDir, transA, transB, mb, n, kb, nnzb,                 \
            as_hiplibs_type(alpha), descrA, as_hiplibs_type(valA), rowPtrA,    \
            colIndA, block_size, as_hiplibs_type(B), ldb,                      \
            as_hiplibs_type(beta), as_hiplibs_type(C), ldc));                  \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

#define GKO_BIND_HIPSPARSE64_BSRMM(ValueType, HipsparseName)                  \
    inline void bsrmm(                                                        \
        hipsparseHandle_t handle, hipsparseOperation_t transA,                \
        hipsparseOperation_t transB, int64 mb, int64 n, int64 kb, int64 nnzb, \
        const ValueType* alpha, const hipsparseMatDescr_t descrA,             \
        const ValueType* valA, const int64* rowPtrA, const int64* colIndA,    \
        int block_size, const ValueType* B, int64 ldb, const ValueType* beta, \
        ValueType* C, int64 ldc) GKO_NOT_IMPLEMENTED;                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_HIPSPARSE32_BSRMM(float, hipsparseSbsrmm);
GKO_BIND_HIPSPARSE32_BSRMM(double, hipsparseDbsrmm);
GKO_BIND_HIPSPARSE32_BSRMM(std::complex<float>, hipsparseCbsrmm);
GKO_BIND_HIPSPARSE32_BSRMM(std::complex<double>, hipsparseZbsrmm);
GKO_BIND_HIPSPARSE64_BSRMM(float, hipsparseSbsrmm);
GKO_BIND_HIPSPARSE64_BSRMM(double, hipsparseDbsrmm);
GKO_BIND_HIPSPARSE64_BSRMM(std::complex<float>, hipsparseCbsrmm);
GKO_BIND_HIPSPARSE64_BSRMM(std::complex<double>, hipsparseZbsrmm);
template <typename ValueType>
GKO_BIND_HIPSPARSE32_BSRMM(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_HIPSPARSE64_BSRMM(ValueType, detail::not_implemented);


#undef GKO_BIND_HIPSPARSE32_BSRMM
#undef GKO_BIND_HIPSPARSE64_BSRMM


}  // namespace hipsparse
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_HIPSPARSE_BLOCK_BINDINGS_HIP_HPP_
