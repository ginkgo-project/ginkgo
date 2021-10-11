/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_CUDA_BASE_CUSOLVER_BINDINGS_HPP_
#define GKO_CUDA_BASE_CUSOLVER_BINDINGS_HPP_


#include <cuda.h>
#include <cusolverSp.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The CuSolver namespace.
 *
 * @ingroup cusolver
 */
namespace cusolver {
/**
 * @brief The detail namespace.
 *
 * @ingroup detail
 */
namespace detail {


template <typename... Args>
inline int64 not_implemented(Args...)
{
    return static_cast<int64>(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
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


inline cusolverSpHandle_t init_sp()
{
    cusolverSpHandle_t handle{};
    GKO_ASSERT_NO_CUSOLVER_ERRORS(cusolverSpCreate(&handle));
    return handle;
}


inline void destroy(cusolverSpHandle_t handle)
{
    GKO_ASSERT_NO_CUSOLVER_ERRORS(cusolverSpDestroy(handle));
}


inline csrqrInfo_t create_csrqr_info()
{
    csrqrInfo_t info = NULL;
    GKO_ASSERT_NO_CUSOLVER_ERRORS(cusolverSpCreateCsrqrInfo(&info));
    return info;
}


inline void destroy(csrqrInfo_t info)
{
    GKO_ASSERT_NO_CUSOLVER_ERRORS(cusolverSpDestroyCsrqrInfo(info));
}


inline void csrqr_batched_analysis(cusolverSpHandle_t handle, int m, int n,
                                   int nnzA, const cusparseMatDescr_t descrA,
                                   const int* csrRowPtrA, const int* csrColIndA,
                                   csrqrInfo_t info)
{
    GKO_ASSERT_NO_CUSOLVER_ERRORS(cusolverSpXcsrqrAnalysisBatched(
        handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info));
}


inline void csrqr_batched_analysis(cusolverSpHandle_t handle, int64 m, int64 n,
                                   int64 nnzA, const cusparseMatDescr_t descrA,
                                   const int64* csrRowPtrA,
                                   const int64* csrColIndA,
                                   csrqrInfo_t info) GKO_NOT_IMPLEMENTED;


#define GKO_BIND_CUSOLVER32_CSRQR_BATCHED_BUFFERINFO(ValueType, CusparseName) \
    inline void csrqr_batched_buffer_info(                                    \
        cusolverSpHandle_t handle, size_type m, size_type n, size_type nnz,   \
        const cusparseMatDescr_t descr, const ValueType* csrVal,              \
        const int32* csrRowPtr, const int32* csrColInd, size_type batch_size, \
        csrqrInfo_t info, size_type* const internal_data_bytes,               \
        size_type* const workspace_bytes)                                     \
    {                                                                         \
        GKO_ASSERT_NO_CUSOLVER_ERRORS(                                        \
            CusparseName(handle, m, n, nnz, descr, as_culibs_type(csrVal),    \
                         csrRowPtr, csrColInd, batch_size, info,              \
                         internal_data_bytes, workspace_bytes));              \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSOLVER64_CSRQR_BATCHED_BUFFERINFO(ValueType)               \
    inline void csrqr_batched_buffer_info(                                    \
        cusolverSpHandle_t handle, size_type m, size_type n, size_type nnz,   \
        const cusparseMatDescr_t descr, const ValueType* csrVal,              \
        const int64* csrRowPtr, const int64* csrColInd, size_type batch_size, \
        csrqrInfo_t info, size_type* const internal_data_bytes,               \
        size_type* const factor_work_size) GKO_NOT_IMPLEMENTED;               \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSOLVER32_CSRQR_BATCHED_BUFFERINFO(float,
                                             cusolverSpScsrqrBufferInfoBatched);
GKO_BIND_CUSOLVER32_CSRQR_BATCHED_BUFFERINFO(double,
                                             cusolverSpDcsrqrBufferInfoBatched);
GKO_BIND_CUSOLVER32_CSRQR_BATCHED_BUFFERINFO(std::complex<float>,
                                             cusolverSpCcsrqrBufferInfoBatched);
GKO_BIND_CUSOLVER32_CSRQR_BATCHED_BUFFERINFO(std::complex<double>,
                                             cusolverSpZcsrqrBufferInfoBatched);
GKO_BIND_CUSOLVER64_CSRQR_BATCHED_BUFFERINFO(float);
GKO_BIND_CUSOLVER64_CSRQR_BATCHED_BUFFERINFO(double);
GKO_BIND_CUSOLVER64_CSRQR_BATCHED_BUFFERINFO(std::complex<float>);
GKO_BIND_CUSOLVER64_CSRQR_BATCHED_BUFFERINFO(std::complex<double>);
template <typename ValueType>
GKO_BIND_CUSOLVER32_CSRQR_BATCHED_BUFFERINFO(ValueType,
                                             detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSOLVER64_CSRQR_BATCHED_BUFFERINFO(ValueType);
#undef GKO_BIND_CUSOLVERE32_CSRQR_BATCHED_BUFFERINFO
#undef GKO_BIND_CUSOLVERE32_CSRQR_BATCHED_BUFFERINFO


#define GKO_BIND_CUSOLVER32_CSRQR_BATCHED_SOLVE(ValueType, CusparseName)      \
    inline void csrqr_batched_solve(                                          \
        cusolverSpHandle_t handle, size_type m, size_type n, size_type nnz,   \
        const cusparseMatDescr_t descr, const ValueType* csrVal,              \
        const int32* csrRowPtr, const int32* csrColInd, const ValueType* rhs, \
        ValueType* x, size_type batch_size, csrqrInfo_t info, void* work_vec) \
    {                                                                         \
        GKO_ASSERT_NO_CUSOLVER_ERRORS(                                        \
            CusparseName(handle, m, n, nnz, descr, as_culibs_type(csrVal),    \
                         csrRowPtr, csrColInd, as_culibs_type(rhs),           \
                         as_culibs_type(x), batch_size, info, work_vec));     \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSOLVER64_CSRQR_BATCHED_SOLVE(ValueType, CusparseName)      \
    inline void csrqr_batched_solve(                                          \
        cusolverSpHandle_t handle, size_type m, size_type n, size_type nnz,   \
        const cusparseMatDescr_t descr, const ValueType* csrVal,              \
        const int64* csrRowPtr, const int64* csrColInd, const ValueType* rhs, \
        ValueType* x, size_type batch_size, csrqrInfo_t info, void* work_vec) \
        GKO_NOT_IMPLEMENTED;                                                  \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSOLVER32_CSRQR_BATCHED_SOLVE(float, cusolverSpScsrqrsvBatched);
GKO_BIND_CUSOLVER32_CSRQR_BATCHED_SOLVE(double, cusolverSpDcsrqrsvBatched);
GKO_BIND_CUSOLVER32_CSRQR_BATCHED_SOLVE(std::complex<float>,
                                        cusolverSpCcsrqrsvBatched);
GKO_BIND_CUSOLVER32_CSRQR_BATCHED_SOLVE(std::complex<double>,
                                        cusolverSpZcsrqrsvBatched);
GKO_BIND_CUSOLVER64_CSRQR_BATCHED_SOLVE(float, cusolverSpScsrqrsvBatched);
GKO_BIND_CUSOLVER64_CSRQR_BATCHED_SOLVE(double, cusolverSpDcsrqrsvBatched);
GKO_BIND_CUSOLVER64_CSRQR_BATCHED_SOLVE(std::complex<float>,
                                        cusolverSpCcsrqrsvBatched);
GKO_BIND_CUSOLVER64_CSRQR_BATCHED_SOLVE(std::complex<double>,
                                        cusolverSpZcsrqrsvBatched);
template <typename ValueType>
GKO_BIND_CUSOLVER32_CSRQR_BATCHED_SOLVE(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSOLVER64_CSRQR_BATCHED_SOLVE(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSOLVER32_CSRQR_BATCHED_SOLVE
#undef GKO_BIND_CUSOLVER64_CSRQR_BATCHED_SOLVE


}  // namespace cusolver
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CUSPARSE_BINDINGS_HPP_
