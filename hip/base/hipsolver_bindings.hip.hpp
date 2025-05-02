// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_HIPSOLVER_BINDINGS_HPP_
#define GKO_HIP_BASE_HIPSOLVER_BINDINGS_HPP_


#if HIP_VERSION >= 50200000
#include <hipsolver/hipsolver.h>
#else
#include <hipsolver.h>
#endif


#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/types.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The HIPSOLVER namespace.
 *
 * @ingroup hipsolver
 */
namespace hipsolver {
/**
 * @brief The detail namespace.
 *
 * @ingroup detail
 */
namespace detail {


template <typename... Args>
inline int64 not_implemented(Args...)
{
    return static_cast<int64>(HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}


}  // namespace detail


template <typename ValueType>
struct is_supported : std::false_type {};

template <>
struct is_supported<float> : std::true_type {};

template <>
struct is_supported<double> : std::true_type {};

template <>
struct is_supported<std::complex<float>> : std::true_type {};

template <>
struct is_supported<std::complex<double>> : std::true_type {};


#define GKO_BIND_HIPSOLVER_SYEVD_BUFFERSIZE(ValueType, HipsolverName)        \
    inline void syevd_buffersize(                                            \
        hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,                 \
        hipsolverFillMode_t uplo, int32 n, ValueType* a, int32 lda,          \
        remove_complex<ValueType>* w, int32* buffer_num_elems)               \
    {                                                                        \
        GKO_ASSERT_NO_HIPSOLVER_ERRORS(                                      \
            HipsolverName(handle, jobz, uplo, n, as_hiplibs_type(a), lda,    \
                          as_hiplibs_type(w), buffer_num_elems));            \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPSOLVER_SYEVD_BUFFERSIZE(float, hipsolverDnSsyevd_bufferSize);
GKO_BIND_HIPSOLVER_SYEVD_BUFFERSIZE(double, hipsolverDnDsyevd_bufferSize);
GKO_BIND_HIPSOLVER_SYEVD_BUFFERSIZE(std::complex<float>,
                                    hipsolverDnCheevd_bufferSize);
GKO_BIND_HIPSOLVER_SYEVD_BUFFERSIZE(std::complex<double>,
                                    hipsolverDnZheevd_bufferSize);
template <typename ValueType>
GKO_BIND_HIPSOLVER_SYEVD_BUFFERSIZE(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPSOLVER_SYEVD_BUFFERSIZE


#define GKO_BIND_HIPSOLVER_SYEVD(ValueType, HipsolverName)                   \
    inline void syevd(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz,   \
                      hipsolverFillMode_t uplo, int32 n, ValueType* a,       \
                      int32 lda, remove_complex<ValueType>* w,               \
                      ValueType* work, int32 buffer_num_elems,               \
                      int32* dev_info)                                       \
    {                                                                        \
        GKO_ASSERT_NO_HIPSOLVER_ERRORS(                                      \
            HipsolverName(handle, jobz, uplo, n, as_hiplibs_type(a), lda,    \
                          as_hiplibs_type(w), as_hiplibs_type(work),         \
                          buffer_num_elems, dev_info));                      \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPSOLVER_SYEVD(float, hipsolverDnSsyevd);
GKO_BIND_HIPSOLVER_SYEVD(double, hipsolverDnDsyevd);
GKO_BIND_HIPSOLVER_SYEVD(std::complex<float>, hipsolverDnCheevd);
GKO_BIND_HIPSOLVER_SYEVD(std::complex<double>, hipsolverDnZheevd);
template <typename ValueType>
GKO_BIND_HIPSOLVER_SYEVD(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPSOLVER_SYEVD


#define GKO_BIND_HIPSOLVER_SYGVD_BUFFERSIZE(ValueType, HipsolverName)        \
    inline void sygvd_buffersize(                                            \
        hipsolverDnHandle_t handle, hipsolverEigType_t itype,                \
        hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int32 n,          \
        ValueType* a, int32 lda, ValueType* b, int32 ldb,                    \
        remove_complex<ValueType>* w, int32* buffer_num_elems)               \
    {                                                                        \
        GKO_ASSERT_NO_HIPSOLVER_ERRORS(HipsolverName(                        \
            handle, itype, jobz, uplo, n, as_hiplibs_type(a), lda,           \
            as_hiplibs_type(b), ldb, as_hiplibs_type(w), buffer_num_elems)); \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPSOLVER_SYGVD_BUFFERSIZE(float, hipsolverDnSsygvd_bufferSize);
GKO_BIND_HIPSOLVER_SYGVD_BUFFERSIZE(double, hipsolverDnDsygvd_bufferSize);
GKO_BIND_HIPSOLVER_SYGVD_BUFFERSIZE(std::complex<float>,
                                    hipsolverDnChegvd_bufferSize);
GKO_BIND_HIPSOLVER_SYGVD_BUFFERSIZE(std::complex<double>,
                                    hipsolverDnZhegvd_bufferSize);
template <typename ValueType>
GKO_BIND_HIPSOLVER_SYGVD_BUFFERSIZE(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPSOLVER_SYGVD_BUFFERSIZE


#define GKO_BIND_HIPSOLVER_SYGVD(ValueType, HipsolverName)                     \
    inline void sygvd(hipsolverDnHandle_t handle, hipsolverEigType_t itype,    \
                      hipsolverEigMode_t jobz, hipsolverFillMode_t uplo,       \
                      int32 n, ValueType* a, int32 lda, ValueType* b,          \
                      int32 ldb, remove_complex<ValueType>* w,                 \
                      ValueType* work, int32 buffer_num_elems,                 \
                      int32* dev_info)                                         \
    {                                                                          \
        GKO_ASSERT_NO_HIPSOLVER_ERRORS(                                        \
            HipsolverName(handle, itype, jobz, uplo, n, as_hiplibs_type(a),    \
                          lda, as_hiplibs_type(b), ldb, as_hiplibs_type(w),    \
                          as_hiplibs_type(work), buffer_num_elems, dev_info)); \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPSOLVER_SYGVD(float, hipsolverDnSsygvd);
GKO_BIND_HIPSOLVER_SYGVD(double, hipsolverDnDsygvd);
GKO_BIND_HIPSOLVER_SYGVD(std::complex<float>, hipsolverDnChegvd);
GKO_BIND_HIPSOLVER_SYGVD(std::complex<double>, hipsolverDnZhegvd);
template <typename ValueType>
GKO_BIND_HIPSOLVER_SYGVD(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPSOLVER_SYGVD


#define GKO_BIND_HIPSOLVER_POTRF_BUFFERSIZE(ValueType, HipsolverName)        \
    inline void potrf_buffersize(                                            \
        hipsolverDnHandle_t handle, hipsolverFillMode_t uplo, int32 n,       \
        ValueType* a, int32 lda, int32* buffer_num_elems)                    \
    {                                                                        \
        GKO_ASSERT_NO_HIPSOLVER_ERRORS(HipsolverName(                        \
            handle, uplo, n, as_hiplibs_type(a), lda, buffer_num_elems));    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPSOLVER_POTRF_BUFFERSIZE(float, hipsolverDnSpotrf_bufferSize);
GKO_BIND_HIPSOLVER_POTRF_BUFFERSIZE(double, hipsolverDnDpotrf_bufferSize);
GKO_BIND_HIPSOLVER_POTRF_BUFFERSIZE(std::complex<float>,
                                    hipsolverDnCpotrf_bufferSize);
GKO_BIND_HIPSOLVER_POTRF_BUFFERSIZE(std::complex<double>,
                                    hipsolverDnZpotrf_bufferSize);
template <typename ValueType>
GKO_BIND_HIPSOLVER_POTRF_BUFFERSIZE(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPSOLVER_POTRF_BUFFERSIZE


#define GKO_BIND_HIPSOLVER_POTRF(ValueType, HipsolverName)                     \
    inline void potrf(hipsolverDnHandle_t handle, hipsolverFillMode_t uplo,    \
                      int32 n, ValueType* a, int32 lda, ValueType* work,       \
                      int32 buffer_num_elems, int32* dev_info)                 \
    {                                                                          \
        GKO_ASSERT_NO_HIPSOLVER_ERRORS(                                        \
            HipsolverName(handle, uplo, n, as_hiplibs_type(a), lda,            \
                          as_hiplibs_type(work), buffer_num_elems, dev_info)); \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_HIPSOLVER_POTRF(float, hipsolverDnSpotrf);
GKO_BIND_HIPSOLVER_POTRF(double, hipsolverDnDpotrf);
GKO_BIND_HIPSOLVER_POTRF(std::complex<float>, hipsolverDnCpotrf);
GKO_BIND_HIPSOLVER_POTRF(std::complex<double>, hipsolverDnZpotrf);
template <typename ValueType>
GKO_BIND_HIPSOLVER_POTRF(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPSOLVER_POTRF


#define GKO_BIND_HIPSOLVER_POTRS(ValueType, HipsolverName)                   \
    inline void potrs(hipsolverDnHandle_t handle, hipsolverFillMode_t uplo,  \
                      int32 n, int32 nrhs, ValueType* a, int32 lda,          \
                      ValueType* b, int32 ldb, int32* dev_info)              \
    {                                                                        \
        GKO_ASSERT_NO_HIPSOLVER_ERRORS(                                      \
            HipsolverName(handle, uplo, n, nrhs, as_hiplibs_type(a), lda,    \
                          as_hiplibs_type(b), ldb, dev_info));               \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HIPSOLVER_POTRS(float, hipsolverDnSpotrs);
GKO_BIND_HIPSOLVER_POTRS(double, hipsolverDnDpotrs);
GKO_BIND_HIPSOLVER_POTRS(std::complex<float>, hipsolverDnCpotrs);
GKO_BIND_HIPSOLVER_POTRS(std::complex<double>, hipsolverDnZpotrs);
template <typename ValueType>
GKO_BIND_HIPSOLVER_POTRS(ValueType, detail::not_implemented);

#undef GKO_BIND_HIPSOLVER_POTRS


}  // namespace hipsolver


namespace dev_lapack {


using namespace hipsolver;


#define LAPACK_EIG_TYPE_1 HIPSOLVER_EIG_TYPE_1
#define LAPACK_EIG_TYPE_2 HIPSOLVER_EIG_TYPE_2
#define LAPACK_EIG_TYPE_3 HIPSOLVER_EIG_TYPE_3

#define LAPACK_EIG_VECTOR HIPSOLVER_EIG_MODE_VECTOR
#define LAPACK_EIG_NOVECTOR HIPSOLVER_EIG_MODE_NOVECTOR

#define LAPACK_FILL_UPPER HIPSOLVER_FILL_MODE_UPPER
#define LAPACK_FILL_LOWER HIPSOLVER_FILL_MODE_LOWER


}  // namespace dev_lapack
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_HIPSOLVER_BINDINGS_HPP_
