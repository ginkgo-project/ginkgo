// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_CUSOLVER_BINDINGS_HPP_
#define GKO_CUDA_BASE_CUSOLVER_BINDINGS_HPP_


#include <cuda.h>
#include <cusolverDn.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The CUSOLVER namespace.
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


#define GKO_BIND_CUSOLVER_SYGVD_BUFFERSIZE(ValueType, CusolverName)          \
    inline void sygvd_buffersize(                                            \
        cusolverDnHandle_t handle, cusolverEigType_t itype,                  \
        cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, ValueType* a,  \
        int lda, ValueType* b, int ldb, remove_complex<ValueType>* w,        \
        int* buffer_num_elems)                                               \
    {                                                                        \
        GKO_ASSERT_NO_CUSOLVER_ERRORS(CusolverName(                          \
            handle, itype, jobz, uplo, n, as_culibs_type(a), lda,            \
            as_culibs_type(b), ldb, as_culibs_type(w), buffer_num_elems));   \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSOLVER_SYGVD_BUFFERSIZE(float, cusolverDnSsygvd_bufferSize);
GKO_BIND_CUSOLVER_SYGVD_BUFFERSIZE(double, cusolverDnDsygvd_bufferSize);
GKO_BIND_CUSOLVER_SYGVD_BUFFERSIZE(std::complex<float>,
                                   cusolverDnChegvd_bufferSize);
GKO_BIND_CUSOLVER_SYGVD_BUFFERSIZE(std::complex<double>,
                                   cusolverDnZhegvd_bufferSize);
template <typename ValueType>
GKO_BIND_CUSOLVER_SYGVD_BUFFERSIZE(ValueType, detail::not_implemented);

#undef GKO_BIND_CUSOLVER_SYGVD_BUFFERSIZE


#define GKO_BIND_CUSOLVER_SYGVD(ValueType, CusolverName)                       \
    inline void sygvd(cusolverDnHandle_t handle, cusolverEigType_t itype,      \
                      cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,    \
                      ValueType* a, int lda, ValueType* b, int ldb,            \
                      remove_complex<ValueType>* w, ValueType* work,           \
                      int buffer_num_elems, int* dev_info)                     \
    {                                                                          \
        GKO_ASSERT_NO_CUSOLVER_ERRORS(                                         \
            CusolverName(handle, itype, jobz, uplo, n, as_culibs_type(a), lda, \
                         as_culibs_type(b), ldb, as_culibs_type(w),            \
                         as_culibs_type(work), buffer_num_elems, dev_info));   \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSOLVER_SYGVD(float, cusolverDnSsygvd);
GKO_BIND_CUSOLVER_SYGVD(double, cusolverDnDsygvd);
GKO_BIND_CUSOLVER_SYGVD(std::complex<float>, cusolverDnChegvd);
GKO_BIND_CUSOLVER_SYGVD(std::complex<double>, cusolverDnZhegvd);
template <typename ValueType>
GKO_BIND_CUSOLVER_SYGVD(ValueType, detail::not_implemented);

#undef GKO_BIND_CUSOLVER_SYGVD


}  // namespace cusolver


namespace dev_lapack {


using namespace cusolver;


#define LAPACK_EIG_TYPE_1 CUSOLVER_EIG_TYPE_1
#define LAPACK_EIG_TYPE_2 CUSOLVER_EIG_TYPE_2
#define LAPACK_EIG_TYPE_3 CUSOLVER_EIG_TYPE_3

#define LAPACK_EIG_VECTOR CUSOLVER_EIG_MODE_VECTOR
#define LAPACK_EIG_NOVECTOR CUSOLVER_EIG_MODE_NOVECTOR

#define LAPACK_FILL_UPPER CUBLAS_FILL_MODE_UPPER
#define LAPACK_FILL_LOWER CUBLAS_FILL_MODE_LOWER


}  // namespace dev_lapack
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CUSOLVER_BINDINGS_HPP_
