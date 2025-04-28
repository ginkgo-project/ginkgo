// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_BASE_BLAS_BINDINGS_HPP_
#define GKO_REFERENCE_BASE_BLAS_BINDINGS_HPP_

#include <ginkgo/core/base/types.hpp>


#if GKO_HAVE_LAPACK


extern "C" {


// Triangular matrix-matrix multiplication
void strmm(const char* side, const char* uplo, const char* transa,
           const char* diag, const std::int32_t* m, const std::int32_t* n,
           const float* alpha, const float* A, const std::int32_t* lda,
           float* B, const std::int32_t* ldb);

void dtrmm(const char* side, const char* uplo, const char* transa,
           const char* diag, const std::int32_t* m, const std::int32_t* n,
           const double* alpha, const double* A, const std::int32_t* lda,
           double* B, const std::int32_t* ldb);

void ctrmm(const char* side, const char* uplo, const char* transa,
           const char* diag, const std::int32_t* m, const std::int32_t* n,
           const std::complex<float>* alpha, const std::complex<float>* A,
           const std::int32_t* lda, std::complex<float>* B,
           const std::int32_t* ldb);

void ztrmm(const char* side, const char* uplo, const char* transa,
           const char* diag, const std::int32_t* m, const std::int32_t* n,
           const std::complex<double>* alpha, const std::complex<double>* A,
           const std::int32_t* lda, std::complex<double>* B,
           const std::int32_t* ldb);
}


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The BLAS namespace.
 *
 * @ingroup lapack
 */
namespace blas {


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


#define GKO_BIND_TRMM(ValueType, BlasName)                                   \
    inline void trmm(const char* side, const char* uplo, const char* transa, \
                     const char* diag, const int32* m, const int32* n,       \
                     const ValueType* alpha, const ValueType* a,             \
                     const int32* lda, ValueType* b, const int32* ldb)       \
    {                                                                        \
        BlasName(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);     \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_TRMM(float, strmm);
GKO_BIND_TRMM(double, dtrmm);
GKO_BIND_TRMM(std::complex<float>, ctrmm);
GKO_BIND_TRMM(std::complex<double>, ztrmm);
template <typename ValueType>
inline void trmm(const char* side, const char* uplo, const char* transa,
                 const char* diag, const int32* m, const int32* n,
                 const ValueType* alpha, const ValueType* a, const int32* lda,
                 ValueType* b, const int32* ldb) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_TRMM


#define BLAS_OP_N 'N'
#define BLAS_OP_T 'T'
#define BLAS_OP_C 'C'

#define BLAS_SIDE_LEFT 'L'
#define BLAS_SIDE_RIGHT 'R'


}  // namespace blas
}  // namespace reference
}  // namespace kernels
}  // namespace gko

#endif  // GKO_HAVE_LAPACK

#endif  // GKO_REFERENCE_BASE_BLAS_BINDINGS_HPP_
