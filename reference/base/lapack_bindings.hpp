// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_BASE_LAPACK_BINDINGS_HPP_
#define GKO_REFERENCE_BASE_LAPACK_BINDINGS_HPP_

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>

#if GKO_HAVE_LAPACK


extern "C" {
void ssygvd(const int* itype, const char* jobz, const char* uplo, const int* n,
            float* A, const int* lda, float* B, const int* ldb, float* w,
            float* work, const int* lwork, int* iwork, const int* liwork,
            int* info);

void dsygvd(const int* itype, const char* jobz, const char* uplo, const int* n,
            double* A, const int* lda, double* B, const int* ldb, double* w,
            double* work, const int* lwork, int* iwork, const int* liwork,
            int* info);

void chegvd(const int* itype, const char* jobz, const char* uplo, const int* n,
            std::complex<float>* A, const int* lda, std::complex<float>* B,
            const int* ldb, float* w, std::complex<float>* work, int* lwork,
            float* rwork, int* lrwork, int* iwork, int* liwork, int* info);

void zhegvd(const int* itype, const char* jobz, const char* uplo, const int* n,
            std::complex<double>* A, const int* lda, std::complex<double>* B,
            const int* ldb, double* w, std::complex<double>* work, int* lwork,
            double* rwork, int* lrwork, int* iwork, int* liwork, int* info);
}

namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The LAPACK namespace.
 *
 * @ingroup lapack
 */
namespace lapack {


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


#define GKO_BIND_SYGVD_BUFFERSIZES(ValueType, LapackName)                    \
    inline void sygvd_buffersizes(                                           \
        const int* itype, const char* jobz, const char* uplo, const int* n,  \
        ValueType* a, const int* lda, ValueType* b, const int* ldb,          \
        ValueType* w, ValueType* work, int* fp_buffer_num_elems, int* iwork, \
        int* int_buffer_num_elems)                                           \
    {                                                                        \
        int info;                                                            \
        *fp_buffer_num_elems = -1;                                           \
        *int_buffer_num_elems = -1;                                          \
        GKO_ASSERT_NO_LAPACK_ERRORS(                                         \
            LapackName(itype, jobz, uplo, n, a, lda, b, ldb, w, work,        \
                       fp_buffer_num_elems, iwork, int_buffer_num_elems,     \
                       &info),                                               \
            info);                                                           \
        *fp_buffer_num_elems = static_cast<int>(work[0]);                    \
        *int_buffer_num_elems = iwork[0];                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_SYGVD_BUFFERSIZES(float, ssygvd);
GKO_BIND_SYGVD_BUFFERSIZES(double, dsygvd);
template <typename ValueType>
inline void sygvd_buffersizes(const int* itype, const char* jobz,
                              const char* uplo, const int* n, ValueType* a,
                              const int* lda, ValueType* b, const int* ldb,
                              ValueType* w, ValueType* work,
                              int* fp_buffer_num_elems, int* iwork,
                              int* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_SYGVD_BUFFERSIZES


#define GKO_BIND_SYGVD(ValueType, LapackName)                                \
    inline void sygvd(const int* itype, const char* jobz, const char* uplo,  \
                      const int* n, ValueType* a, const int* lda,            \
                      ValueType* b, const int* ldb, ValueType* w,            \
                      ValueType* work, int* fp_buffer_num_elems, int* iwork, \
                      int* int_buffer_num_elems)                             \
    {                                                                        \
        int info;                                                            \
        GKO_ASSERT_NO_LAPACK_ERRORS(                                         \
            LapackName(itype, jobz, uplo, n, a, lda, b, ldb, w, work,        \
                       fp_buffer_num_elems, iwork, int_buffer_num_elems,     \
                       &info),                                               \
            info);                                                           \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_SYGVD(float, ssygvd);
GKO_BIND_SYGVD(double, dsygvd);
template <typename ValueType>
inline void sygvd(const int* itype, const char* jobz, const char* uplo,
                  const int* n, ValueType* a, const int* lda, ValueType* b,
                  const int* ldb, ValueType* w, ValueType* work,
                  int* fp_buffer_num_elems, int* iwork,
                  int* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_SYGVD


#define GKO_BIND_HEGVD_BUFFERSIZES(ValueType, LapackName)                    \
    inline void hegvd_buffersizes(                                           \
        const int* itype, const char* jobz, const char* uplo, const int* n,  \
        ValueType* a, const int* lda, ValueType* b, const int* ldb,          \
        gko::remove_complex<ValueType>* w, ValueType* work,                  \
        int* fp_buffer_num_elems, gko::remove_complex<ValueType>* rwork,     \
        int* rfp_buffer_num_elems, int* iwork, int* int_buffer_num_elems)    \
    {                                                                        \
        int info;                                                            \
        *fp_buffer_num_elems = -1;                                           \
        *rfp_buffer_num_elems = -1;                                          \
        *int_buffer_num_elems = -1;                                          \
        GKO_ASSERT_NO_LAPACK_ERRORS(                                         \
            LapackName(itype, jobz, uplo, n, a, lda, b, ldb, w, work,        \
                       fp_buffer_num_elems, rwork, rfp_buffer_num_elems,     \
                       iwork, int_buffer_num_elems, &info),                  \
            info);                                                           \
        *fp_buffer_num_elems = static_cast<int>(work[0].real());             \
        *rfp_buffer_num_elems = static_cast<int>(rwork[0]);                  \
        *int_buffer_num_elems = iwork[0];                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HEGVD_BUFFERSIZES(std::complex<float>, chegvd);
GKO_BIND_HEGVD_BUFFERSIZES(std::complex<double>, zhegvd);
template <typename ValueType>
inline void hegvd_buffersizes(const int* itype, const char* jobz,
                              const char* uplo, const int* n, ValueType* a,
                              const int* lda, ValueType* b, const int* ldb,
                              gko::remove_complex<ValueType>* w,
                              ValueType* work, int* fp_buffer_num_elems,
                              gko::remove_complex<ValueType>* rwork,
                              int* rfp_buffer_num_elems, int* iwork,
                              int* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_HEGVD_BUFFERSIZES


#define GKO_BIND_HEGVD(ValueType, LapackName)                                \
    inline void hegvd(                                                       \
        const int* itype, const char* jobz, const char* uplo, const int* n,  \
        ValueType* a, const int* lda, ValueType* b, const int* ldb,          \
        remove_complex<ValueType>* w, ValueType* work,                       \
        int* fp_buffer_num_elems, remove_complex<ValueType>* rwork,          \
        int* rfp_buffer_num_elems, int* iwork, int* int_buffer_num_elems)    \
    {                                                                        \
        int info;                                                            \
        GKO_ASSERT_NO_LAPACK_ERRORS(                                         \
            LapackName(itype, jobz, uplo, n, a, lda, b, ldb, w, work,        \
                       fp_buffer_num_elems, rwork, rfp_buffer_num_elems,     \
                       iwork, int_buffer_num_elems, &info),                  \
            info);                                                           \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_HEGVD(std::complex<float>, chegvd);
GKO_BIND_HEGVD(std::complex<double>, zhegvd);
template <typename ValueType>
inline void hegvd(const int* itype, const char* jobz, const char* uplo,
                  const int* n, ValueType* a, const int* lda, ValueType* b,
                  const int* ldb, remove_complex<ValueType>* w, ValueType* work,
                  int* fp_buffer_num_elems, remove_complex<ValueType>* rwork,
                  int* rfp_buffer_num_elems, int* iwork,
                  int* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_HEGVD


#define LAPACK_EIG_VECTOR 'V'
#define LAPACK_EIG_NOVECTOR 'N'

#define LAPACK_FILL_UPPER 'U'
#define LAPACK_FILL_LOWER 'L'


}  // namespace lapack
}  // namespace reference
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HAVE_LAPACK

#endif  // GKO_REFERENCE_BASE_LAPACK_BINDINGS_HPP_
