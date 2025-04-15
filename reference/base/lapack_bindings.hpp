// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_BASE_LAPACK_BINDINGS_HPP_
#define GKO_REFERENCE_BASE_LAPACK_BINDINGS_HPP_

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


#if GKO_HAVE_LAPACK


extern "C" {


// Symmetric eigenvalue problem
void ssyevd(const char* jobz, const char* uplo, const std::int32_t* n, float* A,
            const std::int32_t* lda, float* w, float* work, std::int32_t* lwork,
            std::int32_t* iwork, std::int32_t* liwork, std::int32_t* info);

void dsyevd(const char* jobz, const char* uplo, const std::int32_t* n,
            double* A, const std::int32_t* lda, double* w, double* work,
            std::int32_t* lwork, std::int32_t* iwork, std::int32_t* liwork,
            std::int32_t* info);

void cheevd(const char* jobz, const char* uplo, const std::int32_t* n,
            std::complex<float>* A, const std::int32_t* lda, float* w,
            std::complex<float>* work, std::int32_t* lwork, float* rwork,
            std::int32_t* lrwork, std::int32_t* iwork, std::int32_t* liwork,
            std::int32_t* info);

void zheevd(const char* jobz, const char* uplo, const std::int32_t* n,
            std::complex<double>* A, const std::int32_t* lda, double* w,
            std::complex<double>* work, std::int32_t* lwork, double* rwork,
            std::int32_t* lrwork, std::int32_t* iwork, std::int32_t* liwork,
            std::int32_t* info);


// Symmetric generalized eigenvalue problem
void ssygvd(const std::int32_t* itype, const char* jobz, const char* uplo,
            const std::int32_t* n, float* A, const std::int32_t* lda, float* B,
            const std::int32_t* ldb, float* w, float* work, std::int32_t* lwork,
            std::int32_t* iwork, std::int32_t* liwork, std::int32_t* info);

void dsygvd(const std::int32_t* itype, const char* jobz, const char* uplo,
            const std::int32_t* n, double* A, const std::int32_t* lda,
            double* B, const std::int32_t* ldb, double* w, double* work,
            std::int32_t* lwork, std::int32_t* iwork, std::int32_t* liwork,
            std::int32_t* info);

void chegvd(const std::int32_t* itype, const char* jobz, const char* uplo,
            const std::int32_t* n, std::complex<float>* A,
            const std::int32_t* lda, std::complex<float>* B,
            const std::int32_t* ldb, float* w, std::complex<float>* work,
            std::int32_t* lwork, float* rwork, std::int32_t* lrwork,
            std::int32_t* iwork, std::int32_t* liwork, std::int32_t* info);

void zhegvd(const std::int32_t* itype, const char* jobz, const char* uplo,
            const std::int32_t* n, std::complex<double>* A,
            const std::int32_t* lda, std::complex<double>* B,
            const std::int32_t* ldb, double* w, std::complex<double>* work,
            std::int32_t* lwork, double* rwork, std::int32_t* lrwork,
            std::int32_t* iwork, std::int32_t* liwork, std::int32_t* info);
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


#define GKO_BIND_SYEVD_BUFFERSIZES(ValueType, LapackName)                      \
    inline void syevd_buffersizes(                                             \
        const char* jobz, const char* uplo, const int32* n, ValueType* a,      \
        const int32* lda, ValueType* w, ValueType* work,                       \
        int32* fp_buffer_num_elems, int32* iwork, int32* int_buffer_num_elems) \
    {                                                                          \
        int32 info;                                                            \
        *fp_buffer_num_elems = -1;                                             \
        *int_buffer_num_elems = -1;                                            \
        GKO_ASSERT_NO_LAPACK_ERRORS(                                           \
            LapackName(jobz, uplo, n, a, lda, w, work, fp_buffer_num_elems,    \
                       iwork, int_buffer_num_elems, &info),                    \
            info);                                                             \
        *fp_buffer_num_elems = static_cast<int32>(work[0]);                    \
        *int_buffer_num_elems = iwork[0];                                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_SYEVD_BUFFERSIZES(float, ssyevd);
GKO_BIND_SYEVD_BUFFERSIZES(double, dsyevd);
template <typename ValueType>
inline void syevd_buffersizes(const char* jobz, const char* uplo,
                              const int32* n, ValueType* a, const int32* lda,
                              ValueType* w, ValueType* work,
                              int32* fp_buffer_num_elems, int32* iwork,
                              int32* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_SYEVD_BUFFERSIZES


#define GKO_BIND_SYEVD(ValueType, LapackName)                                \
    inline void syevd(const char* jobz, const char* uplo, const int32* n,    \
                      ValueType* a, const int32* lda, ValueType* w,          \
                      ValueType* work, int32* fp_buffer_num_elems,           \
                      int32* iwork, int32* int_buffer_num_elems)             \
    {                                                                        \
        int32 info;                                                          \
        GKO_ASSERT_NO_LAPACK_ERRORS(                                         \
            LapackName(jobz, uplo, n, a, lda, w, work, fp_buffer_num_elems,  \
                       iwork, int_buffer_num_elems, &info),                  \
            info);                                                           \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_SYEVD(float, ssyevd);
GKO_BIND_SYEVD(double, dsyevd);
template <typename ValueType>
inline void syevd(const char* jobz, const char* uplo, const int32* n,
                  ValueType* a, const int32* lda, ValueType* w, ValueType* work,
                  int32* fp_buffer_num_elems, int32* iwork,
                  int32* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_SYEVD


#define GKO_BIND_HEEVD_BUFFERSIZES(ValueType, LapackName)                     \
    inline void heevd_buffersizes(                                            \
        const char* jobz, const char* uplo, const int32* n, ValueType* a,     \
        const int32* lda, gko::remove_complex<ValueType>* w, ValueType* work, \
        int32* fp_buffer_num_elems, gko::remove_complex<ValueType>* rwork,    \
        int32* rfp_buffer_num_elems, int32* iwork,                            \
        int32* int_buffer_num_elems)                                          \
    {                                                                         \
        int32 info;                                                           \
        *fp_buffer_num_elems = -1;                                            \
        *rfp_buffer_num_elems = -1;                                           \
        *int_buffer_num_elems = -1;                                           \
        GKO_ASSERT_NO_LAPACK_ERRORS(                                          \
            LapackName(jobz, uplo, n, a, lda, w, work, fp_buffer_num_elems,   \
                       rwork, rfp_buffer_num_elems, iwork,                    \
                       int_buffer_num_elems, &info),                          \
            info);                                                            \
        *fp_buffer_num_elems = static_cast<int32>(work[0].real());            \
        *rfp_buffer_num_elems = static_cast<int32>(rwork[0]);                 \
        *int_buffer_num_elems = iwork[0];                                     \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_HEEVD_BUFFERSIZES(std::complex<float>, cheevd);
GKO_BIND_HEEVD_BUFFERSIZES(std::complex<double>, zheevd);
template <typename ValueType>
inline void heevd_buffersizes(const char* jobz, const char* uplo,
                              const int32* n, ValueType* a, const int32* lda,
                              gko::remove_complex<ValueType>* w,
                              ValueType* work, int32* fp_buffer_num_elems,
                              gko::remove_complex<ValueType>* rwork,
                              int32* rfp_buffer_num_elems, int32* iwork,
                              int32* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_HEEVD_BUFFERSIZES


#define GKO_BIND_HEEVD(ValueType, LapackName)                                 \
    inline void heevd(                                                        \
        const char* jobz, const char* uplo, const int32* n, ValueType* a,     \
        const int32* lda, gko::remove_complex<ValueType>* w, ValueType* work, \
        int32* fp_buffer_num_elems, gko::remove_complex<ValueType>* rwork,    \
        int32* rfp_buffer_num_elems, int32* iwork,                            \
        int32* int_buffer_num_elems)                                          \
    {                                                                         \
        int32 info;                                                           \
        GKO_ASSERT_NO_LAPACK_ERRORS(                                          \
            LapackName(jobz, uplo, n, a, lda, w, work, fp_buffer_num_elems,   \
                       rwork, rfp_buffer_num_elems, iwork,                    \
                       int_buffer_num_elems, &info),                          \
            info);                                                            \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_HEEVD(std::complex<float>, cheevd);
GKO_BIND_HEEVD(std::complex<double>, zheevd);
template <typename ValueType>
inline void heevd(const char* jobz, const char* uplo, const int32* n,
                  ValueType* a, const int32* lda, remove_complex<ValueType>* w,
                  ValueType* work, int32* fp_buffer_num_elems,
                  remove_complex<ValueType>* rwork, int32* rfp_buffer_num_elems,
                  int32* iwork,
                  int32* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_HEEVD


#define GKO_BIND_SYGVD_BUFFERSIZES(ValueType, LapackName)                      \
    inline void sygvd_buffersizes(                                             \
        const int32* itype, const char* jobz, const char* uplo,                \
        const int32* n, ValueType* a, const int32* lda, ValueType* b,          \
        const int32* ldb, ValueType* w, ValueType* work,                       \
        int32* fp_buffer_num_elems, int32* iwork, int32* int_buffer_num_elems) \
    {                                                                          \
        int32 info;                                                            \
        *fp_buffer_num_elems = -1;                                             \
        *int_buffer_num_elems = -1;                                            \
        GKO_ASSERT_NO_LAPACK_ERRORS(                                           \
            LapackName(itype, jobz, uplo, n, a, lda, b, ldb, w, work,          \
                       fp_buffer_num_elems, iwork, int_buffer_num_elems,       \
                       &info),                                                 \
            info);                                                             \
        *fp_buffer_num_elems = static_cast<int32>(work[0]);                    \
        *int_buffer_num_elems = iwork[0];                                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_SYGVD_BUFFERSIZES(float, ssygvd);
GKO_BIND_SYGVD_BUFFERSIZES(double, dsygvd);
template <typename ValueType>
inline void sygvd_buffersizes(const int32* itype, const char* jobz,
                              const char* uplo, const int32* n, ValueType* a,
                              const int32* lda, ValueType* b, const int32* ldb,
                              ValueType* w, ValueType* work,
                              int32* fp_buffer_num_elems, int32* iwork,
                              int32* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_SYGVD_BUFFERSIZES


#define GKO_BIND_SYGVD(ValueType, LapackName)                                 \
    inline void sygvd(const int32* itype, const char* jobz, const char* uplo, \
                      const int32* n, ValueType* a, const int32* lda,         \
                      ValueType* b, const int32* ldb, ValueType* w,           \
                      ValueType* work, int32* fp_buffer_num_elems,            \
                      int32* iwork, int32* int_buffer_num_elems)              \
    {                                                                         \
        int32 info;                                                           \
        GKO_ASSERT_NO_LAPACK_ERRORS(                                          \
            LapackName(itype, jobz, uplo, n, a, lda, b, ldb, w, work,         \
                       fp_buffer_num_elems, iwork, int_buffer_num_elems,      \
                       &info),                                                \
            info);                                                            \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_SYGVD(float, ssygvd);
GKO_BIND_SYGVD(double, dsygvd);
template <typename ValueType>
inline void sygvd(const int32* itype, const char* jobz, const char* uplo,
                  const int32* n, ValueType* a, const int32* lda, ValueType* b,
                  const int32* ldb, ValueType* w, ValueType* work,
                  int32* fp_buffer_num_elems, int32* iwork,
                  int32* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_SYGVD


#define GKO_BIND_HEGVD_BUFFERSIZES(ValueType, LapackName)                     \
    inline void hegvd_buffersizes(                                            \
        const int32* itype, const char* jobz, const char* uplo,               \
        const int32* n, ValueType* a, const int32* lda, ValueType* b,         \
        const int32* ldb, gko::remove_complex<ValueType>* w, ValueType* work, \
        int32* fp_buffer_num_elems, gko::remove_complex<ValueType>* rwork,    \
        int32* rfp_buffer_num_elems, int32* iwork,                            \
        int32* int_buffer_num_elems)                                          \
    {                                                                         \
        int32 info;                                                           \
        *fp_buffer_num_elems = -1;                                            \
        *rfp_buffer_num_elems = -1;                                           \
        *int_buffer_num_elems = -1;                                           \
        GKO_ASSERT_NO_LAPACK_ERRORS(                                          \
            LapackName(itype, jobz, uplo, n, a, lda, b, ldb, w, work,         \
                       fp_buffer_num_elems, rwork, rfp_buffer_num_elems,      \
                       iwork, int_buffer_num_elems, &info),                   \
            info);                                                            \
        *fp_buffer_num_elems = static_cast<int32>(work[0].real());            \
        *rfp_buffer_num_elems = static_cast<int32>(rwork[0]);                 \
        *int_buffer_num_elems = iwork[0];                                     \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_HEGVD_BUFFERSIZES(std::complex<float>, chegvd);
GKO_BIND_HEGVD_BUFFERSIZES(std::complex<double>, zhegvd);
template <typename ValueType>
inline void hegvd_buffersizes(const int32* itype, const char* jobz,
                              const char* uplo, const int32* n, ValueType* a,
                              const int32* lda, ValueType* b, const int32* ldb,
                              gko::remove_complex<ValueType>* w,
                              ValueType* work, int32* fp_buffer_num_elems,
                              gko::remove_complex<ValueType>* rwork,
                              int32* rfp_buffer_num_elems, int32* iwork,
                              int32* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

#undef GKO_BIND_HEGVD_BUFFERSIZES


#define GKO_BIND_HEGVD(ValueType, LapackName)                                \
    inline void hegvd(                                                       \
        const int32* itype, const char* jobz, const char* uplo,              \
        const int32* n, ValueType* a, const int32* lda, ValueType* b,        \
        const int32* ldb, remove_complex<ValueType>* w, ValueType* work,     \
        int32* fp_buffer_num_elems, remove_complex<ValueType>* rwork,        \
        int32* rfp_buffer_num_elems, int32* iwork,                           \
        int32* int_buffer_num_elems)                                         \
    {                                                                        \
        int32 info;                                                          \
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
inline void hegvd(const int32* itype, const char* jobz, const char* uplo,
                  const int32* n, ValueType* a, const int32* lda, ValueType* b,
                  const int32* ldb, remove_complex<ValueType>* w,
                  ValueType* work, int32* fp_buffer_num_elems,
                  remove_complex<ValueType>* rwork, int32* rfp_buffer_num_elems,
                  int32* iwork,
                  int32* int_buffer_num_elems) GKO_NOT_IMPLEMENTED;

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
