/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_CUDA_BASE_TYPES_HPP_
#define GKO_CUDA_BASE_TYPES_HPP_


#include <ginkgo/core/base/types.hpp>


#include <type_traits>


#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cusparse.h>
#include <thrust/complex.h>


#include <ginkgo/core/base/matrix_data.hpp>


namespace gko {


namespace kernels {
namespace cuda {


namespace detail {


/**
 * @internal
 *
 * replacement for thrust::complex without alignment restrictions.
 */
template <typename T>
struct alignas(std::complex<T>) fake_complex {
    T real;
    T imag;

    GKO_INLINE GKO_ATTRIBUTES constexpr fake_complex() : real{}, imag{} {}

    GKO_INLINE GKO_ATTRIBUTES constexpr fake_complex(thrust::complex<T> val)
        : real{val.real()}, imag{val.imag()}
    {}

    friend bool GKO_INLINE GKO_ATTRIBUTES constexpr operator==(fake_complex a,
                                                               fake_complex b)
    {
        return a.real == b.real && a.imag == b.imag;
    }

    friend bool GKO_INLINE GKO_ATTRIBUTES constexpr operator!=(fake_complex a,
                                                               fake_complex b)
    {
        return !(a == b);
    }
};


template <typename ValueType>
struct fake_complex_unpack_impl {
    using type = ValueType;

    GKO_INLINE GKO_ATTRIBUTES static constexpr ValueType unpack(ValueType v)
    {
        return v;
    }
};

template <typename ValueType>
struct fake_complex_unpack_impl<fake_complex<ValueType>> {
    using type = thrust::complex<ValueType>;

    GKO_INLINE GKO_ATTRIBUTES static constexpr thrust::complex<ValueType>
    unpack(fake_complex<ValueType> v)
    {
        return {v.real, v.imag};
    }
};


template <typename T>
struct culibs_type_impl {
    using type = T;
};

template <typename T>
struct culibs_type_impl<T*> {
    using type = typename culibs_type_impl<T>::type*;
};

template <typename T>
struct culibs_type_impl<T&> {
    using type = typename culibs_type_impl<T>::type&;
};

template <typename T>
struct culibs_type_impl<const T> {
    using type = const typename culibs_type_impl<T>::type;
};

template <typename T>
struct culibs_type_impl<volatile T> {
    using type = volatile typename culibs_type_impl<T>::type;
};

template <>
struct culibs_type_impl<std::complex<float>> {
    using type = cuComplex;
};

template <>
struct culibs_type_impl<std::complex<double>> {
    using type = cuDoubleComplex;
};

template <typename T>
struct culibs_type_impl<thrust::complex<T>> {
    using type = typename culibs_type_impl<std::complex<T>>::type;
};

template <typename T>
struct cuda_type_impl {
    using type = T;
};

template <typename T>
struct cuda_type_impl<T*> {
    using type = typename cuda_type_impl<T>::type*;
};

template <typename T>
struct cuda_type_impl<T&> {
    using type = typename cuda_type_impl<T>::type&;
};

template <typename T>
struct cuda_type_impl<const T> {
    using type = const typename cuda_type_impl<T>::type;
};

template <typename T>
struct cuda_type_impl<volatile T> {
    using type = volatile typename cuda_type_impl<T>::type;
};

template <typename T>
struct cuda_type_impl<std::complex<T>> {
    using type = thrust::complex<T>;
};

template <>
struct cuda_type_impl<cuDoubleComplex> {
    using type = thrust::complex<double>;
};

template <>
struct cuda_type_impl<cuComplex> {
    using type = thrust::complex<float>;
};

template <typename T>
struct cuda_struct_member_type_impl {
    using type = T;
};

template <typename T>
struct cuda_struct_member_type_impl<std::complex<T>> {
    using type = fake_complex<T>;
};

template <typename ValueType, typename IndexType>
struct cuda_type_impl<matrix_data_entry<ValueType, IndexType>> {
    using type = matrix_data_entry<
        typename cuda_struct_member_type_impl<ValueType>::type, IndexType>;
};


template <typename T>
constexpr cudaDataType_t cuda_data_type_impl()
{
    return CUDA_C_8U;
}

template <>
constexpr cudaDataType_t cuda_data_type_impl<float16>()
{
    return CUDA_R_16F;
}

template <>
constexpr cudaDataType_t cuda_data_type_impl<float>()
{
    return CUDA_R_32F;
}

template <>
constexpr cudaDataType_t cuda_data_type_impl<double>()
{
    return CUDA_R_64F;
}

template <>
constexpr cudaDataType_t cuda_data_type_impl<std::complex<float>>()
{
    return CUDA_C_32F;
}

template <>
constexpr cudaDataType_t cuda_data_type_impl<std::complex<double>>()
{
    return CUDA_C_64F;
}

template <>
constexpr cudaDataType_t cuda_data_type_impl<int32>()
{
    return CUDA_R_32I;
}

template <>
constexpr cudaDataType_t cuda_data_type_impl<uint32>()
{
    return CUDA_R_32U;
}

template <>
constexpr cudaDataType_t cuda_data_type_impl<int8>()
{
    return CUDA_R_8I;
}

template <>
constexpr cudaDataType_t cuda_data_type_impl<uint8>()
{
    return CUDA_R_8U;
}


#if defined(CUDA_VERSION) &&  \
    (CUDA_VERSION >= 11000 || \
     ((CUDA_VERSION >= 10020) && !(defined(_WIN32) || defined(__CYGWIN__))))


template <typename T>
constexpr cusparseIndexType_t cusparse_index_type_impl()
{
    return CUSPARSE_INDEX_16U;
}

template <>
constexpr cusparseIndexType_t cusparse_index_type_impl<int32>()
{
    return CUSPARSE_INDEX_32I;
}

template <>
constexpr cusparseIndexType_t cusparse_index_type_impl<int64>()
{
    return CUSPARSE_INDEX_64I;
}


#endif  // defined(CUDA_VERSION) && (CUDA_VERSION >= 11000 || ((CUDA_VERSION >=
        // 10020) && !(defined(_WIN32) || defined(__CYGWIN__))))


}  // namespace detail


/**
 * This is an alias for the `cudaDataType_t` equivalent of `T`. By default,
 * CUDA_C_8U (which is unsupported by C++) is returned.
 *
 * @tparam T  a type
 *
 * @returns the actual `cudaDataType_t`
 */
template <typename T>
constexpr cudaDataType_t cuda_data_type()
{
    return detail::cuda_data_type_impl<T>();
}


#if defined(CUDA_VERSION) &&  \
    (CUDA_VERSION >= 11000 || \
     ((CUDA_VERSION >= 10020) && !(defined(_WIN32) || defined(__CYGWIN__))))


/**
 * This is an alias for the `cudaIndexType_t` equivalent of `T`. By default,
 * CUSPARSE_INDEX_16U is returned.
 *
 * @tparam T  a type
 *
 * @returns the actual `cusparseIndexType_t`
 */
template <typename T>
constexpr cusparseIndexType_t cusparse_index_type()
{
    return detail::cusparse_index_type_impl<T>();
}


#endif  // defined(CUDA_VERSION) && (CUDA_VERSION >= 11000 || ((CUDA_VERSION >=
        // 10020) && !(defined(_WIN32) || defined(__CYGWIN__))))


/**
 * This is an alias for CUDA's equivalent of `T`.
 *
 * @tparam T  a type
 */
template <typename T>
using cuda_type = typename detail::cuda_type_impl<T>::type;


/**
 * Reinterprets the passed in value as a CUDA type.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to CUDA type
 */
template <typename T>
inline std::enable_if_t<
    std::is_pointer<T>::value || std::is_reference<T>::value, cuda_type<T>>
as_cuda_type(T val)
{
    return reinterpret_cast<cuda_type<T>>(val);
}


/**
 * @copydoc as_cuda_type()
 */
template <typename T>
inline std::enable_if_t<
    !std::is_pointer<T>::value && !std::is_reference<T>::value, cuda_type<T>>
as_cuda_type(T val)
{
    return *reinterpret_cast<cuda_type<T>*>(&val);
}


/**
 * This is an alias for equivalent of type T used in CUDA libraries (cuBLAS,
 * cuSPARSE, etc.).
 *
 * @tparam T  a type
 */
template <typename T>
using culibs_type = typename detail::culibs_type_impl<T>::type;


/**
 * Reinterprets the passed in value as an equivalent type used by the CUDA
 * libraries.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to type used by CUDA libraries
 */
template <typename T>
inline culibs_type<T> as_culibs_type(T val)
{
    return reinterpret_cast<culibs_type<T>>(val);
}


/**
 * Casts fake_complex<T> to thrust::complex<T> and leaves any other types
 * unchanged.
 *
 * This is necessary to work around an issue with Thrust shipped in CUDA 9.2,
 * and the fact that thrust::complex has stronger alignment restrictions than
 * std::complex, i.e. structs containing them among other smaller members have
 * different sizes on device and host.
 *
 * @param val  The input value.
 *
 * @return val cast to the correct type.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr
    typename detail::fake_complex_unpack_impl<T>::type
    fake_complex_unpack(T v)
{
    return detail::fake_complex_unpack_impl<T>::unpack(v);
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_TYPES_HPP_
