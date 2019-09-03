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

#ifndef GKO_HIP_BASE_TYPES_HPP_
#define GKO_HIP_BASE_TYPES_HPP_


#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <thrust/complex.h>


namespace gko {


namespace kernels {
namespace hip {


namespace detail {


template <typename T>
struct hiplibs_type_impl {
    using type = T;
};

template <typename T>
struct hiplibs_type_impl<T *> {
    using type = typename hiplibs_type_impl<T>::type *;
};

template <typename T>
struct hiplibs_type_impl<T &> {
    using type = typename hiplibs_type_impl<T>::type &;
};

template <typename T>
struct hiplibs_type_impl<const T> {
    using type = const typename hiplibs_type_impl<T>::type;
};

template <typename T>
struct hiplibs_type_impl<volatile T> {
    using type = volatile typename hiplibs_type_impl<T>::type;
};

template <>
struct hiplibs_type_impl<std::complex<float>> {
    using type = hipComplex;
};

template <>
struct hiplibs_type_impl<std::complex<double>> {
    using type = hipDoubleComplex;
};

template <typename T>
struct hiplibs_type_impl<thrust::complex<T>> {
    using type = typename hiplibs_type_impl<std::complex<T>>::type;
};

template <typename T>
struct hip_type_impl {
    using type = T;
};

template <typename T>
struct hip_type_impl<T *> {
    using type = typename hip_type_impl<T>::type *;
};

template <typename T>
struct hip_type_impl<T &> {
    using type = typename hip_type_impl<T>::type &;
};

template <typename T>
struct hip_type_impl<const T> {
    using type = const typename hip_type_impl<T>::type;
};

template <typename T>
struct hip_type_impl<volatile T> {
    using type = volatile typename hip_type_impl<T>::type;
};

template <typename T>
struct hip_type_impl<std::complex<T>> {
    using type = thrust::complex<T>;
};

template <>
struct hip_type_impl<hipDoubleComplex> {
    using type = thrust::complex<double>;
};

template <>
struct hip_type_impl<hipComplex> {
    using type = thrust::complex<float>;
};


template <typename T>
constexpr hipblasDatatype_t hip_data_type_impl()
{
    return HIPBLAS_C_16F;
}

template <>
constexpr hipblasDatatype_t hip_data_type_impl<float16>()
{
    return HIPBLAS_R_16F;
}

template <>
constexpr hipblasDatatype_t hip_data_type_impl<float>()
{
    return HIPBLAS_R_32F;
}

template <>
constexpr hipblasDatatype_t hip_data_type_impl<double>()
{
    return HIPBLAS_R_64F;
}

template <>
constexpr hipblasDatatype_t hip_data_type_impl<std::complex<float>>()
{
    return HIPBLAS_C_32F;
}

template <>
constexpr hipblasDatatype_t hip_data_type_impl<std::complex<double>>()
{
    return HIPBLAS_C_64F;
}


}  // namespace detail


/**
 * This is an alias for the `hipblasDataType_t` equivalent of `T`. By default,
 * HIPBLAS_C_8U (which is unsupported by C++) is returned.
 *
 * @tparam T  a type
 *
 * @returns the actual `hipblasDatatype_t`
 */
template <typename T>
constexpr hipblasDatatype_t hip_data_type()
{
    return detail::hip_data_type_impl<T>();
}


/**
 * This is an alias for HIP's equivalent of `T`.
 *
 * @tparam T  a type
 */
template <typename T>
using hip_type = typename detail::hip_type_impl<T>::type;


/**
 * Reinterprets the passed in value as a HIP type.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to HIP type
 */
template <typename T>
inline xstd::enable_if_t<
    std::is_pointer<T>::value || std::is_reference<T>::value, hip_type<T>>
as_hip_type(T val)
{
    return reinterpret_cast<hip_type<T>>(val);
}


/**
 * @copydoc as_hip_type()
 */
template <typename T>
inline xstd::enable_if_t<
    !std::is_pointer<T>::value && !std::is_reference<T>::value, hip_type<T>>
as_hip_type(T val)
{
    return *reinterpret_cast<hip_type<T> *>(&val);
}


/**
 * This is an alias for equivalent of type T used in HIP libraries (HIPBLAS,
 * HIPSPARSE, etc.).
 *
 * @tparam T  a type
 */
template <typename T>
using hiplibs_type = typename detail::hiplibs_type_impl<T>::type;


/**
 * Reinterprets the passed in value as an equivalent type used by the HIP
 * libraries.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to type used by HIP libraries
 */
template <typename T>
inline hiplibs_type<T> as_hiplibs_type(T val)
{
    return reinterpret_cast<hiplibs_type<T>>(val);
}


/* TODO: change this */
struct hip_config {
    /**
     * The number of threads within a HIP warp. Note that with HIP the warp_size
     * should not be supposed to be 32.
     */
    static constexpr uint32 warp_size = 32;

    /**
     * The bitmask of the entire warp.
     */
    static constexpr uint32 full_lane_mask = (1ll << warp_size) - 1;

    /**
     * The maximal number of threads allowed in a HIP warp.
     */
    static constexpr uint32 max_block_size = 1024;

    /**
     * The minimal amount of warps that need to be scheduled for each block
     * to maximize GPU occupancy.
     */
    static constexpr uint32 min_warps_per_block = 4;
};


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_BASE_TYPES_HPP_
