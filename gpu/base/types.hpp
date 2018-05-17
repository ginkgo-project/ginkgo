/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_GPU_BASE_TYPES_HPP_
#define GKO_GPU_BASE_TYPES_HPP_


#include <cublas_v2.h>
#include <thrust/complex.h>


namespace gko {


namespace kernels {
namespace gpu {


namespace detail {


template <typename T>
struct culibs_type_impl {
    using type = T;
};

template <typename T>
struct culibs_type_impl<T *> {
    using type = typename culibs_type_impl<T>::type *;
};

template <typename T>
struct culibs_type_impl<T &> {
    using type = typename culibs_type_impl<T>::type &;
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
struct cuda_type_impl {
    using type = T;
};

template <typename T>
struct cuda_type_impl<T *> {
    using type = typename cuda_type_impl<T>::type *;
};

template <typename T>
struct cuda_type_impl<T &> {
    using type = typename cuda_type_impl<T>::type &;
};

template <typename T>
struct cuda_type_impl<const T> {
    using type = const typename cuda_type_impl<T>::type;
};

template <typename T>
struct cuda_type_impl<volatile T> {
    using type = volatile typename cuda_type_impl<T>::type;
};

template <>
struct cuda_type_impl<std::complex<float>> {
    using type = thrust::complex<float>;
};

template <>
struct cuda_type_impl<std::complex<double>> {
    using type = thrust::complex<double>;
};


}  // namespace detail


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
inline cuda_type<T> as_cuda_type(T val)
{
    return reinterpret_cast<cuda_type<T>>(val);
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


struct cuda_config {
    /**
     * The number of threads within a CUDA warp.
     */
    static constexpr uint32 warp_size = 32;

    /**
     * The bitmask of the entire warp.
     */
    static constexpr uint32 full_lane_mask = (1ll << warp_size) - 1;

    /**
     * The maximal number of threads allowed in a CUDA warp.
     */
    static constexpr uint32 max_block_size = 1024;

    /**
     * The minimal amount of warps that need to be scheduled for each block
     * to maximize GPU occupancy.
     */
    static constexpr uint32 min_warps_per_block = 4;
};


}  // namespace gpu
}  // namespace kernels
}  // namespace gko


#endif  // GKO_GPU_BASE_TYPES_HPP_
