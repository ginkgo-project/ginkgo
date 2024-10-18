// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_MATH_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_MATH_HPP_


#include <thrust/complex.h>

#include <ginkgo/core/base/math.hpp>


#ifdef GKO_COMPILING_CUDA


#include <cuda_fp16.h>


#elif defined(GKO_COMPILING_HIP)


#include <hip/hip_fp16.h>


#endif


namespace gko {


// We need this struct, because otherwise we would call a __host__ function in a
// __device__ function (even though it is constexpr)
template <typename T>
struct device_numeric_limits {
    static constexpr auto inf() { return std::numeric_limits<T>::infinity(); }
    static constexpr auto max() { return std::numeric_limits<T>::max(); }
    static constexpr auto min() { return std::numeric_limits<T>::min(); }
};

template <>
struct device_numeric_limits<__half> {
    // from __half documentation, it accepts unsigned short
    // __half does not have constexpr
    static GKO_ATTRIBUTES GKO_INLINE auto inf()
    {
        __half_raw bits;
        bits.x = static_cast<unsigned short>(0b0111110000000000u);
        return __half{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto max()
    {
        __half_raw bits;
        bits.x = static_cast<unsigned short>(0b0111101111111111u);
        return __half{bits};
    }

    static GKO_ATTRIBUTES GKO_INLINE auto min()
    {
        __half_raw bits;
        bits.x = static_cast<unsigned short>(0b0000010000000000u);
        return __half{bits};
    }
};


namespace detail {


template <typename T>
struct remove_complex_impl<thrust::complex<T>> {
    using type = T;
};


template <typename T>
struct truncate_type_impl<thrust::complex<T>> {
    using type = thrust::complex<typename truncate_type_impl<T>::type>;
};


}  // namespace detail
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_BASE_MATH_HPP_
