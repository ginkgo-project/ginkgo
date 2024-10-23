// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_MATH_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_MATH_HPP_


#include <thrust/complex.h>

#include <ginkgo/core/base/math.hpp>


namespace gko {


// We need this struct, because otherwise we would call a __host__ function in a
// __device__ function (even though it is constexpr)
template <typename T>
struct device_numeric_limits {
    static constexpr auto inf = std::numeric_limits<T>::infinity();
    static constexpr auto max = std::numeric_limits<T>::max();
    static constexpr auto min = std::numeric_limits<T>::min();
};

template <>
struct device_numeric_limits<__half> {
    static constexpr auto inf = std::numeric_limits<gko::half>::infinity();
    static constexpr auto max = std::numeric_limits<gko::half>::max();
    static constexpr auto min = std::numeric_limits<gko::half>::min();
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
