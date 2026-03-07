// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_AMP_TYPES_H_
#define GKO_COMMON_UNIFIED_BASE_AMP_TYPES_H_


#include <tuple>

// for device_std
#include "common/unified/base/kernel_launch.hpp"

#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP)

#include "common/cuda_hip/base/types.hpp"

#else

#include "omp/base/math.hpp"

#endif


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace amp {


namespace gkerdev = gko::kernels::GKO_DEVICE_NAMESPACE;

#if GINKGO_ENABLE_BFLOAT16 || GINKGO_ENABLE_HALF

#define GINKGO_HAVE_AMP_HALF 1

#if GINKGO_ENABLE_BFLOAT16

#define GKO_AMP_HALF_IS_BFLOAT16 1
using half = gko::kernels::GKO_DEVICE_NAMESPACE::device_bfloat16;

#else

#define GKO_AMP_HALF_IS_FP16 1
using half = gko::kernels::GKO_DEVICE_NAMESPACE::device_half;

#endif

/**
 * All the real-valued types of different precisions available for adaptive
 * precision algorithms.
 *
 * Note that this is supposed to match the same type in the public header
 * amp_types.hpp.
 */
using supported_precisions = std::tuple<double, float, half>;

#else

using supported_precisions = std::tuple<double, float>;

#endif

/**
 * All the real or complex types of different precisions available for adaptive
 * precision algorithms.
 */
template <typename RealValueType>
struct supported_types {
    using type = supported_precisions;
};

template <typename RealValueType>
struct supported_types<gkerdev::device_std::complex<RealValueType>> {
    using type = gkerdev::to_complex<supported_precisions>;
};

/// Total number of supported precision formats.
constexpr int num_amp_precisions = std::tuple_size<supported_precisions>::value;

/**
 * Metafunction that maps an integer to a supported precision real type.
 */
template <int i>
using real_type_at_idx =
    typename std::tuple_element<static_cast<size_t>(i),
                                supported_precisions>::type;

/**
 * Metafunction that maps an integer to a supported precision
 * real or complex type.
 *
 * @tparam i  The index in the types list.
 * @tparam RealOrComplexType  A type that denotes whether a real type is needed
 *                            or a complex type.
 *
 * Eg.: `type_at_idx<2, std::complex<double>>` will be `std::complex<half>` if
 * half precision is available.
 * @sa supported_precisions
 */
template <int i, typename RealOrComplexType>
using type_at_idx = typename std::tuple_element<
    static_cast<size_t>(i),
    typename supported_types<RealOrComplexType>::type>::type;


namespace detail {


template <typename RealType, int i, typename Enable = void>
struct prec_idx_helper {
    using type = typename prec_idx_helper<RealType, i - 1>::type;
    static constexpr int index = prec_idx_helper<RealType, i - 1>::index;
};

template <typename RealType, int i>
struct prec_idx_helper<
    RealType, i,
    std::enable_if_t<(i >= 0) &&
                     std::is_same<RealType, real_type_at_idx<i>>::value>> {
    using type = RealType;
    static constexpr int index = i;
};

template <typename RealType, int i>
struct prec_idx_helper<RealType, i, std::enable_if_t<(i < 0)>> {};


}  // namespace detail


/**
 * Determines the position of the given scalar type in the list of
 * supported types. Works for both real and complex template arguments.
 *
 * @sa supported_types
 */
template <typename ValueType>
struct precision_index {
    static constexpr int index =
        detail::prec_idx_helper<gko::remove_complex<ValueType>,
                                num_amp_precisions - 1>::index;
};

/**
 * Defines a type that is a tuple of a given type and all the available types
 * with precision narrower than it.
 *
 * @tparam HighestType  The type with the most precision to begin the list.
 */
template <typename HighestType>
struct narrow_types {
    /// Tuple of types including and narrower than HighestType.
    using type = decltype(std::tuple_cat(
        std::make_tuple(
            type_at_idx<precision_index<HighestType>::index, HighestType>{}),
        typename narrow_types<type_at_idx<
            precision_index<HighestType>::index + 1, HighestType>>::type{}));
    /// Number of types in the list of types above.
    static constexpr int num_types = std::tuple_size<type>::value;
};


#ifdef GINKGO_HAVE_AMP_HALF

/**
 * Currently, half is the narrowest precision supported, * if enabled.
 */
template <>
struct narrow_types<half> {
    using type = std::tuple<half>;
    static constexpr int num_types = 1;
};

template <>
struct narrow_types<gkerdev::device_type<std::complex<half>>> {
    using type = std::tuple<gkerdev::device_type<std::complex<half>>>;
    static constexpr int num_types = 1;
};

#else

template <>
struct narrow_types<float> {
    using type = std::tuple<float>;
    static constexpr int num_types = 1;
};

template <>
struct narrow_types<gkerdev::device_type<std::complex<float>>> {
    using type = std::tuple<gkerdev::device_type<std::complex<float>>>;
    static constexpr int num_types = 1;
};

#endif

/**
 * A fixed-size array holding an item for each supported precision starting at
 * the precision of the template parameter ValueType as the highest precision.
 *
 * @tparam T  Type of object to hold for each supported precision.
 * @tparam HighestType  A scalar type of the highest precision needed.
 */
template <typename T, typename HighestType>
using precision_array = std::array<T, narrow_types<HighestType>::num_types>;


}  // namespace amp
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_UNIFIED_BASE_AMP_TYPES_H_
