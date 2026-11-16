// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_ARITHMETIC_ACCESSOR_HPP_
#define GKO_COMMON_UNIFIED_BASE_ARITHMETIC_ACCESSOR_HPP_


#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/device_views.hpp>

#include "accessor/range.hpp"
#include "accessor/reduced_row_major.hpp"
#include "accessor/row_major.hpp"
#include "accessor/scaled_reduced_row_major.hpp"
#include "accessor/utils.hpp"


#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP)
#include "common/cuda_hip/base/types.hpp"
#include "core/emulation/device_arithmetic.hpp"
#elif defined(GKO_COMPILING_DPCPP)
#include "dpcpp/base/types.hpp"
#elif defined(GKO_COMPILING_OMP)
// device_type comes from common/unified/base/kernel_launch.hpp, which
// includes this file
#else
#error "This file should only be used inside Ginkgo device compilation"
#endif


/**
 * @file
 * Kernels compute in the device arithmetic type for the ranges they get as
 * acc::as_device_range(arithmetic<ValueType>(...)).
 */


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


/**
 * @internal
 * The type the kernels compute values of type ValueType in.
 */
#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP)
template <typename ValueType>
using device_arithmetic_type = gko::device_arithmetic_t<ValueType>;
#else
template <typename ValueType>
using device_arithmetic_type = std::remove_cv_t<ValueType>;
#endif


/**
 * @internal
 * The type the kernels compute real values like norms of values of type
 * ValueType in.
 */
template <typename ValueType>
using device_real_arithmetic_type =
    remove_complex<device_arithmetic_type<ValueType>>;


/**
 * @internal
 * The type a kernel on values of type ValueType computes values of type T in,
 * where T is ValueType or its real counterpart (also as device types).
 */
template <typename ValueType, typename T>
using device_arithmetic_type_in =
    std::conditional_t<is_complex_s<std::remove_cv_t<T>>::value,
                       device_arithmetic_type<ValueType>,
                       device_real_arithmetic_type<ValueType>>;


/**
 * @internal
 * The range a kernel on values of type KernelValueType accesses
 * Dimensionality-dimensional data of type ValueType through: a row_major range
 * if it computes in the storage type, otherwise a reduced_row_major range that
 * converts to the arithmetic type on every read and write. With host types,
 * acc::as_device_range maps it to the range the kernel gets; with device types,
 * like the values of the batch structs, it is that range already.
 */
template <typename KernelValueType, typename ValueType,
          std::size_t Dimensionality>
using arithmetic_range = acc::range<std::conditional_t<
    std::is_same_v<
        device_type<device_arithmetic_type_in<KernelValueType, ValueType>>,
        device_type<std::remove_cv_t<ValueType>>>,
    acc::row_major<ValueType, Dimensionality>,
    acc::reduced_row_major<
        Dimensionality, device_arithmetic_type_in<KernelValueType, ValueType>,
        ValueType>>>;


/**
 * @internal
 * The type the elements of Accessor are read and written in.
 */
template <typename Accessor>
struct accessor_arithmetic_type {
    using type = typename Accessor::arithmetic_type;
};

template <typename ValueType, acc::size_type Dimensionality>
struct accessor_arithmetic_type<acc::row_major<ValueType, Dimensionality>> {
    using type = std::remove_cv_t<ValueType>;
};


template <typename Range>
struct range_arithmetic_type {
    using type =
        typename accessor_arithmetic_type<typename Range::accessor>::type;
};

template <typename ValueType>
struct range_arithmetic_type<ValueType*> {
    using type = std::remove_cv_t<ValueType>;
};


/**
 * @internal
 * The type the elements of a range or a pointer are read and written in.
 */
template <typename Range>
using arithmetic_type_t =
    typename range_arithmetic_type<std::decay_t<Range>>::type;


/**
 * @internal
 * Returns size values as a one-dimensional arithmetic_range for a kernel on
 * values of type KernelValueType.
 *
 * @tparam KernelValueType  the value type of the kernel, which decides the
 *                          arithmetic type
 */
template <typename KernelValueType, typename ValueType>
arithmetic_range<KernelValueType, ValueType, 1> arithmetic(ValueType* values,
                                                           size_type size)
{
    return arithmetic_range<KernelValueType, ValueType, 1>{
        std::array<acc::size_type, 1>{static_cast<acc::size_type>(size)},
        values};
}


/**
 * @internal
 * Returns a dense matrix as an arithmetic_range for a kernel on values of type
 * KernelValueType: two-dimensional with the stride of the matrix, or
 * one-dimensional over its stored values for kernels that index them like a
 * pointer, e.g. row vectors.
 *
 * @tparam KernelValueType  the value type of the kernel, which decides the
 *                          arithmetic type
 * @tparam Dimensionality  2, or 1 for the stored values
 */
template <typename KernelValueType, std::size_t Dimensionality = 2,
          typename ValueType>
arithmetic_range<KernelValueType, ValueType, Dimensionality> arithmetic(
    matrix::view::dense<ValueType> mtx)
{
    static_assert(Dimensionality == 1 || Dimensionality == 2,
                  "a dense matrix is one- or two-dimensional");
    if constexpr (Dimensionality == 1) {
        return arithmetic<KernelValueType>(
            mtx.values, mtx.size[0] == 0
                            ? size_type{}
                            : (mtx.size[0] - 1) * mtx.stride + mtx.size[1]);
    } else {
        return arithmetic_range<KernelValueType, ValueType, 2>{
            std::array<acc::size_type, 2>{
                static_cast<acc::size_type>(mtx.size[0]),
                static_cast<acc::size_type>(mtx.size[1])},
            mtx.values,
            std::array<acc::size_type, 1>{
                static_cast<acc::size_type>(mtx.stride)}};
    }
}


/**
 * @internal
 * Returns the values of a CSR matrix as a one-dimensional arithmetic_range for
 * a kernel on values of type KernelValueType.
 *
 * @tparam KernelValueType  the value type of the kernel, which decides the
 *                          arithmetic type
 */
template <typename KernelValueType, typename ValueType, typename IndexType>
arithmetic_range<KernelValueType, ValueType, 1> arithmetic(
    matrix::view::csr<ValueType, IndexType> mtx)
{
    return arithmetic<KernelValueType>(mtx.values, mtx.num_stored_elements);
}


/**
 * @internal
 * Returns a reduced_row_major range over the same storage that computes in the
 * arithmetic type of a kernel on values of type KernelValueType.
 *
 * @tparam KernelValueType  the value type of the kernel, which decides the
 *                          arithmetic type
 */
template <typename KernelValueType, std::size_t Dimensionality,
          typename ArithmeticType, typename StorageType>
auto arithmetic(const acc::range<acc::reduced_row_major<
                    Dimensionality, ArithmeticType, StorageType>>& range)
{
    using accessor = acc::reduced_row_major<
        Dimensionality,
        device_arithmetic_type_in<KernelValueType, ArithmeticType>,
        StorageType>;
    return acc::range<accessor>(range.get_accessor().get_size(),
                                range.get_accessor().get_stored_data(),
                                range.get_accessor().get_stride());
}


/**
 * @internal
 * Returns a scaled_reduced_row_major range over the same storage and scalars
 * that computes in the arithmetic type of a kernel on values of type
 * KernelValueType.
 *
 * @tparam KernelValueType  the value type of the kernel, which decides the
 *                          arithmetic type
 */
template <typename KernelValueType, std::size_t Dimensionality,
          typename ArithmeticType, typename StorageType,
          std::uint64_t ScalarMask, typename ScalarType>
auto arithmetic(
    const acc::range<acc::scaled_reduced_row_major<
        Dimensionality, ArithmeticType, StorageType, ScalarMask, ScalarType>>&
        range)
{
    using accessor = acc::scaled_reduced_row_major<
        Dimensionality,
        device_arithmetic_type_in<KernelValueType, ArithmeticType>, StorageType,
        ScalarMask, ScalarType>;
    return acc::range<accessor>(range.get_accessor().get_size(),
                                range.get_accessor().get_stored_data(),
                                range.get_accessor().get_storage_stride(),
                                range.get_accessor().get_scalar(),
                                range.get_accessor().get_scalar_stride());
}


/**
 * @internal
 * Passes a single value converted to
 * device_arithmetic_type_in<KernelValueType, ValueType>.
 */
template <typename KernelValueType, typename ValueType,
          std::enable_if_t<::gko::detail::is_complex_or_scalar_impl<
              ValueType>::value>* = nullptr>
device_type<device_arithmetic_type_in<KernelValueType, ValueType>> arithmetic(
    const ValueType& value)
{
    return as_device_type(
        static_cast<device_arithmetic_type_in<KernelValueType, ValueType>>(
            value));
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_UNIFIED_BASE_ARITHMETIC_ACCESSOR_HPP_
