// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_EXTENSIONS_KOKKOS_TYPES_HPP
#define GINKGO_EXTENSIONS_KOKKOS_TYPES_HPP

#include <Kokkos_Complex.hpp>


#include <Kokkos_Core.hpp>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/extensions/kokkos/config.hpp>


namespace gko {
namespace ext {
namespace kokkos {
namespace detail {


/**
 * Maps arithmetic types to corresponding Kokkos types.
 *
 * @tparam T  An arithmetic type.
 */
template <typename T>
struct value_type_impl {
    using type = T;
};

template <typename T>
struct value_type_impl<const T> {
    using type = const typename value_type_impl<T>::type;
};

template <typename T>
struct value_type_impl<std::complex<T>> {
    using type = Kokkos::complex<T>;
};


template <typename T>
struct value_type {
    using type = typename value_type_impl<T>::type;

    static_assert(sizeof(std::decay_t<T>) == sizeof(std::decay_t<type>),
                  "Can't handle C++ data type and corresponding Kokkos "
                  "type with mismatching type sizes.");
#if GINKGO_EXTENSION_KOKKOS_CHECK_TYPE_ALIGNMENT
    static_assert(
        alignof(std::decay_t<T>) == alignof(std::decay_t<type>),
        "Can't handle C++ data type and corresponding Kokkos type with "
        "mismatching alignments. If std::complex is used, please make sure "
        "to configure Kokkos with `KOKKOS_ENABLE_COMPLEX_ALIGN=ON`.\n"
        "Alternatively, disable this check by setting the CMake option "
        "-DGINKGO_EXTENSION_KOKKOS_CHECK_TYPE_ALIGNMENT=OFF.");
#endif
};

template <typename T>
using value_type_t = typename value_type<T>::type;


/**
 * Maps ginkgo types to the corresponding kokkos type
 *
 * @tparam T  A ginkgo type to map
 * @tparam MemorySpace  A kokkos MemorySpace which the resulting type will use
 *
 * @warning  The MemorySpace has to match the internally used executor of T.
 *
 * @note  This only describes the interface of the mapper class. The actual
 *        implementation is done via specialization of this class.
 */
template <typename T, typename MemorySpace>
struct mapper {
    static auto map(T&);

    static auto map(const T&);
};


/**
 * Type that maps a Ginkgo array to an unmanaged 1D Kokkos::View.
 *
 * @warning Using std::complex as data type might lead to issues, since the
 *          alignment of Kokkos::complex is not necessarily the same.
 *
 * @tparam MemorySpace  The memory space type the mapped object should use.
 */
template <typename ValueType, typename MemorySpace>
struct mapper<array<ValueType>, MemorySpace> {
    template <typename ValueType_c>
    using type =
        Kokkos::View<typename value_type<ValueType_c>::type*, MemorySpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    /**
     * Constructor based on size and raw pointer
     *
     * @tparam ValueType_c  The same type as ValueType, but possible
     *                      const-qualified
     * @param data  The pointer to the data
     * @param size  The size of the array
     * @return  A Kokkos 1D-view to the data
     */
    template <typename ValueType_c>
    static type<ValueType_c> map(ValueType_c* data, size_type size);

    /**
     * Maps a mutable array.
     */
    static type<ValueType> map(array<ValueType>& arr);

    /**
     * Maps a const array.
     */
    static type<const ValueType> map(const array<ValueType>& arr);

    /**
     * Maps a const_array_view (same as mapping a const array).
     */
    static type<const ValueType> map(
        const ::gko::detail::const_array_view<ValueType>& arr);
};

template <typename ValueType, typename MemorySpace>
template <typename ValueType_c>
typename mapper<array<ValueType>, MemorySpace>::template type<ValueType_c>
mapper<array<ValueType>, MemorySpace>::map(ValueType_c* data, size_type size)
{
    return type<ValueType_c>{reinterpret_cast<value_type_t<ValueType_c>*>(data),
                             size};
}


template <typename ValueType, typename MemorySpace>
typename mapper<array<ValueType>, MemorySpace>::template type<ValueType>
mapper<array<ValueType>, MemorySpace>::map(array<ValueType>& arr)
{
    assert_compatibility<MemorySpace>(arr);

    return map(arr.get_data(), arr.get_size());
}


template <typename ValueType, typename MemorySpace>
typename mapper<array<ValueType>, MemorySpace>::template type<const ValueType>
mapper<array<ValueType>, MemorySpace>::map(const array<ValueType>& arr)
{
    assert_compatibility<MemorySpace>(arr);

    return map(arr.get_const_data(), arr.get_size());
}


template <typename ValueType, typename MemorySpace>
typename mapper<array<ValueType>, MemorySpace>::template type<const ValueType>
mapper<array<ValueType>, MemorySpace>::map(
    const ::gko::detail::const_array_view<ValueType>& arr)
{
    assert_compatibility<MemorySpace>(arr);

    return map(arr.get_const_data(), arr.get_size());
}


/**
 * Type that maps a Ginkgo matrix::Dense to an unmanaged 2D Kokkos::View.
 *
 * @warning Using std::complex as data type might lead to issues, since the
 *          alignment of Kokkos::complex is not necessarily the same.
 *
 * @tparam MemorySpace  The memory space type the mapped object should use.
 */
template <typename ValueType, typename MemorySpace>
struct mapper<matrix::Dense<ValueType>, MemorySpace> {
    template <typename ValueType_c>
    using type = Kokkos::View<typename value_type<ValueType_c>::type**,
                              Kokkos::LayoutStride, MemorySpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    /**
     * Maps a mutable dense matrix.
     */
    static type<ValueType> map(matrix::Dense<ValueType>& m);

    /**
     * Maps a const dense matrix.
     */
    static type<const ValueType> map(const matrix::Dense<ValueType>& m);
};

template <typename ValueType, typename MemorySpace>
typename mapper<matrix::Dense<ValueType>, MemorySpace>::template type<ValueType>
mapper<matrix::Dense<ValueType>, MemorySpace>::map(matrix::Dense<ValueType>& m)
{
    assert_compatibility<MemorySpace>(m);

    auto size = m.get_size();

    return type<ValueType>{
        reinterpret_cast<value_type_t<ValueType>*>(m.get_values()),
        Kokkos::LayoutStride{size[0], m.get_stride(), size[1], 1}};
}


template <typename ValueType, typename MemorySpace>
typename mapper<matrix::Dense<ValueType>,
                MemorySpace>::template type<const ValueType>
mapper<matrix::Dense<ValueType>, MemorySpace>::map(
    const matrix::Dense<ValueType>& m)
{
    assert_compatibility<MemorySpace>(m);

    auto size = m.get_size();

    return type<const ValueType>{
        reinterpret_cast<const value_type_t<ValueType>*>(m.get_const_values()),
        Kokkos::LayoutStride{size[0], m.get_stride(), size[1], 1}};
}


}  // namespace detail


/**
 * Maps Ginkgo object to a type compatible with Kokkos.
 *
 * @tparam T  The Ginkgo type.
 * @tparam MemorySpace  The Kokkos memory space that will be used
 *
 * @param data  The Ginkgo object.
 *
 * @return  A wrapper for the Ginkgo object that is compatible with Kokkos
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
          typename T>
inline auto map_data(T&& data)
{
    return detail::mapper<std::decay_t<T>, MemorySpace>::map(
        std::forward<T>(data));
}

/**
 * @copydoc map_data(T&&, MemorySpace)
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
          typename T>
inline auto map_data(T* data)
{
    return map_data<MemorySpace>(*data);
}

/**
 * @copydoc map_data(T&&, MemorySpace)
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
          typename T>
inline auto map_data(std::unique_ptr<T>& data)
{
    return map_data<MemorySpace>(*data);
}

/**
 * @copydoc map_data(T&&, MemorySpace)
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
          typename T>
inline auto map_data(std::shared_ptr<T>& data)
{
    return map_data<MemorySpace>(*data);
}


}  // namespace kokkos
}  // namespace ext
}  // namespace gko


#endif  // GINKGO_EXTENSIONS_KOKKOS_TYPES_HPP
