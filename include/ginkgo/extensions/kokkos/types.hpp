// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_TYPES_HPP
#define GINKGO_TYPES_HPP

#include <ginkgo/config.hpp>

#if GINKGO_EXTENSION_KOKKOS

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/dense.hpp>


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
struct value_type {
    using type = T;
};

template <typename T>
struct value_type<std::complex<T>> {
    using type = Kokkos::complex<T>;
};

template <typename T>
struct value_type<const std::complex<T>> {
    using type = const Kokkos::complex<T>;
};

template <typename T>
using value_type_t = typename value_type<T>::type;


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

    template <typename ValueType_c>
    static type<ValueType_c> map(ValueType_c* data, size_type size)
    {
        static_assert(sizeof(ValueType_c) == sizeof(value_type_t<ValueType_c>),
                      "Can't handle C++ data type and corresponding Kokkos "
                      "type with mismatching type sizes.");
        // a similar check for alignment is not possible, since the alignment
        // of kokkos::complex can be changed, but not through spack. Thus
        // changing the alignment after an assertion failure requires building
        // kokkos from source.

        return type<ValueType_c>{
            reinterpret_cast<value_type_t<ValueType_c>*>(data), size};
    }

    static type<ValueType> map(array<ValueType>& arr)
    {
        assert_compatibility(arr, MemorySpace{});

        return map(arr.get_data(), arr.get_size());
    }

    static type<const ValueType> map(const array<ValueType>& arr)
    {
        assert_compatibility(arr, MemorySpace{});

        return map(arr.get_const_data(), arr.get_size());
    }


    static type<const ValueType> map(
        const ::gko::detail::const_array_view<ValueType>& arr)
    {
        assert_compatibility(arr, MemorySpace{});

        return map(arr.get_const_data(), arr.get_size());
    }
};


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

    static type<ValueType> map(matrix::Dense<ValueType>& m)
    {
        static_assert(sizeof(ValueType) == sizeof(value_type_t<ValueType>),
                      "Can't handle C++ data type and corresponding Kokkos "
                      "type with mismatching type sizes.");

        assert_compatibility(m, MemorySpace{});

        auto size = m.get_size();

        return type<ValueType>{
            reinterpret_cast<value_type_t<ValueType>*>(m.get_values()),
            Kokkos::LayoutStride{size[0], m.get_stride(), size[1], 1}};
    }

    static type<const ValueType> map(const matrix::Dense<ValueType>& m)
    {
        static_assert(sizeof(ValueType) == sizeof(value_type_t<ValueType>),
                      "Can't handle C++ data type and corresponding Kokkos "
                      "type with mismatching type sizes.");

        assert_compatibility(m, MemorySpace{});

        auto size = m.get_size();

        return type<const ValueType>{
            reinterpret_cast<const value_type_t<ValueType>*>(
                m.get_const_values()),
            Kokkos::LayoutStride{size[0], m.get_stride(), size[1], 1}};
    }
};


template <typename ValueType, typename IndexType, typename MemorySpace>
struct mapper<device_matrix_data<ValueType, IndexType>, MemorySpace> {
    using index_mapper = mapper<array<IndexType>, MemorySpace>;
    using value_mapper = mapper<array<ValueType>, MemorySpace>;

    template <typename ValueType_c, typename IndexType_c>
    struct type {
        using index_array = typename index_mapper::template type<IndexType_c>;
        using value_array = typename value_mapper::template type<ValueType_c>;

        static type map(size_type size, IndexType_c* row_idxs,
                        IndexType_c* col_idxs, ValueType_c* values)
        {
            return {index_mapper::map(row_idxs, size),
                    index_mapper::map(col_idxs, size),
                    value_mapper::map(values, size)};
        }

        index_array row_idxs;
        index_array col_idxs;
        value_array values;
    };

    static type<ValueType, IndexType> map(
        device_matrix_data<ValueType, IndexType>& md)
    {
        assert_compatibility(md, MemorySpace{});
        return type<ValueType, IndexType>::map(
            md.get_num_stored_elements(), md.get_row_idxs(), md.get_col_idxs(),
            md.get_values());
    }

    static type<const ValueType, const IndexType> map(
        const device_matrix_data<ValueType, IndexType>& md)
    {
        assert_compatibility(md, MemorySpace{});
        return type<const ValueType, const IndexType>::map(
            md.get_num_stored_elements(), md.get_const_row_idxs(),
            md.get_const_col_idxs(), md.get_const_values());
    }
};


}  // namespace detail


//!< specialization of gko::native for Kokkos
template <typename MemorySpace>
struct kokkos_type {
    template <typename T>
    static auto map(T* data)
    {
        return map(*data);
    }

    template <typename T>
    static auto map(const std::unique_ptr<T>& data)
    {
        return map(*data);
    }

    template <typename T>
    static auto map(const std::shared_ptr<T>& data)
    {
        return map(*data);
    }

    template <typename T>
    static auto map(T&& data)
    {
        return detail::mapper<std::decay_t<T>, MemorySpace>::map(
            std::forward<T>(data));
    }
};


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
template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
inline auto map_data(T* data, MemorySpace = {})
{
    return kokkos_type<MemorySpace>::map(*data);
}

/**
 * @copydoc map_data(T*, MemorySpace)
 */
template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
inline auto map_data(std::unique_ptr<T>& data, MemorySpace = {})
{
    return kokkos_type<MemorySpace>::map(*data);
}

/**
 * @copydoc map_data(T*, MemorySpace)
 */
template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
inline auto map_data(std::shared_ptr<T>& data, MemorySpace = {})
{
    return kokkos_type<MemorySpace>::map(*data);
}

/**
 * @copydoc map_data(T*, MemorySpace)
 */
template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
inline auto map_data(T&& data, MemorySpace = {})
{
    return kokkos_type<MemorySpace>::map(data);
}


}  // namespace kokkos
}  // namespace ext
}  // namespace gko


#endif  // GINKGO_EXTENSION_KOKKOS
#endif  // GINKGO_TYPES_HPP
