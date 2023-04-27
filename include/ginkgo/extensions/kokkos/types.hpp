//
// Created by marcel on 26.04.23.
//

#ifndef GINKGO_TYPES_HPP
#define GINKGO_TYPES_HPP

#include <ginkgo/config.hpp>


#if GINKGO_EXTENSION_KOKKOS


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>


namespace gko {
namespace ext {
namespace kokkos {
namespace detail {


template <typename T, typename MemorySpace>
struct native_type;


template <typename T>
struct native_value_type {
    using type = T;
};


template <typename T>
struct native_value_type<std::complex<T>> {
    using type = Kokkos::complex<T>;
};


template <typename ValueType, typename MemorySpace>
struct native_type<gko::array<ValueType>, MemorySpace> {
    native_type(gko::array<ValueType>& arr)
        : view(arr.get_data(), arr.get_num_elems())
    {}

    native_type(ValueType* data, gko::size_type num_elements)
        : view(data, num_elements)
    {}

    template <typename I>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const I& i) const
    {
        return view(i);
    }

    Kokkos::View<typename native_value_type<ValueType>::type*, MemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        view;
};


template <typename ValueType, typename MemorySpace>
struct native_type<const gko::array<ValueType>, MemorySpace> {
    native_type(const gko::array<ValueType>& arr)
        : view(arr.get_const_data(), arr.get_num_elems())
    {}

    native_type(const ValueType* data, gko::size_type num_elements)
        : view(data, num_elements)
    {}

    template <typename I>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const I& i) const
    {
        return view(i);
    }

    Kokkos::View<const typename native_value_type<ValueType>::type*,
                 MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        view;
};


template <typename ValueType, typename MemorySpace>
struct native_type<gko::matrix::Dense<ValueType>, MemorySpace> {
    native_type(gko::matrix::Dense<ValueType>& mtx)
        : values(mtx.get_values(), mtx.get_size()[0], mtx.get_size()[1])
    {}

    template <typename I1, typename I2>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const I1& i1,
                                                     const I2& i2) const
    {
        return values(i1, i2);
    }

    Kokkos::View<typename native_value_type<ValueType>::type**,
                 Kokkos::LayoutRight, MemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        values;
};


template <typename ValueType, typename MemorySpace>
struct native_type<const gko::matrix::Dense<ValueType>, MemorySpace> {
    native_type(const gko::matrix::Dense<ValueType>& mtx)
        : values(mtx.get_const_values(), mtx.get_size()[0], mtx.get_size()[1])
    {}

    template <typename I1, typename I2>
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const I1& i1,
                                                     const I2& i2) const
    {
        return values(i1, i2);
    }

    Kokkos::View<const typename native_value_type<ValueType>::type**,
                 Kokkos::LayoutRight, MemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        values;
};


template <typename ValueType, typename IndexType, typename MemorySpace>
struct native_type<gko::device_matrix_data<ValueType, IndexType>, MemorySpace> {
    native_type(gko::device_matrix_data<ValueType, IndexType>& md)
        : row_idxs(md.get_row_idxs(), md.get_num_elems()),
          col_idxs(md.get_col_idxs(), md.get_num_elems()),
          values(md.get_values(), md.get_num_elems())
    {}

    native_type<gko::array<IndexType>, MemorySpace> row_idxs;
    native_type<gko::array<IndexType>, MemorySpace> col_idxs;
    native_type<gko::array<ValueType>, MemorySpace> values;
};


}  // namespace detail


template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
detail::native_type<T, MemorySpace> map_data(T& data, MemorySpace ms = {})
{
    return detail::native_type<T, MemorySpace>{data};
}


template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
detail::native_type<const T, MemorySpace> map_data(const T& data,
                                                   MemorySpace ms = {})
{
    return detail::native_type<const T, MemorySpace>{data};
}


template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
detail::native_type<T, MemorySpace> map_data(T* data, MemorySpace ms = {})
{
    return detail::native_type<T, MemorySpace>{*data};
}


template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
detail::native_type<const T, MemorySpace> map_data(const T* data,
                                                   MemorySpace ms = {})
{
    return detail::native_type<const T, MemorySpace>{*data};
}


}  // namespace kokkos
}  // namespace ext
}  // namespace gko


#endif  // GINKGO_EXTENSION_KOKKOS
#endif  // GINKGO_TYPES_HPP
