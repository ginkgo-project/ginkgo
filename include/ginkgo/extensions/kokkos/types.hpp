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


#include <ginkgo/extensions/kokkos/spaces.hpp>


#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>


namespace gko {
namespace ext {
namespace kokkos {
namespace detail {


template <typename T, typename MemorySpace>
struct native_type {
    using type = T;
};


template <typename T, typename MemorySpace>
struct native_type<std::complex<T>, MemorySpace> {
    using type = Kokkos::complex<T>;
};


template <typename ValueType, typename MemorySpace>
struct native_type<gko::array<ValueType>, MemorySpace> {
    struct type {
        type(gko::array<ValueType>& arr)
            : view(arr.get_data(), arr.get_num_elems())
        {}

        type(ValueType* data, gko::size_type num_elements)
            : view(data, num_elements)
        {}

        template <typename I>
        KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const I& i) const
        {
            return view(i);
        }

        Kokkos::View<typename native_type<ValueType, MemorySpace>::type*,
                     MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            view;
    };
};


template <typename ValueType, typename MemorySpace>
struct native_type<const gko::array<ValueType>, MemorySpace> {
    struct type {
        type(const gko::array<ValueType>& arr)
            : view(arr.get_const_data(), arr.get_num_elems())
        {}

        type(const ValueType* data, gko::size_type num_elements)
            : view(data, num_elements)
        {}

        template <typename I>
        KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const I& i) const
        {
            return view(i);
        }

        Kokkos::View<const typename native_type<ValueType, MemorySpace>::type*,
                     MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            view;
    };
};


template <typename ValueType, typename MemorySpace>
struct native_type<gko::matrix::Dense<ValueType>, MemorySpace> {
    struct type {
        type(gko::matrix::Dense<ValueType>& mtx)
            : values(mtx.get_values(), mtx.get_size()[0], mtx.get_size()[1])
        {}

        template <typename I1, typename I2>
        KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const I1& i1,
                                                         const I2& i2) const
        {
            return values(i1, i2);
        }

        Kokkos::View<typename native_type<ValueType, MemorySpace>::type**,
                     Kokkos::LayoutRight, MemorySpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            values;
    };
};


template <typename ValueType, typename MemorySpace>
struct native_type<const gko::matrix::Dense<ValueType>, MemorySpace> {
    struct type {
        type(const gko::matrix::Dense<ValueType>& mtx)
            : values(mtx.get_const_values(), mtx.get_size()[0],
                     mtx.get_size()[1])
        {}

        template <typename I1, typename I2>
        KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const I1& i1,
                                                         const I2& i2) const
        {
            return values(i1, i2);
        }

        Kokkos::View<const typename native_type<ValueType, MemorySpace>::type**,
                     Kokkos::LayoutRight, MemorySpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            values;
    };
};


template <typename ValueType, typename IndexType, typename MemorySpace>
struct native_type<gko::device_matrix_data<ValueType, IndexType>, MemorySpace> {
    struct type {
        type(gko::device_matrix_data<ValueType, IndexType>& md)
            : row_idxs(md.get_row_idxs(), md.get_num_elems()),
              col_idxs(md.get_col_idxs(), md.get_num_elems()),
              values(md.get_values(), md.get_num_elems())
        {}

        typename native_type<gko::array<IndexType>, MemorySpace>::type row_idxs;
        typename native_type<gko::array<IndexType>, MemorySpace>::type col_idxs;
        typename native_type<gko::array<ValueType>, MemorySpace>::type values;
    };
};


}  // namespace detail


template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
decltype(auto) map_data(T* data, MemorySpace ms = {})
{
    if constexpr (has_executor_v<T>) {
        //        ensure_compatibility(data, ms);
        return typename detail::native_type<T, MemorySpace>::type{*data};
    } else {
        return data;
    }
}


template <typename T,
          typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
decltype(auto) map_data(T&& data, MemorySpace ms = {})
{
    if constexpr (has_executor_v<T>) {
        //        ensure_compatibility(data, ms);
        return typename detail::native_type<T, MemorySpace>::type{
            std::forward<T>(data)};
    } else {
        return std::forward<T>(data);
    }
}


template <typename ConcreteType>
struct EnableGinkgoData {
    template <typename... Args>
    static ConcreteType create(Args&&... args)
    {
        return {map_data(std::forward<Args>(args))...};
    }
};


}  // namespace kokkos
}  // namespace ext
}  // namespace gko


#endif  // GINKGO_EXTENSION_KOKKOS
#endif  // GINKGO_TYPES_HPP
