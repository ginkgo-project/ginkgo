/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GINKGO_TYPES_HPP
#define GINKGO_TYPES_HPP

#include <ginkgo/config.hpp>


#if GINKGO_EXTENSION_KOKKOS


#include <ginkgo/core/base/native_type.hpp>


#include <ginkgo/extensions/kokkos/spaces.hpp>


#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>


namespace gko {
namespace ext {
namespace kokkos {


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


namespace detail {


template <typename MemorySpace>
struct EnableKokkosCompatibility {
    template <typename T>
    static void check_compatibility(T&& obj)
    {
        detail::assert_compatibility(std::forward<T>(obj), MemorySpace{});
    }
};


}  // namespace detail


/**
 * Type that maps a Ginkgo array to an unmanaged 1D Kokkos::View.
 *
 * @warning Using std::complex as data type might lead to issues, since the
 *          alignment of Kokkos::complex is not necessarily the same.
 *
 * @tparam MemorySpace  The memory space type the mapped object should use.
 */
template <typename MemorySpace>
struct array_mapper : detail::EnableKokkosCompatibility<MemorySpace> {
    template <typename ValueType>
    using type =
        Kokkos::View<typename value_type<ValueType>::type*, MemorySpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    template <typename ValueType>
    static type<ValueType> map(ValueType* data, size_type size)
    {
        static_assert(sizeof(ValueType) == sizeof(value_type_t<ValueType>),
                      "Can't handle C++ data type and corresponding Kokkos "
                      "type with mismatching type sizes.");
        // a similar check for alignment is not possible, since the alignment
        // of kokkos::complex can be changed, but not through spack. Thus
        // changing the alignment after an assertion failure requires building
        // kokkos from source.

        return type<ValueType>{reinterpret_cast<value_type_t<ValueType>*>(data),
                               size};
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
template <typename MemorySpace>
struct dense_mapper : detail::EnableKokkosCompatibility<MemorySpace> {
    template <typename ValueType>
    using type = Kokkos::View<typename value_type<ValueType>::type**,
                              Kokkos::LayoutStride, MemorySpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    template <typename ValueType>
    static type<ValueType> map(ValueType* data, gko::dim<2> size,
                               size_type stride)
    {
        static_assert(sizeof(ValueType) == sizeof(value_type_t<ValueType>),
                      "Can't handle C++ data type and corresponding Kokkos "
                      "type with mismatching type sizes.");

        return type<ValueType>{
            reinterpret_cast<value_type_t<ValueType>*>(data),
            Kokkos::LayoutStride{size[0], stride, size[1], 1}};
    }
};


//!< specialization of gko::native for Kokkos
template <typename MemorySpace>
using kokkos_type =
    gko::native<array_mapper<MemorySpace>, dense_mapper<MemorySpace>>;


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
