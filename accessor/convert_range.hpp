// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_CONVERT_RANGE_HPP_
#define GKO_ACCESSOR_CONVERT_RANGE_HPP_


#include "block_col_major.hpp"
#include "range.hpp"
#include "reduced_row_major.hpp"
#include "row_major.hpp"
#include "scaled_reduced_row_major.hpp"
#include "utils.hpp"


namespace gko {
namespace acc {
namespace detail {


/**
 * Reinterprets a pointer to `T` as a pointer to `TypeMap<T>::type`.
 * `TypeMap` is a backend type trait such as `cuda_type`, which also
 * keeps const qualifiers.
 */
template <template <typename> class TypeMap, typename T>
GKO_ACC_INLINE auto convert_pointer(T* ptr)
{
    return reinterpret_cast<typename TypeMap<T>::type*>(ptr);
}


/**
 * Returns a range of the same accessor kind, with every value type `T`
 * replaced by `TypeMap<T>::type` and every pointer reinterpreted the same way.
 * Each backend (CUDA, HIP, SYCL) passes in only its type trait.
 *
 * Dimensions are deduced as `auto` because the accessors disagree on the type:
 * row_major uses `size_type`, the others use `std::size_t`.
 */
template <template <typename> class TypeMap, typename T, auto dim>
GKO_ACC_INLINE auto convert_range(const range<row_major<T, dim>>& r)
{
    return range<row_major<typename TypeMap<T>::type, dim>>(
        r.get_accessor().lengths,
        convert_pointer<TypeMap>(r.get_accessor().data),
        r.get_accessor().stride);
}

template <template <typename> class TypeMap, typename T, auto dim>
GKO_ACC_INLINE auto convert_range(const range<block_col_major<T, dim>>& r)
{
    return range<block_col_major<typename TypeMap<T>::type, dim>>(
        r.get_accessor().lengths,
        convert_pointer<TypeMap>(r.get_accessor().data),
        r.get_accessor().stride);
}

template <template <typename> class TypeMap, auto dim, typename Type1,
          typename Type2>
GKO_ACC_INLINE auto convert_range(
    const range<reduced_row_major<dim, Type1, Type2>>& r)
{
    return range<reduced_row_major<dim, typename TypeMap<Type1>::type,
                                   typename TypeMap<Type2>::type>>(
        r.get_accessor().get_size(),
        convert_pointer<TypeMap>(r.get_accessor().get_stored_data()),
        r.get_accessor().get_stride());
}

template <template <typename> class TypeMap, auto dim, typename Type1,
          typename Type2, std::uint64_t mask>
GKO_ACC_INLINE auto convert_range(
    const range<scaled_reduced_row_major<dim, Type1, Type2, mask>>& r)
{
    return range<scaled_reduced_row_major<dim, typename TypeMap<Type1>::type,
                                          typename TypeMap<Type2>::type, mask>>(
        r.get_accessor().get_size(),
        convert_pointer<TypeMap>(r.get_accessor().get_stored_data()),
        r.get_accessor().get_storage_stride(),
        convert_pointer<TypeMap>(r.get_accessor().get_scalar()),
        r.get_accessor().get_scalar_stride());
}


}  // namespace detail
}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_CONVERT_RANGE_HPP_
