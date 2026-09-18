// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_UTILS_HPP_
#define GKO_CORE_BASE_UTILS_HPP_


#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {


template <typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE ValueType checked_load(const ValueType* p,
                                                 IndexType i, IndexType size,
                                                 ValueType sentinel)
{
    return i < size ? p[i] : sentinel;
}


}  // namespace kernels


namespace detail {


template <typename Dest>
struct conversion_sort_helper {};

template <typename ValueType, typename IndexType>
struct conversion_sort_helper<matrix::Csr<ValueType, IndexType>> {
    using mtx_type = matrix::Csr<ValueType, IndexType>;
    template <typename Source>
    static std::unique_ptr<mtx_type> get_sorted_conversion(
        std::shared_ptr<const Executor>& exec, Source* source)
    {
        auto editable_mtx = mtx_type::create(exec);
        as<ConvertibleTo<mtx_type>>(source)->convert_to(editable_mtx);
        editable_mtx->sort_by_column_index();
        return editable_mtx;
    }
};


template <typename Dest, typename Source>
std::unique_ptr<Dest, std::function<void(Dest*)>> convert_to_with_sorting_impl(
    std::shared_ptr<const Executor>& exec, Source* obj, bool skip_sorting)
{
    if (skip_sorting) {
        return copy_and_convert_to<Dest>(exec, obj);
    } else {
        using decay_dest = std::decay_t<Dest>;
        auto sorted_mtx =
            detail::conversion_sort_helper<decay_dest>::get_sorted_conversion(
                exec, obj);
        return {sorted_mtx.release(), std::default_delete<Dest>()};
    }
}

template <typename Dest, typename Source>
std::shared_ptr<Dest> convert_to_with_sorting_impl(
    std::shared_ptr<const Executor>& exec, std::shared_ptr<Source> obj,
    bool skip_sorting)
{
    if (skip_sorting) {
        return copy_and_convert_to<Dest>(exec, obj);
    } else {
        using decay_dest = std::decay_t<Dest>;
        auto sorted_mtx =
            detail::conversion_sort_helper<decay_dest>::get_sorted_conversion(
                exec, obj.get());
        return {std::move(sorted_mtx)};
    }
}


}  // namespace detail


/**
 * @internal
 *
 * Helper function that converts the given matrix to the Dest format with
 * additional sorting if requested.
 *
 * If the given matrix was already sorted, is on the same executor and with a
 * dynamic type of `Dest`, the same pointer is returned with an empty
 * deleter.
 * In all other cases, a new matrix is created, which stores the converted
 * matrix.
 *
 * @tparam Dest  the type to which the object should be converted
 * @tparam Source  the type of the source object
 *
 * @param exec  the executor where the result should be placed
 * @param obj  the source object that should be converted
 * @param skip_sorting  indicator if the resulting matrix should be sorted or
 *                      not
 */
template <typename Dest, typename Source>
std::unique_ptr<Dest, std::function<void(Dest*)>> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, Source* obj, bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<Dest>(exec, obj, skip_sorting);
}

/**
 * @copydoc convert_to_with_sorting(std::shared_ptr<const Executor>,
 * Source *, bool)
 *
 * @note This version adds the const qualifier for the result since the input is
 *       also const
 */
template <typename Dest, typename Source>
std::unique_ptr<const Dest, std::function<void(const Dest*)>>
convert_to_with_sorting(std::shared_ptr<const Executor> exec, const Source* obj,
                        bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<const Dest>(exec, obj,
                                                            skip_sorting);
}

/**
 * @copydoc convert_to_with_sorting(std::shared_ptr<const Executor>,
 * Source *, bool)
 *
 * @note This version has a unique_ptr as the source instead of a plain pointer
 */
template <typename Dest, typename Source>
std::unique_ptr<Dest, std::function<void(Dest*)>> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, const std::unique_ptr<Source>& obj,
    bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<Dest>(exec, obj.get(),
                                                      skip_sorting);
}

/**
 * @internal
 *
 * Helper function that converts the given matrix to the Dest format with
 * additional sorting if requested.
 *
 * If the given matrix was already sorted, is on the same executor and with a
 * dynamic type of `Dest`, the same pointer is returned.
 * In all other cases, a new matrix is created, which stores the converted
 * matrix.
 *
 * @tparam Dest  the type to which the object should be converted
 * @tparam Source  the type of the source object
 *
 * @param exec  the executor where the result should be placed
 * @param obj  the source object that should be converted
 * @param skip_sorting  indicator if the resulting matrix should be sorted or
 *                      not
 */
template <typename Dest, typename Source>
std::shared_ptr<Dest> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, std::shared_ptr<Source> obj,
    bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<Dest>(exec, obj, skip_sorting);
}

/**
 * @copydoc convert_to_with_sorting(std::shared_ptr<const Executor>,
 * std::shared_ptr<Source>, bool)
 *
 * @note This version adds the const qualifier for the result since the input is
 *       also const
 */
template <typename Dest, typename Source>
std::shared_ptr<const Dest> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, std::shared_ptr<const Source> obj,
    bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<const Dest>(exec, obj,
                                                            skip_sorting);
}

/**
 * Converts the given arguments into an array of entries of the requested
 * template type.
 *
 * @tparam T  The requested type of entries in the output array.
 *
 * @param args  Entities to be filled into an array after casting to type T.
 */
template <typename T, typename... Args>
constexpr std::array<T, sizeof...(Args)> to_std_array(Args&&... args)
{
    return {static_cast<T>(args)...};
}


/**
 * Checks whether the sum of two non-negative integers is representable in
 * IndexType, without overflowing during the check.
 */
template <typename IndexType>
constexpr bool sum_fits(std::uint64_t a, std::uint64_t b)
{
    constexpr auto limit =
        static_cast<std::uint64_t>(std::numeric_limits<IndexType>::max());
    return a <= limit && b <= limit - a;
}


/**
 * Throws OverflowError if the sum of two non-negative integers cannot be
 * represented in IndexType. Unlike assertion-only checks, this also runs in
 * release builds.
 */
template <typename IndexType>
void ensure_sum_fits(std::uint64_t a, std::uint64_t b)
{
    if (!sum_fits<IndexType>(a, b)) {
        throw OverflowError{__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(IndexType))};
    }
}


/**
 * Checks whether the product of two non-negative integers is representable in
 * IndexType, without overflowing. A single argument checks that value itself.
 */
template <typename IndexType>
constexpr bool product_fits(std::uint64_t a, std::uint64_t b = 1)
{
    constexpr auto limit =
        static_cast<std::uint64_t>(std::numeric_limits<IndexType>::max());
    return a == 0 || b <= limit / a;
}


/**
 * Checks the last accessed offset of a dense view, excluding trailing padding.
 * Dimensions and stride are non-negative; check their storage types separately.
 */
template <typename IndexType>
constexpr bool dense_access_fits(std::uint64_t rows, std::uint64_t cols,
                                 std::uint64_t stride)
{
    return rows == 0 || cols == 0 ||
           (product_fits<IndexType>(rows - 1, stride) &&
            sum_fits<IndexType>((rows - 1) * stride, cols - 1));
}


/**
 * Checks the last accessed offset of contiguous square blocks. Block count
 * and size are non-negative; check dimension and stride storage separately.
 */
template <typename IndexType>
constexpr bool block_access_fits(std::uint64_t blocks, std::uint64_t block_size)
{
    return blocks == 0 || block_size == 0 ||
           (product_fits<std::uint64_t>(block_size, block_size) &&
            dense_access_fits<IndexType>(blocks, block_size * block_size,
                                         block_size * block_size));
}


/**
 * Checks both metadata storage and element offsets for a 2-D row-major
 * accessor. All dimensions and the stride must be non-negative.
 */
template <typename Accessor>
constexpr bool dense_accessor_fits(std::uint64_t rows, std::uint64_t cols,
                                   std::uint64_t stride)
{
    using size_type = typename Accessor::size_type;
    return product_fits<size_type>(rows) && product_fits<size_type>(cols) &&
           product_fits<size_type>(stride) &&
           dense_access_fits<typename Accessor::index_type>(rows, cols, stride);
}


/**
 * Checks metadata storage and element offsets for a 3-D block-column-major
 * accessor with contiguous square blocks. Counts and sizes are non-negative.
 */
template <typename Accessor>
constexpr bool block_accessor_fits(std::uint64_t blocks,
                                   std::uint64_t block_size)
{
    using size_type = typename Accessor::size_type;
    return product_fits<size_type>(blocks) &&
           product_fits<size_type>(block_size, block_size) &&
           block_access_fits<typename Accessor::index_type>(blocks, block_size);
}


}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_
