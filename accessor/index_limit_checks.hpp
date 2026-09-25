// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_INDEX_LIMIT_CHECKS_HPP_
#define GKO_ACCESSOR_INDEX_LIMIT_CHECKS_HPP_


#include <cstdint>
#include <limits>


namespace gko {
namespace acc {


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


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_INDEX_LIMIT_CHECKS_HPP_
