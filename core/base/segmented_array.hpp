// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_SEGMENTED_ARRAY_HPP
#define GINKGO_SEGMENTED_ARRAY_HPP


#include <ginkgo/core/base/segmented_array.hpp>


namespace gko {


/**
 * Helper struct storing an array segment
 *
 * @tparam T  The value type of the array
 */
template <typename T>
struct array_segment {
    T* begin;
    T* end;
};


/**
 * Helper function to create a device-compatible view of an array segment.
 */
template <typename T>
constexpr array_segment<T> get_array_segment(segmented_array<T>& sarr,
                                             size_type segment_id)
{
    assert(segment_id < sarr.get_segment_count());
    auto offsets = sarr.get_offsets().get_const_data();
    auto data = sarr.get_flat_data();
    return {data + offsets[segment_id], data + offsets[segment_id + 1]};
}


/**
 * Helper function to create a device-compatible view of a const array segment.
 */
template <typename T>
constexpr array_segment<const T> get_array_segment(
    const segmented_array<T>& sarr, size_type segment_id)
{
    assert(segment_id < sarr.get_segment_count());
    auto offsets = sarr.get_offsets().get_const_data();
    auto data = sarr.get_const_flat_data();
    return {data + offsets[segment_id], data + offsets[segment_id + 1]};
}


}  // namespace gko

#endif  // GINKGO_SEGMENTED_ARRAY_HPP
