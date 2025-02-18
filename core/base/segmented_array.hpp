// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_SEGMENTED_ARRAY_HPP
#define GINKGO_SEGMENTED_ARRAY_HPP


#include <ginkgo/core/base/segmented_array.hpp>


namespace gko {


/**
 * Helper struct to handle segmented arrays in device kernels
 *
 * @TODO: replace by segmented ranges when available
 *
 * @tparam T  the value type of the array, maybe const-qualified
 */
template <typename T>
struct device_segmented_array {
    /**
     * Helper struct storing a single segment
     */
    struct segment {
        T* begin;
        T* end;
    };

    constexpr segment get_segment(size_type segment_id)
    {
        GKO_ASSERT(segment_id < (offsets_end - offsets_begin));
        return {flat_begin + offsets_begin[segment_id],
                flat_begin + offsets_begin[segment_id + 1]};
    }

    T* flat_begin;
    T* flat_end;
    const int64* offsets_begin;
    const int64* offsets_end;
};

/**
 * Create device_segmented_array from a segmented_array.
 */
template <typename T>
constexpr device_segmented_array<T> to_device(segmented_array<T>& sarr)
{
    return {sarr.get_flat_data(), sarr.get_flat_data() + sarr.get_size(),
            sarr.get_offsets().get_const_data(),
            sarr.get_offsets().get_const_data() + sarr.get_segment_count()};
}


/**
 * Create device_segmented_array from a segmented_array.
 */
template <typename T>
constexpr device_segmented_array<const T> to_device(
    const segmented_array<T>& sarr)
{
    return {sarr.get_const_flat_data(),
            sarr.get_const_flat_data() + sarr.get_size(),
            sarr.get_offsets().get_const_data(),
            sarr.get_offsets().get_const_data() + sarr.get_segment_count()};
}

/**
 * Explicitly create a const version of device_segmented_array.
 *
 * This is mostly relevant for tests.
 */
template <typename T>
constexpr device_segmented_array<const T> to_device_const(
    const segmented_array<T>& sarr)
{
    return {sarr.get_const_flat_data(),
            sarr.get_const_flat_data() + sarr.get_size(),
            sarr.get_offsets().get_const_data(),
            sarr.get_offsets().get_const_data() + sarr.get_segment_count()};
}


}  // namespace gko

#endif  // GINKGO_SEGMENTED_ARRAY_HPP
