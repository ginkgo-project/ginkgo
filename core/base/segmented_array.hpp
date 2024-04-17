// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_SEGMENTED_ARRAY_HPP
#define GINKGO_SEGMENTED_ARRAY_HPP

#include <ginkgo/core/base/segmented_array.hpp>


#include "core/base/segmented_range.hpp"


namespace gko {
namespace device {


template <typename T>
using segmented_array = segmented_value_range<T*, const int64*>;


}


template <typename T>
device::segmented_array<T> map_to_device(segmented_array<T>& seg_array)
{
    return device::segmented_array<T>{
        seg_array.get_flat().get_data(),
        seg_array.get_offsets().get_const_data(),
        seg_array.get_offsets().get_const_data() + 1,
        static_cast<int64>(seg_array.size())};
}


template <typename T>
device::segmented_array<const T> map_to_device(
    const segmented_array<T>& seg_array)
{
    return device::segmented_array<const T>{
        seg_array.get_flat().get_const_data(),
        seg_array.get_offsets().get_const_data(),
        seg_array.get_offsets().get_const_data() + 1,
        static_cast<int64>(seg_array.size())};
}


template <typename T>
device::segmented_array<const T> map_to_device_const(
    segmented_array<T>& seg_array)
{
    return map_to_device(const_cast<const segmented_array<T>&>(seg_array));
}


}  // namespace gko


#endif  // GINKGO_SEGMENTED_ARRAY_HPP
