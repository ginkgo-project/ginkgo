// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_ARRAY_ACCESS_HPP_
#define GKO_CORE_BASE_ARRAY_ACCESS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


/**
 * Returns a single value from an array.
 *
 * This involves a bounds check, polymorphic calls and potentially a
 * device-to-host copy, so it is not suitable for accessing many elements
 * in performance-critical code.
 *
 * @param array  the array to get the element from.
 * @param index  the array element index.
 * @tparam ValueType  the value type of the array.
 * @return the value at index.
 */
template <typename ValueType>
ValueType get_element(const array<ValueType>& array, size_type index)
{
    // TODO2.0 add bounds check for negative indices
    GKO_ENSURE_IN_BOUNDS(index, array.get_size());
    return array.get_executor()->copy_val_to_host(array.get_const_data() +
                                                  index);
}


/**
 * Sets a single entry in the array to a new value.
 *
 * This involves a bounds check, polymorphic calls and potentially a
 * host-to-device copy, so it is not suitable for accessing many elements
 * in performance-critical code.
 *
 * @param array  the array to set the element in.
 * @param index  the array element index.
 * @param value  the new value.
 * @tparam ValueType  the value type of the array.
 * @tparam ParameterType  the type of the value to be assigned.
 */
template <typename ValueType, typename ParameterType>
void set_element(array<ValueType>& array, size_type index, ParameterType value)
{
    auto converted_value = static_cast<ValueType>(value);
    // TODO2.0 add bounds check for negative indices
    GKO_ENSURE_IN_BOUNDS(index, array.get_size());
    auto exec = array.get_executor();
    exec->copy_from(exec->get_master(), 1, &converted_value,
                    array.get_data() + index);
}


}  // namespace gko


#endif  // GKO_CORE_BASE_ARRAY_ACCESS_HPP_
