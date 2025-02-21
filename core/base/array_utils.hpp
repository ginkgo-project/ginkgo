// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_ARRAY_UTILS_HPP_
#define GKO_CORE_BASE_ARRAY_UTILS_HPP_


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


/**
 * Resizes the array so that it is able to hold the specified number of
 * elements, copying over existing elements into the new allocation.
 * For a view or other non-owning array types, this throws an exception
 * since these types cannot be resized.
 * The first `copy_size` elements stored in the array will be copied to the
 * beginning of the new allocation.
 *
 * If the array is not assigned an executor, an exception will be thrown.
 *
 * @param arr  the array to extend
 * @param new_size  the number of elements in the new allocation
 * @param copy_size  the number of elements to copy over from the old
 *                   allocation.
 */
template <typename ValueType>
void extend_array(array<ValueType>& arr, size_type new_size,
                  size_type copy_size)
{
    if (new_size == arr.get_size()) {
        return;
    }
    const auto exec = arr.get_executor();
    if (exec == nullptr) {
        throw gko::NotSupported(__FILE__, __LINE__, __func__,
                                "gko::Executor (nullptr)");
    }
    if (!arr.is_owning()) {
        throw gko::NotSupported(__FILE__, __LINE__, __func__,
                                "Non owning gko::array cannot be extended.");
    }
    if (new_size < arr.get_size()) {
        throw gko::NotSupported(__FILE__, __LINE__, __func__,
                                "array::extend cannot shrink an array.");
    }
    if (copy_size > arr.get_size()) {
        throw gko::NotSupported(
            __FILE__, __LINE__, __func__,
            "Attempting to copy more elements than available.");
    }
    // TODO2.0 check for negative size
    array<ValueType> new_array{exec, new_size};
    exec->copy(copy_size, arr.get_const_data(), new_array.get_data());
    arr = std::move(new_array);
}


/**
 * Resizes the array so that it is able to hold the specified number of
 * elements, copying over all existing elements into the new allocation.
 * For a view or other non-owning array types, this throws an exception
 * since these types cannot be resized.
 *
 * If the array is not assigned an executor, an exception will be thrown.
 *
 * @param arr  the array to extend
 * @param new_size  the number of elements in the new allocation.
 */
template <typename ValueType>
void extend_array(array<ValueType>& arr, size_type new_size)
{
    extend_array(arr, new_size, arr.get_size());
}


}  // namespace gko


#endif  // GKO_CORE_BASE_ARRAY_UTILS_HPP_
