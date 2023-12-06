// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_DEVICE_REFERENCE_HPP_
#define GKO_PUBLIC_CORE_BASE_DEVICE_REFERENCE_HPP_


#include <ginkgo/core/base/executor.hpp>


namespace gko {


/**
 * Reference-like type referencing a value in an executor's memory space.
 *
 * @tparam ValueType  the referenced type
 */
template <typename ValueType>
class device_reference {
public:
    using value_type = ValueType;

    /** Constructs a device reference from its executor and pointer. */
    explicit device_reference(const Executor* exec, ValueType* ptr)
        : exec_{exec}, ptr_{ptr}
    {}

    /** Copies the value at the referenced location to the host to return it.*/
    operator value_type() const { return exec_->copy_val_to_host(ptr_); }

    /** Copies the given value to the referenced location. */
    device_reference& operator=(const value_type& other)
    {
        exec_->copy_from(exec_->get_master(), 1, &other, ptr_);
        return *this;
    }

    /** Copies the value from one referenced location to another. */
    device_reference& operator=(const device_reference& other)
    {
        exec_->copy_from(other.exec_, 1, other.ptr_, ptr_);
        return *this;
    }

    value_type copy() const { return *this; }

private:
    const Executor* exec_;
    ValueType* ptr_;
};


/**
 * Swap function for device references. It takes care of creating a
 * non-reference temporary to avoid the problem of a normal std::swap():
 * ```
 * // a and b are reference-like objects pointing to different entries
 * auto tmp = a; // tmp is a reference-like type, so this is not a copy!
 * a = b;        // copies value at b to a, which also modifies tmp
 * b = tmp;      // copies value at tmp (= a) to b
 * // now both a and b point to the same value that was originally at b
 * ```
 * It is modelled after the behavior of std::vector<bool> bit references.
 * To swap in generic code, use the pattern `using std::swap; swap(a, b);`
 *
 * @tparam ValueType  the value type inside the corresponding device_iterator
 */
template <typename ValueType>
void swap(device_reference<ValueType> a, device_reference<ValueType> b)
{
    auto tmp = a.copy();
    a = b;
    b = tmp;
}


/**
 * @copydoc swap(device_reference, device_reference)
 */
template <typename ValueType>
void swap(typename device_reference<ValueType>::value_type& a,
          device_reference<ValueType> b)
{
    auto tmp = a;
    a = b;
    b = tmp;
}


/**
 * @copydoc swap(device_reference, device_reference)
 */
template <typename ValueType>
void swap(device_reference<ValueType> a,
          typename device_reference<ValueType>::value_type& b)
{
    auto tmp = a.copy();
    a = b;
    b = tmp;
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_DEVICE_REFERENCE_HPP_
