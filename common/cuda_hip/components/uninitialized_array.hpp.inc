// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * Stores an array with uninitialized contents.
 *
 * This class is needed for datatypes that do have a non-empty constructor when
 * using them as shared memory, for example `thrust::complex<float>`.
 *
 * @tparam ValueType the type of values
 * @tparam size the size of the array
 */
template <typename ValueType, size_type size>
class uninitialized_array {
public:
    /**
     * Operator for casting an uninitialized_array into its constexpr value
     * pointer.
     *
     * @return the constexpr pointer to the first entry of the array.
     */
    constexpr GKO_ATTRIBUTES operator const ValueType*() const noexcept
    {
        return &(*this)[0];
    }

    /**
     * Operator for casting an uninitialized_array into its non-const value
     * pointer.
     *
     * @return the non-const pointer to the first entry of the array.
     */
    GKO_ATTRIBUTES operator ValueType*() noexcept { return &(*this)[0]; }

    /**
     * constexpr array access operator.
     *
     * @param pos The array index. Using a value outside [0, size) is undefined
     * behavior.
     *
     * @return a reference to the array entry at the given index.
     */
    constexpr GKO_ATTRIBUTES const ValueType& operator[](
        size_type pos) const noexcept
    {
        return reinterpret_cast<const ValueType*>(data_)[pos];
    }

    /**
     * Non-const array access operator.
     *
     * @param pos The array index. Using a value outside [0, size) is undefined
     * behavior.
     *
     * @return a reference to the array entry at the given index.
     */
    GKO_ATTRIBUTES ValueType& operator[](size_type pos) noexcept
    {
        return reinterpret_cast<ValueType*>(data_)[pos];
    }

private:
    unsigned char data_[sizeof(ValueType) / sizeof(unsigned char) * size];
};
