// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_UNINITIALIZED_ARRAY_HPP_
#define GKO_DPCPP_COMPONENTS_UNINITIALIZED_ARRAY_HPP_


#include <ginkgo/core/base/types.hpp>


#include "dpcpp/base/dpct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


// TODO: porting - consider directly use the array as shared memory


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
    constexpr __dpct_inline__ operator const ValueType*() const noexcept
    {
        return &(*this)[0];
    }

    /**
     * Operator for casting an uninitialized_array into its non-const value
     * pointer.
     *
     * @return the non-const pointer to the first entry of the array.
     */
    __dpct_inline__ operator ValueType*() noexcept { return &(*this)[0]; }

    /**
     * constexpr array access operator.
     *
     * @param pos The array index. Using a value outside [0, size) is undefined
     * behavior.
     *
     * @return a reference to the array entry at the given index.
     */
    constexpr __dpct_inline__ const ValueType& operator[](
        size_type pos) const noexcept
    {
        return data_[pos];
    }

    /**
     * Non-const array access operator.
     *
     * @param pos The array index. Using a value outside [0, size) is undefined
     * behavior.
     *
     * @return a reference to the array entry at the given index.
     */
    __dpct_inline__ ValueType& operator[](size_type pos) noexcept
    {
        return data_[pos];
    }

private:
    // if dpcpp uses char to represent data in char, compiling gives error.
    // Thanksfully, dpcpp support complex data allocation directly.
    ValueType data_[size];
};


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_UNINITIALIZED_ARRAY_HPP_
