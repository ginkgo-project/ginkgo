// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_UNINITIALIZED_ARRAY_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_UNINITIALIZED_ARRAY_HPP_


#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/thrust.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


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
        return data_;
    }

    /**
     * Operator for casting an uninitialized_array into its non-const value
     * pointer.
     *
     * @return the non-const pointer to the first entry of the array.
     */
    GKO_ATTRIBUTES operator ValueType*() noexcept { return data_; }

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
    GKO_ATTRIBUTES ValueType& operator[](size_type pos) noexcept
    {
        return data_[pos];
    }

private:
    ValueType data_[size];
};


template <typename ValueType, size_type size>
class uninitialized_array<thrust::complex<ValueType>, size> {
public:
    constexpr GKO_ATTRIBUTES operator const thrust::complex<ValueType>*()
        const noexcept
    {
        return &(*this)[0];
    }

    GKO_ATTRIBUTES operator thrust::complex<ValueType>*() noexcept
    {
        return &(*this)[0];
    }

    constexpr GKO_ATTRIBUTES const thrust::complex<ValueType>& operator[](
        size_type pos) const noexcept
    {
        return reinterpret_cast<const thrust::complex<ValueType>*>(data_)[pos];
    }

    GKO_ATTRIBUTES thrust::complex<ValueType>& operator[](
        size_type pos) noexcept
    {
        return reinterpret_cast<thrust::complex<ValueType>*>(data_)[pos];
    }

private:
    ValueType data_[2 * size];
};


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_UNINITIALIZED_ARRAY_HPP_
