// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_DEVICE_DENSE_HPP_
#define GKO_PUBLIC_CORE_MATRIX_DEVICE_DENSE_HPP_

#include <ginkgo/core/base/dim.hpp>


namespace gko {
namespace matrix {
namespace view {


/**
 * Non-owning view of a matrix::Dense to be used inside device kernels.
 * This type is used to provide a simple and stable ABI for passing data between
 * libraries.
 *
 * @tparam ValueType  the value type used to store matrix entries.
 */
template <typename ValueType>
struct dense {
    dim<2> size;
    size_type stride;
    ValueType* data;

    constexpr dense(dim<2> size, size_type stride, ValueType* data)
        : size{size}, stride{stride}, data{data}
    {}

    constexpr dense<const ValueType> as_const() const
    {
        return dense<const ValueType>{size, stride, data};
    }

    constexpr ValueType& operator()(size_type row, size_type col) const
    {
        return data[row * stride + col];
    }
};


}  // namespace view
}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_DEVICE_DENSE_HPP_
