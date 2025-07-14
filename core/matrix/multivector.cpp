// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/multivector.hpp>

namespace gko {
namespace matrix {


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_config_of(
    ptr_param<const MultiVector> other)
{
    return other->create_with_same_config_impl();
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_type_of(
    ptr_param<const MultiVector> other, std::shared_ptr<const Executor> exec)
{
    return other->create_with_type_of_impl(std::move(exec), {}, {}, 0);
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_type_of(
    ptr_param<const MultiVector> other, std::shared_ptr<const Executor> exec,
    const dim<2>& global_size, const dim<2>& local_size)
{
    GKO_ASSERT_EQUAL_COLS(global_size, local_size);
    return other->create_with_type_of_impl(std::move(exec), global_size,
                                           local_size, global_size[1]);
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_type_of(
    ptr_param<const MultiVector> other, std::shared_ptr<const Executor> exec,
    const dim<2>& global_size, const dim<2>& local_size, size_type stride)
{
    return other->create_with_type_of_impl(std::move(exec), global_size,
                                           local_size, stride);
}


template <typename ValueType>
dim<2> MultiVector<ValueType>::get_size() const noexcept
{
    return LinOp::get_size();
}


template <typename ValueType>
void MultiVector<ValueType>::set_size(const dim<2>& size) noexcept
{
    LinOp::set_size(size);
}


}  // namespace matrix
}  // namespace gko
