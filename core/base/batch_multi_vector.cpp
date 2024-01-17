// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/batch_multi_vector.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/base/batch_multi_vector_kernels.hpp"


namespace gko {
namespace batch {
namespace multi_vector {
namespace {


GKO_REGISTER_OPERATION(scale, batch_multi_vector::scale);
GKO_REGISTER_OPERATION(add_scaled, batch_multi_vector::add_scaled);
GKO_REGISTER_OPERATION(compute_dot, batch_multi_vector::compute_dot);
GKO_REGISTER_OPERATION(compute_conj_dot, batch_multi_vector::compute_conj_dot);
GKO_REGISTER_OPERATION(compute_norm2, batch_multi_vector::compute_norm2);
GKO_REGISTER_OPERATION(copy, batch_multi_vector::copy);


}  // namespace
}  // namespace multi_vector


namespace detail {


template <typename ValueType>
batch_dim<2> compute_batch_size(
    const std::vector<gko::matrix::Dense<ValueType>*>& matrices)
{
    auto common_size = matrices[0]->get_size();
    for (size_type i = 1; i < matrices.size(); ++i) {
        GKO_ASSERT_EQUAL_DIMENSIONS(common_size, matrices[i]->get_size());
    }
    return batch_dim<2>{matrices.size(), common_size};
}


}  // namespace detail


template <typename ValueType>
std::unique_ptr<gko::matrix::Dense<ValueType>>
MultiVector<ValueType>::create_view_for_item(size_type item_id)
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto stride = this->get_common_size()[1];
    auto mat = unbatch_type::create(
        exec, this->get_common_size(),
        make_array_view(exec, num_rows * stride,
                        this->get_values_for_item(item_id)),
        stride);
    return mat;
}


template <typename ValueType>
std::unique_ptr<const gko::matrix::Dense<ValueType>>
MultiVector<ValueType>::create_const_view_for_item(size_type item_id) const
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto stride = this->get_common_size()[1];
    auto mat = unbatch_type::create_const(
        exec, this->get_common_size(),
        make_const_array_view(exec, num_rows * stride,
                              this->get_const_values_for_item(item_id)),
        stride);
    return mat;
}


template <typename ValueType>
MultiVector<ValueType>::MultiVector(std::shared_ptr<const Executor> exec,
                                    const batch_dim<2>& size)
    : EnablePolymorphicObject<MultiVector<ValueType>>(exec),
      batch_size_(size),
      values_(exec, compute_num_elems(size))
{}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_config_of(
    ptr_param<const MultiVector> other)
{
    // De-referencing `other` before calling the functions (instead of
    // using operator `->`) is currently required to be compatible with
    // CUDA 10.1.
    // Otherwise, it results in a compile error.
    return (*other).create_with_same_config();
}


template <typename ValueType>
std::unique_ptr<const MultiVector<ValueType>>
MultiVector<ValueType>::create_const(
    std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
    gko::detail::const_array_view<ValueType>&& values)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const MultiVector>(new MultiVector{
        exec, sizes, gko::detail::array_const_cast(std::move(values))});
}


template <typename ValueType>
void MultiVector<ValueType>::fill(ValueType value)
{
    GKO_ASSERT(this->values_.get_size() > 0);
    this->values_.fill(value);
}


template <typename ValueType>
void MultiVector<ValueType>::set_size(const batch_dim<2>& value) noexcept
{
    batch_size_ = value;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_same_config() const
{
    return MultiVector<ValueType>::create(this->get_executor(),
                                          this->get_size());
}


template <typename ValueType>
void MultiVector<ValueType>::scale(
    ptr_param<const MultiVector<ValueType>> alpha)
{
    GKO_ASSERT_EQ(alpha->get_num_batch_items(), this->get_num_batch_items());
    if (alpha->get_common_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this->get_common_size(),
                              alpha->get_common_size());
    }
    // element wise scaling requires same size
    if (alpha->get_common_size()[0] != 1) {
        GKO_ASSERT_EQUAL_DIMENSIONS(this->get_common_size(),
                                    alpha->get_common_size());
    }
    auto exec = this->get_executor();
    exec->run(multi_vector::make_scale(make_temporary_clone(exec, alpha).get(),
                                       this));
}


template <typename ValueType>
void MultiVector<ValueType>::add_scaled(
    ptr_param<const MultiVector<ValueType>> alpha,
    ptr_param<const MultiVector<ValueType>> b)
{
    GKO_ASSERT_EQ(alpha->get_num_batch_items(), this->get_num_batch_items());
    GKO_ASSERT_EQUAL_ROWS(alpha->get_common_size(), dim<2>(1, 1));
    if (alpha->get_common_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this->get_common_size(),
                              alpha->get_common_size());
    }
    GKO_ASSERT_EQ(b->get_num_batch_items(), this->get_num_batch_items());
    GKO_ASSERT_EQUAL_DIMENSIONS(this->get_common_size(), b->get_common_size());

    auto exec = this->get_executor();
    exec->run(multi_vector::make_add_scaled(
        make_temporary_clone(exec, alpha).get(),
        make_temporary_clone(exec, b).get(), this));
}


inline const batch_dim<2> get_col_sizes(const batch_dim<2>& sizes)
{
    return batch_dim<2>(sizes.get_num_batch_items(),
                        dim<2>(1, sizes.get_common_size()[1]));
}


template <typename ValueType>
void MultiVector<ValueType>::compute_conj_dot(
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<MultiVector<ValueType>> result) const
{
    GKO_ASSERT_EQ(b->get_num_batch_items(), this->get_num_batch_items());
    GKO_ASSERT_EQUAL_DIMENSIONS(this->get_common_size(), b->get_common_size());
    GKO_ASSERT_EQ(this->get_num_batch_items(), result->get_num_batch_items());
    GKO_ASSERT_EQUAL_DIMENSIONS(
        result->get_common_size(),
        get_col_sizes(this->get_size()).get_common_size());
    auto exec = this->get_executor();
    exec->run(multi_vector::make_compute_conj_dot(
        this, make_temporary_clone(exec, b).get(),
        make_temporary_output_clone(exec, result).get()));
}


template <typename ValueType>
void MultiVector<ValueType>::compute_dot(
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<MultiVector<ValueType>> result) const
{
    GKO_ASSERT_EQ(b->get_num_batch_items(), this->get_num_batch_items());
    GKO_ASSERT_EQUAL_DIMENSIONS(this->get_common_size(), b->get_common_size());
    GKO_ASSERT_EQ(this->get_num_batch_items(), result->get_num_batch_items());
    GKO_ASSERT_EQUAL_DIMENSIONS(
        result->get_common_size(),
        get_col_sizes(this->get_size()).get_common_size());
    auto exec = this->get_executor();
    exec->run(multi_vector::make_compute_dot(
        this, make_temporary_clone(exec, b).get(),
        make_temporary_output_clone(exec, result).get()));
}


template <typename ValueType>
void MultiVector<ValueType>::compute_norm2(
    ptr_param<MultiVector<remove_complex<ValueType>>> result) const
{
    GKO_ASSERT_EQ(this->get_num_batch_items(), result->get_num_batch_items());
    GKO_ASSERT_EQUAL_DIMENSIONS(
        result->get_common_size(),
        get_col_sizes(this->get_size()).get_common_size());

    auto exec = this->get_executor();
    exec->run(multi_vector::make_compute_norm2(
        this, make_temporary_output_clone(exec, result).get()));
}


template <typename ValueType>
void MultiVector<ValueType>::convert_to(
    MultiVector<next_precision<ValueType>>* result) const
{
    result->values_ = this->values_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void MultiVector<ValueType>::move_to(
    MultiVector<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


#define GKO_DECLARE_BATCH_MULTI_VECTOR(_type) class MultiVector<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_MULTI_VECTOR);


}  // namespace batch
}  // namespace gko
