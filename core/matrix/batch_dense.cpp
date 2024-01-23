// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/batch_dense.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/batch_dense_kernels.hpp"


namespace gko {
namespace batch {
namespace matrix {
namespace dense {
namespace {


GKO_REGISTER_OPERATION(simple_apply, batch_dense::simple_apply);
GKO_REGISTER_OPERATION(advanced_apply, batch_dense::advanced_apply);
GKO_REGISTER_OPERATION(scale, batch_dense::scale);
GKO_REGISTER_OPERATION(scale_add, batch_dense::scale_add);
GKO_REGISTER_OPERATION(add_scaled_identity, batch_dense::add_scaled_identity);


}  // namespace
}  // namespace dense


template <typename ValueType>
std::unique_ptr<gko::matrix::Dense<ValueType>>
Dense<ValueType>::create_view_for_item(size_type item_id)
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
Dense<ValueType>::create_const_view_for_item(size_type item_id) const
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
std::unique_ptr<const Dense<ValueType>> Dense<ValueType>::create_const(
    std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
    gko::detail::const_array_view<ValueType>&& values)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const Dense>(new Dense{
        exec, sizes, gko::detail::array_const_cast(std::move(values))});
}


template <typename ValueType>
Dense<ValueType>::Dense(std::shared_ptr<const Executor> exec,
                        const batch_dim<2>& size)
    : EnableBatchLinOp<Dense<ValueType>>(exec, size),
      values_(exec, compute_num_elems(size))
{}


template <typename ValueType>
Dense<ValueType>* Dense<ValueType>::apply(
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<MultiVector<ValueType>> x)
{
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    return this;
}


template <typename ValueType>
const Dense<ValueType>* Dense<ValueType>::apply(
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<MultiVector<ValueType>> x) const
{
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    return this;
}


template <typename ValueType>
Dense<ValueType>* Dense<ValueType>::apply(
    ptr_param<const MultiVector<ValueType>> alpha,
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<const MultiVector<ValueType>> beta,
    ptr_param<MultiVector<ValueType>> x)
{
    this->validate_application_parameters(alpha.get(), b.get(), beta.get(),
                                          x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, alpha).get(),
                     make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, beta).get(),
                     make_temporary_clone(exec, x).get());
    return this;
}


template <typename ValueType>
const Dense<ValueType>* Dense<ValueType>::apply(
    ptr_param<const MultiVector<ValueType>> alpha,
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<const MultiVector<ValueType>> beta,
    ptr_param<MultiVector<ValueType>> x) const
{
    this->validate_application_parameters(alpha.get(), b.get(), beta.get(),
                                          x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, alpha).get(),
                     make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, beta).get(),
                     make_temporary_clone(exec, x).get());
    return this;
}


template <typename ValueType>
void Dense<ValueType>::apply_impl(const MultiVector<ValueType>* b,
                                  MultiVector<ValueType>* x) const
{
    this->get_executor()->run(dense::make_simple_apply(this, b, x));
}


template <typename ValueType>
void Dense<ValueType>::apply_impl(const MultiVector<ValueType>* alpha,
                                  const MultiVector<ValueType>* b,
                                  const MultiVector<ValueType>* beta,
                                  MultiVector<ValueType>* x) const
{
    this->get_executor()->run(
        dense::make_advanced_apply(alpha, this, b, beta, x));
}


template <typename ValueType>
void Dense<ValueType>::scale(const array<ValueType>& row_scale,
                             const array<ValueType>& col_scale)
{
    GKO_ASSERT_EQ(col_scale.get_size(),
                  (this->get_common_size()[1] * this->get_num_batch_items()));
    GKO_ASSERT_EQ(row_scale.get_size(),
                  (this->get_common_size()[0] * this->get_num_batch_items()));
    auto exec = this->get_executor();
    exec->run(dense::make_scale(make_temporary_clone(exec, &col_scale).get(),
                                make_temporary_clone(exec, &row_scale).get(),
                                this));
}


template <typename ValueType>
void Dense<ValueType>::scale_add(
    ptr_param<const MultiVector<ValueType>> alpha,
    ptr_param<const batch::matrix::Dense<ValueType>> b)
{
    GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(alpha, b);
    GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(this, b);
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(this, b);
    auto exec = this->get_executor();
    exec->run(dense::make_scale_add(make_temporary_clone(exec, alpha).get(),
                                    make_temporary_clone(exec, b).get(), this));
}


template <typename ValueType>
void Dense<ValueType>::add_scaled_identity(
    ptr_param<const MultiVector<ValueType>> alpha,
    ptr_param<const MultiVector<ValueType>> beta)
{
    GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(alpha, beta);
    GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(this, beta);
    GKO_ASSERT_EQUAL_DIMENSIONS(alpha->get_common_size(), gko::dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(beta->get_common_size(), gko::dim<2>(1, 1));
    auto exec = this->get_executor();
    exec->run(dense::make_add_scaled_identity(
        make_temporary_clone(exec, alpha).get(),
        make_temporary_clone(exec, beta).get(), this));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(
    Dense<next_precision<ValueType>>* result) const
{
    result->values_ = this->values_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


#define GKO_DECLARE_BATCH_DENSE_MATRIX(_type) class Dense<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_MATRIX);


}  // namespace matrix
}  // namespace batch
}  // namespace gko
