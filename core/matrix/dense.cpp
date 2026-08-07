// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/matrix/multivector_kernels.hpp"


namespace gko {
namespace matrix {
namespace dense {


GKO_REGISTER_OPERATION(simple_apply, dense::simple_apply);
GKO_REGISTER_OPERATION(advanced_apply, dense::apply);


}  // namespace dense
namespace multivector {


GKO_REGISTER_OPERATION(copy, multivector::copy);
GKO_REGISTER_OPERATION(fill, multivector::fill);
GKO_REGISTER_OPERATION(fill_in_matrix_data, multivector::fill_in_matrix_data);


}  // namespace multivector


template <typename ValueType>
void Dense<ValueType>::convert_to(MultiVector<ValueType>* result) const
{
    if (result->get_size() != this->get_size()) {
        result->set_size(this->get_size());
        result->stride_ = stride_;
        result->values_.resize_and_reset(result->get_size()[0] *
                                         result->stride_);
    }
    auto exec = this->get_executor();
    exec->run(multivector::make_copy(
        this->get_const_device_view(),
        make_temporary_output_clone(exec, result)->get_device_view()));
}


template <typename ValueType>
void Dense<ValueType>::move_to(MultiVector<ValueType>* result)
{
    result->set_size(this->get_size());
    this->set_size(dim<2>{0, 0});
    result->stride_ = std::exchange(stride_, 0);
    result->values_ = std::move(values_);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(
    Dense<next_precision<ValueType>>* result) const
{
    if (result->get_size() != this->get_size()) {
        result->set_size(this->get_size());
        result->stride_ = stride_;
        result->values_.resize_and_reset(result->get_size()[0] *
                                         result->stride_);
    }
    auto exec = this->get_executor();
    exec->run(multivector::make_copy(
        this->get_const_device_view(),
        make_temporary_output_clone(exec, result)->get_device_view()));
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense<next_precision<ValueType>>* result)
{
    this->convert_to(result);
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
}


#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
template <typename ValueType>
void Dense<ValueType>::convert_to(
    Dense<next_precision<ValueType, 2>>* result) const
{
    if (result->get_size() != this->get_size()) {
        result->set_size(this->get_size());
        result->stride_ = stride_;
        result->values_.resize_and_reset(result->get_size()[0] *
                                         result->stride_);
    }
    auto exec = this->get_executor();
    exec->run(multivector::make_copy(
        this->get_const_device_view(),
        make_temporary_output_clone(exec, result)->get_device_view()));
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense<next_precision<ValueType, 2>>* result)
{
    this->convert_to(result);
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
}
#endif


#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
template <typename ValueType>
void Dense<ValueType>::convert_to(
    Dense<next_precision<ValueType, 3>>* result) const
{
    if (result->get_size() != this->get_size()) {
        result->set_size(this->get_size());
        result->stride_ = stride_;
        result->values_.resize_and_reset(result->get_size()[0] *
                                         result->stride_);
    }
    auto exec = this->get_executor();
    exec->run(multivector::make_copy(
        this->get_const_device_view(),
        make_temporary_output_clone(exec, result)->get_device_view()));
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense<next_precision<ValueType, 3>>* result)
{
    this->convert_to(result);
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
}
#endif


template <typename ValueType>
void Dense<ValueType>::read(const mat_data32& data)
{
    this->read(device_mat_data32::create_from_host(this->get_executor(), data));
}


template <typename ValueType>
void Dense<ValueType>::read(const mat_data64& data)
{
    this->read(device_mat_data64::create_from_host(this->get_executor(), data));
}


template <typename ValueType>
void Dense<ValueType>::read(const device_mat_data32& data)
{
    auto exec = this->get_executor();
    this->set_size(data.get_size());
    this->stride_ = data.get_size()[1];
    this->values_.resize_and_reset(data.get_size()[0] * this->stride_);
    this->values_.fill(zero<ValueType>());
    exec->run(multivector::make_fill_in_matrix_data(
        *make_temporary_clone(exec, &data), this->get_device_view()));
}


template <typename ValueType>
void Dense<ValueType>::read(const device_mat_data64& data)
{
    auto exec = this->get_executor();
    this->set_size(data.get_size());
    this->stride_ = data.get_size()[1];
    this->values_.resize_and_reset(data.get_size()[0] * this->stride_);
    this->values_.fill(zero<ValueType>());
    exec->run(multivector::make_fill_in_matrix_data(
        *make_temporary_clone(exec, &data), this->get_device_view()));
}


template <typename ValueType>
void Dense<ValueType>::read(device_mat_data32&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType>
void Dense<ValueType>::read(device_mat_data64&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType>
void Dense<ValueType>::write(matrix_data<ValueType, int32>& data) const
{
    this->as_const_multivector_view()->write(data);
}


template <typename ValueType>
void Dense<ValueType>::write(matrix_data<ValueType, int64>& data) const
{
    this->as_const_multivector_view()->write(data);
}


template <typename ValueType>
void Dense<ValueType>::validate_data() const
{
    this->as_const_multivector_view()->validate_data();
}


template <typename ValueType>
void Dense<ValueType>::fill(ValueType value)
{
    this->get_executor()->run(
        multivector::make_fill(this->get_device_view(), value));
}


template <typename ValueType>
void Dense<ValueType>::extract_diagonal(
    ptr_param<Diagonal<ValueType>> output) const
{
    this->as_const_multivector_view()->extract_diagonal(output);
}


template <typename ValueType>
std::unique_ptr<Diagonal<ValueType>> Dense<ValueType>::extract_diagonal() const
{
    return this->as_const_multivector_view()->extract_diagonal();
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::transpose() const
{
    auto result =
        Dense::create(this->get_executor(), gko::transpose(this->get_size()));
    this->transpose(result);
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::conj_transpose() const
{
    auto result =
        Dense::create(this->get_executor(), gko::transpose(this->get_size()));
    this->conj_transpose(result);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::transpose(ptr_param<Dense> output) const
{
    this->as_const_multivector_view()->transpose(output->as_multivector_view());
}


template <typename ValueType>
void Dense<ValueType>::conj_transpose(ptr_param<Dense> output) const
{
    this->as_const_multivector_view()->conj_transpose(
        output->as_multivector_view());
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::create(
    std::shared_ptr<const Executor> exec, const dim<2>& size, size_type stride)
{
    return std::unique_ptr<Dense>{new Dense{std::move(exec), size, stride}};
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::create(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    array<value_type> values, size_type stride)
{
    return std::unique_ptr<Dense>{
        new Dense{std::move(exec), size, std::move(values), stride}};
}


template <typename ValueType>
std::unique_ptr<const Dense<ValueType>> Dense<ValueType>::create_const(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    ::gko::detail::const_array_view<ValueType>&& values, size_type stride)
{
    return std::unique_ptr<const Dense>{new Dense{
        exec, size, gko::detail::array_const_cast(std::move(values)), stride}};
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::create_subview(span rows,
                                                                   span cols)
{
    row_major_range range_this{this->get_values(), this->get_size()[0],
                               this->get_size()[1], this->get_stride()};
    auto sub_range = range_this(rows, cols);
    size_type storage_size =
        rows.length() > 0 ? sub_range.length(1) +
                                (sub_range.length(0) - 1) * this->get_stride()
                          : 0;
    return Dense::create(
        this->get_executor(), dim<2>{sub_range.length(0), sub_range.length(1)},
        make_array_view(this->get_executor(), storage_size, sub_range->data),
        this->get_stride());
}


template <typename ValueType>
std::unique_ptr<const Dense<ValueType>> Dense<ValueType>::create_subview(
    span rows, span cols) const
{
    return const_cast<Dense*>(this)->create_subview(rows, cols);
}


template <typename ValueType>
std::unique_ptr<const Dense<ValueType>> Dense<ValueType>::create_const_subview(
    span rows, span cols) const
{
    return this->create_subview(rows, cols);
}


template <typename ValueType>
std::unique_ptr<const MultiVector<ValueType>>
Dense<ValueType>::as_const_multivector_view() const
{
    return MultiVector<ValueType>::create_const(
        this->get_executor(), this->get_size(), this->values_.as_const_view(),
        stride_);
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> Dense<ValueType>::as_multivector_view()
{
    return MultiVector<ValueType>::create(this->get_executor(),
                                          this->get_size(),
                                          this->values_.as_view(), stride_);
}


template <typename ValueType>
typename Dense<ValueType>::device_view Dense<ValueType>::get_device_view()
{
    return device_view{this->get_size(), this->stride_,
                       this->values_.get_data()};
}


template <typename ValueType>
typename Dense<ValueType>::const_device_view
Dense<ValueType>::get_const_device_view() const
{
    return const_device_view{this->get_size(), this->stride_,
                             this->values_.get_const_data()};
}


template <typename ValueType>
ValueType& Dense<ValueType>::at(size_type row, size_type col)
{
    return values_.get_data()[linearize_index(row, col)];
}


template <typename ValueType>
ValueType Dense<ValueType>::at(size_type row, size_type col) const
{
    return values_.get_const_data()[linearize_index(row, col)];
}


template <typename ValueType>
size_type Dense<ValueType>::get_stride() const noexcept
{
    return stride_;
}


template <typename ValueType>
size_type Dense<ValueType>::get_num_stored_elements() const noexcept
{
    return this->values_.get_size();
}


template <typename ValueType>
Dense<ValueType>::Dense(const Dense& other) : LinOp(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
Dense<ValueType>::Dense(Dense&& other) : LinOp(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType>
Dense<ValueType>& Dense<ValueType>::operator=(const Dense& other)
{
    if (&other != this) {
        auto old_size = this->get_size();
        LinOp::operator=(other);
        // NOTE: keep this consistent with resize(...)
        if (old_size != other.get_size()) {
            this->stride_ = this->get_size()[1];
            this->values_.resize_and_reset(this->get_size()[0] * this->stride_);
        }
        // we need to create a executor-local clone of the target data, that
        // will be copied back later. Need temporary_clone, not
        // temporary_output_clone to avoid overwriting padding
        auto exec = other.get_executor();
        auto exec_values_array =
            make_temporary_output_clone(exec, &this->values_);
        exec->run(
            multivector::make_copy(other.get_const_device_view(),
                                   device_view{this->get_size(), this->stride_,
                                               exec_values_array->get_data()}));
    }
    return *this;
}


template <typename ValueType>
Dense<ValueType>& Dense<ValueType>::operator=(Dense&& other)
{
    if (&other != this) {
        LinOp::operator=(std::move(other));
        stride_ = std::exchange(other.stride_, 0);
        values_ = std::move(other.values_);
    }
    return *this;
}


template <typename ValueType>
Dense<ValueType>::Dense(std::shared_ptr<const Executor> exec,
                        const dim<2>& size, size_type stride)
    : LinOp(exec, size),
      stride_(stride == 0 ? size[1] : stride),
      values_(exec, size[0] * stride_)
{}


template <typename ValueType>
Dense<ValueType>::Dense(std::shared_ptr<const Executor> exec,
                        const dim<2>& size, array<value_type> values,
                        size_type stride)
    : LinOp(exec, size),
      stride_(stride == 0 ? size[1] : stride),
      values_(exec, std::move(values))
{
    if (size[0] > 0 && size[1] > 0) {
        GKO_ENSURE_IN_BOUNDS((size[0] - 1) * stride_ + size[1] - 1,
                             values_.get_size());
    }
}


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(dense::make_simple_apply(
                this->get_const_device_view(), dense_b->get_const_device_view(),
                dense_x->get_device_view()));
        },
        b, x);
}


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                  const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            this->get_executor()->run(dense::make_advanced_apply(
                dense_alpha->get_const_device_view(),
                this->get_const_device_view(), dense_b->get_const_device_view(),
                dense_beta->get_const_device_view(),
                dense_x->get_device_view()));
        },
        alpha, b, beta, x);
}


template <typename ValueType>
size_type Dense<ValueType>::linearize_index(size_type row,
                                            size_type col) const noexcept
{
    return row * stride_ + col;
}


#define GKO_DECLARE_DENSE(ValueType) class Dense<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE);


}  // namespace matrix
}  // namespace gko
