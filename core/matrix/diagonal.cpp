// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/diagonal.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/utils.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/base/validation.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/matrix/diagonal_kernels.hpp"


namespace gko {
namespace matrix {
namespace diagonal {
namespace {


GKO_REGISTER_OPERATION(apply_to_dense, diagonal::apply_to_dense);
GKO_REGISTER_OPERATION(right_apply_to_dense, diagonal::right_apply_to_dense);
GKO_REGISTER_OPERATION(apply_to_csr, diagonal::apply_to_csr);
GKO_REGISTER_OPERATION(right_apply_to_csr, diagonal::right_apply_to_csr);
GKO_REGISTER_OPERATION(fill_in_matrix_data, diagonal::fill_in_matrix_data);
GKO_REGISTER_OPERATION(convert_to_csr, diagonal::convert_to_csr);
GKO_REGISTER_OPERATION(conj_transpose, diagonal::conj_transpose);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // anonymous namespace
}  // namespace diagonal


template <typename ValueType>
void Diagonal<ValueType>::validate_data() const
{
    GKO_VALIDATE(validation::sparse_matrix_values_are_finite(values_),
                 "matrix must contain only finite values");
}


template <typename ValueType>
void Diagonal<ValueType>::apply_impl(const AbstractMultiVector* b,
                                     AbstractMultiVector* x) const
{
    apply_precision_dispatch<ValueType>(
        [this](auto view_b, auto view_x) {
            this->get_executor()->run(
                diagonal::make_apply_to_dense(this, view_b, view_x, false));
        },
        b, x);
}


template <typename ValueType>
void Diagonal<ValueType>::apply(ptr_param<const Csr<ValueType, int32>> b,
                                ptr_param<Csr<ValueType, int32>> x) const
{
    LinOp::validate_application_parameters(b.get(), x.get());
    x->copy_from(b);
    this->get_executor()->run(diagonal::make_apply_to_csr(
        this, b->get_const_device_view(), x->get_device_view(), false));
}


template <typename ValueType>
void Diagonal<ValueType>::apply(ptr_param<const Csr<ValueType, int64>> b,
                                ptr_param<Csr<ValueType, int64>> x) const
{
    LinOp::validate_application_parameters(b.get(), x.get());
    x->copy_from(b);
    this->get_executor()->run(diagonal::make_apply_to_csr(
        this, b->get_const_device_view(), x->get_device_view(), false));
}


template <typename ValueType>
std::unique_ptr<LinOp> Diagonal<ValueType>::transpose() const
{
    return this->clone();
}


template <typename ValueType>
std::unique_ptr<LinOp> Diagonal<ValueType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto tmp = Diagonal<ValueType>::create(exec, this->get_size()[0]);

    exec->run(diagonal::make_conj_transpose(this, tmp.get()));
    return std::move(tmp);
}


template <typename ValueType>
void Diagonal<ValueType>::convert_to(
    Diagonal<next_precision<ValueType>>* result) const
{
    result->values_ = this->values_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void Diagonal<ValueType>::move_to(Diagonal<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
template <typename ValueType>
void Diagonal<ValueType>::convert_to(
    Diagonal<next_precision<ValueType, 2>>* result) const
{
    result->values_ = this->values_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void Diagonal<ValueType>::move_to(
    Diagonal<next_precision<ValueType, 2>>* result)
{
    this->convert_to(result);
}
#endif


#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
template <typename ValueType>
void Diagonal<ValueType>::convert_to(
    Diagonal<next_precision<ValueType, 3>>* result) const
{
    result->values_ = this->values_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void Diagonal<ValueType>::move_to(
    Diagonal<next_precision<ValueType, 3>>* result)
{
    this->convert_to(result);
}
#endif


template <typename ValueType>
void Diagonal<ValueType>::convert_to(Csr<ValueType, int32>* result) const
{
    auto exec = this->get_executor();
    {
        auto tmp = make_temporary_clone(exec, result);
        tmp->row_ptrs_.resize_and_reset(this->get_size()[0] + 1);
        tmp->col_idxs_.resize_and_reset(this->get_size()[0]);
        tmp->values_.resize_and_reset(this->get_size()[0]);
        tmp->set_size(this->get_size());
        exec->run(diagonal::make_convert_to_csr(this, tmp->get_device_view()));
    }
    result->make_srow();
}


template <typename ValueType>
void Diagonal<ValueType>::move_to(Csr<ValueType, int32>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Diagonal<ValueType>::convert_to(Csr<ValueType, int64>* result) const
{
    auto exec = this->get_executor();
    {
        auto tmp = make_temporary_clone(exec, result);
        tmp->row_ptrs_.resize_and_reset(this->get_size()[0] + 1);
        tmp->col_idxs_.resize_and_reset(this->get_size()[0]);
        tmp->values_.resize_and_reset(this->get_size()[0]);
        tmp->set_size(this->get_size());
        exec->run(diagonal::make_convert_to_csr(this, tmp->get_device_view()));
    }
    result->make_srow();
}


template <typename ValueType>
void Diagonal<ValueType>::move_to(Csr<ValueType, int64>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Diagonal<ValueType>::read(const device_mat_data& data)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(data.get_size());
    this->set_size(data.get_size());
    values_.resize_and_reset(data.get_size()[0]);
    values_.fill(zero<ValueType>());
    auto exec = this->get_executor();
    exec->run(diagonal::make_fill_in_matrix_data(
        *make_temporary_clone(exec, &data), this));
}


template <typename ValueType>
void Diagonal<ValueType>::read(const device_mat_data32& data)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(data.get_size());
    this->set_size(data.get_size());
    values_.resize_and_reset(data.get_size()[0]);
    values_.fill(zero<ValueType>());
    auto exec = this->get_executor();
    exec->run(diagonal::make_fill_in_matrix_data(
        *make_temporary_clone(exec, &data), this));
}


template <typename ValueType>
void Diagonal<ValueType>::read(device_mat_data&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType>
void Diagonal<ValueType>::read(device_mat_data32&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType>
void Diagonal<ValueType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_from_host(this->get_executor(), data));
}


template <typename ValueType>
void Diagonal<ValueType>::read(const mat_data32& data)
{
    this->read(device_mat_data32::create_from_host(this->get_executor(), data));
}


namespace {


template <typename MatrixType, typename MatrixData>
inline void write_impl(const MatrixType* mtx, MatrixData& data)
{
    auto tmp = make_temporary_clone(mtx->get_executor()->get_master(), mtx);

    data = {tmp->get_size(), {}};
    const auto values = tmp->get_const_values();

    for (size_type row = 0; row < data.size[0]; ++row) {
        data.nonzeros.emplace_back(row, row, values[row]);
    }
}


}  // namespace


template <typename ValueType>
void Diagonal<ValueType>::write(mat_data& data) const
{
    write_impl(this, data);
}


template <typename ValueType>
void Diagonal<ValueType>::write(mat_data32& data) const
{
    write_impl(this, data);
}


template <typename ValueType>
void Diagonal<ValueType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(diagonal::make_inplace_absolute_array(this->get_values(),
                                                    this->get_size()[0]));
}


template <typename ValueType, typename T, typename U>
void validate_reverse_application_parameters(const Diagonal<ValueType>* op, T b,
                                             U x)
{
    GKO_ASSERT_REVERSE_CONFORMANT(op, b);
    GKO_ASSERT_EQUAL_ROWS(b, x);
    GKO_ASSERT_EQUAL_COLS(op, x);
}


template <typename ValueType>
void Diagonal<ValueType>::rapply(ptr_param<const AbstractMultiVector> b,
                                 ptr_param<AbstractMultiVector> x) const
{
    validate_reverse_application_parameters(this, b, x);
    apply_precision_dispatch<ValueType>(
        [this](auto view_b, auto view_x) {
            this->get_executor()->run(
                diagonal::make_right_apply_to_dense(this, view_b, view_x));
        },
        b.get(), x.get());
}


template <typename ValueType>
void Diagonal<ValueType>::rapply(ptr_param<const Csr<ValueType, int32>> b,
                                 ptr_param<Csr<ValueType, int32>> x) const
{
    validate_reverse_application_parameters(this, b, x);
    x->copy_from(b);
    this->get_executor()->run(diagonal::make_right_apply_to_csr(
        this, b->get_const_device_view(), x->get_device_view()));
}


template <typename ValueType>
void Diagonal<ValueType>::rapply(ptr_param<const Csr<ValueType, int64>> b,
                                 ptr_param<Csr<ValueType, int64>> x) const
{
    validate_reverse_application_parameters(this, b, x);
    x->copy_from(b);
    this->get_executor()->run(diagonal::make_right_apply_to_csr(
        this, b->get_const_device_view(), x->get_device_view()));
}


template <typename ValueType, typename T, typename U>
void validate_inverse_application_parameters(const Diagonal<ValueType>* op, T b,
                                             U x)
{
    GKO_ASSERT_CONFORMANT(op, b);
    GKO_ASSERT_EQUAL_ROWS(b, x);
    GKO_ASSERT_EQUAL_ROWS(op, x);
}


template <typename ValueType>
void Diagonal<ValueType>::inverse_apply(ptr_param<const AbstractMultiVector> b,
                                        ptr_param<AbstractMultiVector> x) const
{
    validate_inverse_application_parameters(this, b, x);
    apply_precision_dispatch<ValueType>(
        [this](auto view_b, auto view_x) {
            this->get_executor()->run(
                diagonal::make_apply_to_dense(this, view_b, view_x, true));
        },
        b.get(), x.get());
}


template <typename ValueType>
void Diagonal<ValueType>::inverse_apply(
    ptr_param<const Csr<ValueType, int32>> b,
    ptr_param<Csr<ValueType, int32>> x) const
{
    validate_inverse_application_parameters(this, b, x);
    x->copy_from(b);
    this->get_executor()->run(diagonal::make_apply_to_csr(
        this, b->get_const_device_view(), x->get_device_view(), true));
}


template <typename ValueType>
void Diagonal<ValueType>::inverse_apply(
    ptr_param<const Csr<ValueType, int64>> b,
    ptr_param<Csr<ValueType, int64>> x) const
{
    validate_inverse_application_parameters(this, b, x);
    x->copy_from(b);
    this->get_executor()->run(diagonal::make_apply_to_csr(
        this, b->get_const_device_view(), x->get_device_view(), true));
}


template <typename ValueType>
std::unique_ptr<typename Diagonal<ValueType>::absolute_type>
Diagonal<ValueType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_diagonal = absolute_type::create(exec, this->get_size()[0]);

    exec->run(diagonal::make_outplace_absolute_array(
        this->get_const_values(), this->get_size()[0],
        abs_diagonal->get_values()));

    return abs_diagonal;
}


template <typename ValueType>
Diagonal<ValueType>::Diagonal(std::shared_ptr<const Executor> exec,
                              size_type size)
    : LinOp(exec, dim<2>{size}), values_(exec, size)
{}


template <typename ValueType>
Diagonal<ValueType>::Diagonal(std::shared_ptr<const Executor> exec,
                              const size_type size, array<value_type> values)
    : LinOp(exec, dim<2>(size)), values_{exec, std::move(values)}
{
    GKO_ENSURE_COMPATIBLE_BOUNDS(size, values_.get_size());
}


template <typename ValueType>
std::unique_ptr<Diagonal<ValueType>> Diagonal<ValueType>::create(
    std::shared_ptr<const Executor> exec, size_type size)
{
    return std::unique_ptr<Diagonal>{new Diagonal{exec, size}};
}


template <typename ValueType>
std::unique_ptr<Diagonal<ValueType>> Diagonal<ValueType>::create(
    std::shared_ptr<const Executor> exec, const size_type size,
    array<value_type> values)
{
    return std::unique_ptr<Diagonal>{
        new Diagonal{exec, size, std::move(values)}};
}


template <typename ValueType>
std::unique_ptr<const Diagonal<ValueType>> Diagonal<ValueType>::create_const(
    std::shared_ptr<const Executor> exec, size_type size,
    gko::detail::const_array_view<ValueType>&& values)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const Diagonal>{new Diagonal{
        exec, size, gko::detail::array_const_cast(std::move(values))}};
}


#define GKO_DECLARE_DIAGONAL_MATRIX(value_type) class Diagonal<value_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_MATRIX);


}  // namespace matrix


// Implement DiagonalExtractable for LinOp when Diagonal is complete class
template <typename ValueType>
std::unique_ptr<LinOp> DiagonalExtractable<ValueType>::extract_diagonal_linop()
    const
{
    return this->extract_diagonal();
}


#define GKO_DECLARE_DIAGONAL_EXTRACTABLE(value_type) \
    std::unique_ptr<LinOp>                           \
    DiagonalExtractable<value_type>::extract_diagonal_linop() const
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_EXTRACTABLE);


}  // namespace gko
