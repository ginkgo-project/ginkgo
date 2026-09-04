// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/multivector.hpp"

#include <algorithm>
#include <type_traits>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/base/validation.hpp"
#include "core/matrix/multivector_kernels.hpp"
#include "core/matrix/permutation.hpp"


namespace gko {
namespace matrix {
namespace multivector {
namespace {


GKO_REGISTER_OPERATION(copy, multivector::copy);
GKO_REGISTER_OPERATION(fill, multivector::fill);
GKO_REGISTER_OPERATION(scale, multivector::scale);
GKO_REGISTER_OPERATION(inv_scale, multivector::inv_scale);
GKO_REGISTER_OPERATION(add_scaled, multivector::add_scaled);
GKO_REGISTER_OPERATION(sub_scaled, multivector::sub_scaled);
GKO_REGISTER_OPERATION(compute_dot, multivector::compute_dot_dispatch);
GKO_REGISTER_OPERATION(compute_conj_dot,
                       multivector::compute_conj_dot_dispatch);
GKO_REGISTER_OPERATION(compute_norm2, multivector::compute_norm2_dispatch);
GKO_REGISTER_OPERATION(compute_norm1, multivector::compute_norm1);
GKO_REGISTER_OPERATION(compute_mean, multivector::compute_mean);
GKO_REGISTER_OPERATION(compute_squared_norm2,
                       multivector::compute_squared_norm2);
GKO_REGISTER_OPERATION(compute_sqrt, multivector::compute_sqrt);
GKO_REGISTER_OPERATION(transpose, multivector::transpose);
GKO_REGISTER_OPERATION(conj_transpose, multivector::conj_transpose);
GKO_REGISTER_OPERATION(symm_permute, multivector::symm_permute);
GKO_REGISTER_OPERATION(inv_symm_permute, multivector::inv_symm_permute);
GKO_REGISTER_OPERATION(nonsymm_permute, multivector::nonsymm_permute);
GKO_REGISTER_OPERATION(inv_nonsymm_permute, multivector::inv_nonsymm_permute);
GKO_REGISTER_OPERATION(row_gather, multivector::row_gather);
GKO_REGISTER_OPERATION(advanced_row_gather, multivector::advanced_row_gather);
GKO_REGISTER_OPERATION(col_permute, multivector::col_permute);
GKO_REGISTER_OPERATION(inverse_row_permute, multivector::inv_row_permute);
GKO_REGISTER_OPERATION(inverse_col_permute, multivector::inv_col_permute);
GKO_REGISTER_OPERATION(symm_scale_permute, multivector::symm_scale_permute);
GKO_REGISTER_OPERATION(inv_symm_scale_permute,
                       multivector::inv_symm_scale_permute);
GKO_REGISTER_OPERATION(nonsymm_scale_permute,
                       multivector::nonsymm_scale_permute);
GKO_REGISTER_OPERATION(inv_nonsymm_scale_permute,
                       multivector::inv_nonsymm_scale_permute);
GKO_REGISTER_OPERATION(row_scale_permute, multivector::row_scale_permute);
GKO_REGISTER_OPERATION(col_scale_permute, multivector::col_scale_permute);
GKO_REGISTER_OPERATION(inv_row_scale_permute,
                       multivector::inv_row_scale_permute);
GKO_REGISTER_OPERATION(inv_col_scale_permute,
                       multivector::inv_col_scale_permute);
GKO_REGISTER_OPERATION(fill_in_matrix_data, multivector::fill_in_matrix_data);
GKO_REGISTER_OPERATION(inplace_absolute_dense,
                       multivector::inplace_absolute_dense);
GKO_REGISTER_OPERATION(outplace_absolute_dense,
                       multivector::outplace_absolute_dense);
GKO_REGISTER_OPERATION(make_complex, multivector::make_complex);
GKO_REGISTER_OPERATION(get_real, multivector::get_real);
GKO_REGISTER_OPERATION(get_imag, multivector::get_imag);


}  // anonymous namespace
}  // namespace multivector


template <typename ValueType>
validation::ValidationResult dense_matrix_values_are_finite(
    const MultiVector<ValueType>* mtx)
{
    if constexpr (std::is_integral<ValueType>::value) {
        return {true, ""};
    } else {
        const auto host_mtx = mtx->clone(mtx->get_executor()->get_master());
        const auto num_rows = host_mtx->get_size()[0];
        const auto num_cols = host_mtx->get_size()[1];
        const auto host_values = host_mtx->get_const_values();
        const auto stride = host_mtx->get_stride();
        for (size_t i = 0; i < num_rows; ++i) {
            for (size_t j = 0; j < num_cols; ++j) {
                if (!is_finite(host_values[i * stride + j])) {
                    return {false, "index " + std::to_string(j) + " in row " +
                                       std::to_string(i) + " with stride " +
                                       std::to_string(stride)};
                }
            }
        }
        return {true, ""};
    }
}


template <typename ValueType>
void MultiVector<ValueType>::validate_data() const
{
    GKO_VALIDATE(dense_matrix_values_are_finite(this),
                 "matrix must contain only finite values");
}


template <typename ValueType>
void MultiVector<ValueType>::compute_mean(
    ptr_param<AbstractMultiVector> result) const
{
    auto exec = this->get_executor();
    this->compute_mean_impl(make_temporary_output_clone(exec, result).get());
}


template <typename ValueType>
void MultiVector<ValueType>::compute_mean(ptr_param<AbstractMultiVector> result,
                                          array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_COLS(result, this);
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    auto dense_res = as<MultiVector>(result->as_precision(this));
    exec->run(multivector::make_compute_mean(
        this->get_const_device_view(), dense_res->get_device_view(), tmp));
}


template <typename ValueType>
void MultiVector<ValueType>::compute_mean_impl(
    AbstractMultiVector* result) const
{
    auto exec = this->get_executor();
    array<char> tmp{exec};
    this->compute_mean(make_temporary_output_clone(exec, result).get(), tmp);
}


template <typename ValueType>
MultiVector<ValueType>& MultiVector<ValueType>::operator=(
    const MultiVector& other)
{
    if (&other != this) {
        auto old_size = this->get_size();
        AbstractMultiVector::operator=(other);
        // NOTE: keep this consistent with resize(...)
        if (old_size != other.get_size()) {
            this->stride_ = this->get_size()[1];
            this->values_.resize_and_reset(this->get_size()[0] * this->stride_);
        }
        // we need to create a executor-local clone of the target data, that
        // will be copied back later. Need temporary_clone, not
        // temporary_output_clone to avoid overwriting padding
        auto exec = other.get_executor();
        auto exec_values_array = make_temporary_clone(exec, &this->values_);
        // create a (value, not pointer to avoid allocation overhead) view
        // matrix on the array to avoid special-casing cross-executor copies
        auto exec_this_view =
            MultiVector{exec, this->get_size(),
                        make_array_view(exec, exec_values_array->get_size(),
                                        exec_values_array->get_data()),
                        this->get_stride()};
        exec->run(multivector::make_copy(other.get_const_device_view(),
                                         exec_this_view.get_device_view()));
    }
    return *this;
}


template <typename ValueType>
MultiVector<ValueType>& MultiVector<ValueType>::operator=(
    MultiVector<ValueType>&& other)
{
    if (&other != this) {
        AbstractMultiVector::operator=(std::move(other));
        values_ = std::move(other.values_);
        stride_ = std::exchange(other.stride_, 0);
    }
    return *this;
}


template <typename ValueType>
MultiVector<ValueType>::MultiVector(const MultiVector<ValueType>& other)
    : MultiVector(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
MultiVector<ValueType>::MultiVector(MultiVector<ValueType>&& other)
    : MultiVector(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_type_of(
    ptr_param<const MultiVector> other, std::shared_ptr<const Executor> exec,
    const dim<2>& size, size_type stride)
{
    return other->create_with_type_of_impl(exec, size, stride);
}


template <typename ValueType>
void MultiVector<ValueType>::convert_to(
    MultiVector<next_precision<ValueType>>* result) const
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
void MultiVector<ValueType>::move_to(
    MultiVector<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
template <typename ValueType>
void MultiVector<ValueType>::convert_to(
    MultiVector<next_precision<ValueType, 2>>* result) const
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
void MultiVector<ValueType>::move_to(
    MultiVector<next_precision<ValueType, 2>>* result)
{
    this->convert_to(result);
}
#endif


#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
template <typename ValueType>
void MultiVector<ValueType>::convert_to(
    MultiVector<next_precision<ValueType, 3>>* result) const
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
void MultiVector<ValueType>::move_to(
    MultiVector<next_precision<ValueType, 3>>* result)
{
    this->convert_to(result);
}
#endif


template <typename ValueType>
void MultiVector<ValueType>::convert_to(Dense<ValueType>* result) const
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
void MultiVector<ValueType>::move_to(Dense<ValueType>* result)
{
    result->set_size(this->get_size());
    this->set_size(dim<2>{0, 0});
    result->stride_ = std::exchange(stride_, 0);
    result->values_ = std::move(values_);
}


template <typename ValueType>
void MultiVector<ValueType>::resize(gko::dim<2> new_size)
{
    if (this->get_size() != new_size) {
        this->set_size(new_size);
        this->stride_ = new_size[1];
        this->values_.resize_and_reset(new_size[0] * this->get_stride());
    }
}


template <typename ValueType>
void MultiVector<ValueType>::read(const device_mat_data64& data)
{
    auto exec = this->get_executor();
    this->resize(data.get_size());
    this->fill(zero<ValueType>());
    exec->run(multivector::make_fill_in_matrix_data(
        *make_temporary_clone(exec, &data), this->get_device_view()));
}


template <typename ValueType>
void MultiVector<ValueType>::read(const device_mat_data32& data)
{
    auto exec = this->get_executor();
    this->resize(data.get_size());
    this->fill(zero<ValueType>());
    exec->run(multivector::make_fill_in_matrix_data(
        *make_temporary_clone(exec, &data), this->get_device_view()));
}


template <typename ValueType>
void MultiVector<ValueType>::read(device_mat_data64&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType>
void MultiVector<ValueType>::read(device_mat_data32&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType>
void MultiVector<ValueType>::read(const mat_data64& data)
{
    this->read(device_mat_data64::create_from_host(this->get_executor(), data));
}


template <typename ValueType>
void MultiVector<ValueType>::read(const mat_data32& data)
{
    this->read(device_mat_data32::create_from_host(this->get_executor(), data));
}


namespace {


template <typename MatrixType, typename MatrixData>
inline void write_impl(const MatrixType* mtx, MatrixData& data)
{
    auto tmp = make_temporary_clone(mtx->get_executor()->get_master(), mtx);

    data = {mtx->get_size(), {}};

    for (size_type row = 0; row < data.size[0]; ++row) {
        for (size_type col = 0; col < data.size[1]; ++col) {
            if (is_nonzero(tmp->at(row, col))) {
                data.nonzeros.emplace_back(row, col, tmp->at(row, col));
            }
        }
    }
}


}  // namespace


template <typename ValueType>
void MultiVector<ValueType>::write(mat_data64& data) const
{
    write_impl(this, data);
}


template <typename ValueType>
void MultiVector<ValueType>::write(mat_data32& data) const
{
    write_impl(this, data);
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::transpose()
    const
{
    auto result = MultiVector::create(this->get_executor(),
                                      gko::transpose(this->get_size()));
    this->transpose(result);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::conj_transpose()
    const
{
    auto result = MultiVector::create(this->get_executor(),
                                      gko::transpose(this->get_size()));
    this->conj_transpose(result);
    return result;
}


template <typename ValueType>
void MultiVector<ValueType>::transpose(
    ptr_param<MultiVector<ValueType>> output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(output, gko::transpose(this->get_size()));
    auto exec = this->get_executor();
    exec->run(multivector::make_transpose(
        this->get_const_device_view(),
        make_temporary_output_clone(exec, output)->get_device_view()));
}


template <typename ValueType>
void MultiVector<ValueType>::conj_transpose(
    ptr_param<MultiVector<ValueType>> output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(output, gko::transpose(this->get_size()));
    auto exec = this->get_executor();
    exec->run(multivector::make_conj_transpose(
        this->get_const_device_view(),
        make_temporary_output_clone(exec, output)->get_device_view()));
}


template <typename ValueType>
template <typename IndexType>
void MultiVector<ValueType>::permute_impl(
    const Permutation<IndexType>* permutation, permute_mode mode,
    MultiVector* output) const
{
    const auto exec = this->get_executor();
    const auto size = this->get_size();
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    validate_permute_dimensions(size, permutation->get_size(), mode);
    if ((mode & permute_mode::symmetric) == permute_mode::none) {
        output->copy_from(this);
        return;
    }
    auto local_output = make_temporary_output_clone(exec, output);
    auto local_perm = make_temporary_clone(exec, permutation);
    switch (mode) {
    case permute_mode::rows:
        exec->run(multivector::make_row_gather(
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    case permute_mode::columns:
        exec->run(multivector::make_col_permute(
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    case permute_mode::symmetric:
        exec->run(multivector::make_symm_permute(
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    case permute_mode::inverse_rows:
        exec->run(multivector::make_inverse_row_permute(
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    case permute_mode::inverse_columns:
        exec->run(multivector::make_inverse_col_permute(
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    case permute_mode::inverse_symmetric:
        exec->run(multivector::make_inv_symm_permute(
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    default:
        GKO_INVALID_STATE("Invalid permute mode");
    }
}


template <typename ValueType>
template <typename IndexType>
void MultiVector<ValueType>::permute_impl(
    const Permutation<IndexType>* row_permutation,
    const Permutation<IndexType>* col_permutation, bool invert,
    MultiVector* output) const
{
    auto exec = this->get_executor();
    auto size = this->get_size();
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    GKO_ASSERT_EQUAL_ROWS(this, row_permutation);
    GKO_ASSERT_EQUAL_COLS(this, col_permutation);
    auto local_output = make_temporary_output_clone(exec, output);
    auto local_row_perm = make_temporary_clone(exec, row_permutation);
    auto local_col_perm = make_temporary_clone(exec, col_permutation);
    if (invert) {
        exec->run(multivector::make_inv_nonsymm_permute(
            local_row_perm->get_const_permutation(),
            local_col_perm->get_const_permutation(),
            this->get_const_device_view(), local_output->get_device_view()));
    } else {
        exec->run(multivector::make_nonsymm_permute(
            local_row_perm->get_const_permutation(),
            local_col_perm->get_const_permutation(),
            this->get_const_device_view(), local_output->get_device_view()));
    }
}


template <typename ValueType>
template <typename IndexType>
void MultiVector<ValueType>::scale_permute_impl(
    const ScaledPermutation<ValueType, IndexType>* permutation,
    permute_mode mode, MultiVector* output) const
{
    const auto exec = this->get_executor();
    const auto size = this->get_size();
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    validate_permute_dimensions(size, permutation->get_size(), mode);
    if ((mode & permute_mode::symmetric) == permute_mode::none) {
        output->copy_from(this);
        return;
    }
    auto local_output = make_temporary_output_clone(exec, output);
    auto local_perm = make_temporary_clone(exec, permutation);
    switch (mode) {
    case permute_mode::rows:
        exec->run(multivector::make_row_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    case permute_mode::columns:
        exec->run(multivector::make_col_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    case permute_mode::symmetric:
        exec->run(multivector::make_symm_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    case permute_mode::inverse_rows:
        exec->run(multivector::make_inv_row_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    case permute_mode::inverse_columns:
        exec->run(multivector::make_inv_col_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    case permute_mode::inverse_symmetric:
        exec->run(multivector::make_inv_symm_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this->get_const_device_view(),
            local_output->get_device_view()));
        break;
    default:
        GKO_INVALID_STATE("Invalid permute mode");
    }
}


template <typename ValueType>
template <typename IndexType>
void MultiVector<ValueType>::scale_permute_impl(
    const ScaledPermutation<ValueType, IndexType>* row_permutation,
    const ScaledPermutation<ValueType, IndexType>* col_permutation, bool invert,
    MultiVector* output) const
{
    auto exec = this->get_executor();
    auto size = this->get_size();
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    GKO_ASSERT_EQUAL_ROWS(this, row_permutation);
    GKO_ASSERT_EQUAL_COLS(this, col_permutation);
    auto local_output = make_temporary_output_clone(exec, output);
    auto local_row_perm = make_temporary_clone(exec, row_permutation);
    auto local_col_perm = make_temporary_clone(exec, col_permutation);
    if (invert) {
        exec->run(multivector::make_inv_nonsymm_scale_permute(
            local_row_perm->get_const_scaling_factors(),
            local_row_perm->get_const_permutation(),
            local_col_perm->get_const_scaling_factors(),
            local_col_perm->get_const_permutation(),
            this->get_const_device_view(), local_output->get_device_view()));
    } else {
        exec->run(multivector::make_nonsymm_scale_permute(
            local_row_perm->get_const_scaling_factors(),
            local_row_perm->get_const_permutation(),
            local_col_perm->get_const_scaling_factors(),
            local_col_perm->get_const_permutation(),
            this->get_const_device_view(), local_output->get_device_view()));
    }
}


template <typename ValueType>
template <typename OutputType, typename IndexType>
void MultiVector<ValueType>::row_gather_impl(
    const array<IndexType>* row_idxs,
    MultiVector<OutputType>* row_collection) const
{
    auto exec = this->get_executor();
    dim<2> expected_dim{row_idxs->get_size(), this->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(expected_dim, row_collection);

    exec->run(multivector::make_row_gather(
        make_temporary_clone(exec, row_idxs)->get_const_data(),
        this->get_const_device_view(),
        make_temporary_output_clone(exec, row_collection)->get_device_view()));
}

template <typename ValueType>
template <typename OutputType, typename IndexType>
void MultiVector<ValueType>::row_gather_impl(
    const MultiVector<ValueType>* alpha, const array<IndexType>* row_idxs,
    const MultiVector<ValueType>* beta,
    MultiVector<OutputType>* row_collection) const
{
    auto exec = this->get_executor();
    dim<2> expected_dim{row_idxs->get_size(), this->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(expected_dim, row_collection);

    exec->run(multivector::make_advanced_row_gather(
        make_temporary_clone(exec, alpha)->get_const_device_view(),
        make_temporary_clone(exec, row_idxs)->get_const_data(),
        this->get_const_device_view(),
        make_temporary_clone(exec, beta)->get_const_device_view(),
        make_temporary_clone(exec, row_collection)->get_device_view()));
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::permute(
    const array<int32>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::permute(
    const array<int64>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::permute(
    ptr_param<const Permutation<int32>> permutation, permute_mode mode) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->permute(permutation, result, mode);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::permute(
    ptr_param<const Permutation<int64>> permutation, permute_mode mode) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->permute(permutation, result, mode);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::permute(
    ptr_param<const Permutation<int32>> row_permutation,
    ptr_param<const Permutation<int32>> col_permutation, bool invert) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->permute(row_permutation, col_permutation, result, invert);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::permute(
    ptr_param<const Permutation<int64>> row_permutation,
    ptr_param<const Permutation<int64>> col_permutation, bool invert) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->permute(row_permutation, col_permutation, result, invert);
    return result;
}


template <typename ValueType>
void MultiVector<ValueType>::permute(
    ptr_param<const Permutation<int32>> permutation,
    ptr_param<MultiVector<ValueType>> result, permute_mode mode) const
{
    this->permute_impl(permutation.get(), mode, result.get());
}


template <typename ValueType>
void MultiVector<ValueType>::permute(
    ptr_param<const Permutation<int64>> permutation,
    ptr_param<MultiVector<ValueType>> result, permute_mode mode) const
{
    this->permute_impl(permutation.get(), mode, result.get());
}


template <typename ValueType>
void MultiVector<ValueType>::permute(
    ptr_param<const Permutation<int32>> row_permutation,
    ptr_param<const Permutation<int32>> col_permutation,
    ptr_param<MultiVector<ValueType>> result, bool invert) const
{
    this->permute_impl(row_permutation.get(), col_permutation.get(), invert,
                       result.get());
}


template <typename ValueType>
void MultiVector<ValueType>::permute(
    ptr_param<const Permutation<int64>> row_permutation,
    ptr_param<const Permutation<int64>> col_permutation,
    ptr_param<MultiVector<ValueType>> result, bool invert) const
{
    this->permute_impl(row_permutation.get(), col_permutation.get(), invert,
                       result.get());
}


template <typename IndexType>
std::unique_ptr<const Permutation<IndexType>> create_permutation_view(
    const array<IndexType>& indices)
{
    return Permutation<IndexType>::create_const(indices.get_executor(),
                                                indices.as_const_view());
}


template <typename ValueType>
void MultiVector<ValueType>::permute(
    const array<int32>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::symmetric, output.get());
}


template <typename ValueType>
void MultiVector<ValueType>::permute(
    const array<int64>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::symmetric, output.get());
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::inverse_permute(
    const array<int32>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->inverse_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::inverse_permute(
    const array<int64>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->inverse_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
void MultiVector<ValueType>::inverse_permute(
    const array<int32>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_symmetric, output.get());
}


template <typename ValueType>
void MultiVector<ValueType>::inverse_permute(
    const array<int64>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_symmetric, output.get());
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::row_permute(
    const array<int32>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->row_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::row_permute(
    const array<int64>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->row_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
void MultiVector<ValueType>::row_permute(
    const array<int32>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::rows, output.get());
}


template <typename ValueType>
void MultiVector<ValueType>::row_permute(
    const array<int64>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::rows, output.get());
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::row_gather(
    const array<int32>* row_idxs) const
{
    auto exec = this->get_executor();
    dim<2> out_dim{row_idxs->get_size(), this->get_size()[1]};
    auto result = MultiVector::create(exec, out_dim);
    this->row_gather(row_idxs, result);
    return result;
}

template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::row_gather(
    const array<int64>* row_idxs) const
{
    auto exec = this->get_executor();
    dim<2> out_dim{row_idxs->get_size(), this->get_size()[1]};
    auto result = MultiVector::create(exec, out_dim);
    this->row_gather(row_idxs, result);
    return result;
}


namespace {


template <typename ValueType, typename Function>
void gather_mixed_real_complex(Function fn, AbstractMultiVector* out)
{
#ifdef GINKGO_MIXED_PRECISION
    run<matrix::MultiVector, ValueType, next_precision<ValueType>,
        next_precision<ValueType, 2>, next_precision<ValueType, 3>>(out, fn);
#else
    fn(as<MultiVector<ValueType>>(out->as_precision(precision_v<ValueType>))
           .get());
#endif
}


}  // namespace


template <typename ValueType>
void MultiVector<ValueType>::row_gather(
    const array<int32>* row_idxs,
    ptr_param<AbstractMultiVector> row_collection) const
{
    gather_mixed_real_complex<ValueType>(
        [&](auto dense) { this->row_gather_impl(row_idxs, dense); },
        row_collection.get());
}


template <typename ValueType>
void MultiVector<ValueType>::row_gather(
    const array<int64>* row_idxs,
    ptr_param<AbstractMultiVector> row_collection) const
{
    gather_mixed_real_complex<ValueType>(
        [&](auto dense) { this->row_gather_impl(row_idxs, dense); },
        row_collection.get());
}


template <typename ValueType>
void MultiVector<ValueType>::row_gather(
    ptr_param<const AbstractMultiVector> alpha,
    const array<int32>* gather_indices,
    ptr_param<const AbstractMultiVector> beta,
    ptr_param<AbstractMultiVector> out) const
{
    auto dense_alpha = as<MultiVector>(alpha->as_precision(this));
    auto dense_beta = as<MultiVector>(beta->as_precision(this));
    GKO_ASSERT_EQUAL_DIMENSIONS(dense_alpha, gko::dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(dense_beta, gko::dim<2>(1, 1));
    gather_mixed_real_complex<ValueType>(
        [&](auto dense) {
            this->row_gather_impl(dense_alpha.get(), gather_indices,
                                  dense_beta.get(), dense);
        },
        out.get());
}

template <typename ValueType>
void MultiVector<ValueType>::row_gather(
    ptr_param<const AbstractMultiVector> alpha,
    const array<int64>* gather_indices,
    ptr_param<const AbstractMultiVector> beta,
    ptr_param<AbstractMultiVector> out) const
{
    auto dense_alpha = as<MultiVector>(alpha->as_precision(this));
    auto dense_beta = as<MultiVector>(beta->as_precision(this));
    GKO_ASSERT_EQUAL_DIMENSIONS(dense_alpha, gko::dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(dense_beta, gko::dim<2>(1, 1));
    gather_mixed_real_complex<ValueType>(
        [&](auto dense) {
            this->row_gather_impl(dense_alpha.get(), gather_indices,
                                  dense_beta.get(), dense);
        },
        out.get());
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::column_permute(
    const array<int32>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->column_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::column_permute(
    const array<int64>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->column_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
void MultiVector<ValueType>::column_permute(
    const array<int32>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::columns, output.get());
}


template <typename ValueType>
void MultiVector<ValueType>::column_permute(
    const array<int64>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::columns, output.get());
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::inverse_row_permute(
    const array<int32>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->inverse_row_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::inverse_row_permute(
    const array<int64>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->inverse_row_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
void MultiVector<ValueType>::inverse_row_permute(
    const array<int32>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_rows, output.get());
}


template <typename ValueType>
void MultiVector<ValueType>::inverse_row_permute(
    const array<int64>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_rows, output.get());
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::inverse_column_permute(
    const array<int32>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->inverse_column_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::inverse_column_permute(
    const array<int64>* permutation_indices) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->inverse_column_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
void MultiVector<ValueType>::inverse_column_permute(
    const array<int32>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_columns, output.get());
}


template <typename ValueType>
void MultiVector<ValueType>::inverse_column_permute(
    const array<int64>* permutation_indices,
    ptr_param<MultiVector<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_columns, output.get());
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int32>> permutation,
    permute_mode mode) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->scale_permute(permutation, result, mode);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int64>> permutation,
    permute_mode mode) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->scale_permute(permutation, result, mode);
    return result;
}


template <typename ValueType>
void MultiVector<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int32>> permutation,
    ptr_param<MultiVector> output, permute_mode mode) const
{
    this->scale_permute_impl(permutation.get(), mode, output.get());
}


template <typename ValueType>
void MultiVector<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int64>> permutation,
    ptr_param<MultiVector> output, permute_mode mode) const
{
    this->scale_permute_impl(permutation.get(), mode, output.get());
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int32>> row_permutation,
    ptr_param<const ScaledPermutation<value_type, int32>> col_permutation,
    bool invert) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->scale_permute(row_permutation, col_permutation, result, invert);
    return result;
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int64>> row_permutation,
    ptr_param<const ScaledPermutation<value_type, int64>> col_permutation,
    bool invert) const
{
    auto result = MultiVector::create(this->get_executor(), this->get_size());
    this->scale_permute(row_permutation, col_permutation, result, invert);
    return result;
}


template <typename ValueType>
void MultiVector<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int32>> row_permutation,
    ptr_param<const ScaledPermutation<value_type, int32>> col_permutation,
    ptr_param<MultiVector> output, bool invert) const
{
    this->scale_permute_impl(row_permutation.get(), col_permutation.get(),
                             invert, output.get());
}


template <typename ValueType>
void MultiVector<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int64>> row_permutation,
    ptr_param<const ScaledPermutation<value_type, int64>> col_permutation,
    ptr_param<MultiVector> output, bool invert) const
{
    this->scale_permute_impl(row_permutation.get(), col_permutation.get(),
                             invert, output.get());
}


template <typename ValueType>
typename MultiVector<ValueType>::device_view
MultiVector<ValueType>::get_device_view()
{
    return device_view{this->get_size(), this->get_stride(),
                       this->get_values()};
}


template <typename ValueType>
typename MultiVector<ValueType>::const_device_view
MultiVector<ValueType>::get_const_device_view() const
{
    return const_device_view{this->get_size(), this->get_stride(),
                             this->get_const_values()};
}


template <typename ValueType>
template <typename OtherValueType, typename>
temporary_conversion<MultiVector<OtherValueType>>
MultiVector<ValueType>::as_precision()
{
    if constexpr (is_complex<ValueType>() == is_complex<OtherValueType>()) {
        // The value types are either both real or both complex
        return temporary_conversion<MultiVector<OtherValueType>>::create(this);
    } else {
        // Conversions from real to complex (or vice versa) are not allowed.
        GKO_NOT_IMPLEMENTED;
    }
}

#define GKO_DECLARE_MULTIVECTOR_AS_PRECISION(ValueType, OtherValueType) \
    auto MultiVector<ValueType>::as_precision()                         \
        ->temporary_conversion<MultiVector<OtherValueType>>
#define GKO_DECLARE_MULTIVECTOR_AS_PRECISION_same(ValueType) \
    GKO_DECLARE_MULTIVECTOR_AS_PRECISION(ValueType, ValueType)
GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(GKO_DECLARE_MULTIVECTOR_AS_PRECISION);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIVECTOR_AS_PRECISION_same);


template <typename ValueType>
template <typename OtherValueType, typename>
temporary_conversion<const MultiVector<OtherValueType>>
MultiVector<ValueType>::as_precision() const
{
    if constexpr (is_complex<ValueType>() == is_complex<OtherValueType>()) {
        // The value types are either both real or both complex
        return temporary_conversion<const MultiVector<OtherValueType>>::create(
            this);
    } else {
        // Conversions from real to complex (or vice versa) are not allowed.
        GKO_NOT_IMPLEMENTED;
    }
}

#define GKO_DECLARE_MULTIVECTOR_CONST_AS_PRECISION(ValueType, OtherValueType) \
    auto MultiVector<ValueType>::as_precision()                               \
        const->temporary_conversion<const MultiVector<OtherValueType>>
#define GKO_DECLARE_MULTIVECTOR_CONST_AS_PRECISION_same(ValueType) \
    GKO_DECLARE_MULTIVECTOR_CONST_AS_PRECISION(ValueType, ValueType)
GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(
    GKO_DECLARE_MULTIVECTOR_CONST_AS_PRECISION);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_CONST_AS_PRECISION_same);


template <typename ValueType>
void MultiVector<ValueType>::compute_absolute_inplace_impl()
{
    this->get_executor()->run(
        multivector::make_inplace_absolute_dense(this->get_device_view()));
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_same_config_impl() const
{
    return MultiVector::create(this->get_executor(), this->get_size(),
                               this->get_stride());
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_with_type_of_impl(
    std::shared_ptr<const Executor> exec, const dim<2>& global_size,
    const dim<2>& local_size, size_type stride) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(global_size, local_size);
    return create_with_type_of_impl(std::move(exec), global_size, stride);
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_subview_impl(local_span rows, local_span columns)
{
    return create_subview_impl(rows, columns,
                               dim<2>(rows.length(), columns.length()));
}


template <typename ValueType>
std::unique_ptr<const MultiVector<ValueType>>
MultiVector<ValueType>::create_subview_impl(local_span rows,
                                            local_span columns) const
{
    return const_cast<MultiVector&>(*this).create_subview(rows, columns);
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>>
MultiVector<ValueType>::create_subview_impl(local_span rows, local_span columns,
                                            dim<2> global_size)
{
    dim<2> actual_size{rows.length(), columns.length()};
    GKO_ASSERT_EQUAL_DIMENSIONS(actual_size, global_size);

    row_major_range range_this{this->get_values(), this->get_size()[0],
                               this->get_size()[1], this->get_stride()};
    auto sub_range = range_this(rows, columns);
    size_type storage_size =
        rows.length() > 0 ? sub_range.length(1) +
                                (sub_range.length(0) - 1) * this->get_stride()
                          : 0;
    return MultiVector::create(
        this->get_executor(), dim<2>{sub_range.length(0), sub_range.length(1)},
        make_array_view(this->get_executor(), storage_size, sub_range->data),
        this->get_stride());
}


template <typename ValueType>
std::unique_ptr<const MultiVector<ValueType>>
MultiVector<ValueType>::create_subview_impl(local_span rows, local_span columns,
                                            dim<2> global_size) const
{
    dim<2> actual_size{rows.length(), columns.length()};
    GKO_ASSERT_EQUAL_DIMENSIONS(actual_size, global_size);
    return const_cast<MultiVector&>(*this).create_subview(rows, columns);
}


template <typename ValueType>
std::unique_ptr<const typename MultiVector<ValueType>::real_type>
MultiVector<ValueType>::create_real_view_impl() const
{
    const auto num_rows = this->get_size()[0];
    constexpr bool complex = is_complex<ValueType>();
    const auto num_cols =
        complex ? 2 * this->get_size()[1] : this->get_size()[1];
    const auto stride = complex ? 2 * this->get_stride() : this->get_stride();

    return MultiVector<remove_complex<ValueType>>::create_const(
        this->get_executor(), dim<2>{num_rows, num_cols},
        make_const_array_view(
            this->get_executor(), num_rows * stride,
            reinterpret_cast<const remove_complex<ValueType>*>(
                this->get_const_values())),
        stride);
}


template <typename ValueType>
std::unique_ptr<typename MultiVector<ValueType>::real_type>
MultiVector<ValueType>::create_real_view_impl()
{
    const auto num_rows = this->get_size()[0];
    constexpr bool complex = is_complex<ValueType>();
    const auto num_cols =
        complex ? 2 * this->get_size()[1] : this->get_size()[1];
    const auto stride = complex ? 2 * this->get_stride() : this->get_stride();

    return MultiVector<remove_complex<ValueType>>::create(
        this->get_executor(), dim<2>{num_rows, num_cols},
        make_array_view(
            this->get_executor(), num_rows * stride,
            reinterpret_cast<remove_complex<ValueType>*>(this->get_values())),
        stride);
}


template <typename ValueType>
std::unique_ptr<typename MultiVector<ValueType>::absolute_type>
MultiVector<ValueType>::compute_absolute_impl() const
{
    // do not inherit the stride
    auto result = absolute_type::create(this->get_executor(), this->get_size());
    this->compute_absolute(result);
    return result;
}


template <typename ValueType>
void MultiVector<ValueType>::compute_absolute_impl(absolute_type* result) const
{
    auto exec = this->get_executor();

    exec->run(multivector::make_outplace_absolute_dense(
        this->get_const_device_view(),
        make_temporary_output_clone(exec, result)->get_device_view()));
}


template <typename ValueType>
std::unique_ptr<typename MultiVector<ValueType>::complex_type>
MultiVector<ValueType>::make_complex_impl() const
{
    auto result = complex_type::create(this->get_executor(), this->get_size());
    this->make_complex(result);
    return result;
}


template <typename ValueType>
std::unique_ptr<typename MultiVector<ValueType>::real_type>
MultiVector<ValueType>::get_real_impl() const
{
    auto result = real_type::create(this->get_executor(), this->get_size());
    this->get_real(result);
    return result;
}


template <typename ValueType>
std::unique_ptr<typename MultiVector<ValueType>::real_type>
MultiVector<ValueType>::get_imag_impl() const
{
    auto result = real_type::create(this->get_executor(), this->get_size());
    this->get_imag(result);
    return result;
}


template <typename ValueType>
void MultiVector<ValueType>::make_complex_impl(complex_type* result) const
{
    auto exec = this->get_executor();

    exec->run(multivector::make_make_complex(
        this->get_const_device_view(),
        make_temporary_output_clone(exec, result)->get_device_view()));
}


template <typename ValueType>
void MultiVector<ValueType>::get_real_impl(real_type* result) const
{
    auto exec = this->get_executor();

    exec->run(multivector::make_get_real(this->get_const_device_view(),
                                         result->get_device_view()));
}


template <typename ValueType>
void MultiVector<ValueType>::get_imag_impl(real_type* result) const
{
    auto exec = this->get_executor();

    exec->run(multivector::make_get_imag(this->get_const_device_view(),
                                         result->get_device_view()));
}


template <typename ValueType>
void MultiVector<ValueType>::fill_impl(value_type value)
{
    this->get_executor()->run(
        multivector::make_fill(this->get_device_view(), value));
}


template <typename ValueType>
void MultiVector<ValueType>::scale_impl(scaling_param<value_type> alpha)
{
    std::visit(
        [this](auto alpha_v) {
            auto exec = this->get_executor();
            exec->run(multivector::make_scale(alpha_v->get_const_device_view(),
                                              this->get_device_view()));
        },
        alpha.variant);
}


template <typename ValueType>
void MultiVector<ValueType>::inv_scale_impl(scaling_param<value_type> alpha)
{
    std::visit(
        [this](auto alpha_v) {
            auto exec = this->get_executor();
            exec->run(multivector::make_inv_scale(
                alpha_v->get_const_device_view(), this->get_device_view()));
        },
        alpha.variant);
}


template <typename ValueType>
void MultiVector<ValueType>::add_scaled_impl(scaling_param<value_type> alpha,
                                             const MultiVector* b)
{
    std::visit(
        [this, b](auto alpha_v) {
            auto exec = this->get_executor();
            exec->run(multivector::make_add_scaled(
                alpha_v->get_const_device_view(), b->get_const_device_view(),
                this->get_device_view()));
        },
        alpha.variant);
}


template <typename ValueType>
void MultiVector<ValueType>::sub_scaled_impl(scaling_param<value_type> alpha,
                                             const MultiVector* b)
{
    std::visit(
        [this, b](auto alpha_v) {
            auto exec = this->get_executor();

            exec->run(multivector::make_sub_scaled(
                alpha_v->get_const_device_view(), b->get_const_device_view(),
                this->get_device_view()));
        },
        alpha.variant);
}


template <typename ValueType>
void MultiVector<ValueType>::compute_dot_impl(
    const MultiVector* b, matrix::MultiVector<value_type>* result,
    array<char>& tmp) const
{
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    exec->run(multivector::make_compute_dot(this->get_const_device_view(),
                                            b->get_const_device_view(),
                                            result->get_device_view(), tmp));
}


template <typename ValueType>
void MultiVector<ValueType>::compute_conj_dot_impl(
    const MultiVector* b, matrix::MultiVector<value_type>* result,
    array<char>& tmp) const
{
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    exec->run(multivector::make_compute_conj_dot(
        this->get_const_device_view(), b->get_const_device_view(),
        result->get_device_view(), tmp));
}


template <typename ValueType>
void MultiVector<ValueType>::compute_norm2_impl(norm_type* result,
                                                array<char>& tmp) const
{
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    exec->run(multivector::make_compute_norm2(this->get_const_device_view(),
                                              result->get_device_view(), tmp));
}


template <typename ValueType>
void MultiVector<ValueType>::compute_squared_norm2_impl(norm_type* result,
                                                        array<char>& tmp) const
{
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    exec->run(multivector::make_compute_squared_norm2(
        this->get_const_device_view(), result->get_device_view(), tmp));
}


template <typename ValueType>
void MultiVector<ValueType>::compute_norm1_impl(norm_type* result,
                                                array<char>& tmp) const
{
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    exec->run(multivector::make_compute_norm1(this->get_const_device_view(),
                                              result->get_device_view(), tmp));
}


template <typename ValueType>
AbstractMultiVector::device_view<typename MultiVector<ValueType>::value_type>
MultiVector<ValueType>::get_local_device_view_impl()
{
    return this->get_device_view();
}


template <typename ValueType>
AbstractMultiVector::device_view<
    const typename MultiVector<ValueType>::value_type>
MultiVector<ValueType>::get_const_local_device_view_impl() const
{
    return this->get_const_device_view();
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::create(
    std::shared_ptr<const Executor> exec, const dim<2>& size, size_type stride)
{
    return std::unique_ptr<MultiVector>{new MultiVector{exec, size, stride}};
}


template <typename ValueType>
std::unique_ptr<MultiVector<ValueType>> MultiVector<ValueType>::create(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    array<value_type> values, size_type stride)
{
    return std::unique_ptr<MultiVector>{
        new MultiVector{exec, size, std::move(values), stride}};
}


template <typename ValueType>
std::unique_ptr<const MultiVector<ValueType>>
MultiVector<ValueType>::create_const(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    gko::detail::const_array_view<ValueType>&& values, size_type stride)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const MultiVector>{new MultiVector{
        exec, size, gko::detail::array_const_cast(std::move(values)), stride}};
}


template <typename ValueType>
std::unique_ptr<const Dense<ValueType>>
MultiVector<ValueType>::as_const_dense_view() const
{
    return Dense<ValueType>::create_const(this->get_executor(),
                                          this->get_size(),
                                          values_.as_const_view(), stride_);
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> MultiVector<ValueType>::as_dense_view()
{
    return Dense<ValueType>::create(this->get_executor(), this->get_size(),
                                    values_.as_view(), stride_);
}


template <typename ValueType>
MultiVector<ValueType>::MultiVector(std::shared_ptr<const Executor> exec,
                                    const dim<2>& size, size_type stride)
    : EnableMultiVector<MultiVector>(exec, size),
      stride_(stride == 0 ? size[1] : stride),
      values_(exec, size[0] * stride_)
{}


template <typename ValueType>
MultiVector<ValueType>::MultiVector(std::shared_ptr<const Executor> exec,
                                    const dim<2>& size,
                                    array<value_type> values, size_type stride)
    : EnableMultiVector<MultiVector>(exec, size),
      stride_{stride},
      values_{exec, std::move(values)}
{
    if (size[0] > 0 && size[1] > 0) {
        GKO_ENSURE_IN_BOUNDS((size[0] - 1) * stride + size[1] - 1,
                             values_.get_size());
    }
}


#define GKO_DECLARE_MULTIVECTOR_MATRIX(ValueType) class MultiVector<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIVECTOR_MATRIX);


}  // namespace matrix
}  // namespace gko
