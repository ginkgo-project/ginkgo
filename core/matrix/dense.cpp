// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/dense.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/array_access.hpp"
#include "core/base/dispatch_helper.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/matrix/hybrid_kernels.hpp"
#include "core/matrix/permutation.hpp"


namespace gko {
namespace matrix {
namespace dense {
namespace {


GKO_REGISTER_OPERATION(simple_apply, dense::simple_apply);
GKO_REGISTER_OPERATION(apply, dense::apply);
GKO_REGISTER_OPERATION(copy, dense::copy);
GKO_REGISTER_OPERATION(fill, dense::fill);
GKO_REGISTER_OPERATION(scale, dense::scale);
GKO_REGISTER_OPERATION(inv_scale, dense::inv_scale);
GKO_REGISTER_OPERATION(add_scaled, dense::add_scaled);
GKO_REGISTER_OPERATION(sub_scaled, dense::sub_scaled);
GKO_REGISTER_OPERATION(add_scaled_diag, dense::add_scaled_diag);
GKO_REGISTER_OPERATION(sub_scaled_diag, dense::sub_scaled_diag);
GKO_REGISTER_OPERATION(compute_dot, dense::compute_dot_dispatch);
GKO_REGISTER_OPERATION(compute_conj_dot, dense::compute_conj_dot_dispatch);
GKO_REGISTER_OPERATION(compute_norm2, dense::compute_norm2_dispatch);
GKO_REGISTER_OPERATION(compute_norm1, dense::compute_norm1);
GKO_REGISTER_OPERATION(compute_mean, dense::compute_mean);
GKO_REGISTER_OPERATION(compute_squared_norm2, dense::compute_squared_norm2);
GKO_REGISTER_OPERATION(compute_sqrt, dense::compute_sqrt);
GKO_REGISTER_OPERATION(compute_max_nnz_per_row, dense::compute_max_nnz_per_row);
GKO_REGISTER_OPERATION(compute_hybrid_coo_row_ptrs,
                       hybrid::compute_coo_row_ptrs);
GKO_REGISTER_OPERATION(count_nonzeros_per_row, dense::count_nonzeros_per_row);
GKO_REGISTER_OPERATION(count_nonzero_blocks_per_row,
                       dense::count_nonzero_blocks_per_row);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(compute_slice_sets, dense::compute_slice_sets);
GKO_REGISTER_OPERATION(transpose, dense::transpose);
GKO_REGISTER_OPERATION(conj_transpose, dense::conj_transpose);
GKO_REGISTER_OPERATION(symm_permute, dense::symm_permute);
GKO_REGISTER_OPERATION(inv_symm_permute, dense::inv_symm_permute);
GKO_REGISTER_OPERATION(nonsymm_permute, dense::nonsymm_permute);
GKO_REGISTER_OPERATION(inv_nonsymm_permute, dense::inv_nonsymm_permute);
GKO_REGISTER_OPERATION(row_gather, dense::row_gather);
GKO_REGISTER_OPERATION(advanced_row_gather, dense::advanced_row_gather);
GKO_REGISTER_OPERATION(col_permute, dense::col_permute);
GKO_REGISTER_OPERATION(inverse_row_permute, dense::inv_row_permute);
GKO_REGISTER_OPERATION(inverse_col_permute, dense::inv_col_permute);
GKO_REGISTER_OPERATION(symm_scale_permute, dense::symm_scale_permute);
GKO_REGISTER_OPERATION(inv_symm_scale_permute, dense::inv_symm_scale_permute);
GKO_REGISTER_OPERATION(nonsymm_scale_permute, dense::nonsymm_scale_permute);
GKO_REGISTER_OPERATION(inv_nonsymm_scale_permute,
                       dense::inv_nonsymm_scale_permute);
GKO_REGISTER_OPERATION(row_scale_permute, dense::row_scale_permute);
GKO_REGISTER_OPERATION(col_scale_permute, dense::col_scale_permute);
GKO_REGISTER_OPERATION(inv_row_scale_permute, dense::inv_row_scale_permute);
GKO_REGISTER_OPERATION(inv_col_scale_permute, dense::inv_col_scale_permute);
GKO_REGISTER_OPERATION(fill_in_matrix_data, dense::fill_in_matrix_data);
GKO_REGISTER_OPERATION(convert_to_coo, dense::convert_to_coo);
GKO_REGISTER_OPERATION(convert_to_csr, dense::convert_to_csr);
GKO_REGISTER_OPERATION(convert_to_ell, dense::convert_to_ell);
GKO_REGISTER_OPERATION(convert_to_fbcsr, dense::convert_to_fbcsr);
GKO_REGISTER_OPERATION(convert_to_hybrid, dense::convert_to_hybrid);
GKO_REGISTER_OPERATION(convert_to_sellp, dense::convert_to_sellp);
GKO_REGISTER_OPERATION(convert_to_sparsity_csr, dense::convert_to_sparsity_csr);
GKO_REGISTER_OPERATION(extract_diagonal, dense::extract_diagonal);
GKO_REGISTER_OPERATION(inplace_absolute_dense, dense::inplace_absolute_dense);
GKO_REGISTER_OPERATION(outplace_absolute_dense, dense::outplace_absolute_dense);
GKO_REGISTER_OPERATION(make_complex, dense::make_complex);
GKO_REGISTER_OPERATION(get_real, dense::get_real);
GKO_REGISTER_OPERATION(get_imag, dense::get_imag);
GKO_REGISTER_OPERATION(add_scaled_identity, dense::add_scaled_identity);


}  // anonymous namespace
}  // namespace dense


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(
                dense::make_simple_apply(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                  const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            this->get_executor()->run(dense::make_apply(
                dense_alpha, this, dense_b, dense_beta, dense_x));
        },
        alpha, b, beta, x);
}


template <typename ValueType>
void Dense<ValueType>::fill(const ValueType value)
{
    this->get_executor()->run(dense::make_fill(this, value));
}


template <typename ValueType>
void Dense<ValueType>::scale(ptr_param<const LinOp> alpha)
{
    auto exec = this->get_executor();
    this->scale_impl(make_temporary_clone(exec, alpha).get());
}


template <typename ValueType>
void Dense<ValueType>::inv_scale(ptr_param<const LinOp> alpha)
{
    auto exec = this->get_executor();
    this->inv_scale_impl(make_temporary_clone(exec, alpha).get());
}


template <typename ValueType>
void Dense<ValueType>::add_scaled(ptr_param<const LinOp> alpha,
                                  ptr_param<const LinOp> b)
{
    auto exec = this->get_executor();
    this->add_scaled_impl(make_temporary_clone(exec, alpha).get(),
                          make_temporary_clone(exec, b).get());
}


template <typename ValueType>
void Dense<ValueType>::sub_scaled(ptr_param<const LinOp> alpha,
                                  ptr_param<const LinOp> b)
{
    auto exec = this->get_executor();
    this->sub_scaled_impl(make_temporary_clone(exec, alpha).get(),
                          make_temporary_clone(exec, b).get());
}


template <typename ValueType>
void Dense<ValueType>::compute_dot(ptr_param<const LinOp> b,
                                   ptr_param<LinOp> result) const
{
    auto exec = this->get_executor();
    this->compute_dot_impl(make_temporary_clone(exec, b).get(),
                           make_temporary_output_clone(exec, result).get());
}


template <typename ValueType>
void Dense<ValueType>::compute_conj_dot(ptr_param<const LinOp> b,
                                        ptr_param<LinOp> result) const
{
    auto exec = this->get_executor();
    this->compute_conj_dot_impl(
        make_temporary_clone(exec, b).get(),
        make_temporary_output_clone(exec, result).get());
}


template <typename ValueType>
void Dense<ValueType>::compute_norm2(ptr_param<LinOp> result) const
{
    auto exec = this->get_executor();
    this->compute_norm2_impl(make_temporary_output_clone(exec, result).get());
}


template <typename ValueType>
void Dense<ValueType>::compute_norm1(ptr_param<LinOp> result) const
{
    auto exec = this->get_executor();
    this->compute_norm1_impl(make_temporary_output_clone(exec, result).get());
}


template <typename ValueType>
void Dense<ValueType>::compute_squared_norm2(ptr_param<LinOp> result) const
{
    auto exec = this->get_executor();
    this->compute_squared_norm2_impl(
        make_temporary_output_clone(exec, result).get());
}


template <typename ValueType>
void Dense<ValueType>::inv_scale_impl(const LinOp* alpha)
{
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    auto exec = this->get_executor();
    // if alpha is real (convertible to real) and ValueType complex
    if (dynamic_cast<const ConvertibleTo<Dense<>>*>(alpha) &&
        is_complex<ValueType>()) {
        // use the real-complex kernel
        exec->run(dense::make_inv_scale(
            make_temporary_conversion<remove_complex<ValueType>>(alpha).get(),
            dynamic_cast<complex_type*>(this)));
        // this last cast is a no-op for complex value type and the branch is
        // never taken for real value type
    } else {
        // otherwise: use the normal kernel
        exec->run(dense::make_inv_scale(
            make_temporary_conversion<ValueType>(alpha).get(), this));
    }
}


template <typename ValueType>
void Dense<ValueType>::scale_impl(const LinOp* alpha)
{
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    auto exec = this->get_executor();
    // if alpha is real (convertible to real) and ValueType complex
    if (dynamic_cast<const ConvertibleTo<Dense<>>*>(alpha) &&
        is_complex<ValueType>()) {
        // use the real-complex kernel
        exec->run(dense::make_scale(
            make_temporary_conversion<remove_complex<ValueType>>(alpha).get(),
            dynamic_cast<complex_type*>(this)));
        // this last cast is a no-op for complex value type and the branch is
        // never taken for real value type
    } else {
        // otherwise: use the normal kernel
        exec->run(dense::make_scale(
            make_temporary_conversion<ValueType>(alpha).get(), this));
    }
}


template <typename ValueType>
void Dense<ValueType>::add_scaled_impl(const LinOp* alpha, const LinOp* b)
{
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    auto exec = this->get_executor();

    // if alpha is real and value type complex
    if (dynamic_cast<const ConvertibleTo<Dense<>>*>(alpha) &&
        is_complex<ValueType>()) {
        exec->run(dense::make_add_scaled(
            make_temporary_conversion<remove_complex<ValueType>>(alpha).get(),
            make_temporary_conversion<to_complex<ValueType>>(b).get(),
            dynamic_cast<complex_type*>(this)));
    } else {
        if (dynamic_cast<const Diagonal<ValueType>*>(b)) {
            exec->run(dense::make_add_scaled_diag(
                make_temporary_conversion<ValueType>(alpha).get(),
                dynamic_cast<const Diagonal<ValueType>*>(b), this));
        } else {
            exec->run(dense::make_add_scaled(
                make_temporary_conversion<ValueType>(alpha).get(),
                make_temporary_conversion<ValueType>(b).get(), this));
        }
    }
}


template <typename ValueType>
void Dense<ValueType>::sub_scaled_impl(const LinOp* alpha, const LinOp* b)
{
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    auto exec = this->get_executor();

    if (dynamic_cast<const ConvertibleTo<Dense<>>*>(alpha) &&
        is_complex<ValueType>()) {
        exec->run(dense::make_sub_scaled(
            make_temporary_conversion<remove_complex<ValueType>>(alpha).get(),
            make_temporary_conversion<to_complex<ValueType>>(b).get(),
            dynamic_cast<complex_type*>(this)));
    } else {
        if (dynamic_cast<const Diagonal<ValueType>*>(b)) {
            exec->run(dense::make_sub_scaled_diag(
                make_temporary_conversion<ValueType>(alpha).get(),
                dynamic_cast<const Diagonal<ValueType>*>(b), this));
        } else {
            exec->run(dense::make_sub_scaled(
                make_temporary_conversion<ValueType>(alpha).get(),
                make_temporary_conversion<ValueType>(b).get(), this));
        }
    }
}


template <typename ValueType>
void Dense<ValueType>::compute_dot(ptr_param<const LinOp> b,
                                   ptr_param<LinOp> result,
                                   array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    auto local_b = make_temporary_clone(exec, b);
    auto local_res = make_temporary_clone(exec, result);
    auto dense_b = make_temporary_conversion<ValueType>(local_b.get());
    auto dense_res = make_temporary_conversion<ValueType>(local_res.get());
    exec->run(
        dense::make_compute_dot(this, dense_b.get(), dense_res.get(), tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_dot_impl(const LinOp* b, LinOp* result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    auto dense_b = make_temporary_conversion<ValueType>(b);
    auto dense_res = make_temporary_conversion<ValueType>(result);
    array<char> tmp{exec};
    exec->run(
        dense::make_compute_dot(this, dense_b.get(), dense_res.get(), tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_conj_dot(ptr_param<const LinOp> b,
                                        ptr_param<LinOp> result,
                                        array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    auto local_b = make_temporary_clone(exec, b);
    auto local_res = make_temporary_clone(exec, result);
    auto dense_b = make_temporary_conversion<ValueType>(local_b.get());
    auto dense_res = make_temporary_conversion<ValueType>(local_res.get());
    exec->run(dense::make_compute_conj_dot(this, dense_b.get(), dense_res.get(),
                                           tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_conj_dot_impl(const LinOp* b,
                                             LinOp* result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    auto dense_b = make_temporary_conversion<ValueType>(b);
    auto dense_res = make_temporary_conversion<ValueType>(result);
    array<char> tmp{exec};
    exec->run(dense::make_compute_conj_dot(this, dense_b.get(), dense_res.get(),
                                           tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_norm2(ptr_param<LinOp> result,
                                     array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    auto local_result = make_temporary_clone(exec, result);
    auto dense_res = make_temporary_conversion<remove_complex<ValueType>>(
        local_result.get());
    exec->run(dense::make_compute_norm2(this, dense_res.get(), tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_norm2_impl(LinOp* result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    auto dense_res =
        make_temporary_conversion<remove_complex<ValueType>>(result);
    array<char> tmp{exec};
    exec->run(dense::make_compute_norm2(this, dense_res.get(), tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_norm1(ptr_param<LinOp> result,
                                     array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    auto local_result = make_temporary_clone(exec, result);
    auto dense_res = make_temporary_conversion<remove_complex<ValueType>>(
        local_result.get());
    exec->run(dense::make_compute_norm1(this, dense_res.get(), tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_norm1_impl(LinOp* result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    auto dense_res =
        make_temporary_conversion<remove_complex<ValueType>>(result);
    array<char> tmp{exec};
    exec->run(dense::make_compute_norm1(this, dense_res.get(), tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_squared_norm2(ptr_param<LinOp> result,
                                             array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    auto local_result = make_temporary_clone(exec, result);
    auto dense_res = make_temporary_conversion<remove_complex<ValueType>>(
        local_result.get());
    exec->run(dense::make_compute_squared_norm2(this, dense_res.get(), tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_mean(ptr_param<LinOp> result) const
{
    auto exec = this->get_executor();
    this->compute_mean_impl(make_temporary_output_clone(exec, result).get());
}


template <typename ValueType>
void Dense<ValueType>::compute_mean(ptr_param<LinOp> result,
                                    array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_COLS(result, this);
    auto exec = this->get_executor();
    if (tmp.get_executor() != exec) {
        tmp.clear();
        tmp.set_executor(exec);
    }
    auto dense_res = make_temporary_conversion<ValueType>(result);
    exec->run(dense::make_compute_mean(this, dense_res.get(), tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_squared_norm2_impl(LinOp* result) const
{
    auto exec = this->get_executor();
    array<char> tmp{exec};
    this->compute_squared_norm2(make_temporary_output_clone(exec, result).get(),
                                tmp);
}


template <typename ValueType>
void Dense<ValueType>::compute_mean_impl(LinOp* result) const
{
    auto exec = this->get_executor();
    array<char> tmp{exec};
    this->compute_mean(make_temporary_output_clone(exec, result).get(), tmp);
}


template <typename ValueType>
Dense<ValueType>& Dense<ValueType>::operator=(const Dense& other)
{
    if (&other != this) {
        auto old_size = this->get_size();
        EnableLinOp<Dense>::operator=(other);
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
            Dense{exec, this->get_size(),
                  make_array_view(exec, exec_values_array->get_size(),
                                  exec_values_array->get_data()),
                  this->get_stride()};
        exec->run(dense::make_copy(&other, &exec_this_view));
    }
    return *this;
}


template <typename ValueType>
Dense<ValueType>& Dense<ValueType>::operator=(Dense<ValueType>&& other)
{
    if (&other != this) {
        EnableLinOp<Dense>::operator=(std::move(other));
        values_ = std::move(other.values_);
        stride_ = std::exchange(other.stride_, 0);
    }
    return *this;
}


template <typename ValueType>
Dense<ValueType>::Dense(const Dense<ValueType>& other)
    : Dense(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
Dense<ValueType>::Dense(Dense<ValueType>&& other) : Dense(other.get_executor())
{
    *this = std::move(other);
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
    exec->run(dense::make_copy(
        this, make_temporary_output_clone(exec, result).get()));
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::convert_impl(Coo<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto num_rows = this->get_size()[0];

    array<int64> row_ptrs{exec, num_rows + 1};
    exec->run(dense::make_count_nonzeros_per_row(this, row_ptrs.get_data()));
    exec->run(
        dense::make_prefix_sum_nonnegative(row_ptrs.get_data(), num_rows + 1));
    const auto nnz = get_element(row_ptrs, num_rows);
    result->resize(this->get_size(), nnz);
    exec->run(
        dense::make_convert_to_coo(this, row_ptrs.get_const_data(),
                                   make_temporary_clone(exec, result).get()));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Coo<ValueType, int32>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Coo<ValueType, int32>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Coo<ValueType, int64>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Coo<ValueType, int64>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::convert_impl(Csr<ValueType, IndexType>* result) const
{
    {
        auto exec = this->get_executor();
        const auto num_rows = this->get_size()[0];
        auto tmp = make_temporary_clone(exec, result);
        tmp->row_ptrs_.resize_and_reset(num_rows + 1);
        exec->run(
            dense::make_count_nonzeros_per_row(this, tmp->get_row_ptrs()));
        exec->run(dense::make_prefix_sum_nonnegative(tmp->get_row_ptrs(),
                                                     num_rows + 1));
        const auto nnz =
            exec->copy_val_to_host(tmp->get_const_row_ptrs() + num_rows);
        tmp->col_idxs_.resize_and_reset(nnz);
        tmp->values_.resize_and_reset(nnz);
        tmp->set_size(this->get_size());
        exec->run(dense::make_convert_to_csr(this, tmp.get()));
    }
    result->make_srow();
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int32>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int32>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int64>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int64>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::convert_impl(Fbcsr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto bs = result->get_block_size();
    const auto row_blocks = detail::get_num_blocks(bs, this->get_size()[0]);
    const auto col_blocks = detail::get_num_blocks(bs, this->get_size()[1]);
    auto tmp = make_temporary_clone(exec, result);
    tmp->row_ptrs_.resize_and_reset(row_blocks + 1);
    exec->run(dense::make_count_nonzero_blocks_per_row(this, bs,
                                                       tmp->get_row_ptrs()));
    exec->run(dense::make_prefix_sum_nonnegative(tmp->get_row_ptrs(),
                                                 row_blocks + 1));
    const auto nnz_blocks =
        exec->copy_val_to_host(tmp->get_const_row_ptrs() + row_blocks);
    tmp->col_idxs_.resize_and_reset(nnz_blocks);
    tmp->values_.resize_and_reset(nnz_blocks * bs * bs);
    tmp->values_.fill(zero<ValueType>());
    tmp->set_size(this->get_size());
    exec->run(dense::make_convert_to_fbcsr(this, tmp.get()));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Fbcsr<ValueType, int32>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Fbcsr<ValueType, int32>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Fbcsr<ValueType, int64>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Fbcsr<ValueType, int64>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::convert_impl(Ell<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    size_type num_stored_elements_per_row{};
    exec->run(
        dense::make_compute_max_nnz_per_row(this, num_stored_elements_per_row));
    result->resize(this->get_size(), num_stored_elements_per_row);
    exec->run(dense::make_convert_to_ell(
        this, make_temporary_clone(exec, result).get()));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Ell<ValueType, int32>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Ell<ValueType, int32>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Ell<ValueType, int64>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Ell<ValueType, int64>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::convert_impl(Hybrid<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto num_rows = this->get_size()[0];
    const auto num_cols = this->get_size()[1];
    array<size_type> row_nnz{exec, num_rows};
    array<int64> coo_row_ptrs{exec, num_rows + 1};
    exec->run(dense::make_count_nonzeros_per_row(this, row_nnz.get_data()));
    size_type ell_lim{};
    size_type coo_nnz{};
    result->get_strategy()->compute_hybrid_config(row_nnz, &ell_lim, &coo_nnz);
    if (ell_lim > num_cols) {
        // TODO remove temporary fix after ELL gains true structural zeros
        ell_lim = num_cols;
    }
    exec->run(dense::make_compute_hybrid_coo_row_ptrs(row_nnz, ell_lim,
                                                      coo_row_ptrs.get_data()));
    coo_nnz = get_element(coo_row_ptrs, num_rows);
    auto tmp = make_temporary_clone(exec, result);
    tmp->resize(this->get_size(), ell_lim, coo_nnz);
    exec->run(dense::make_convert_to_hybrid(this, coo_row_ptrs.get_const_data(),
                                            tmp.get()));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Hybrid<ValueType, int32>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Hybrid<ValueType, int32>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Hybrid<ValueType, int64>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Hybrid<ValueType, int64>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::convert_impl(Sellp<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto num_rows = this->get_size()[0];
    const auto stride_factor = result->get_stride_factor();
    const auto slice_size = result->get_slice_size();
    const auto num_slices = ceildiv(num_rows, slice_size);
    auto tmp = make_temporary_clone(exec, result);
    tmp->stride_factor_ = stride_factor;
    tmp->slice_size_ = slice_size;
    tmp->slice_sets_.resize_and_reset(num_slices + 1);
    tmp->slice_lengths_.resize_and_reset(num_slices);
    exec->run(dense::make_compute_slice_sets(this, slice_size, stride_factor,
                                             tmp->get_slice_sets(),
                                             tmp->get_slice_lengths()));
    auto total_cols =
        exec->copy_val_to_host(tmp->get_slice_sets() + num_slices);
    tmp->col_idxs_.resize_and_reset(total_cols * slice_size);
    tmp->values_.resize_and_reset(total_cols * slice_size);
    tmp->set_size(this->get_size());
    exec->run(dense::make_convert_to_sellp(this, tmp.get()));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Sellp<ValueType, int32>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Sellp<ValueType, int32>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Sellp<ValueType, int64>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Sellp<ValueType, int64>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::convert_impl(
    SparsityCsr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto num_rows = this->get_size()[0];
    auto tmp = make_temporary_clone(exec, result);
    tmp->row_ptrs_.resize_and_reset(num_rows + 1);
    exec->run(
        dense::make_count_nonzeros_per_row(this, tmp->row_ptrs_.get_data()));
    exec->run(dense::make_prefix_sum_nonnegative(tmp->row_ptrs_.get_data(),
                                                 num_rows + 1));
    const auto nnz = get_element(tmp->row_ptrs_, num_rows);
    tmp->col_idxs_.resize_and_reset(nnz);
    tmp->value_.fill(one<ValueType>());
    tmp->set_size(this->get_size());
    exec->run(dense::make_convert_to_sparsity_csr(this, tmp.get()));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(SparsityCsr<ValueType, int32>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(SparsityCsr<ValueType, int32>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(SparsityCsr<ValueType, int64>* result) const
{
    this->convert_impl(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(SparsityCsr<ValueType, int64>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::resize(gko::dim<2> new_size)
{
    if (this->get_size() != new_size) {
        this->set_size(new_size);
        this->stride_ = new_size[1];
        this->values_.resize_and_reset(new_size[0] * this->get_stride());
    }
}


template <typename ValueType>
void Dense<ValueType>::read(const device_mat_data& data)
{
    auto exec = this->get_executor();
    this->resize(data.get_size());
    this->fill(zero<ValueType>());
    exec->run(dense::make_fill_in_matrix_data(
        *make_temporary_clone(exec, &data), this));
}


template <typename ValueType>
void Dense<ValueType>::read(const device_mat_data32& data)
{
    auto exec = this->get_executor();
    this->resize(data.get_size());
    this->fill(zero<ValueType>());
    exec->run(dense::make_fill_in_matrix_data(
        *make_temporary_clone(exec, &data), this));
}


template <typename ValueType>
void Dense<ValueType>::read(device_mat_data&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType>
void Dense<ValueType>::read(device_mat_data32&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType>
void Dense<ValueType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_from_host(this->get_executor(), data));
}


template <typename ValueType>
void Dense<ValueType>::read(const mat_data32& data)
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
void Dense<ValueType>::write(mat_data& data) const
{
    write_impl(this, data);
}


template <typename ValueType>
void Dense<ValueType>::write(mat_data32& data) const
{
    write_impl(this, data);
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
void Dense<ValueType>::transpose(ptr_param<Dense<ValueType>> output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(output, gko::transpose(this->get_size()));
    auto exec = this->get_executor();
    exec->run(dense::make_transpose(
        this, make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
void Dense<ValueType>::conj_transpose(ptr_param<Dense<ValueType>> output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(output, gko::transpose(this->get_size()));
    auto exec = this->get_executor();
    exec->run(dense::make_conj_transpose(
        this, make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::permute_impl(const Permutation<IndexType>* permutation,
                                    permute_mode mode, Dense* output) const
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
        exec->run(dense::make_row_gather(local_perm->get_const_permutation(),
                                         this, local_output.get()));
        break;
    case permute_mode::columns:
        exec->run(dense::make_col_permute(local_perm->get_const_permutation(),
                                          this, local_output.get()));
        break;
    case permute_mode::symmetric:
        exec->run(dense::make_symm_permute(local_perm->get_const_permutation(),
                                           this, local_output.get()));
        break;
    case permute_mode::inverse_rows:
        exec->run(dense::make_inverse_row_permute(
            local_perm->get_const_permutation(), this, local_output.get()));
        break;
    case permute_mode::inverse_columns:
        exec->run(dense::make_inverse_col_permute(
            local_perm->get_const_permutation(), this, local_output.get()));
        break;
    case permute_mode::inverse_symmetric:
        exec->run(dense::make_inv_symm_permute(
            local_perm->get_const_permutation(), this, local_output.get()));
        break;
    default:
        GKO_INVALID_STATE("Invalid permute mode");
    }
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::permute_impl(
    const Permutation<IndexType>* row_permutation,
    const Permutation<IndexType>* col_permutation, bool invert,
    Dense* output) const
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
        exec->run(dense::make_inv_nonsymm_permute(
            local_row_perm->get_const_permutation(),
            local_col_perm->get_const_permutation(), this, local_output.get()));
    } else {
        exec->run(dense::make_nonsymm_permute(
            local_row_perm->get_const_permutation(),
            local_col_perm->get_const_permutation(), this, local_output.get()));
    }
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::scale_permute_impl(
    const ScaledPermutation<ValueType, IndexType>* permutation,
    permute_mode mode, Dense* output) const
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
        exec->run(dense::make_row_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this, local_output.get()));
        break;
    case permute_mode::columns:
        exec->run(dense::make_col_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this, local_output.get()));
        break;
    case permute_mode::symmetric:
        exec->run(dense::make_symm_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this, local_output.get()));
        break;
    case permute_mode::inverse_rows:
        exec->run(dense::make_inv_row_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this, local_output.get()));
        break;
    case permute_mode::inverse_columns:
        exec->run(dense::make_inv_col_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this, local_output.get()));
        break;
    case permute_mode::inverse_symmetric:
        exec->run(dense::make_inv_symm_scale_permute(
            local_perm->get_const_scaling_factors(),
            local_perm->get_const_permutation(), this, local_output.get()));
        break;
    default:
        GKO_INVALID_STATE("Invalid permute mode");
    }
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::scale_permute_impl(
    const ScaledPermutation<ValueType, IndexType>* row_permutation,
    const ScaledPermutation<ValueType, IndexType>* col_permutation, bool invert,
    Dense* output) const
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
        exec->run(dense::make_inv_nonsymm_scale_permute(
            local_row_perm->get_const_scaling_factors(),
            local_row_perm->get_const_permutation(),
            local_col_perm->get_const_scaling_factors(),
            local_col_perm->get_const_permutation(), this, local_output.get()));
    } else {
        exec->run(dense::make_nonsymm_scale_permute(
            local_row_perm->get_const_scaling_factors(),
            local_row_perm->get_const_permutation(),
            local_col_perm->get_const_scaling_factors(),
            local_col_perm->get_const_permutation(), this, local_output.get()));
    }
}


template <typename ValueType>
template <typename OutputType, typename IndexType>
void Dense<ValueType>::row_gather_impl(const array<IndexType>* row_idxs,
                                       Dense<OutputType>* row_collection) const
{
    auto exec = this->get_executor();
    dim<2> expected_dim{row_idxs->get_size(), this->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(expected_dim, row_collection);

    exec->run(dense::make_row_gather(
        make_temporary_clone(exec, row_idxs)->get_const_data(), this,
        make_temporary_output_clone(exec, row_collection).get()));
}

template <typename ValueType>
template <typename OutputType, typename IndexType>
void Dense<ValueType>::row_gather_impl(const Dense<ValueType>* alpha,
                                       const array<IndexType>* row_idxs,
                                       const Dense<ValueType>* beta,
                                       Dense<OutputType>* row_collection) const
{
    auto exec = this->get_executor();
    dim<2> expected_dim{row_idxs->get_size(), this->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(expected_dim, row_collection);

    exec->run(dense::make_advanced_row_gather(
        make_temporary_clone(exec, alpha).get(),
        make_temporary_clone(exec, row_idxs)->get_const_data(), this,
        make_temporary_clone(exec, beta).get(),
        make_temporary_clone(exec, row_collection).get()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::permute(
    const array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::permute(
    const array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::permute(
    ptr_param<const Permutation<int32>> permutation, permute_mode mode) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->permute(permutation, result, mode);
    return result;
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::permute(
    ptr_param<const Permutation<int64>> permutation, permute_mode mode) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->permute(permutation, result, mode);
    return result;
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::permute(
    ptr_param<const Permutation<int32>> row_permutation,
    ptr_param<const Permutation<int32>> col_permutation, bool invert) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->permute(row_permutation, col_permutation, result, invert);
    return result;
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::permute(
    ptr_param<const Permutation<int64>> row_permutation,
    ptr_param<const Permutation<int64>> col_permutation, bool invert) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->permute(row_permutation, col_permutation, result, invert);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::permute(ptr_param<const Permutation<int32>> permutation,
                               ptr_param<Dense<ValueType>> result,
                               permute_mode mode) const
{
    this->permute_impl(permutation.get(), mode, result.get());
}


template <typename ValueType>
void Dense<ValueType>::permute(ptr_param<const Permutation<int64>> permutation,
                               ptr_param<Dense<ValueType>> result,
                               permute_mode mode) const
{
    this->permute_impl(permutation.get(), mode, result.get());
}


template <typename ValueType>
void Dense<ValueType>::permute(
    ptr_param<const Permutation<int32>> row_permutation,
    ptr_param<const Permutation<int32>> col_permutation,
    ptr_param<Dense<ValueType>> result, bool invert) const
{
    this->permute_impl(row_permutation.get(), col_permutation.get(), invert,
                       result.get());
}


template <typename ValueType>
void Dense<ValueType>::permute(
    ptr_param<const Permutation<int64>> row_permutation,
    ptr_param<const Permutation<int64>> col_permutation,
    ptr_param<Dense<ValueType>> result, bool invert) const
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
void Dense<ValueType>::permute(const array<int32>* permutation_indices,
                               ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::symmetric, output.get());
}


template <typename ValueType>
void Dense<ValueType>::permute(const array<int64>* permutation_indices,
                               ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::symmetric, output.get());
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_permute(
    const array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_permute(
    const array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::inverse_permute(const array<int32>* permutation_indices,
                                       ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_symmetric, output.get());
}


template <typename ValueType>
void Dense<ValueType>::inverse_permute(const array<int64>* permutation_indices,
                                       ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_symmetric, output.get());
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::row_permute(
    const array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->row_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::row_permute(
    const array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->row_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::row_permute(const array<int32>* permutation_indices,
                                   ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::rows, output.get());
}


template <typename ValueType>
void Dense<ValueType>::row_permute(const array<int64>* permutation_indices,
                                   ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::rows, output.get());
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::row_gather(
    const array<int32>* row_idxs) const
{
    auto exec = this->get_executor();
    dim<2> out_dim{row_idxs->get_size(), this->get_size()[1]};
    auto result = Dense::create(exec, out_dim);
    this->row_gather(row_idxs, result);
    return result;
}

template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::row_gather(
    const array<int64>* row_idxs) const
{
    auto exec = this->get_executor();
    dim<2> out_dim{row_idxs->get_size(), this->get_size()[1]};
    auto result = Dense::create(exec, out_dim);
    this->row_gather(row_idxs, result);
    return result;
}


namespace {


template <typename ValueType, typename Function>
void gather_mixed_real_complex(Function fn, LinOp* out)
{
#ifdef GINKGO_MIXED_PRECISION
    using fst_type = matrix::Dense<ValueType>;
    using snd_type = matrix::Dense<next_precision<ValueType>>;
    run<fst_type*, snd_type*>(out, fn);
#else
    precision_dispatch<ValueType>(fn, out);
#endif
}


}  // namespace


template <typename ValueType>
void Dense<ValueType>::row_gather(const array<int32>* row_idxs,
                                  ptr_param<LinOp> row_collection) const
{
    gather_mixed_real_complex<ValueType>(
        [&](auto dense) { this->row_gather_impl(row_idxs, dense); },
        row_collection.get());
}


template <typename ValueType>
void Dense<ValueType>::row_gather(const array<int64>* row_idxs,
                                  ptr_param<LinOp> row_collection) const
{
    gather_mixed_real_complex<ValueType>(
        [&](auto dense) { this->row_gather_impl(row_idxs, dense); },
        row_collection.get());
}


template <typename ValueType>
void Dense<ValueType>::row_gather(ptr_param<const LinOp> alpha,
                                  const array<int32>* gather_indices,
                                  ptr_param<const LinOp> beta,
                                  ptr_param<LinOp> out) const
{
    auto dense_alpha = make_temporary_conversion<ValueType>(alpha);
    auto dense_beta = make_temporary_conversion<ValueType>(beta);
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
void Dense<ValueType>::row_gather(ptr_param<const LinOp> alpha,
                                  const array<int64>* gather_indices,
                                  ptr_param<const LinOp> beta,
                                  ptr_param<LinOp> out) const
{
    auto dense_alpha = make_temporary_conversion<ValueType>(alpha);
    auto dense_beta = make_temporary_conversion<ValueType>(beta);
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
std::unique_ptr<LinOp> Dense<ValueType>::column_permute(
    const array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->column_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::column_permute(
    const array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->column_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::column_permute(const array<int32>* permutation_indices,
                                      ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::columns, output.get());
}


template <typename ValueType>
void Dense<ValueType>::column_permute(const array<int64>* permutation_indices,
                                      ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::columns, output.get());
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_row_permute(
    const array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_row_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_row_permute(
    const array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_row_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::inverse_row_permute(
    const array<int32>* permutation_indices,
    ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_rows, output.get());
}


template <typename ValueType>
void Dense<ValueType>::inverse_row_permute(
    const array<int64>* permutation_indices,
    ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_rows, output.get());
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_column_permute(
    const array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_column_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_column_permute(
    const array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_column_permute(permutation_indices, result);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::inverse_column_permute(
    const array<int32>* permutation_indices,
    ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_columns, output.get());
}


template <typename ValueType>
void Dense<ValueType>::inverse_column_permute(
    const array<int64>* permutation_indices,
    ptr_param<Dense<ValueType>> output) const
{
    this->permute_impl(create_permutation_view(*permutation_indices).get(),
                       permute_mode::inverse_columns, output.get());
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int32>> permutation,
    permute_mode mode) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->scale_permute(permutation, result, mode);
    return result;
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int64>> permutation,
    permute_mode mode) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->scale_permute(permutation, result, mode);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int32>> permutation,
    ptr_param<Dense> output, permute_mode mode) const
{
    this->scale_permute_impl(permutation.get(), mode, output.get());
}


template <typename ValueType>
void Dense<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int64>> permutation,
    ptr_param<Dense> output, permute_mode mode) const
{
    this->scale_permute_impl(permutation.get(), mode, output.get());
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int32>> row_permutation,
    ptr_param<const ScaledPermutation<value_type, int32>> col_permutation,
    bool invert) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->scale_permute(row_permutation, col_permutation, result, invert);
    return result;
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int64>> row_permutation,
    ptr_param<const ScaledPermutation<value_type, int64>> col_permutation,
    bool invert) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->scale_permute(row_permutation, col_permutation, result, invert);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int32>> row_permutation,
    ptr_param<const ScaledPermutation<value_type, int32>> col_permutation,
    ptr_param<Dense> output, bool invert) const
{
    this->scale_permute_impl(row_permutation.get(), col_permutation.get(),
                             invert, output.get());
}


template <typename ValueType>
void Dense<ValueType>::scale_permute(
    ptr_param<const ScaledPermutation<value_type, int64>> row_permutation,
    ptr_param<const ScaledPermutation<value_type, int64>> col_permutation,
    ptr_param<Dense> output, bool invert) const
{
    this->scale_permute_impl(row_permutation.get(), col_permutation.get(),
                             invert, output.get());
}


template <typename ValueType>
void Dense<ValueType>::extract_diagonal(
    ptr_param<Diagonal<ValueType>> output) const
{
    auto exec = this->get_executor();
    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    GKO_ASSERT_EQ(output->get_size()[0], diag_size);

    exec->run(dense::make_extract_diagonal(
        this, make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
std::unique_ptr<Diagonal<ValueType>> Dense<ValueType>::extract_diagonal() const
{
    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(this->get_executor(), diag_size);
    this->extract_diagonal(diag);
    return diag;
}


template <typename ValueType>
void Dense<ValueType>::compute_absolute_inplace()
{
    this->get_executor()->run(dense::make_inplace_absolute_dense(this));
}


template <typename ValueType>
std::unique_ptr<typename Dense<ValueType>::absolute_type>
Dense<ValueType>::compute_absolute() const
{
    // do not inherit the stride
    auto result = absolute_type::create(this->get_executor(), this->get_size());
    this->compute_absolute(result);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::compute_absolute(ptr_param<absolute_type> output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    auto exec = this->get_executor();

    exec->run(dense::make_outplace_absolute_dense(
        this, make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
std::unique_ptr<typename Dense<ValueType>::complex_type>
Dense<ValueType>::make_complex() const
{
    auto result = complex_type::create(this->get_executor(), this->get_size());
    this->make_complex(result);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::make_complex(ptr_param<complex_type> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
    auto exec = this->get_executor();

    exec->run(dense::make_make_complex(
        this, make_temporary_output_clone(exec, result).get()));
}


template <typename ValueType>
std::unique_ptr<typename Dense<ValueType>::real_type>
Dense<ValueType>::get_real() const
{
    auto result = real_type::create(this->get_executor(), this->get_size());
    this->get_real(result);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::get_real(ptr_param<real_type> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
    auto exec = this->get_executor();

    exec->run(dense::make_get_real(
        this, make_temporary_output_clone(exec, result).get()));
}


template <typename ValueType>
std::unique_ptr<typename Dense<ValueType>::real_type>
Dense<ValueType>::get_imag() const
{
    auto result = real_type::create(this->get_executor(), this->get_size());
    this->get_imag(result);
    return result;
}


template <typename ValueType>
void Dense<ValueType>::get_imag(ptr_param<real_type> result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
    auto exec = this->get_executor();

    exec->run(dense::make_get_imag(
        this, make_temporary_output_clone(exec, result).get()));
}


template <typename ValueType>
void Dense<ValueType>::add_scaled_identity_impl(const LinOp* const a,
                                                const LinOp* const b)
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_beta, auto dense_x) {
            this->get_executor()->run(dense::make_add_scaled_identity(
                dense_alpha, dense_beta, dense_x));
        },
        a, b, this);
}


template <typename ValueType>
std::unique_ptr<typename Dense<ValueType>::real_type>
Dense<ValueType>::create_real_view()
{
    const auto num_rows = this->get_size()[0];
    constexpr bool complex = is_complex<ValueType>();
    const auto num_cols =
        complex ? 2 * this->get_size()[1] : this->get_size()[1];
    const auto stride = complex ? 2 * this->get_stride() : this->get_stride();

    return Dense<remove_complex<ValueType>>::create(
        this->get_executor(), dim<2>{num_rows, num_cols},
        make_array_view(
            this->get_executor(), num_rows * stride,
            reinterpret_cast<remove_complex<ValueType>*>(this->get_values())),
        stride);
}


template <typename ValueType>
std::unique_ptr<const typename Dense<ValueType>::real_type>
Dense<ValueType>::create_real_view() const
{
    const auto num_rows = this->get_size()[0];
    constexpr bool complex = is_complex<ValueType>();
    const auto num_cols =
        complex ? 2 * this->get_size()[1] : this->get_size()[1];
    const auto stride = complex ? 2 * this->get_stride() : this->get_stride();

    return Dense<remove_complex<ValueType>>::create_const(
        this->get_executor(), dim<2>{num_rows, num_cols},
        make_const_array_view(
            this->get_executor(), num_rows * stride,
            reinterpret_cast<const remove_complex<ValueType>*>(
                this->get_const_values())),
        stride);
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::create_submatrix_impl(
    const span& rows, const span& columns, const size_type stride)
{
    row_major_range range_this{this->get_values(), this->get_size()[0],
                               this->get_size()[1], this->get_stride()};
    auto sub_range = range_this(rows, columns);
    size_type storage_size =
        rows.length() > 0 ? sub_range.length(1) +
                                (sub_range.length(0) - 1) * this->get_stride()
                          : 0;
    return Dense::create(
        this->get_executor(), dim<2>{sub_range.length(0), sub_range.length(1)},
        make_array_view(this->get_executor(), storage_size, sub_range->data),
        stride);
}


#define GKO_DECLARE_DENSE_MATRIX(_type) class Dense<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_MATRIX);


}  // namespace matrix
}  // namespace gko
