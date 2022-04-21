/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/matrix/dense.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/dispatch_helper.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/matrix/hybrid_kernels.hpp"


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
GKO_REGISTER_OPERATION(compute_squared_norm2, dense::compute_squared_norm2);
GKO_REGISTER_OPERATION(compute_sqrt, dense::compute_sqrt);
GKO_REGISTER_OPERATION(compute_max_nnz_per_row, dense::compute_max_nnz_per_row);
GKO_REGISTER_OPERATION(compute_hybrid_coo_row_ptrs,
                       hybrid::compute_coo_row_ptrs);
GKO_REGISTER_OPERATION(count_nonzeros_per_row, dense::count_nonzeros_per_row);
GKO_REGISTER_OPERATION(count_nonzero_blocks_per_row,
                       dense::count_nonzero_blocks_per_row);
GKO_REGISTER_OPERATION(prefix_sum, components::prefix_sum);
GKO_REGISTER_OPERATION(compute_slice_sets, dense::compute_slice_sets);
GKO_REGISTER_OPERATION(transpose, dense::transpose);
GKO_REGISTER_OPERATION(conj_transpose, dense::conj_transpose);
GKO_REGISTER_OPERATION(symm_permute, dense::symm_permute);
GKO_REGISTER_OPERATION(inv_symm_permute, dense::inv_symm_permute);
GKO_REGISTER_OPERATION(row_gather, dense::row_gather);
GKO_REGISTER_OPERATION(advanced_row_gather, dense::advanced_row_gather);
GKO_REGISTER_OPERATION(column_permute, dense::column_permute);
GKO_REGISTER_OPERATION(inverse_row_permute, dense::inverse_row_permute);
GKO_REGISTER_OPERATION(inverse_column_permute, dense::inverse_column_permute);
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
void Dense<ValueType>::scale(const LinOp* alpha)
{
    auto exec = this->get_executor();
    this->scale_impl(make_temporary_clone(exec, alpha).get());
}


template <typename ValueType>
void Dense<ValueType>::inv_scale(const LinOp* alpha)
{
    auto exec = this->get_executor();
    this->inv_scale_impl(make_temporary_clone(exec, alpha).get());
}


template <typename ValueType>
void Dense<ValueType>::add_scaled(const LinOp* alpha, const LinOp* b)
{
    auto exec = this->get_executor();
    this->add_scaled_impl(make_temporary_clone(exec, alpha).get(),
                          make_temporary_clone(exec, b).get());
}


template <typename ValueType>
void Dense<ValueType>::sub_scaled(const LinOp* alpha, const LinOp* b)
{
    auto exec = this->get_executor();
    this->sub_scaled_impl(make_temporary_clone(exec, alpha).get(),
                          make_temporary_clone(exec, b).get());
}


template <typename ValueType>
void Dense<ValueType>::compute_dot(const LinOp* b, LinOp* result) const
{
    auto exec = this->get_executor();
    this->compute_dot_impl(make_temporary_clone(exec, b).get(),
                           make_temporary_output_clone(exec, result).get());
}


template <typename ValueType>
void Dense<ValueType>::compute_conj_dot(const LinOp* b, LinOp* result) const
{
    auto exec = this->get_executor();
    this->compute_conj_dot_impl(
        make_temporary_clone(exec, b).get(),
        make_temporary_output_clone(exec, result).get());
}


template <typename ValueType>
void Dense<ValueType>::compute_norm2(LinOp* result) const
{
    auto exec = this->get_executor();
    this->compute_norm2_impl(make_temporary_output_clone(exec, result).get());
}


template <typename ValueType>
void Dense<ValueType>::compute_norm1(LinOp* result) const
{
    auto exec = this->get_executor();
    this->compute_norm1_impl(make_temporary_output_clone(exec, result).get());
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
void Dense<ValueType>::compute_dot(const LinOp* b, LinOp* result,
                                   Array<char>& tmp) const
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
    Array<char> tmp{exec};
    exec->run(
        dense::make_compute_dot(this, dense_b.get(), dense_res.get(), tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_conj_dot(const LinOp* b, LinOp* result,
                                        Array<char>& tmp) const
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
    Array<char> tmp{exec};
    exec->run(dense::make_compute_conj_dot(this, dense_b.get(), dense_res.get(),
                                           tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_norm2(LinOp* result, Array<char>& tmp) const
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
    Array<char> tmp{exec};
    exec->run(dense::make_compute_norm2(this, dense_res.get(), tmp));
}


template <typename ValueType>
void Dense<ValueType>::compute_norm1(LinOp* result, Array<char>& tmp) const
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
    Array<char> tmp{exec};
    exec->run(dense::make_compute_norm1(this, dense_res.get(), tmp));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Dense<ValueType>* result) const
{
    if (this->get_size() && result->get_size() == this->get_size()) {
        // we need to create a executor-local clone of the target data, that
        // will be copied back later.
        auto exec = this->get_executor();
        auto result_array = make_temporary_output_clone(exec, &result->values_);
        // create a (value, not pointer to avoid allocation overhead) view
        // matrix on the array to avoid special-casing cross-executor copies
        auto tmp_result =
            Dense{exec, result->get_size(),
                  Array<ValueType>::view(exec, result_array->get_num_elems(),
                                         result_array->get_data()),
                  result->get_stride()};
        exec->run(dense::make_copy(this, &tmp_result));
    } else {
        result->values_ = this->values_;
        result->stride_ = this->stride_;
        result->set_size(this->get_size());
    }
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense<ValueType>* result)
{
    if (this != result) {
        result->values_ = std::move(this->values_);
        result->stride_ = this->stride_;
        result->set_size(this->get_size());
    }
}


template <typename ValueType>
void Dense<ValueType>::convert_to(
    Dense<next_precision<ValueType>>* result) const
{
    if (result->get_size() == this->get_size()) {
        auto exec = this->get_executor();
        exec->run(dense::make_copy(
            this, make_temporary_output_clone(exec, result).get()));
    } else {
        result->values_ = this->values_;
        result->stride_ = this->stride_;
        result->set_size(this->get_size());
    }
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

    Array<int64> row_ptrs{exec, num_rows + 1};
    exec->run(dense::make_count_nonzeros_per_row(this, row_ptrs.get_data()));
    exec->run(dense::make_prefix_sum(row_ptrs.get_data(), num_rows + 1));
    const auto nnz =
        exec->copy_val_to_host(row_ptrs.get_const_data() + num_rows);
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
        exec->run(dense::make_prefix_sum(tmp->get_row_ptrs(), num_rows + 1));
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
    exec->run(dense::make_prefix_sum(tmp->get_row_ptrs(), row_blocks + 1));
    const auto nnz_blocks =
        exec->copy_val_to_host(tmp->get_const_row_ptrs() + row_blocks);
    tmp->col_idxs_.resize_and_reset(nnz_blocks);
    tmp->values_.resize_and_reset(nnz_blocks * bs * bs);
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
    Array<size_type> row_nnz{exec, num_rows};
    Array<int64> coo_row_ptrs{exec, num_rows + 1};
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
    coo_nnz = exec->copy_val_to_host(coo_row_ptrs.get_const_data() + num_rows);
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
    exec->run(dense::make_prefix_sum(tmp->row_ptrs_.get_data(), num_rows + 1));
    const auto nnz =
        exec->copy_val_to_host(tmp->row_ptrs_.get_const_data() + num_rows);
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
    this->transpose(result.get());
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::conj_transpose() const
{
    auto result =
        Dense::create(this->get_executor(), gko::transpose(this->get_size()));
    this->conj_transpose(result.get());
    return result;
}


template <typename ValueType>
void Dense<ValueType>::transpose(Dense<ValueType>* output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(output, gko::transpose(this->get_size()));
    auto exec = this->get_executor();
    exec->run(dense::make_transpose(
        this, make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
void Dense<ValueType>::conj_transpose(Dense<ValueType>* output) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(output, gko::transpose(this->get_size()));
    auto exec = this->get_executor();
    exec->run(dense::make_conj_transpose(
        this, make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::permute_impl(const Array<IndexType>* permutation_indices,
                                    Dense<ValueType>* output) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();

    exec->run(dense::make_symm_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::inverse_permute_impl(
    const Array<IndexType>* permutation_indices, Dense<ValueType>* output) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();

    exec->run(dense::make_inv_symm_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::row_permute_impl(
    const Array<IndexType>* permutation_indices, Dense<ValueType>* output) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    auto exec = this->get_executor();

    exec->run(dense::make_row_gather(
        make_temporary_clone(exec, permutation_indices).get(), this,
        make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
template <typename OutputType, typename IndexType>
void Dense<ValueType>::row_gather_impl(const Array<IndexType>* row_idxs,
                                       Dense<OutputType>* row_collection) const
{
    auto exec = this->get_executor();
    dim<2> expected_dim{row_idxs->get_num_elems(), this->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(expected_dim, row_collection);

    exec->run(dense::make_row_gather(
        make_temporary_clone(exec, row_idxs).get(), this,
        make_temporary_output_clone(exec, row_collection).get()));
}

template <typename ValueType>
template <typename OutputType, typename IndexType>
void Dense<ValueType>::row_gather_impl(const Dense<ValueType>* alpha,
                                       const Array<IndexType>* row_idxs,
                                       const Dense<ValueType>* beta,
                                       Dense<OutputType>* row_collection) const
{
    auto exec = this->get_executor();
    dim<2> expected_dim{row_idxs->get_num_elems(), this->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(expected_dim, row_collection);

    exec->run(dense::make_advanced_row_gather(
        make_temporary_clone(exec, alpha).get(),
        make_temporary_clone(exec, row_idxs).get(), this,
        make_temporary_clone(exec, beta).get(),
        make_temporary_clone(exec, row_collection).get()));
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::column_permute_impl(
    const Array<IndexType>* permutation_indices, Dense<ValueType>* output) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    auto exec = this->get_executor();

    exec->run(dense::make_column_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::inverse_row_permute_impl(
    const Array<IndexType>* permutation_indices, Dense<ValueType>* output) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    auto exec = this->get_executor();

    exec->run(dense::make_inverse_row_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::inverse_column_permute_impl(
    const Array<IndexType>* permutation_indices, Dense<ValueType>* output) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    GKO_ASSERT_EQUAL_DIMENSIONS(this, output);
    auto exec = this->get_executor();

    exec->run(dense::make_inverse_column_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        make_temporary_output_clone(exec, output).get()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::permute(
    const Array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::permute(
    const Array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
void Dense<ValueType>::permute(const Array<int32>* permutation_indices,
                               Dense<ValueType>* output) const
{
    this->permute_impl(permutation_indices, output);
}


template <typename ValueType>
void Dense<ValueType>::permute(const Array<int64>* permutation_indices,
                               Dense<ValueType>* output) const
{
    this->permute_impl(permutation_indices, output);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_permute(
    const Array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_permute(
    const Array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
void Dense<ValueType>::inverse_permute(const Array<int32>* permutation_indices,
                                       Dense<ValueType>* output) const
{
    this->inverse_permute_impl(permutation_indices, output);
}


template <typename ValueType>
void Dense<ValueType>::inverse_permute(const Array<int64>* permutation_indices,
                                       Dense<ValueType>* output) const
{
    this->inverse_permute_impl(permutation_indices, output);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::row_permute(
    const Array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->row_permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::row_permute(
    const Array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->row_permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
void Dense<ValueType>::row_permute(const Array<int32>* permutation_indices,
                                   Dense<ValueType>* output) const
{
    this->row_permute_impl(permutation_indices, output);
}


template <typename ValueType>
void Dense<ValueType>::row_permute(const Array<int64>* permutation_indices,
                                   Dense<ValueType>* output) const
{
    this->row_permute_impl(permutation_indices, output);
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::row_gather(
    const Array<int32>* row_idxs) const
{
    auto exec = this->get_executor();
    dim<2> out_dim{row_idxs->get_num_elems(), this->get_size()[1]};
    auto result = Dense::create(exec, out_dim);
    this->row_gather(row_idxs, result.get());
    return result;
}

template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::row_gather(
    const Array<int64>* row_idxs) const
{
    auto exec = this->get_executor();
    dim<2> out_dim{row_idxs->get_num_elems(), this->get_size()[1]};
    auto result = Dense::create(exec, out_dim);
    this->row_gather(row_idxs, result.get());
    return result;
}

template <typename ValueType>
void Dense<ValueType>::row_gather(const Array<int32>* row_idxs,
                                  Dense<ValueType>* row_collection) const
{
    this->row_gather_impl(row_idxs, row_collection);
}

template <typename ValueType>
void Dense<ValueType>::row_gather(const Array<int64>* row_idxs,
                                  Dense<ValueType>* row_collection) const
{
    this->row_gather_impl(row_idxs, row_collection);
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
void Dense<ValueType>::row_gather(const Array<int32>* row_idxs,
                                  LinOp* row_collection) const
{
    gather_mixed_real_complex<ValueType>(
        [&](auto dense) { this->row_gather_impl(row_idxs, dense); },
        row_collection);
}


template <typename ValueType>
void Dense<ValueType>::row_gather(const Array<int64>* row_idxs,
                                  LinOp* row_collection) const
{
    gather_mixed_real_complex<ValueType>(
        [&](auto dense) { this->row_gather_impl(row_idxs, dense); },
        row_collection);
}


template <typename ValueType>
void Dense<ValueType>::row_gather(const LinOp* alpha,
                                  const Array<int32>* gather_indices,
                                  const LinOp* beta, LinOp* out) const
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
        out);
}

template <typename ValueType>
void Dense<ValueType>::row_gather(const LinOp* alpha,
                                  const Array<int64>* gather_indices,
                                  const LinOp* beta, LinOp* out) const
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
        out);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::column_permute(
    const Array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->column_permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::column_permute(
    const Array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->column_permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
void Dense<ValueType>::column_permute(const Array<int32>* permutation_indices,
                                      Dense<ValueType>* output) const
{
    this->column_permute_impl(permutation_indices, output);
}


template <typename ValueType>
void Dense<ValueType>::column_permute(const Array<int64>* permutation_indices,
                                      Dense<ValueType>* output) const
{
    this->column_permute_impl(permutation_indices, output);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_row_permute(
    const Array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_row_permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_row_permute(
    const Array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_row_permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
void Dense<ValueType>::inverse_row_permute(
    const Array<int32>* permutation_indices, Dense<ValueType>* output) const
{
    this->inverse_row_permute_impl(permutation_indices, output);
}


template <typename ValueType>
void Dense<ValueType>::inverse_row_permute(
    const Array<int64>* permutation_indices, Dense<ValueType>* output) const
{
    this->inverse_row_permute_impl(permutation_indices, output);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_column_permute(
    const Array<int32>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_column_permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_column_permute(
    const Array<int64>* permutation_indices) const
{
    auto result = Dense::create(this->get_executor(), this->get_size());
    this->inverse_column_permute(permutation_indices, result.get());
    return result;
}


template <typename ValueType>
void Dense<ValueType>::inverse_column_permute(
    const Array<int32>* permutation_indices, Dense<ValueType>* output) const
{
    this->inverse_column_permute_impl(permutation_indices, output);
}


template <typename ValueType>
void Dense<ValueType>::inverse_column_permute(
    const Array<int64>* permutation_indices, Dense<ValueType>* output) const
{
    this->inverse_column_permute_impl(permutation_indices, output);
}


template <typename ValueType>
void Dense<ValueType>::extract_diagonal(Diagonal<ValueType>* output) const
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
    this->extract_diagonal(diag.get());
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
    this->compute_absolute(result.get());
    return result;
}


template <typename ValueType>
void Dense<ValueType>::compute_absolute(
    Dense<ValueType>::absolute_type* output) const
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
    this->make_complex(result.get());
    return result;
}


template <typename ValueType>
void Dense<ValueType>::make_complex(
    typename Dense<ValueType>::complex_type* result) const
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
    this->get_real(result.get());
    return result;
}


template <typename ValueType>
void Dense<ValueType>::get_real(
    typename Dense<ValueType>::real_type* result) const
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
    this->get_imag(result.get());
    return result;
}


template <typename ValueType>
void Dense<ValueType>::get_imag(
    typename Dense<ValueType>::real_type* result) const
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


#define GKO_DECLARE_DENSE_MATRIX(_type) class Dense<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_MATRIX);


}  // namespace matrix
}  // namespace gko
