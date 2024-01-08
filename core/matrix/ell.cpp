// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/ell.hpp>


#include <algorithm>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"
#include "core/base/array_access.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"


namespace gko {
namespace matrix {
namespace ell {
namespace {


GKO_REGISTER_OPERATION(spmv, ell::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, ell::advanced_spmv);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(compute_max_row_nnz, ell::compute_max_row_nnz);
GKO_REGISTER_OPERATION(fill_in_matrix_data, ell::fill_in_matrix_data);
GKO_REGISTER_OPERATION(fill_in_dense, ell::fill_in_dense);
GKO_REGISTER_OPERATION(copy, ell::copy);
GKO_REGISTER_OPERATION(convert_to_csr, ell::convert_to_csr);
GKO_REGISTER_OPERATION(count_nonzeros_per_row, ell::count_nonzeros_per_row);
GKO_REGISTER_OPERATION(extract_diagonal, ell::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // anonymous namespace
}  // namespace ell


template <typename ValueType, typename IndexType>
Ell<ValueType, IndexType>& Ell<ValueType, IndexType>::operator=(
    const Ell& other)
{
    if (&other != this) {
        const auto old_size = this->get_size();
        EnableLinOp<Ell>::operator=(other);
        // NOTE: keep this consistent with resize(...)
        if (old_size != other.get_size() ||
            this->get_num_stored_elements_per_row() !=
                other.get_num_stored_elements_per_row()) {
            this->num_stored_elements_per_row_ =
                other.get_num_stored_elements_per_row();
            this->stride_ = other.get_size()[0];
            const auto alloc_size =
                this->stride_ * this->num_stored_elements_per_row_;
            this->values_.resize_and_reset(alloc_size);
            this->col_idxs_.resize_and_reset(alloc_size);
        }
        // we need to create a executor-local clone of the target data, that
        // will be copied back later. Need temporary_clone, not
        // temporary_output_clone to avoid overwriting padding
        auto exec = other.get_executor();
        auto exec_values_array = make_temporary_clone(exec, &this->values_);
        auto exec_cols_array = make_temporary_clone(exec, &this->col_idxs_);
        // create a (value, not pointer to avoid allocation overhead) view
        // matrix on the array to avoid special-casing cross-executor copies
        auto exec_this_view =
            Ell{exec,
                this->get_size(),
                make_array_view(exec, exec_values_array->get_size(),
                                exec_values_array->get_data()),
                make_array_view(exec, exec_cols_array->get_size(),
                                exec_cols_array->get_data()),
                this->get_num_stored_elements_per_row(),
                this->get_stride()};
        exec->run(ell::make_copy(&other, &exec_this_view));
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Ell<ValueType, IndexType>& Ell<ValueType, IndexType>::operator=(Ell&& other)
{
    if (&other != this) {
        EnableLinOp<Ell>::operator=(std::move(other));
        values_ = std::move(other.values_);
        col_idxs_ = std::move(other.col_idxs_);
        num_stored_elements_per_row_ =
            std::exchange(other.num_stored_elements_per_row_, 0);
        stride_ = std::exchange(other.stride_, 0);
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Ell<ValueType, IndexType>::Ell(const Ell& other) : Ell(other.get_executor())
{
    *this = other;
}


template <typename ValueType, typename IndexType>
Ell<ValueType, IndexType>::Ell(Ell&& other) : Ell(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    mixed_precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(ell::make_spmv(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
{
    mixed_precision_dispatch_real_complex<ValueType>(
        [this, alpha, beta](auto dense_b, auto dense_x) {
            auto dense_alpha = make_temporary_conversion<ValueType>(alpha);
            auto dense_beta = make_temporary_conversion<
                typename std::decay_t<decltype(*dense_x)>::value_type>(beta);
            this->get_executor()->run(ell::make_advanced_spmv(
                dense_alpha.get(), this, dense_b, dense_beta.get(), dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(
    Ell<next_precision<ValueType>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->num_stored_elements_per_row_ = this->num_stored_elements_per_row_;
    result->stride_ = this->stride_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(
    Ell<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp_result = make_temporary_output_clone(exec, result);
    tmp_result->resize(this->get_size());
    tmp_result->fill(zero<ValueType>());
    exec->run(ell::make_fill_in_dense(this, tmp_result.get()));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto num_rows = this->get_size()[0];
    {
        auto tmp = make_temporary_clone(exec, result);
        tmp->row_ptrs_.resize_and_reset(num_rows + 1);
        exec->run(
            ell::make_count_nonzeros_per_row(this, tmp->row_ptrs_.get_data()));
        exec->run(ell::make_prefix_sum_nonnegative(tmp->row_ptrs_.get_data(),
                                                   num_rows + 1));
        const auto nnz =
            static_cast<size_type>(get_element(tmp->row_ptrs_, num_rows));
        tmp->col_idxs_.resize_and_reset(nnz);
        tmp->values_.resize_and_reset(nnz);
        tmp->set_size(this->get_size());
        exec->run(ell::make_convert_to_csr(this, tmp.get()));
    }
    result->make_srow();
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(Csr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::resize(dim<2> new_size, size_type max_row_nnz)
{
    if (this->get_size() != new_size ||
        this->get_num_stored_elements_per_row() != max_row_nnz) {
        this->stride_ = new_size[0];
        values_.resize_and_reset(this->stride_ * max_row_nnz);
        col_idxs_.resize_and_reset(this->stride_ * max_row_nnz);
        this->num_stored_elements_per_row_ = max_row_nnz;
        this->set_size(new_size);
    }
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::read(const device_mat_data& data)
{
    auto exec = this->get_executor();
    array<int64> row_ptrs(exec, data.get_size()[0] + 1);
    auto local_data = make_temporary_clone(exec, &data);
    exec->run(ell::make_convert_idxs_to_ptrs(
        local_data->get_const_row_idxs(), local_data->get_num_stored_elements(),
        data.get_size()[0], row_ptrs.get_data()));
    size_type max_nnz{};
    exec->run(ell::make_compute_max_row_nnz(row_ptrs, max_nnz));
    this->resize(data.get_size(), max_nnz);
    exec->run(ell::make_fill_in_matrix_data(*local_data,
                                            row_ptrs.get_const_data(), this));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::read(device_mat_data&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_from_host(this->get_executor(), data));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::write(mat_data& data) const
{
    auto tmp = make_temporary_clone(this->get_executor()->get_master(), this);

    data = {tmp->get_size(), {}};

    for (size_type row = 0; row < tmp->get_size()[0]; ++row) {
        for (size_type i = 0; i < tmp->num_stored_elements_per_row_; ++i) {
            const auto val = tmp->val_at(row, i);
            const auto col = tmp->col_at(row, i);
            if (col != invalid_index<IndexType>()) {
                data.nonzeros.emplace_back(row, col, val);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Ell<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(ell::make_fill_array(diag->get_values(), diag->get_size()[0],
                                   zero<ValueType>()));
    exec->run(ell::make_extract_diagonal(this, diag.get()));
    return diag;
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(ell::make_inplace_absolute_array(
        this->get_values(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Ell<ValueType, IndexType>::absolute_type>
Ell<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_ell = absolute_type::create(
        exec, this->get_size(), this->get_num_stored_elements_per_row(),
        this->get_stride());

    abs_ell->col_idxs_ = col_idxs_;
    exec->run(ell::make_outplace_absolute_array(this->get_const_values(),
                                                this->get_num_stored_elements(),
                                                abs_ell->get_values()));

    return abs_ell;
}


#define GKO_DECLARE_ELL_MATRIX(ValueType, IndexType) \
    class Ell<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_MATRIX);


}  // namespace matrix
}  // namespace gko
