// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/sellp.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
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
#include "core/matrix/sellp_kernels.hpp"


namespace gko {
namespace matrix {
namespace sellp {
namespace {


GKO_REGISTER_OPERATION(spmv, sellp::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, sellp::advanced_spmv);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(compute_slice_sets, sellp::compute_slice_sets);
GKO_REGISTER_OPERATION(fill_in_matrix_data, sellp::fill_in_matrix_data);
GKO_REGISTER_OPERATION(fill_in_dense, sellp::fill_in_dense);
GKO_REGISTER_OPERATION(convert_to_csr, sellp::convert_to_csr);
GKO_REGISTER_OPERATION(count_nonzeros_per_row, sellp::count_nonzeros_per_row);
GKO_REGISTER_OPERATION(extract_diagonal, sellp::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // anonymous namespace
}  // namespace sellp


template <typename ValueType, typename IndexType>
Sellp<ValueType, IndexType>& Sellp<ValueType, IndexType>::operator=(
    const Sellp& other)
{
    if (&other != this) {
        EnableLinOp<Sellp>::operator=(other);
        values_ = other.values_;
        col_idxs_ = other.col_idxs_;
        slice_lengths_ = other.slice_lengths_;
        slice_sets_ = other.slice_sets_;
        slice_size_ = other.slice_size_;
        stride_factor_ = other.stride_factor_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Sellp<ValueType, IndexType>& Sellp<ValueType, IndexType>::operator=(
    Sellp&& other)
{
    if (&other != this) {
        EnableLinOp<Sellp>::operator=(std::move(other));
        values_ = std::move(other.values_);
        col_idxs_ = std::move(other.col_idxs_);
        slice_lengths_ = std::move(other.slice_lengths_);
        slice_sets_ = std::move(other.slice_sets_);
        // slice_size and stride_factor are immutable
        slice_size_ = other.slice_size_;
        stride_factor_ = other.stride_factor_;
        // restore other invariant
        other.slice_sets_.resize_and_reset(1);
        other.slice_sets_.fill(0);
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Sellp<ValueType, IndexType>::Sellp(const Sellp& other)
    : Sellp(other.get_executor())
{
    *this = other;
}


template <typename ValueType, typename IndexType>
Sellp<ValueType, IndexType>::Sellp(Sellp&& other) : Sellp(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(sellp::make_spmv(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                             const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            this->get_executor()->run(sellp::make_advanced_spmv(
                dense_alpha, this, dense_b, dense_beta, dense_x));
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::convert_to(
    Sellp<next_precision<ValueType>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->slice_lengths_ = this->slice_lengths_;
    result->slice_sets_ = this->slice_sets_;
    result->slice_size_ = this->slice_size_;
    result->stride_factor_ = this->stride_factor_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::move_to(
    Sellp<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp_result = make_temporary_output_clone(exec, result);
    tmp_result->resize(this->get_size());
    tmp_result->fill(zero<ValueType>());
    exec->run(sellp::make_fill_in_dense(this, tmp_result.get()));
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto num_rows = this->get_size()[0];
    {
        auto tmp = make_temporary_clone(exec, result);
        tmp->row_ptrs_.resize_and_reset(num_rows + 1);
        exec->run(sellp::make_count_nonzeros_per_row(
            this, tmp->row_ptrs_.get_data()));
        exec->run(sellp::make_prefix_sum_nonnegative(tmp->row_ptrs_.get_data(),
                                                     num_rows + 1));
        const auto nnz =
            static_cast<size_type>(get_element(tmp->row_ptrs_, num_rows));
        tmp->col_idxs_.resize_and_reset(nnz);
        tmp->values_.resize_and_reset(nnz);
        tmp->set_size(this->get_size());
        exec->run(sellp::make_convert_to_csr(this, tmp.get()));
    }
    result->make_srow();
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::move_to(Csr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::read(const device_mat_data& data)
{
    auto exec = this->get_executor();
    const auto size = data.get_size();
    slice_lengths_.resize_and_reset(ceildiv(size[0], slice_size_));
    slice_sets_.resize_and_reset(ceildiv(size[0], slice_size_) + 1);
    this->set_size(size);
    array<int64> row_ptrs{exec, size[0] + 1};
    auto local_data = make_temporary_clone(exec, &data);
    exec->run(sellp::make_convert_idxs_to_ptrs(
        local_data->get_const_row_idxs(), local_data->get_num_stored_elements(),
        size[0], row_ptrs.get_data()));
    exec->run(sellp::make_compute_slice_sets(
        row_ptrs, this->get_slice_size(), this->get_stride_factor(),
        slice_sets_.get_data(), slice_lengths_.get_data()));
    const auto total_cols =
        get_element(slice_sets_, slice_sets_.get_size() - 1);
    values_.resize_and_reset(total_cols * slice_size_);
    col_idxs_.resize_and_reset(total_cols * slice_size_);
    exec->run(sellp::make_fill_in_matrix_data(*local_data,
                                              row_ptrs.get_const_data(), this));
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::read(device_mat_data&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_from_host(this->get_executor(), data));
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::write(mat_data& data) const
{
    auto tmp = make_temporary_clone(this->get_executor()->get_master(), this);

    data = {tmp->get_size(), {}};

    auto slice_size = tmp->get_slice_size();
    size_type slice_num = static_cast<index_type>(
        (tmp->get_size()[0] + slice_size - 1) / slice_size);
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row_in_slice = 0; row_in_slice < slice_size;
             row_in_slice++) {
            auto row = slice * slice_size + row_in_slice;
            if (row < tmp->get_size()[0]) {
                const auto slice_len = tmp->get_const_slice_lengths()[slice];
                const auto slice_offset = tmp->get_const_slice_sets()[slice];
                for (size_type i = 0; i < slice_len; i++) {
                    const auto col = tmp->col_at(row_in_slice, slice_offset, i);
                    const auto val = tmp->val_at(row_in_slice, slice_offset, i);
                    if (col != invalid_index<IndexType>()) {
                        data.nonzeros.emplace_back(row, col, val);
                    }
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Sellp<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(sellp::make_fill_array(diag->get_values(), diag->get_size()[0],
                                     zero<ValueType>()));
    exec->run(sellp::make_extract_diagonal(this, diag.get()));
    return diag;
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(sellp::make_inplace_absolute_array(
        this->get_values(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Sellp<ValueType, IndexType>::absolute_type>
Sellp<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_sellp = absolute_type::create(
        exec, this->get_size(), this->get_slice_size(),
        this->get_stride_factor(), this->get_total_cols());

    abs_sellp->col_idxs_ = col_idxs_;
    abs_sellp->slice_lengths_ = slice_lengths_;
    abs_sellp->slice_sets_ = slice_sets_;
    exec->run(sellp::make_outplace_absolute_array(
        this->get_const_values(), this->get_num_stored_elements(),
        abs_sellp->get_values()));

    return abs_sellp;
}


#define GKO_DECLARE_SELLP_MATRIX(ValueType, IndexType) \
    class Sellp<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_MATRIX);


}  // namespace matrix
}  // namespace gko
