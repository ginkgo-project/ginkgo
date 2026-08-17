// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/multivector.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/array_access.hpp"
#include "core/base/dispatch_helper.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/matrix/hybrid_kernels.hpp"
#include "core/matrix/multivector_kernels.hpp"
#include "ginkgo/core/matrix/fbcsr.hpp"


namespace gko {
namespace matrix {
namespace dense {


GKO_REGISTER_OPERATION(simple_apply, dense::simple_apply);
GKO_REGISTER_OPERATION(advanced_apply, dense::apply);
GKO_REGISTER_OPERATION(convert_to_coo, dense::convert_to_coo);
GKO_REGISTER_OPERATION(convert_to_csr, dense::convert_to_csr);
GKO_REGISTER_OPERATION(convert_to_ell, dense::convert_to_ell);
GKO_REGISTER_OPERATION(convert_to_fbcsr, dense::convert_to_fbcsr);
GKO_REGISTER_OPERATION(convert_to_hybrid, dense::convert_to_hybrid);
GKO_REGISTER_OPERATION(convert_to_sellp, dense::convert_to_sellp);
GKO_REGISTER_OPERATION(convert_to_sparsity_csr, dense::convert_to_sparsity_csr);
GKO_REGISTER_OPERATION(compute_max_nnz_per_row, dense::compute_max_nnz_per_row);
GKO_REGISTER_OPERATION(compute_hybrid_coo_row_ptrs,
                       hybrid::compute_coo_row_ptrs);
GKO_REGISTER_OPERATION(count_nonzeros_per_row, dense::count_nonzeros_per_row);
GKO_REGISTER_OPERATION(count_nonzero_blocks_per_row,
                       dense::count_nonzero_blocks_per_row);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(compute_slice_sets, dense::compute_slice_sets);
GKO_REGISTER_OPERATION(extract_diagonal, dense::extract_diagonal);
GKO_REGISTER_OPERATION(add_scaled_diag, dense::add_scaled_diag);
GKO_REGISTER_OPERATION(sub_scaled_diag, dense::sub_scaled_diag);
GKO_REGISTER_OPERATION(add_scaled_identity, dense::add_scaled_identity);


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
template <typename IndexType>
void Dense<ValueType>::convert_impl(Coo<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto num_rows = this->get_size()[0];

    array<int64> row_ptrs{exec, num_rows + 1};
    exec->run(dense::make_count_nonzeros_per_row(this->get_const_device_view(),
                                                 row_ptrs.get_data()));
    exec->run(
        dense::make_prefix_sum_nonnegative(row_ptrs.get_data(), num_rows + 1));
    const auto nnz = get_element(row_ptrs, num_rows);
    result->resize(this->get_size(), nnz);
    exec->run(dense::make_convert_to_coo(
        this->get_const_device_view(), row_ptrs.get_const_data(),
        make_temporary_clone(exec, result)->get_device_view()));
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
        exec->run(dense::make_count_nonzeros_per_row(
            this->get_const_device_view(), tmp->get_row_ptrs()));
        exec->run(dense::make_prefix_sum_nonnegative(tmp->get_row_ptrs(),
                                                     num_rows + 1));
        const auto nnz =
            exec->copy_val_to_host(tmp->get_const_row_ptrs() + num_rows);
        tmp->col_idxs_.resize_and_reset(nnz);
        tmp->values_.resize_and_reset(nnz);
        tmp->set_size(this->get_size());
        exec->run(dense::make_convert_to_csr(this->get_const_device_view(),
                                             tmp->get_device_view()));
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
    exec->run(dense::make_count_nonzero_blocks_per_row(
        this->get_const_device_view(), bs, tmp->get_row_ptrs()));
    exec->run(dense::make_prefix_sum_nonnegative(tmp->get_row_ptrs(),
                                                 row_blocks + 1));
    const auto nnz_blocks =
        exec->copy_val_to_host(tmp->get_const_row_ptrs() + row_blocks);
    tmp->col_idxs_.resize_and_reset(nnz_blocks);
    tmp->values_.resize_and_reset(nnz_blocks * bs * bs);
    tmp->values_.fill(zero<ValueType>());
    tmp->set_size(this->get_size());
    exec->run(
        dense::make_convert_to_fbcsr(this->get_const_device_view(), tmp.get()));
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
}


template <typename ValueType>
template <typename IndexType>
void Dense<ValueType>::convert_impl(Ell<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    size_type num_stored_elements_per_row{};
    exec->run(dense::make_compute_max_nnz_per_row(this->get_const_device_view(),
                                                  num_stored_elements_per_row));
    result->resize(this->get_size(), num_stored_elements_per_row);
    exec->run(dense::make_convert_to_ell(
        this->get_const_device_view(),
        make_temporary_clone(exec, result)->get_device_view()));
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
    exec->run(dense::make_count_nonzeros_per_row(this->get_const_device_view(),
                                                 row_nnz.get_data()));
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
    exec->run(dense::make_convert_to_hybrid(this->get_const_device_view(),
                                            coo_row_ptrs.get_const_data(),
                                            tmp->get_device_view()));
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
    exec->run(dense::make_compute_slice_sets(
        this->get_const_device_view(), slice_size, stride_factor,
        tmp->get_slice_sets(), tmp->get_slice_lengths()));
    auto total_cols =
        exec->copy_val_to_host(tmp->get_slice_sets() + num_slices);
    tmp->col_idxs_.resize_and_reset(total_cols * slice_size);
    tmp->values_.resize_and_reset(total_cols * slice_size);
    tmp->set_size(this->get_size());
    exec->run(dense::make_convert_to_sellp(this->get_const_device_view(),
                                           tmp->get_device_view()));
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
    exec->run(dense::make_count_nonzeros_per_row(this->get_const_device_view(),
                                                 tmp->row_ptrs_.get_data()));
    exec->run(dense::make_prefix_sum_nonnegative(tmp->row_ptrs_.get_data(),
                                                 num_rows + 1));
    const auto nnz = get_element(tmp->row_ptrs_, num_rows);
    tmp->col_idxs_.resize_and_reset(nnz);
    tmp->value_.fill(one<ValueType>());
    tmp->set_size(this->get_size());
    exec->run(dense::make_convert_to_sparsity_csr(this->get_const_device_view(),
                                                  tmp.get()));
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
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
    this->set_size(dim<2>{0, 0});
    this->stride_ = 0;
    this->values_.resize_and_reset(0);
}


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
    this->resize(data.get_size());
    this->fill(zero<ValueType>());
    exec->run(multivector::make_fill_in_matrix_data(
        *make_temporary_clone(exec, &data), this->get_device_view()));
}


template <typename ValueType>
void Dense<ValueType>::read(const device_mat_data64& data)
{
    auto exec = this->get_executor();
    this->resize(data.get_size());
    this->fill(zero<ValueType>());
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
    auto exec = this->get_executor();
    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    GKO_ASSERT_EQ(output->get_size()[0], diag_size);

    exec->run(dense::make_extract_diagonal(
        this->get_const_device_view(),
        make_temporary_output_clone(exec, output).get()));
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
void Dense<ValueType>::add_scaled(ptr_param<const LinOp> alpha,
                                  ptr_param<const Diagonal<value_type>> diag)
{
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this, diag);
    auto exec = this->get_executor();
    exec->run(dense::make_add_scaled_diag(
        make_temporary_conversion<ValueType>(alpha)->get_const_device_view(),
        diag.get(), this->get_device_view()));
}


template <typename ValueType>
void Dense<ValueType>::sub_scaled(ptr_param<const LinOp> alpha,
                                  ptr_param<const Diagonal<value_type>> diag)
{
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this, diag);
    auto exec = this->get_executor();
    exec->run(dense::make_sub_scaled_diag(
        make_temporary_conversion<ValueType>(alpha)->get_const_device_view(),
        diag.get(), this->get_device_view()));
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
void Dense<ValueType>::add_scaled_identity_impl(const AbstractMultiVector* a,
                                                const AbstractMultiVector* b)
{
    this->get_executor()->run(dense::make_add_scaled_identity(
        a->as_precision(this->get_precision())
            ->template get_const_local_device_view<ValueType>(),
        b->as_precision(this->get_precision())
            ->template get_const_local_device_view<ValueType>(),
        this->get_device_view()));
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


template <typename ValueType>
void Dense<ValueType>::resize(dim<2> new_size)
{
    if (this->get_size() != new_size) {
        this->set_size(new_size);
        this->stride_ = new_size[1];
        this->values_.resize_and_reset(new_size[0] * this->get_stride());
    }
}


#define GKO_DECLARE_DENSE(ValueType) class Dense<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE);


}  // namespace matrix
}  // namespace gko
