// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/csr.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/array_access.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "core/matrix/hybrid_kernels.hpp"
#include "core/matrix/permutation.hpp"
#include "core/matrix/sellp_kernels.hpp"


namespace gko {
namespace matrix {
namespace csr {
namespace {


GKO_REGISTER_OPERATION(spmv, csr::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, csr::advanced_spmv);
GKO_REGISTER_OPERATION(spgemm, csr::spgemm);
GKO_REGISTER_OPERATION(advanced_spgemm, csr::advanced_spgemm);
GKO_REGISTER_OPERATION(spgeam, csr::spgeam);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(convert_ptrs_to_idxs, components::convert_ptrs_to_idxs);
GKO_REGISTER_OPERATION(fill_in_dense, csr::fill_in_dense);
GKO_REGISTER_OPERATION(compute_slice_sets, sellp::compute_slice_sets);
GKO_REGISTER_OPERATION(convert_to_sellp, csr::convert_to_sellp);
GKO_REGISTER_OPERATION(compute_max_row_nnz, ell::compute_max_row_nnz);
GKO_REGISTER_OPERATION(convert_to_ell, csr::convert_to_ell);
GKO_REGISTER_OPERATION(convert_to_fbcsr, csr::convert_to_fbcsr);
GKO_REGISTER_OPERATION(compute_hybrid_coo_row_ptrs,
                       hybrid::compute_coo_row_ptrs);
GKO_REGISTER_OPERATION(convert_to_hybrid, csr::convert_to_hybrid);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row_in_span,
                       csr::calculate_nonzeros_per_row_in_span);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row_in_index_set,
                       csr::calculate_nonzeros_per_row_in_index_set);
GKO_REGISTER_OPERATION(compute_submatrix, csr::compute_submatrix);
GKO_REGISTER_OPERATION(compute_submatrix_from_index_set,
                       csr::compute_submatrix_from_index_set);
GKO_REGISTER_OPERATION(transpose, csr::transpose);
GKO_REGISTER_OPERATION(conj_transpose, csr::conj_transpose);
GKO_REGISTER_OPERATION(inv_symm_permute, csr::inv_symm_permute);
GKO_REGISTER_OPERATION(row_permute, csr::row_permute);
GKO_REGISTER_OPERATION(inv_row_permute, csr::inv_row_permute);
GKO_REGISTER_OPERATION(inv_col_permute, csr::inv_col_permute);
GKO_REGISTER_OPERATION(inv_nonsymm_permute, csr::inv_nonsymm_permute);
GKO_REGISTER_OPERATION(inv_symm_scale_permute, csr::inv_symm_scale_permute);
GKO_REGISTER_OPERATION(row_scale_permute, csr::row_scale_permute);
GKO_REGISTER_OPERATION(inv_row_scale_permute, csr::inv_row_scale_permute);
GKO_REGISTER_OPERATION(inv_col_scale_permute, csr::inv_col_scale_permute);
GKO_REGISTER_OPERATION(inv_nonsymm_scale_permute,
                       csr::inv_nonsymm_scale_permute);
GKO_REGISTER_OPERATION(convert_ptrs_to_sizes,
                       components::convert_ptrs_to_sizes);
GKO_REGISTER_OPERATION(sort_by_column_index, csr::sort_by_column_index);
GKO_REGISTER_OPERATION(is_sorted_by_column_index,
                       csr::is_sorted_by_column_index);
GKO_REGISTER_OPERATION(extract_diagonal, csr::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);
GKO_REGISTER_OPERATION(scale, csr::scale);
GKO_REGISTER_OPERATION(inv_scale, csr::inv_scale);
GKO_REGISTER_OPERATION(add_scaled_identity, csr::add_scaled_identity);
GKO_REGISTER_OPERATION(check_diagonal_entries,
                       csr::check_diagonal_entries_exist);
GKO_REGISTER_OPERATION(aos_to_soa, components::aos_to_soa);


}  // anonymous namespace
}  // namespace csr


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>& Csr<ValueType, IndexType>::operator=(
    const Csr<ValueType, IndexType>& other)
{
    if (&other != this) {
        EnableLinOp<Csr>::operator=(other);
        // NOTE: as soon as strategies are improved, this can be reverted
        values_ = other.values_;
        col_idxs_ = other.col_idxs_;
        row_ptrs_ = other.row_ptrs_;
        srow_ = other.srow_;
        if (this->get_executor() != other.get_executor()) {
            other.convert_strategy_helper(this);
        } else {
            this->set_strategy(other.get_strategy()->copy());
        }
        // END NOTE
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>& Csr<ValueType, IndexType>::operator=(
    Csr<ValueType, IndexType>&& other)
{
    if (&other != this) {
        EnableLinOp<Csr>::operator=(std::move(other));
        values_ = std::move(other.values_);
        col_idxs_ = std::move(other.col_idxs_);
        row_ptrs_ = std::move(other.row_ptrs_);
        srow_ = std::move(other.srow_);
        strategy_ = other.strategy_;
        if (this->get_executor() != other.get_executor()) {
            detail::strategy_rebuild_helper(this);
        }
        // restore other invariant
        other.row_ptrs_.resize_and_reset(1);
        other.row_ptrs_.fill(0);
        other.make_srow();
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::Csr(const Csr<ValueType, IndexType>& other)
    : Csr{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::Csr(Csr<ValueType, IndexType>&& other)
    : Csr{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    using ComplexDense = Dense<to_complex<ValueType>>;
    using TCsr = Csr<ValueType, IndexType>;
    if (auto b_csr = dynamic_cast<const TCsr*>(b)) {
        // if b is a CSR matrix, we compute a SpGeMM
        auto x_csr = as<TCsr>(x);
        this->get_executor()->run(csr::make_spgemm(this, b_csr, x_csr));
    } else {
        mixed_precision_dispatch_real_complex<ValueType>(
            [this](auto dense_b, auto dense_x) {
                this->get_executor()->run(
                    csr::make_spmv(this, dense_b, dense_x));
            },
            b, x);
    }
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
{
    using ComplexDense = Dense<to_complex<ValueType>>;
    using RealDense = Dense<remove_complex<ValueType>>;
    using TCsr = Csr<ValueType, IndexType>;
    if (auto b_csr = dynamic_cast<const TCsr*>(b)) {
        // if b is a CSR matrix, we compute a SpGeMM
        auto x_csr = as<TCsr>(x);
        auto x_copy = x_csr->clone();
        this->get_executor()->run(csr::make_advanced_spgemm(
            as<Dense<ValueType>>(alpha), this, b_csr,
            as<Dense<ValueType>>(beta), x_copy.get(), x_csr));
    } else if (dynamic_cast<const Identity<ValueType>*>(b)) {
        // if b is an identity matrix, we compute an SpGEAM
        auto x_csr = as<TCsr>(x);
        auto x_copy = x_csr->clone();
        this->get_executor()->run(
            csr::make_spgeam(as<Dense<ValueType>>(alpha), this,
                             as<Dense<ValueType>>(beta), x_copy.get(), x_csr));
    } else {
        mixed_precision_dispatch_real_complex<ValueType>(
            [this, alpha, beta](auto dense_b, auto dense_x) {
                auto dense_alpha = make_temporary_conversion<ValueType>(alpha);
                auto dense_beta = make_temporary_conversion<
                    typename std::decay_t<decltype(*dense_x)>::value_type>(
                    beta);
                this->get_executor()->run(
                    csr::make_advanced_spmv(dense_alpha.get(), this, dense_b,
                                            dense_beta.get(), dense_x));
            },
            b, x);
    }
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Csr<next_precision<ValueType>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    result->set_size(this->get_size());
    convert_strategy_helper(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(
    Csr<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Coo<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    auto tmp = make_temporary_clone(exec, result);
    tmp->values_ = this->values_;
    tmp->col_idxs_ = this->col_idxs_;
    tmp->row_idxs_.resize_and_reset(this->get_num_stored_elements());
    tmp->set_size(this->get_size());
    exec->run(csr::make_convert_ptrs_to_idxs(
        this->get_const_row_ptrs(), this->get_size()[0], tmp->get_row_idxs()));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Coo<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp_result = make_temporary_output_clone(exec, result);
    tmp_result->resize(this->get_size());
    tmp_result->fill(zero<ValueType>());
    exec->run(csr::make_fill_in_dense(this, tmp_result.get()));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Hybrid<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto num_rows = this->get_size()[0];
    const auto num_cols = this->get_size()[1];
    array<size_type> row_nnz{exec, num_rows};
    array<int64> coo_row_ptrs{exec, num_rows + 1};
    exec->run(csr::make_convert_ptrs_to_sizes(this->get_const_row_ptrs(),
                                              num_rows, row_nnz.get_data()));
    size_type ell_lim{};
    size_type coo_nnz{};
    result->get_strategy()->compute_hybrid_config(row_nnz, &ell_lim, &coo_nnz);
    if (ell_lim > num_cols) {
        // TODO remove temporary fix after ELL gains true structural zeros
        ell_lim = num_cols;
    }
    exec->run(csr::make_compute_hybrid_coo_row_ptrs(row_nnz, ell_lim,
                                                    coo_row_ptrs.get_data()));
    coo_nnz = get_element(coo_row_ptrs, num_rows);
    auto tmp = make_temporary_clone(exec, result);
    tmp->resize(this->get_size(), ell_lim, coo_nnz);
    exec->run(csr::make_convert_to_hybrid(this, coo_row_ptrs.get_const_data(),
                                          tmp.get()));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Hybrid<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Sellp<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto stride_factor = result->get_stride_factor();
    const auto slice_size = result->get_slice_size();
    const auto num_rows = this->get_size()[0];
    const auto num_slices = ceildiv(num_rows, slice_size);
    auto tmp = make_temporary_clone(exec, result);
    tmp->slice_sets_.resize_and_reset(num_slices + 1);
    tmp->slice_lengths_.resize_and_reset(num_slices);
    tmp->stride_factor_ = stride_factor;
    tmp->slice_size_ = slice_size;
    exec->run(csr::make_compute_slice_sets(this->row_ptrs_, slice_size,
                                           stride_factor, tmp->get_slice_sets(),
                                           tmp->get_slice_lengths()));
    auto total_cols =
        exec->copy_val_to_host(tmp->get_slice_sets() + num_slices);
    tmp->col_idxs_.resize_and_reset(total_cols * slice_size);
    tmp->values_.resize_and_reset(total_cols * slice_size);
    tmp->set_size(this->get_size());
    exec->run(csr::make_convert_to_sellp(this, tmp.get()));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Sellp<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    SparsityCsr<ValueType, IndexType>* result) const
{
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    if (!result->value_.get_data()) {
        result->value_ =
            array<ValueType>(result->get_executor(), {one<ValueType>()});
    }
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(
    SparsityCsr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Ell<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    size_type max_nnz_per_row{};
    exec->run(csr::make_compute_max_row_nnz(this->row_ptrs_, max_nnz_per_row));
    auto tmp = make_temporary_clone(exec, result);
    if (tmp->get_size() != this->get_size() ||
        tmp->num_stored_elements_per_row_ != max_nnz_per_row) {
        tmp->num_stored_elements_per_row_ = max_nnz_per_row;
        tmp->stride_ = this->get_size()[0];
        const auto storage = tmp->num_stored_elements_per_row_ * tmp->stride_;
        tmp->col_idxs_.resize_and_reset(storage);
        tmp->values_.resize_and_reset(storage);
        tmp->set_size(this->get_size());
    }
    exec->run(csr::make_convert_to_ell(this, tmp.get()));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Ell<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Fbcsr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto bs = result->get_block_size();
    const auto row_blocks = detail::get_num_blocks(bs, this->get_size()[0]);
    const auto col_blocks = detail::get_num_blocks(bs, this->get_size()[1]);
    auto tmp = make_temporary_clone(exec, result);
    tmp->row_ptrs_.resize_and_reset(row_blocks + 1);
    tmp->set_size(this->get_size());
    exec->run(csr::make_convert_to_fbcsr(this, bs, tmp->row_ptrs_,
                                         tmp->col_idxs_, tmp->values_));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Fbcsr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::read(const mat_data& data)
{
    auto size = data.size;
    auto exec = this->get_executor();
    this->set_size(size);
    this->row_ptrs_.resize_and_reset(size[0] + 1);
    this->col_idxs_.resize_and_reset(data.nonzeros.size());
    this->values_.resize_and_reset(data.nonzeros.size());
    // the device matrix data contains views on the column indices
    // and values array of this matrix, and an owning array for the
    // row indices (which doesn't exist in this matrix)
    device_mat_data view{exec, size,
                         array<IndexType>{exec, data.nonzeros.size()},
                         this->col_idxs_.as_view(), this->values_.as_view()};
    const auto host_data =
        make_array_view(exec->get_master(), data.nonzeros.size(),
                        const_cast<matrix_data_entry<ValueType, IndexType>*>(
                            data.nonzeros.data()));
    exec->run(
        csr::make_aos_to_soa(*make_temporary_clone(exec, &host_data), view));
    exec->run(csr::make_convert_idxs_to_ptrs(view.get_const_row_idxs(),
                                             view.get_num_stored_elements(),
                                             size[0], this->get_row_ptrs()));
    this->make_srow();
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::read(const device_mat_data& data)
{
    auto size = data.get_size();
    auto exec = this->get_executor();
    this->row_ptrs_.resize_and_reset(size[0] + 1);
    this->set_size(size);
    // copy the column indices and values array from the device matrix data
    // into this. Compared to the read(device_mat_data&&) version, the internal
    // arrays keep their current ownership status.
    this->values_ = make_const_array_view(data.get_executor(),
                                          data.get_num_stored_elements(),
                                          data.get_const_values());
    this->col_idxs_ = make_const_array_view(data.get_executor(),
                                            data.get_num_stored_elements(),
                                            data.get_const_col_idxs());
    const auto row_idxs = make_const_array_view(data.get_executor(),
                                                data.get_num_stored_elements(),
                                                data.get_const_row_idxs())
                              .copy_to_array();
    auto local_row_idxs = make_temporary_clone(exec, &row_idxs);
    exec->run(csr::make_convert_idxs_to_ptrs(local_row_idxs->get_const_data(),
                                             local_row_idxs->get_size(),
                                             size[0], this->get_row_ptrs()));
    this->make_srow();
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::read(device_mat_data&& data)
{
    auto size = data.get_size();
    auto exec = this->get_executor();
    auto arrays = data.empty_out();
    this->row_ptrs_.resize_and_reset(size[0] + 1);
    this->set_size(size);
    this->values_ = std::move(arrays.values);
    this->col_idxs_ = std::move(arrays.col_idxs);
    const auto row_idxs = std::move(arrays.row_idxs);
    auto local_row_idxs = make_temporary_clone(exec, &row_idxs);
    exec->run(csr::make_convert_idxs_to_ptrs(local_row_idxs->get_const_data(),
                                             local_row_idxs->get_size(),
                                             size[0], this->get_row_ptrs()));
    this->make_srow();
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::write(mat_data& data) const
{
    auto tmp = make_temporary_clone(this->get_executor()->get_master(), this);

    data = {tmp->get_size(), {}};

    for (size_type row = 0; row < tmp->get_size()[0]; ++row) {
        const auto start = tmp->row_ptrs_.get_const_data()[row];
        const auto end = tmp->row_ptrs_.get_const_data()[row + 1];
        for (auto i = start; i < end; ++i) {
            const auto col = tmp->col_idxs_.get_const_data()[i];
            const auto val = tmp->values_.get_const_data()[i];
            data.nonzeros.emplace_back(row, col, val);
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy =
        Csr::create(exec, gko::transpose(this->get_size()),
                    this->get_num_stored_elements(), this->get_strategy());

    exec->run(csr::make_transpose(this, trans_cpy.get()));
    trans_cpy->make_srow();
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy =
        Csr::create(exec, gko::transpose(this->get_size()),
                    this->get_num_stored_elements(), this->get_strategy());

    exec->run(csr::make_conj_transpose(this, trans_cpy.get()));
    trans_cpy->make_srow();
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>> Csr<ValueType, IndexType>::permute(
    ptr_param<const Permutation<IndexType>> permutation,
    permute_mode mode) const
{
    const auto exec = this->get_executor();
    const auto size = this->get_size();
    const auto nnz = this->get_num_stored_elements();
    validate_permute_dimensions(size, permutation->get_size(), mode);
    if ((mode & permute_mode::symmetric) == permute_mode::none) {
        return this->clone();
    }
    auto result = Csr::create(exec, size, nnz, this->get_strategy()->copy());
    auto local_permutation = make_temporary_clone(exec, permutation);
    std::unique_ptr<const Permutation<IndexType>> inv_permutation;
    const auto perm_idxs = local_permutation->get_const_permutation();
    const IndexType* inv_perm_idxs{};
    // Due to the sparse storage, we can only inverse-permute columns, so we
    // need to compute the inverse for forward-permutations.
    bool needs_inverse =
        (mode & permute_mode::inverse_columns) == permute_mode::columns;
    if (needs_inverse) {
        inv_permutation = local_permutation->compute_inverse();
        inv_perm_idxs = inv_permutation->get_const_permutation();
    }
    switch (mode) {
    case permute_mode::rows:
        exec->run(csr::make_row_permute(perm_idxs, this, result.get()));
        break;
    case permute_mode::columns:
        exec->run(csr::make_inv_col_permute(inv_perm_idxs, this, result.get()));
        break;
    case permute_mode::inverse_rows:
        exec->run(csr::make_inv_row_permute(perm_idxs, this, result.get()));
        break;
    case permute_mode::inverse_columns:
        exec->run(csr::make_inv_col_permute(perm_idxs, this, result.get()));
        break;
    case permute_mode::symmetric:
        exec->run(
            csr::make_inv_symm_permute(inv_perm_idxs, this, result.get()));
        break;
    case permute_mode::inverse_symmetric:
        exec->run(csr::make_inv_symm_permute(perm_idxs, this, result.get()));
        break;
    default:
        GKO_INVALID_STATE("Invalid permute mode");
    }
    result->make_srow();
    if ((mode & permute_mode::columns) == permute_mode::columns) {
        result->sort_by_column_index();
    }
    return result;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>> Csr<ValueType, IndexType>::permute(
    ptr_param<const Permutation<IndexType>> row_permutation,
    ptr_param<const Permutation<IndexType>> col_permutation, bool invert) const
{
    const auto exec = this->get_executor();
    const auto size = this->get_size();
    const auto nnz = this->get_num_stored_elements();
    GKO_ASSERT_EQUAL_ROWS(this, row_permutation);
    GKO_ASSERT_EQUAL_COLS(this, col_permutation);
    auto result = Csr::create(exec, size, nnz, this->get_strategy()->copy());
    auto local_row_permutation = make_temporary_clone(exec, row_permutation);
    auto local_col_permutation = make_temporary_clone(exec, col_permutation);
    if (invert) {
        exec->run(csr::make_inv_nonsymm_permute(
            local_row_permutation->get_const_permutation(),
            local_col_permutation->get_const_permutation(), this,
            result.get()));
    } else {
        const auto inv_row_perm = local_row_permutation->compute_inverse();
        const auto inv_col_perm = local_col_permutation->compute_inverse();
        exec->run(csr::make_inv_nonsymm_permute(
            inv_row_perm->get_const_permutation(),
            inv_col_perm->get_const_permutation(), this, result.get()));
    }
    result->make_srow();
    result->sort_by_column_index();
    return result;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::scale_permute(
    ptr_param<const ScaledPermutation<ValueType, IndexType>> permutation,
    permute_mode mode) const
{
    const auto exec = this->get_executor();
    const auto size = this->get_size();
    const auto nnz = this->get_num_stored_elements();
    validate_permute_dimensions(size, permutation->get_size(), mode);
    if ((mode & permute_mode::symmetric) == permute_mode::none) {
        return this->clone();
    }
    auto result = Csr::create(exec, size, nnz, this->get_strategy()->copy());
    auto local_permutation = make_temporary_clone(exec, permutation);
    std::unique_ptr<const ScaledPermutation<ValueType, IndexType>>
        inv_permutation;
    const auto perm_idxs = local_permutation->get_const_permutation();
    const auto scale_factors = local_permutation->get_const_scaling_factors();
    const ValueType* inv_scale_factors{};
    const IndexType* inv_perm_idxs{};
    // to permute columns, we need to know the inverse permutation
    bool needs_inverse =
        (mode & permute_mode::inverse_columns) == permute_mode::columns;
    if (needs_inverse) {
        inv_permutation = local_permutation->compute_inverse();
        inv_scale_factors = inv_permutation->get_const_scaling_factors();
        inv_perm_idxs = inv_permutation->get_const_permutation();
    }
    switch (mode) {
    case permute_mode::rows:
        exec->run(csr::make_row_scale_permute(scale_factors, perm_idxs, this,
                                              result.get()));
        break;
    case permute_mode::columns:
        exec->run(csr::make_inv_col_scale_permute(
            inv_scale_factors, inv_perm_idxs, this, result.get()));
        break;
    case permute_mode::inverse_rows:
        exec->run(csr::make_inv_row_scale_permute(scale_factors, perm_idxs,
                                                  this, result.get()));
        break;
    case permute_mode::inverse_columns:
        exec->run(csr::make_inv_col_scale_permute(scale_factors, perm_idxs,
                                                  this, result.get()));
        break;
    case permute_mode::symmetric:
        exec->run(csr::make_inv_symm_scale_permute(
            inv_scale_factors, inv_perm_idxs, this, result.get()));
        break;
    case permute_mode::inverse_symmetric:
        exec->run(csr::make_inv_symm_scale_permute(scale_factors, perm_idxs,
                                                   this, result.get()));
        break;
    default:
        GKO_INVALID_STATE("Invalid permute mode");
    }
    result->make_srow();
    if ((mode & permute_mode::columns) == permute_mode::columns) {
        result->sort_by_column_index();
    }
    return result;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::scale_permute(
    ptr_param<const ScaledPermutation<ValueType, IndexType>> row_permutation,
    ptr_param<const ScaledPermutation<ValueType, IndexType>> col_permutation,
    bool invert) const
{
    const auto exec = this->get_executor();
    const auto size = this->get_size();
    const auto nnz = this->get_num_stored_elements();
    GKO_ASSERT_EQUAL_ROWS(this, row_permutation);
    GKO_ASSERT_EQUAL_COLS(this, col_permutation);
    auto result = Csr::create(exec, size, nnz, this->get_strategy()->copy());
    auto local_row_permutation = make_temporary_clone(exec, row_permutation);
    auto local_col_permutation = make_temporary_clone(exec, col_permutation);
    if (invert) {
        exec->run(csr::make_inv_nonsymm_scale_permute(
            local_row_permutation->get_const_scaling_factors(),
            local_row_permutation->get_const_permutation(),
            local_col_permutation->get_const_scaling_factors(),
            local_col_permutation->get_const_permutation(), this,
            result.get()));
    } else {
        const auto inv_row_perm = local_row_permutation->compute_inverse();
        const auto inv_col_perm = local_col_permutation->compute_inverse();
        exec->run(csr::make_inv_nonsymm_scale_permute(
            inv_row_perm->get_const_scaling_factors(),
            inv_row_perm->get_const_permutation(),
            inv_col_perm->get_const_scaling_factors(),
            inv_col_perm->get_const_permutation(), this, result.get()));
    }
    result->make_srow();
    result->sort_by_column_index();
    return result;
}


template <typename IndexType>
std::unique_ptr<const Permutation<IndexType>> create_permutation_view(
    const array<IndexType>& indices)
{
    return Permutation<IndexType>::create_const(indices.get_executor(),
                                                indices.as_const_view());
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::permute(
    const array<IndexType>* permutation_indices) const
{
    return permute(create_permutation_view(*permutation_indices),
                   permute_mode::symmetric);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::inverse_permute(
    const array<IndexType>* permutation_indices) const
{
    return permute(create_permutation_view(*permutation_indices),
                   permute_mode::inverse_symmetric);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::row_permute(
    const array<IndexType>* permutation_indices) const
{
    return permute(create_permutation_view(*permutation_indices),
                   permute_mode::rows);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::column_permute(
    const array<IndexType>* permutation_indices) const
{
    return permute(create_permutation_view(*permutation_indices),
                   permute_mode::columns);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::inverse_row_permute(
    const array<IndexType>* permutation_indices) const
{
    return permute(create_permutation_view(*permutation_indices),
                   permute_mode::inverse_rows);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::inverse_column_permute(
    const array<IndexType>* permutation_indices) const
{
    return permute(create_permutation_view(*permutation_indices),
                   permute_mode::inverse_columns);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::sort_by_column_index()
{
    auto exec = this->get_executor();
    exec->run(csr::make_sort_by_column_index(this));
}


template <typename ValueType, typename IndexType>
bool Csr<ValueType, IndexType>::is_sorted_by_column_index() const
{
    auto exec = this->get_executor();
    bool is_sorted;
    exec->run(csr::make_is_sorted_by_column_index(this, &is_sorted));
    return is_sorted;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::create_submatrix(const gko::span& row_span,
                                            const gko::span& column_span) const
{
    using Mat = Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    auto sub_mat_size = gko::dim<2>(row_span.length(), column_span.length());
    array<IndexType> row_ptrs(exec, row_span.length() + 1);
    exec->run(csr::make_calculate_nonzeros_per_row_in_span(
        this, row_span, column_span, &row_ptrs));
    exec->run(csr::make_prefix_sum_nonnegative(row_ptrs.get_data(),
                                               row_span.length() + 1));
    auto num_nnz = get_element(row_ptrs, sub_mat_size[0]);
    auto sub_mat = Mat::create(exec, sub_mat_size,
                               std::move(array<ValueType>(exec, num_nnz)),
                               std::move(array<IndexType>(exec, num_nnz)),
                               std::move(row_ptrs), this->get_strategy());
    exec->run(csr::make_compute_submatrix(this, row_span, column_span,
                                          sub_mat.get()));
    sub_mat->make_srow();
    return sub_mat;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::create_submatrix(
    const index_set<IndexType>& row_index_set,
    const index_set<IndexType>& col_index_set) const
{
    using Mat = Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    if (!row_index_set.get_size() || !col_index_set.get_size()) {
        return Mat::create(exec);
    }
    if (row_index_set.is_contiguous() && col_index_set.is_contiguous()) {
        auto row_st = row_index_set.get_executor()->copy_val_to_host(
            row_index_set.get_subsets_begin());
        auto row_end = row_index_set.get_executor()->copy_val_to_host(
            row_index_set.get_subsets_end());
        auto col_st = col_index_set.get_executor()->copy_val_to_host(
            col_index_set.get_subsets_begin());
        auto col_end = col_index_set.get_executor()->copy_val_to_host(
            col_index_set.get_subsets_end());

        return this->create_submatrix(span(row_st, row_end),
                                      span(col_st, col_end));
    } else {
        auto submat_num_rows = row_index_set.get_num_elems();
        auto submat_num_cols = col_index_set.get_num_elems();
        auto sub_mat_size = gko::dim<2>(submat_num_rows, submat_num_cols);
        array<IndexType> row_ptrs(exec, submat_num_rows + 1);
        exec->run(csr::make_calculate_nonzeros_per_row_in_index_set(
            this, row_index_set, col_index_set, row_ptrs.get_data()));
        exec->run(csr::make_prefix_sum_nonnegative(row_ptrs.get_data(),
                                                   submat_num_rows + 1));
        auto num_nnz = get_element(row_ptrs, sub_mat_size[0]);
        auto sub_mat = Mat::create(exec, sub_mat_size,
                                   std::move(array<ValueType>(exec, num_nnz)),
                                   std::move(array<IndexType>(exec, num_nnz)),
                                   std::move(row_ptrs), this->get_strategy());
        exec->run(csr::make_compute_submatrix_from_index_set(
            this, row_index_set, col_index_set, sub_mat.get()));
        sub_mat->make_srow();
        return sub_mat;
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Csr<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(csr::make_fill_array(diag->get_values(), diag->get_size()[0],
                                   zero<ValueType>()));
    exec->run(csr::make_extract_diagonal(this, diag.get()));
    return diag;
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(csr::make_inplace_absolute_array(
        this->get_values(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Csr<ValueType, IndexType>::absolute_type>
Csr<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_csr = absolute_type::create(exec, this->get_size(),
                                         this->get_num_stored_elements());

    abs_csr->col_idxs_ = col_idxs_;
    abs_csr->row_ptrs_ = row_ptrs_;
    exec->run(csr::make_outplace_absolute_array(this->get_const_values(),
                                                this->get_num_stored_elements(),
                                                abs_csr->get_values()));

    convert_strategy_helper(abs_csr.get());
    return abs_csr;
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::scale_impl(const LinOp* alpha)
{
    auto exec = this->get_executor();
    exec->run(csr::make_scale(make_temporary_conversion<ValueType>(alpha).get(),
                              this));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::inv_scale_impl(const LinOp* alpha)
{
    auto exec = this->get_executor();
    exec->run(csr::make_inv_scale(
        make_temporary_conversion<ValueType>(alpha).get(), this));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::add_scaled_identity_impl(const LinOp* const a,
                                                         const LinOp* const b)
{
    bool has_diags{false};
    this->get_executor()->run(
        csr::make_check_diagonal_entries(this, has_diags));
    if (!has_diags) {
        GKO_UNSUPPORTED_MATRIX_PROPERTY(
            "The matrix has one or more structurally zero diagonal entries!");
    }
    this->get_executor()->run(csr::make_add_scaled_identity(
        make_temporary_conversion<ValueType>(a).get(),
        make_temporary_conversion<ValueType>(b).get(), this));
}


#define GKO_DECLARE_CSR_MATRIX(ValueType, IndexType) \
    class Csr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_MATRIX);


}  // namespace matrix
}  // namespace gko
