// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/csr.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
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
#include "core/components/precision_conversion_kernels.hpp"
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
GKO_REGISTER_OPERATION(spgemm_reuse, csr::spgemm_reuse);
GKO_REGISTER_OPERATION(advanced_spgemm_reuse, csr::advanced_spgemm_reuse);
GKO_REGISTER_OPERATION(spgeam, csr::spgeam);
GKO_REGISTER_OPERATION(spgeam_numeric, csr::spgeam_numeric);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(convert_ptrs_to_idxs, components::convert_ptrs_to_idxs);
GKO_REGISTER_OPERATION(fill_in_dense, csr::fill_in_dense);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);
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
GKO_REGISTER_OPERATION(convert_precision, components::convert_precision);
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
std::unique_ptr<const Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::create_const(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    gko::detail::const_array_view<ValueType>&& values,
    gko::detail::const_array_view<IndexType>&& col_idxs,
    gko::detail::const_array_view<IndexType>&& row_ptrs,
    csr::spmv_strategy strategy)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return create(exec, size, gko::detail::array_const_cast(std::move(values)),
                  gko::detail::array_const_cast(std::move(col_idxs)),
                  gko::detail::array_const_cast(std::move(row_ptrs)), strategy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>> Csr<ValueType, IndexType>::create(
    std::shared_ptr<const Executor> exec, csr::spmv_strategy strategy)
{
    return create(exec, dim<2>{}, size_type{}, std::move(strategy));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>> Csr<ValueType, IndexType>::create(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    size_type num_nonzeros, csr::spmv_strategy strategy)
{
    return std::unique_ptr<Csr>{
        new Csr{exec, size, num_nonzeros, std::move(strategy)}};
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>> Csr<ValueType, IndexType>::create(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    array<value_type> values, array<index_type> col_idxs,
    array<index_type> row_ptrs, csr::spmv_strategy strategy)
{
    return std::unique_ptr<Csr>{
        new Csr{exec, size, std::move(values), std::move(col_idxs),
                std::move(row_ptrs), std::move(strategy)}};
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::Csr(std::shared_ptr<const Executor> exec,
                               const dim<2>& size, size_type num_nonzeros,
                               csr::spmv_strategy strategy)
    : LinOp(exec, size),
      strategy_(strategy),
      values_(exec, num_nonzeros),
      col_idxs_(exec, num_nonzeros),
      row_ptrs_(exec, size[0] + 1),
      srow_(exec)
{
    row_ptrs_.fill(0);
    // this->make_srow();
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::Csr(std::shared_ptr<const Executor> exec,
                               const dim<2>& size, array<value_type> values,
                               array<index_type> col_idxs,
                               array<index_type> row_ptrs,
                               csr::spmv_strategy strategy)
    : LinOp(exec, size),
      strategy_(strategy),
      values_{exec, std::move(values)},
      col_idxs_{exec, std::move(col_idxs)},
      row_ptrs_{exec, std::move(row_ptrs)},
      srow_(exec)
{
    GKO_ASSERT_EQ(values_.get_size(), col_idxs_.get_size());
    GKO_ASSERT_EQ(this->get_size()[0] + 1, row_ptrs_.get_size());
    this->make_srow();
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>& Csr<ValueType, IndexType>::operator=(
    const Csr<ValueType, IndexType>& other)
{
    if (&other != this) {
        LinOp::operator=(other);
        // NOTE: as soon as strategies are improved, this can be reverted
        values_ = other.values_;
        col_idxs_ = other.col_idxs_;
        row_ptrs_ = other.row_ptrs_;
        srow_ = other.srow_;
        this->set_strategy(other.get_strategy());
        // END NOTE
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>& Csr<ValueType, IndexType>::operator=(
    Csr<ValueType, IndexType>&& other)
{
    if (&other != this) {
        LinOp::operator=(std::move(other));
        values_ = std::move(other.values_);
        col_idxs_ = std::move(other.col_idxs_);
        row_ptrs_ = std::move(other.row_ptrs_);
        srow_ = std::move(other.srow_);
        strategy_ = other.strategy_;
        if (this->get_executor() != other.get_executor()) {
            this->make_srow();
            // detail::strategy_rebuild_helper(this);
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
        auto builder = CsrBuilder<ValueType, IndexType>(x_csr);
        this->get_executor()->run(csr::make_spgemm(this, b_csr, &builder));
    } else {
        mixed_precision_dispatch_real_complex<ValueType>(
            [this](auto dense_b, auto dense_x) {
                this->get_executor()->run(csr::make_spmv(
                    this->get_actual_strategy(), max_nnz_per_row_, this,
                    dense_b->get_const_device_view(),
                    dense_x->get_device_view()));
            },
            b, x);
    }
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::make_srow()
{
    size_type srow_size = 0;
    int warp_size = 0;
    int64_t nwarps = 0;
    max_nnz_per_row_ = 0;
    auto exec = this->get_executor();
    array<IndexType> row_ptrs_host(exec->get_master());
    const IndexType* row_ptrs{nullptr};
    if (exec == exec->get_master()) {
        row_ptrs = row_ptrs_.get_const_data();
    } else {
        row_ptrs_host = row_ptrs_;
        row_ptrs = row_ptrs_host.get_const_data();
    }
    // calculate the max_nnz_per_row in host
    for (int i = 0; i < this->get_size()[0]; i++) {
        max_nnz_per_row_ =
            std::max(max_nnz_per_row_, row_ptrs[i + 1] - row_ptrs[i]);
    }

    if (auto dexec = std::dynamic_pointer_cast<const CudaExecutor>(exec)) {
        nwarps = dexec->get_num_warps();
        warp_size = dexec->get_warp_size();
    } else if (auto dexec =
                   std::dynamic_pointer_cast<const HipExecutor>(exec)) {
        nwarps = dexec->get_num_warps();
        warp_size = dexec->get_warp_size();
    } else if (auto dexec =
                   std::dynamic_pointer_cast<const DpcppExecutor>(exec)) {
        nwarps = dexec->get_num_subgroups();
        warp_size = 32;
    }
    auto load_balance_size = [&](const int64_t nnz) -> int64_t {
        int multiple = 8;
        if (std::dynamic_pointer_cast<const CudaExecutor>(exec)) {
            if (nnz >= static_cast<int64_t>(2e8)) {
                multiple = 2048;
            } else if (nnz >= static_cast<int64_t>(2e7)) {
                multiple = 512;
            } else if (nnz >= static_cast<int64_t>(2e6)) {
                multiple = 128;
            } else if (nnz >= static_cast<int64_t>(2e5)) {
                multiple = 32;
            }
        } else if (std::dynamic_pointer_cast<const HipExecutor>(exec)) {
            // only for AMD GPU
            if (nnz >= static_cast<int64_t>(1e7)) {
                multiple = 64;
            } else if (nnz >= static_cast<int64_t>(1e6)) {
                multiple = 16;
            }
        } else if (std::dynamic_pointer_cast<const DpcppExecutor>(exec)) {
            if (nnz >= static_cast<int64_t>(2e8)) {
                multiple = 256;
            } else if (nnz >= static_cast<int64_t>(2e7)) {
                multiple = 32;
            }
        } else {
            return 0;
        }
        return static_cast<int64_t>(std::min(
            ceildiv(nnz, warp_size), static_cast<int64_t>(nwarps * multiple)));
    };

    if (strategy_ == csr::spmv_strategy::load_balance ||
        strategy_ == csr::spmv_strategy::automatical) {
        srow_size = load_balance_size(this->get_num_stored_elements());
    }
    // just to make load_balance(2) works
    // srow_size = std::max(srow_size, size_type{1});
    srow_.resize_and_reset(srow_size);
    if (srow_size != 0) {
        srow_.set_executor(exec->get_master());
        const auto num_rows = this->get_size()[0];
        for (size_type i = 0; i < srow_size; i++) {
            srow_.get_data()[i] = 0;
        }
        const auto num_elems = this->get_num_stored_elements();
        const auto bucket_divider =
            num_elems > 0 ? ceildiv(num_elems, warp_size) : 1;
        for (size_type i = 0; i < num_rows; i++) {
            auto bucket =
                ceildiv((ceildiv(row_ptrs[i + 1], warp_size) * srow_size),
                        bucket_divider);
            if (bucket < srow_size) {
                srow_.get_data()[bucket]++;
            }
        }
        // find starting row for thread i
        for (size_type i = 1; i < srow_size; i++) {
            srow_.get_data()[i] += srow_.get_data()[i - 1];
        }
        srow_.set_executor(exec);
    }
    row_ptrs_.set_executor(exec);
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
        auto builder = CsrBuilder<ValueType, IndexType>(x_csr);
        this->get_executor()->run(csr::make_advanced_spgemm(
            as<Dense<ValueType>>(alpha)->get_const_device_view(), this, b_csr,
            as<Dense<ValueType>>(beta)->get_const_device_view(), x_copy.get(),
            &builder));
    } else if (dynamic_cast<const Identity<ValueType>*>(b)) {
        // if b is an identity matrix, we compute an SpGEAM
        auto x_csr = as<TCsr>(x);
        auto x_copy = x_csr->clone();
        auto builder = CsrBuilder<ValueType, IndexType>(x_csr);
        this->get_executor()->run(csr::make_spgeam(
            as<Dense<ValueType>>(alpha)->get_const_device_view(), this,
            as<Dense<ValueType>>(beta)->get_const_device_view(), x_copy.get(),
            &builder));
    } else {
        mixed_precision_dispatch_real_complex<ValueType>(
            [this, alpha, beta](auto dense_b, auto dense_x) {
                auto dense_alpha = make_temporary_conversion<ValueType>(alpha);
                auto dense_beta = make_temporary_conversion<
                    typename std::decay_t<decltype(*dense_x)>::value_type>(
                    beta);
                this->get_executor()->run(csr::make_advanced_spmv(
                    this->get_actual_strategy(), max_nnz_per_row_,
                    dense_alpha->get_const_device_view(), this,
                    dense_b->get_const_device_view(),
                    dense_beta->get_const_device_view(),
                    dense_x->get_device_view()));
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
    result->set_strategy(this->get_strategy());
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(
    Csr<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}

#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Csr<next_precision<ValueType, 2>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    result->set_size(this->get_size());
    result->set_strategy(this->get_strategy());
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(
    Csr<next_precision<ValueType, 2>, IndexType>* result)
{
    this->convert_to(result);
}
#endif


#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Csr<next_precision<ValueType, 3>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    result->set_size(this->get_size());
    result->set_strategy(this->get_strategy());
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(
    Csr<next_precision<ValueType, 3>, IndexType>* result)
{
    this->convert_to(result);
}
#endif


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
    exec->run(csr::make_fill_in_dense(this, tmp_result->get_device_view()));
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
                                          tmp->get_device_view()));
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
    exec->run(csr::make_convert_to_sellp(this, tmp->get_device_view()));
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
    exec->run(csr::make_convert_to_ell(this, tmp->get_device_view()));
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
std::unique_ptr<Csr<ValueType, IndexType>> Csr<ValueType, IndexType>::multiply(
    ptr_param<const Csr> other) const
{
    GKO_ASSERT_CONFORMANT(this, other);
    auto result_size = gko::dim<2>{this->get_size()[0], other->get_size()[1]};
    auto exec = this->get_executor();
    auto local_other = make_temporary_clone(exec, other);
    auto result = Csr::create(exec, result_size);
    {
        auto builder = CsrBuilder<ValueType, IndexType>(result);
        exec->run(csr::make_spgemm(this, local_other.get(), &builder));
    }
    return result;
}


template <typename ValueType, typename IndexType>
struct Csr<ValueType, IndexType>::multiply_reuse_info::lookup_data {
    dim<2> size1;
    dim<2> size2;
    dim<2> size_out;
    size_type nnz1;
    size_type nnz2;
    size_type nnz_out;
    csr::lookup_data<IndexType> data;
};


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::multiply_reuse_info::~multiply_reuse_info() =
    default;


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::multiply_reuse_info::multiply_reuse_info(
    multiply_reuse_info&&) noexcept = default;


template <typename ValueType, typename IndexType>
typename Csr<ValueType, IndexType>::multiply_reuse_info&
Csr<ValueType, IndexType>::multiply_reuse_info::operator=(
    multiply_reuse_info&&) noexcept = default;


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::multiply_reuse_info::multiply_reuse_info() = default;


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::multiply_reuse_info::multiply_reuse_info(
    std::unique_ptr<lookup_data> data)
    : internal{std::move(data)}
{}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::multiply_reuse_info::update_values(
    ptr_param<const Csr> mtx1, ptr_param<const Csr> mtx2,
    ptr_param<Csr> out) const
{
    if (!internal) {
        throw InvalidStateError{
            __FILE__, __LINE__, __func__,
            "Attempting to use uninitialized multiply_reuse_info"};
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(mtx1, internal->size1);
    GKO_ASSERT_EQUAL_DIMENSIONS(mtx2, internal->size2);
    GKO_ASSERT_EQUAL_DIMENSIONS(out, internal->size_out);
    GKO_ASSERT_EQ(mtx1->get_num_stored_elements(), internal->nnz1);
    GKO_ASSERT_EQ(mtx2->get_num_stored_elements(), internal->nnz2);
    GKO_ASSERT_EQ(out->get_num_stored_elements(), internal->nnz_out);
    auto exec = internal->data.storage.get_executor();
    auto local_mtx1 = make_temporary_clone(exec, mtx1);
    auto local_mtx2 = make_temporary_clone(exec, mtx2);
    auto local_out = make_temporary_clone(exec, out);
    exec->run(csr::make_spgemm_reuse(local_mtx1.get(), local_mtx2.get(),
                                     internal->data, local_out.get()));
}


template <typename ValueType, typename IndexType>
std::pair<std::unique_ptr<Csr<ValueType, IndexType>>,
          typename Csr<ValueType, IndexType>::multiply_reuse_info>
Csr<ValueType, IndexType>::multiply_reuse(ptr_param<const Csr> other) const
{
    GKO_ASSERT_CONFORMANT(this, other);
    auto result_size = gko::dim<2>{this->get_size()[0], other->get_size()[1]};
    auto exec = this->get_executor();
    auto local_other = make_temporary_clone(exec, other);
    auto result = Csr::create(exec, result_size);
    {
        auto builder = CsrBuilder<ValueType, IndexType>(result);
        exec->run(csr::make_spgemm(this, local_other.get(), &builder));
    }
    auto lookup = csr::build_lookup(result.get());
    auto reuse_info = multiply_reuse_info{
        std::make_unique<typename multiply_reuse_info::lookup_data>(
            typename multiply_reuse_info::lookup_data{
                this->get_size(), other->get_size(), result_size,
                this->get_num_stored_elements(),
                other->get_num_stored_elements(),
                result->get_num_stored_elements(), std::move(lookup)})};
    return std::make_pair(std::move(result), std::move(reuse_info));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::multiply_add(
    ptr_param<const Dense<value_type>> scale_mult,
    ptr_param<const Csr> mtx_mult, ptr_param<const Dense<value_type>> scale_add,
    ptr_param<const Csr> mtx_add) const
{
    GKO_ASSERT_CONFORMANT(this, mtx_mult);
    auto result_size =
        gko::dim<2>{this->get_size()[0], mtx_mult->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(mtx_add, result_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(scale_mult, dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(scale_add, dim<2>(1, 1));
    auto exec = this->get_executor();
    auto local_scale_mult = make_temporary_clone(exec, scale_mult);
    auto local_mtx_mult = make_temporary_clone(exec, mtx_mult);
    auto local_scale_add = make_temporary_clone(exec, scale_add);
    auto local_mtx_add = make_temporary_clone(exec, mtx_add);
    auto result = Csr::create(exec, result_size);
    {
        auto builder = CsrBuilder<ValueType, IndexType>(result);
        exec->run(csr::make_advanced_spgemm(
            local_scale_mult->get_const_device_view(), this,
            local_mtx_mult.get(), local_scale_add->get_const_device_view(),
            local_mtx_add.get(), &builder));
    }
    return result;
}


template <typename ValueType, typename IndexType>
struct Csr<ValueType, IndexType>::multiply_add_reuse_info::lookup_data {
    dim<2> size1;
    dim<2> size2;
    dim<2> size_out;
    size_type nnz1;
    size_type nnz2;
    size_type nnz3;
    size_type nnz_out;
    csr::lookup_data<IndexType> data;
};


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::multiply_add_reuse_info::~multiply_add_reuse_info() =
    default;


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::multiply_add_reuse_info::multiply_add_reuse_info(
    multiply_add_reuse_info&&) noexcept = default;


template <typename ValueType, typename IndexType>
typename Csr<ValueType, IndexType>::multiply_add_reuse_info&
Csr<ValueType, IndexType>::multiply_add_reuse_info::operator=(
    multiply_add_reuse_info&&) noexcept = default;


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::multiply_add_reuse_info::multiply_add_reuse_info() =
    default;


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::multiply_add_reuse_info::multiply_add_reuse_info(
    std::unique_ptr<lookup_data> data)
    : internal{std::move(data)}
{}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::multiply_add_reuse_info::update_values(
    ptr_param<const Csr> mtx1, ptr_param<const Dense<value_type>> alpha,
    ptr_param<const Csr> mtx2, ptr_param<const Dense<value_type>> beta,
    ptr_param<const Csr> mtx3, ptr_param<Csr> out) const
{
    if (!internal) {
        throw InvalidStateError{
            __FILE__, __LINE__, __func__,
            "Attempting to use uninitialized multiply_add_reuse_info"};
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(mtx1, internal->size1);
    GKO_ASSERT_EQUAL_DIMENSIONS(mtx2, internal->size2);
    GKO_ASSERT_EQUAL_DIMENSIONS(mtx3, internal->size_out);
    GKO_ASSERT_EQUAL_DIMENSIONS(out, internal->size_out);
    GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(beta, dim<2>(1, 1));
    GKO_ASSERT_EQ(mtx1->get_num_stored_elements(), internal->nnz1);
    GKO_ASSERT_EQ(mtx2->get_num_stored_elements(), internal->nnz2);
    GKO_ASSERT_EQ(mtx3->get_num_stored_elements(), internal->nnz3);
    GKO_ASSERT_EQ(out->get_num_stored_elements(), internal->nnz_out);
    auto exec = internal->data.storage.get_executor();
    auto local_mtx1 = make_temporary_clone(exec, mtx1);
    auto local_mtx2 = make_temporary_clone(exec, mtx2);
    auto local_mtx3 = make_temporary_clone(exec, mtx3);
    auto local_out = make_temporary_clone(exec, out);
    auto local_alpha = make_temporary_clone(exec, alpha);
    auto local_beta = make_temporary_clone(exec, beta);
    exec->run(csr::make_advanced_spgemm_reuse(
        local_alpha->get_const_device_view(), local_mtx1.get(),
        local_mtx2.get(), local_beta->get_const_device_view(), local_mtx3.get(),
        internal->data, local_out.get()));
}


template <typename ValueType, typename IndexType>
std::pair<std::unique_ptr<Csr<ValueType, IndexType>>,
          typename Csr<ValueType, IndexType>::multiply_add_reuse_info>
Csr<ValueType, IndexType>::multiply_add_reuse(
    ptr_param<const Dense<value_type>> scale_mult,
    ptr_param<const Csr> mtx_mult, ptr_param<const Dense<value_type>> scale_add,
    ptr_param<const Csr> mtx_add) const
{
    GKO_ASSERT_CONFORMANT(this, mtx_mult);
    auto result_size =
        gko::dim<2>{this->get_size()[0], mtx_mult->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(mtx_add, result_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(scale_mult, dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(scale_add, dim<2>(1, 1));
    auto exec = this->get_executor();
    auto local_scale_mult = make_temporary_clone(exec, scale_mult);
    auto local_mtx_mult = make_temporary_clone(exec, mtx_mult);
    auto local_scale_add = make_temporary_clone(exec, scale_add);
    auto local_mtx_add = make_temporary_clone(exec, mtx_add);
    auto result = Csr::create(exec, result_size);
    {
        auto builder = CsrBuilder<ValueType, IndexType>(result);
        exec->run(csr::make_advanced_spgemm(
            local_scale_mult->get_const_device_view(), this,
            local_mtx_mult.get(), local_scale_add->get_const_device_view(),
            local_mtx_add.get(), &builder));
    }
    auto lookup = csr::build_lookup(result.get());
    auto reuse_info = multiply_add_reuse_info{
        std::make_unique<typename multiply_add_reuse_info::lookup_data>(
            typename multiply_add_reuse_info::lookup_data{
                this->get_size(), mtx_mult->get_size(), result_size,
                this->get_num_stored_elements(),
                mtx_mult->get_num_stored_elements(),
                mtx_add->get_num_stored_elements(),
                result->get_num_stored_elements(), std::move(lookup)})};
    return std::make_pair(std::move(result), std::move(reuse_info));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>> Csr<ValueType, IndexType>::scale_add(
    ptr_param<const Dense<value_type>> scale_this,
    ptr_param<const Dense<value_type>> scale_other,
    ptr_param<const Csr> mtx_other) const
{
    auto exec = this->get_executor();
    GKO_ASSERT_EQUAL_DIMENSIONS(this, mtx_other);
    GKO_ASSERT_EQUAL_DIMENSIONS(scale_this, dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(scale_other, dim<2>(1, 1));
    auto local_scale_this = make_temporary_clone(exec, scale_this);
    auto local_scale_other = make_temporary_clone(exec, scale_other);
    auto local_mtx_other = make_temporary_clone(exec, mtx_other);
    auto result = Csr::create(exec, this->get_size());
    {
        auto builder = CsrBuilder<ValueType, IndexType>(result);
        exec->run(csr::make_spgeam(local_scale_this->get_const_device_view(),
                                   this,
                                   local_scale_other->get_const_device_view(),
                                   local_mtx_other.get(), &builder));
    }
    return result;
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::scale_add_reuse_info::scale_add_reuse_info() =
    default;


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::scale_add_reuse_info::scale_add_reuse_info(
    std::unique_ptr<lookup_data> data)
    : internal{std::move(data)}
{}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::scale_add_reuse_info::~scale_add_reuse_info() =
    default;


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::scale_add_reuse_info::scale_add_reuse_info(
    scale_add_reuse_info&&) noexcept = default;


template <typename ValueType, typename IndexType>
typename Csr<ValueType, IndexType>::scale_add_reuse_info&
Csr<ValueType, IndexType>::scale_add_reuse_info::operator=(
    scale_add_reuse_info&&) noexcept = default;


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::scale_add_reuse_info::update_values(
    ptr_param<const Dense<value_type>> scale1, ptr_param<const Csr> mtx1,
    ptr_param<const Dense<value_type>> scale2, ptr_param<const Csr> mtx2,
    ptr_param<Csr> out) const
{
    if (!internal) {
        throw InvalidStateError{
            __FILE__, __LINE__, __func__,
            "Attempting to use uninitialized scale_add_reuse_info"};
    }
    auto exec = internal->exec;
    GKO_ASSERT_EQUAL_DIMENSIONS(mtx1, internal->size);
    GKO_ASSERT_EQUAL_DIMENSIONS(mtx2, internal->size);
    GKO_ASSERT_EQUAL_DIMENSIONS(out, internal->size);
    GKO_ASSERT_EQUAL_DIMENSIONS(scale1, dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(scale2, dim<2>(1, 1));
    GKO_ASSERT_EQ(mtx1->get_num_stored_elements(), internal->nnz1);
    GKO_ASSERT_EQ(mtx2->get_num_stored_elements(), internal->nnz2);
    GKO_ASSERT_EQ(out->get_num_stored_elements(), internal->nnz_out);
    auto local_scale1 = make_temporary_clone(exec, scale1);
    auto local_scale2 = make_temporary_clone(exec, scale2);
    auto local_mtx1 = make_temporary_clone(exec, mtx1);
    auto local_mtx2 = make_temporary_clone(exec, mtx2);
    auto local_mtx_out = make_temporary_clone(exec, out);
    exec->run(csr::make_spgeam_numeric(local_scale1->get_const_device_view(),
                                       local_mtx1.get(),
                                       local_scale2->get_const_device_view(),
                                       local_mtx2.get(), local_mtx_out.get()));
}


template <typename ValueType, typename IndexType>
struct Csr<ValueType, IndexType>::scale_add_reuse_info::lookup_data {
    std::shared_ptr<const Executor> exec;
    dim<2> size;
    size_type nnz1;
    size_type nnz2;
    size_type nnz_out;
    // any potential future optimization data for repeated SpGEAM goes here
};


template <typename ValueType, typename IndexType>
std::pair<std::unique_ptr<Csr<ValueType, IndexType>>,
          typename Csr<ValueType, IndexType>::scale_add_reuse_info>
Csr<ValueType, IndexType>::add_scale_reuse(
    ptr_param<const Dense<value_type>> scale_this,
    ptr_param<const Dense<value_type>> scale_other,
    ptr_param<const Csr> mtx_other) const
{
    auto exec = this->get_executor();
    GKO_ASSERT_EQUAL_DIMENSIONS(this, mtx_other);
    GKO_ASSERT_EQUAL_DIMENSIONS(scale_this, dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(scale_other, dim<2>(1, 1));
    auto local_scale_this = make_temporary_clone(exec, scale_this);
    auto local_scale_other = make_temporary_clone(exec, scale_other);
    auto local_mtx_other = make_temporary_clone(exec, mtx_other);
    auto result = Csr::create(exec, this->get_size());
    {
        auto builder = CsrBuilder<ValueType, IndexType>(result);
        exec->run(csr::make_spgeam(local_scale_this->get_const_device_view(),
                                   this,
                                   local_scale_other->get_const_device_view(),
                                   local_mtx_other.get(), &builder));
    }
    return std::make_pair(
        std::move(result),
        scale_add_reuse_info{
            std::make_unique<typename scale_add_reuse_info::lookup_data>(
                typename scale_add_reuse_info::lookup_data{
                    exec, this->get_size(), this->get_num_stored_elements(),
                    mtx_other->get_num_stored_elements(),
                    result->get_num_stored_elements()})});
}


template <typename ValueType, typename IndexType, typename TransformClosure>
std::pair<std::unique_ptr<Csr<ValueType, IndexType>>,
          typename Csr<ValueType, IndexType>::permuting_reuse_info>
transform_reusable(const Csr<ValueType, IndexType>* input, gko::dim<2> out_size,
                   size_type nnz, TransformClosure closure)
{
    using FloatIndexType =
        std::conditional_t<std::is_same_v<IndexType, int32>, float, double>;
    static_assert(sizeof(FloatIndexType) == sizeof(IndexType));
    static_assert(alignof(FloatIndexType) == alignof(IndexType));
    auto exec = input->get_executor();
    auto in_size = input->get_size();
    auto transformed = Csr<ValueType, IndexType>::create(exec, out_size, nnz);
    // transform matrix with integer values from 0 to nnz - 1 reinterpret_cast
    // as float
    array<IndexType> iota_values{exec, nnz};
    exec->run(csr::make_fill_seq_array(iota_values.get_data(), nnz));
    auto iota_float_view = make_array_view(
        exec, nnz, reinterpret_cast<FloatIndexType*>(iota_values.get_data()));
    auto iota_mtx = Csr<FloatIndexType, IndexType>::create_const(
        exec, input->get_size(), iota_float_view.as_const_view(),
        make_const_array_view(exec, nnz, input->get_const_col_idxs()),
        make_const_array_view(exec, in_size[0] + 1,
                              input->get_const_row_ptrs()),
        csr::spmv_strategy::sparselib);
    auto transformed_iota = closure(iota_mtx.get());
    exec->copy(out_size[0] + 1, transformed_iota->get_const_row_ptrs(),
               transformed->get_row_ptrs());
    exec->copy(nnz, transformed_iota->get_const_col_idxs(),
               transformed->get_col_idxs());
    exec->copy(nnz,
               reinterpret_cast<const IndexType*>(
                   transformed_iota->get_const_values()),
               iota_values.get_data());
    auto transform_permutation =
        Permutation<IndexType>::create(exec, std::move(iota_values));
    transformed->set_strategy(input->get_strategy());
    // permute values into output matrix
    input->create_const_value_view()->permute(transform_permutation,
                                              transformed->create_value_view(),
                                              permute_mode::rows);

    return std::make_pair(
        std::move(transformed),
        typename Csr<ValueType, IndexType>::permuting_reuse_info{
            std::move(transform_permutation)});
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::permuting_reuse_info::permuting_reuse_info()
    : permuting_reuse_info{nullptr}
{}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::permuting_reuse_info::permuting_reuse_info(
    std::unique_ptr<Permutation<index_type>> value_permutation)
    : value_permutation{std::move(value_permutation)}
{}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::permuting_reuse_info::update_values(
    ptr_param<const Csr> input, ptr_param<Csr> output) const
{
    if (!value_permutation) {
        GKO_NOT_SUPPORTED(value_permutation);
    }
    input->create_const_value_view()->permute(
        value_permutation, output->create_value_view(), permute_mode::rows);
}


template <typename ValueType, typename IndexType>
auto Csr<ValueType, IndexType>::transpose_reuse() const
    -> std::pair<std::unique_ptr<Csr>, Csr::permuting_reuse_info>
{
    return transform_reusable(
        this, gko::transpose(this->get_size()), this->get_num_stored_elements(),
        [](auto mtx) {
            return as<gko::detail::pointee<decltype(mtx)>>(mtx->transpose());
        });
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
    auto result = Csr::create(exec, size, nnz, this->get_strategy());
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
    auto result = Csr::create(exec, size, nnz, this->get_strategy());
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
auto Csr<ValueType, IndexType>::permute_reuse(
    ptr_param<const Permutation<index_type>> permutation,
    permute_mode mode) const
    -> std::pair<std::unique_ptr<Csr>, permuting_reuse_info>
{
    return transform_reusable(
        this, this->get_size(), this->get_num_stored_elements(),
        [&](auto mtx) { return mtx->permute(permutation, mode); });
}


template <typename ValueType, typename IndexType>
auto Csr<ValueType, IndexType>::permute_reuse(
    ptr_param<const Permutation<index_type>> row_permutation,
    ptr_param<const Permutation<index_type>> column_permutation,
    bool invert) const -> std::pair<std::unique_ptr<Csr>, permuting_reuse_info>
{
    return transform_reusable(
        this, this->get_size(), this->get_num_stored_elements(), [&](auto mtx) {
            return mtx->permute(row_permutation, column_permutation, invert);
        });
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
    auto result = Csr::create(exec, size, nnz, this->get_strategy());
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
    auto result = Csr::create(exec, size, nnz, this->get_strategy());
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
    exec->run(csr::make_is_sorted_by_column_index(this, is_sorted));
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
        this, row_span, column_span, row_ptrs));
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
std::unique_ptr<Dense<ValueType>> Csr<ValueType, IndexType>::create_value_view()
{
    const auto nnz = this->get_num_stored_elements();
    const auto exec = this->get_executor();
    return Dense<ValueType>::create(
        exec, gko::dim<2>{nnz, 1},
        make_array_view(exec, nnz, this->get_values()), 1);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<const Dense<ValueType>>
Csr<ValueType, IndexType>::create_const_value_view() const
{
    const auto nnz = this->get_num_stored_elements();
    const auto exec = this->get_executor();
    return Dense<ValueType>::create_const(
        exec, gko::dim<2>{nnz, 1},
        make_const_array_view(exec, nnz, this->get_const_values()), 1);
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

    abs_csr->make_srow();
    return abs_csr;
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::scale_impl(const LinOp* alpha)
{
    auto exec = this->get_executor();
    exec->run(csr::make_scale(
        make_temporary_conversion<ValueType>(alpha)->get_const_device_view(),
        this));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::inv_scale_impl(const LinOp* alpha)
{
    auto exec = this->get_executor();
    exec->run(csr::make_inv_scale(
        make_temporary_conversion<ValueType>(alpha)->get_const_device_view(),
        this));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::add_scaled_identity_impl(const LinOp* a,
                                                         const LinOp* b)
{
    bool has_diags{false};
    this->get_executor()->run(
        csr::make_check_diagonal_entries(this, has_diags));
    if (!has_diags) {
        GKO_UNSUPPORTED_MATRIX_PROPERTY(
            "The matrix has one or more structurally zero diagonal entries!");
    }
    this->get_executor()->run(csr::make_add_scaled_identity(
        make_temporary_conversion<ValueType>(a)->get_const_device_view(),
        make_temporary_conversion<ValueType>(b)->get_const_device_view(),
        this));
}

template <typename ValueType, typename IndexType>
csr::spmv_strategy Csr<ValueType, IndexType>::get_strategy() const noexcept
{
    return strategy_;
}

template <typename ValueType, typename IndexType>
csr::spmv_strategy Csr<ValueType, IndexType>::get_actual_strategy()
    const noexcept
{
    auto strategy = this->get_strategy();
    if (strategy != csr::spmv_strategy::automatical) {
        return strategy;
    }
    auto exec = this->get_executor();
    // If the number of stored elements is larger than <nnz_limit> or the
    // maximum
    // number of stored elements per row is larger than <row_len_limit>, use
    // load_balance otherwise use classical
    /* Use imbalance strategy when the maximum number of nonzero per row
     * is more than 1024 on NVIDIA hardware */
    const int64_t nvidia_row_len_limit = 1024;
    /* Use imbalance strategy when the matrix has more more than 1e6 on
     * NVIDIA hardware */
    const int64_t nvidia_nnz_limit{static_cast<int64_t>(1e6)};
    /* Use imbalance strategy when the maximum number of nonzero per row
     * is more than 768 on AMD hardware */
    const int64_t amd_row_len_limit = 768;
    /* Use imbalance strategy when the matrix has more more than 1e8 on
     * AMD hardware */
    const int64_t amd_nnz_limit{static_cast<int64_t>(1e8)};
    /* Use imbalance strategy when the maximum number of nonzero per row
     * is more than 25600 on Intel hardware */
    const int64_t intel_row_len_limit = 25600;
    /* Use imbalance strategy when the matrix has more more than 3e8 on
     * Intel hardware */
    const int64_t intel_nnz_limit{static_cast<int64_t>(3e8)};
    auto nnz_limit = nvidia_nnz_limit;
    auto row_len_limit = nvidia_row_len_limit;
    if (std::dynamic_pointer_cast<const DpcppExecutor>(exec)) {
        nnz_limit = intel_nnz_limit;
        row_len_limit = intel_row_len_limit;
    } else if (std::dynamic_pointer_cast<const HipExecutor>(exec)) {
        nnz_limit = amd_nnz_limit;
        row_len_limit = amd_row_len_limit;
    } else if (!std::dynamic_pointer_cast<const CudaExecutor>(exec)) {
        // we do not have load balance on reference and omp executor.
        return csr::spmv_strategy::classical;
    }
    if (this->get_num_stored_elements() > nnz_limit ||
        max_nnz_per_row_ > row_len_limit) {
        return csr::spmv_strategy::load_balance;
    } else {
        return csr::spmv_strategy::classical;
    }
}


#define GKO_DECLARE_CSR_MATRIX(ValueType, IndexType) \
    class Csr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_MATRIX);


}  // namespace matrix
}  // namespace gko
