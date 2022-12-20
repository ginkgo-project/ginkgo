/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "core/matrix/hybrid_kernels.hpp"
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
GKO_REGISTER_OPERATION(inverse_row_permute, csr::inverse_row_permute);
GKO_REGISTER_OPERATION(inverse_column_permute, csr::inverse_column_permute);
GKO_REGISTER_OPERATION(invert_permutation, csr::invert_permutation);
GKO_REGISTER_OPERATION(convert_ptrs_to_sizes,
                       components::convert_ptrs_to_sizes);
GKO_REGISTER_OPERATION(sort_by_column_index, csr::sort_by_column_index);
GKO_REGISTER_OPERATION(is_sorted_by_column_index,
                       csr::is_sorted_by_column_index);
GKO_REGISTER_OPERATION(extract_diagonal, csr::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(prefix_sum, components::prefix_sum);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);
GKO_REGISTER_OPERATION(scale, csr::scale);
GKO_REGISTER_OPERATION(inv_scale, csr::inv_scale);
GKO_REGISTER_OPERATION(add_scaled_identity, csr::add_scaled_identity);
GKO_REGISTER_OPERATION(check_diagonal_entries,
                       csr::check_diagonal_entries_exist);


}  // anonymous namespace
}  // namespace csr


namespace detail {
/**
 *
 * Helper function that extends the sparsity pattern of the matrix M to M^n
 * without changing its values.
 *
 * The input matrix must be sorted and on the correct executor for this to work.
 * If `power` is 1, the matrix will be returned unchanged.
 */
template <typename ValueType, typename IndexType>
std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> extend_sparsity(
    std::shared_ptr<const Executor>& exec,
    std::shared_ptr<const gko::matrix::Csr<ValueType, IndexType>> mtx,
    int power)
{
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    GKO_ASSERT(power >= 1);
    if (power == 1) {
        // copy the matrix, as it will be used to store the inverse
        return {std::move(mtx->clone())};
    }
    auto id_power = mtx->clone();
    auto tmp = csr::create(exec, mtx->get_size());
    // accumulates mtx * the remainder from odd powers
    auto acc = mtx->clone();
    // compute id^(n-1) using square-and-multiply
    int i = power - 1;
    while (i > 1) {
        if (i % 2 != 0) {
            // store one power in acc:
            // i^(2n+1) -> i*i^2n
            id_power->apply(lend(acc), lend(tmp));
            std::swap(acc, tmp);
            i--;
        }
        // square id_power: i^2n -> (i^2)^n
        id_power->apply(lend(id_power), lend(tmp));
        std::swap(id_power, tmp);
        i /= 2;
    }
    // combine acc and id_power again
    id_power->apply(lend(acc), lend(tmp));
    return {std::move(tmp)};
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_EXTEND_SPARSITY);

}  // namespace detail


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
        precision_dispatch_real_complex<ValueType>(
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
                             as<Dense<ValueType>>(beta), lend(x_copy), x_csr));
    } else {
        precision_dispatch_real_complex<ValueType>(
            [this](auto dense_alpha, auto dense_b, auto dense_beta,
                   auto dense_x) {
                this->get_executor()->run(csr::make_advanced_spmv(
                    dense_alpha, this, dense_b, dense_beta, dense_x));
            },
            alpha, b, beta, x);
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
    coo_nnz = exec->copy_val_to_host(coo_row_ptrs.get_const_data() + num_rows);
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
    this->read(device_mat_data::create_from_host(this->get_executor(), data));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::read(const device_mat_data& data)
{
    // make a copy, read the data in
    this->read(device_mat_data{this->get_executor(), data});
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
                                             local_row_idxs->get_num_elems(),
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
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::permute(
    const array<IndexType>* permutation_indices) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());
    array<IndexType> inv_permutation(exec, this->get_size()[1]);

    exec->run(csr::make_invert_permutation(
        this->get_size()[1],
        make_temporary_clone(exec, permutation_indices)->get_const_data(),
        inv_permutation.get_data()));
    exec->run(csr::make_inv_symm_permute(inv_permutation.get_const_data(), this,
                                         permute_cpy.get()));
    permute_cpy->make_srow();
    return std::move(permute_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::inverse_permute(
    const array<IndexType>* permutation_indices) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());

    exec->run(csr::make_inv_symm_permute(
        make_temporary_clone(exec, permutation_indices)->get_const_data(), this,
        permute_cpy.get()));
    permute_cpy->make_srow();
    return std::move(permute_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::row_permute(
    const array<IndexType>* permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());

    exec->run(csr::make_row_permute(
        make_temporary_clone(exec, permutation_indices)->get_const_data(), this,
        permute_cpy.get()));
    permute_cpy->make_srow();
    return std::move(permute_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::column_permute(
    const array<IndexType>* permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    auto exec = this->get_executor();
    auto permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());
    array<IndexType> inv_permutation(exec, this->get_size()[1]);

    exec->run(csr::make_invert_permutation(
        this->get_size()[1],
        make_temporary_clone(exec, permutation_indices)->get_const_data(),
        inv_permutation.get_data()));
    exec->run(csr::make_inverse_column_permute(inv_permutation.get_const_data(),
                                               this, permute_cpy.get()));
    permute_cpy->make_srow();
    permute_cpy->sort_by_column_index();
    return std::move(permute_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::inverse_row_permute(
    const array<IndexType>* permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());

    exec->run(csr::make_inverse_row_permute(
        make_temporary_clone(exec, permutation_indices)->get_const_data(), this,
        inverse_permute_cpy.get()));
    inverse_permute_cpy->make_srow();
    return std::move(inverse_permute_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::inverse_column_permute(
    const array<IndexType>* permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());

    exec->run(csr::make_inverse_column_permute(
        make_temporary_clone(exec, permutation_indices)->get_const_data(), this,
        inverse_permute_cpy.get()));
    inverse_permute_cpy->make_srow();
    inverse_permute_cpy->sort_by_column_index();
    return std::move(inverse_permute_cpy);
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
    exec->run(csr::make_prefix_sum(row_ptrs.get_data(), row_span.length() + 1));
    auto num_nnz =
        exec->copy_val_to_host(row_ptrs.get_data() + sub_mat_size[0]);
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
    if (!row_index_set.get_num_elems() || !col_index_set.get_num_elems()) {
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
        exec->run(
            csr::make_prefix_sum(row_ptrs.get_data(), submat_num_rows + 1));
        auto num_nnz =
            exec->copy_val_to_host(row_ptrs.get_data() + sub_mat_size[0]);
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
    exec->run(csr::make_extract_diagonal(this, lend(diag)));
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


#define GKO_DECLARE_BLOCK_DIAG_CSR_MATRIX(ValueType, IndexType)                \
    std::unique_ptr<matrix::Csr<ValueType, IndexType>>                         \
    create_block_diagonal_matrix(                                              \
        std::shared_ptr<const Executor> exec,                                  \
        const std::vector<std::unique_ptr<matrix::Csr<ValueType, IndexType>>>& \
            matrices)

template <typename ValueType, typename IndexType>
GKO_DECLARE_BLOCK_DIAG_CSR_MATRIX(ValueType, IndexType)
{
    using mtx_type = matrix::Csr<ValueType, IndexType>;
    size_type total_rows = 0, total_nnz = 0;
    for (size_type imat = 0; imat < matrices.size(); imat++) {
        GKO_ASSERT_IS_SQUARE_MATRIX(matrices[imat]);
        total_rows += matrices[imat]->get_size()[0];
        total_nnz += matrices[imat]->get_num_stored_elements();
    }
    array<IndexType> h_row_ptrs(exec->get_master(), total_rows + 1);
    array<IndexType> h_col_idxs(exec->get_master(), total_nnz);
    array<ValueType> h_values(exec->get_master(), total_nnz);
    size_type roffset = 0, nzoffset = 0;
    for (size_type im = 0; im < matrices.size(); im++) {
        auto imatrix = mtx_type::create(exec->get_master());
        imatrix->copy_from(matrices[im].get());
        const auto irowptrs = imatrix->get_const_row_ptrs();
        const auto icolidxs = imatrix->get_const_col_idxs();
        const auto ivalues = imatrix->get_const_values();
        for (size_type irow = 0; irow < imatrix->get_size()[0]; irow++) {
            h_row_ptrs.get_data()[irow + roffset] = irowptrs[irow] + nzoffset;
            for (size_type inz = irowptrs[irow]; inz < irowptrs[irow + 1];
                 inz++) {
                h_col_idxs.get_data()[nzoffset + inz] = icolidxs[inz] + roffset;
                h_values.get_data()[nzoffset + inz] = ivalues[inz];
            }
        }
        roffset += imatrix->get_size()[0];
        nzoffset += irowptrs[imatrix->get_size()[0]];
    }
    h_row_ptrs.get_data()[roffset] = nzoffset;
    assert(nzoffset == total_nnz);
    assert(roffset == total_rows);
    auto outm = mtx_type::create(exec, dim<2>(total_rows, total_rows), h_values,
                                 h_col_idxs, h_row_ptrs);
    return outm;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_DIAG_CSR_MATRIX);


}  // namespace gko
