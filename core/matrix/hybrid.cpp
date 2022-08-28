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

#include <ginkgo/core/matrix/hybrid.hpp>


#include <algorithm>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/coo_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "core/matrix/hybrid_kernels.hpp"


namespace gko {
namespace matrix {
namespace hybrid {
namespace {


GKO_REGISTER_OPERATION(compute_row_nnz, hybrid::compute_row_nnz);
GKO_REGISTER_OPERATION(fill_in_matrix_data, hybrid::fill_in_matrix_data);
GKO_REGISTER_OPERATION(ell_fill_in_dense, ell::fill_in_dense);
GKO_REGISTER_OPERATION(coo_fill_in_dense, coo::fill_in_dense);
GKO_REGISTER_OPERATION(ell_extract_diagonal, ell::extract_diagonal);
GKO_REGISTER_OPERATION(coo_extract_diagonal, coo::extract_diagonal);
GKO_REGISTER_OPERATION(ell_count_nonzeros_per_row, ell::count_nonzeros_per_row);
GKO_REGISTER_OPERATION(compute_coo_row_ptrs, hybrid::compute_coo_row_ptrs);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(convert_to_csr, hybrid::convert_to_csr);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(prefix_sum, components::prefix_sum);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // anonymous namespace
}  // namespace hybrid


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::strategy_type::strategy_type()
    : ell_num_stored_elements_per_row_(zero<size_type>()),
      coo_nnz_(zero<size_type>())
{}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::strategy_type::compute_hybrid_config(
    const array<size_type>& row_nnz, size_type* ell_num_stored_elements_per_row,
    size_type* coo_nnz)
{
    array<size_type> ref_row_nnz(row_nnz.get_executor()->get_master(),
                                 row_nnz.get_num_elems());
    ref_row_nnz = row_nnz;
    ell_num_stored_elements_per_row_ =
        this->compute_ell_num_stored_elements_per_row(&ref_row_nnz);
    coo_nnz_ = this->compute_coo_nnz(ref_row_nnz);
    *ell_num_stored_elements_per_row = ell_num_stored_elements_per_row_;
    *coo_nnz = coo_nnz_;
}


template <typename ValueType, typename IndexType>
size_type
Hybrid<ValueType,
       IndexType>::strategy_type::get_ell_num_stored_elements_per_row() const
    noexcept
{
    return ell_num_stored_elements_per_row_;
}


template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::strategy_type::get_coo_nnz() const
    noexcept
{
    return coo_nnz_;
}


template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::strategy_type::compute_coo_nnz(
    const array<size_type>& row_nnz) const
{
    size_type coo_nnz = 0;
    auto row_nnz_val = row_nnz.get_const_data();
    for (size_type i = 0; i < row_nnz.get_num_elems(); i++) {
        if (row_nnz_val[i] > ell_num_stored_elements_per_row_) {
            coo_nnz += row_nnz_val[i] - ell_num_stored_elements_per_row_;
        }
    }
    return coo_nnz;
}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::column_limit::column_limit(size_type num_column)
    : num_columns_(num_column)
{}

template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::column_limit::
    compute_ell_num_stored_elements_per_row(array<size_type>* row_nnz) const
{
    return num_columns_;
}


template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::column_limit::get_num_columns() const
{
    return num_columns_;
}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::imbalance_limit::imbalance_limit(double percent)
    : percent_(percent)
{
    percent_ = std::min(percent_, 1.0);
    percent_ = std::max(percent_, 0.0);
}

template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::imbalance_limit::
    compute_ell_num_stored_elements_per_row(array<size_type>* row_nnz) const
{
    auto row_nnz_val = row_nnz->get_data();
    auto num_rows = row_nnz->get_num_elems();
    if (num_rows == 0) {
        return 0;
    }
    std::sort(row_nnz_val, row_nnz_val + num_rows);
    if (percent_ < 1) {
        auto percent_pos = static_cast<size_type>(num_rows * percent_);
        return row_nnz_val[percent_pos];
    } else {
        return row_nnz_val[num_rows - 1];
    }
}

template <typename ValueType, typename IndexType>
double Hybrid<ValueType, IndexType>::imbalance_limit::get_percentage() const
{
    return percent_;
}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::imbalance_bounded_limit::imbalance_bounded_limit(
    double percent, double ratio)
    : strategy_(imbalance_limit(percent)), ratio_(ratio)
{}

template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::imbalance_bounded_limit::
    compute_ell_num_stored_elements_per_row(array<size_type>* row_nnz) const
{
    auto num_rows = row_nnz->get_num_elems();
    auto ell_cols = strategy_.compute_ell_num_stored_elements_per_row(row_nnz);
    return std::min(ell_cols, static_cast<size_type>(num_rows * ratio_));
}


template <typename ValueType, typename IndexType>
double Hybrid<ValueType, IndexType>::imbalance_bounded_limit::get_percentage()
    const
{
    return strategy_.get_percentage();
}


template <typename ValueType, typename IndexType>
double Hybrid<ValueType, IndexType>::imbalance_bounded_limit::get_ratio() const
{
    return ratio_;
}

template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::minimal_storage_limit::minimal_storage_limit()
    : strategy_(imbalance_limit(static_cast<double>(sizeof(IndexType)) /
                                (sizeof(ValueType) + 2 * sizeof(IndexType))))
{}

template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::minimal_storage_limit::
    compute_ell_num_stored_elements_per_row(array<size_type>* row_nnz) const
{
    return strategy_.compute_ell_num_stored_elements_per_row(row_nnz);
}


template <typename ValueType, typename IndexType>
double Hybrid<ValueType, IndexType>::minimal_storage_limit::get_percentage()
    const
{
    return strategy_.get_percentage();
}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::automatic::automatic()
    : strategy_(imbalance_bounded_limit(1.0 / 3.0, 0.001))
{}


template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::automatic::
    compute_ell_num_stored_elements_per_row(array<size_type>* row_nnz) const
{
    return strategy_.compute_ell_num_stored_elements_per_row(row_nnz);
}


template <typename ValueType, typename IndexType>
ValueType* Hybrid<ValueType, IndexType>::get_ell_values() noexcept
{
    return ell_->get_values();
}


template <typename ValueType, typename IndexType>
const ValueType* Hybrid<ValueType, IndexType>::get_const_ell_values() const
    noexcept
{
    return ell_->get_const_values();
}


template <typename ValueType, typename IndexType>
IndexType* Hybrid<ValueType, IndexType>::get_ell_col_idxs() noexcept
{
    return ell_->get_col_idxs();
}


template <typename ValueType, typename IndexType>
const IndexType* Hybrid<ValueType, IndexType>::get_const_ell_col_idxs() const
    noexcept
{
    return ell_->get_const_col_idxs();
}


template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::get_ell_num_stored_elements_per_row()
    const noexcept
{
    return ell_->get_num_stored_elements_per_row();
}


template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::get_ell_stride() const noexcept
{
    return ell_->get_stride();
}


template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::get_ell_num_stored_elements() const
    noexcept
{
    return ell_->get_num_stored_elements();
}


template <typename ValueType, typename IndexType>
const Ell<ValueType, IndexType>* Hybrid<ValueType, IndexType>::get_ell() const
    noexcept
{
    return ell_.get();
}


template <typename ValueType, typename IndexType>
ValueType* Hybrid<ValueType, IndexType>::get_coo_values() noexcept
{
    return coo_->get_values();
}


template <typename ValueType, typename IndexType>
const ValueType* Hybrid<ValueType, IndexType>::get_const_coo_values() const
    noexcept
{
    return coo_->get_const_values();
}


template <typename ValueType, typename IndexType>
IndexType* Hybrid<ValueType, IndexType>::get_coo_col_idxs() noexcept
{
    return coo_->get_col_idxs();
}


template <typename ValueType, typename IndexType>
const IndexType* Hybrid<ValueType, IndexType>::get_const_coo_col_idxs() const
    noexcept
{
    return coo_->get_const_col_idxs();
}


template <typename ValueType, typename IndexType>
IndexType* Hybrid<ValueType, IndexType>::get_coo_row_idxs() noexcept
{
    return coo_->get_row_idxs();
}


template <typename ValueType, typename IndexType>
const IndexType* Hybrid<ValueType, IndexType>::get_const_coo_row_idxs() const
    noexcept
{
    return coo_->get_const_row_idxs();
}


template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::get_coo_num_stored_elements() const
    noexcept
{
    return coo_->get_num_stored_elements();
}


template <typename ValueType, typename IndexType>
const Coo<ValueType, IndexType>* Hybrid<ValueType, IndexType>::get_coo() const
    noexcept
{
    return coo_.get();
}


template <typename ValueType, typename IndexType>
size_type Hybrid<ValueType, IndexType>::get_num_stored_elements() const noexcept
{
    return coo_->get_num_stored_elements() + ell_->get_num_stored_elements();
}


template <typename ValueType, typename IndexType>
std::shared_ptr<typename Hybrid<ValueType, IndexType>::strategy_type>
Hybrid<ValueType, IndexType>::get_strategy() const noexcept
{
    return strategy_;
}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::Hybrid(std::shared_ptr<const Executor> exec,
                                     std::shared_ptr<strategy_type> strategy)
    : Hybrid(std::move(exec), dim<2>{}, std::move(strategy))
{}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::Hybrid(std::shared_ptr<const Executor> exec,
                                     const dim<2>& size,
                                     std::shared_ptr<strategy_type> strategy)
    : Hybrid(std::move(exec), size, size[1], std::move(strategy))
{}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::Hybrid(std::shared_ptr<const Executor> exec,
                                     const dim<2>& size,
                                     size_type num_stored_elements_per_row,
                                     std::shared_ptr<strategy_type> strategy)
    : Hybrid(std::move(exec), size, num_stored_elements_per_row, size[0], {},
             std::move(strategy))
{}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::Hybrid(std::shared_ptr<const Executor> exec,
                                     const dim<2>& size,
                                     size_type num_stored_elements_per_row,
                                     size_type stride,
                                     std::shared_ptr<strategy_type> strategy)
    : Hybrid(std::move(exec), size, num_stored_elements_per_row, stride, {},
             std::move(strategy))
{}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::Hybrid(std::shared_ptr<const Executor> exec,
                                     const dim<2>& size,
                                     size_type num_stored_elements_per_row,
                                     size_type stride, size_type num_nonzeros,
                                     std::shared_ptr<strategy_type> strategy)
    : EnableLinOp<Hybrid>(exec, size),
      ell_(ell_type::create(exec, size, num_stored_elements_per_row, stride)),
      coo_(coo_type::create(exec, size, num_nonzeros)),
      strategy_(std::move(strategy))
{}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>& Hybrid<ValueType, IndexType>::operator=(
    const Hybrid& other)
{
    if (&other != this) {
        EnableLinOp<Hybrid>::operator=(other);
        auto exec = this->get_executor();
        *coo_ = *other.coo_;
        *ell_ = *other.ell_;
        strategy_ = other.strategy_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>& Hybrid<ValueType, IndexType>::operator=(
    Hybrid&& other)
{
    if (&other != this) {
        EnableLinOp<Hybrid>::operator=(std::move(other));
        auto exec = this->get_executor();
        *coo_ = std::move(*other.coo_);
        *ell_ = std::move(*other.ell_);
        strategy_ = other.strategy_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::Hybrid(const Hybrid& other)
    : Hybrid(other.get_executor())
{
    *this = other;
}


template <typename ValueType, typename IndexType>
Hybrid<ValueType, IndexType>::Hybrid(Hybrid&& other)
    : Hybrid(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            auto ell_mtx = this->get_ell();
            auto coo_mtx = this->get_coo();
            ell_mtx->apply(dense_b, dense_x);
            coo_mtx->apply2(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                              const LinOp* b, const LinOp* beta,
                                              LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto ell_mtx = this->get_ell();
            auto coo_mtx = this->get_coo();
            ell_mtx->apply(dense_alpha, dense_b, dense_beta, dense_x);
            coo_mtx->apply2(dense_alpha, dense_b, dense_x);
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::convert_to(
    Hybrid<next_precision<ValueType>, IndexType>* result) const
{
    this->ell_->convert_to(result->ell_.get());
    this->coo_->convert_to(result->coo_.get());
    // TODO set strategy correctly
    // There is no way to correctly clone the strategy like in
    // Csr::convert_to
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::move_to(
    Hybrid<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    result->resize(this->get_size());
    result->fill(zero<ValueType>());
    auto result_local = make_temporary_clone(exec, result);
    exec->run(
        hybrid::make_ell_fill_in_dense(this->get_ell(), result_local.get()));
    exec->run(
        hybrid::make_coo_fill_in_dense(this->get_coo(), result_local.get()));
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto num_rows = this->get_size()[0];
    {
        auto tmp = make_temporary_clone(exec, result);
        array<IndexType> ell_row_ptrs{exec, num_rows + 1};
        array<IndexType> coo_row_ptrs{exec, num_rows + 1};
        exec->run(hybrid::make_ell_count_nonzeros_per_row(
            this->get_ell(), ell_row_ptrs.get_data()));
        exec->run(
            hybrid::make_prefix_sum(ell_row_ptrs.get_data(), num_rows + 1));
        exec->run(hybrid::make_convert_idxs_to_ptrs(
            this->get_const_coo_row_idxs(), this->get_coo_num_stored_elements(),
            num_rows, coo_row_ptrs.get_data()));
        const auto nnz = static_cast<size_type>(
            exec->copy_val_to_host(ell_row_ptrs.get_const_data() + num_rows) +
            exec->copy_val_to_host(coo_row_ptrs.get_const_data() + num_rows));
        tmp->row_ptrs_.resize_and_reset(num_rows + 1);
        tmp->col_idxs_.resize_and_reset(nnz);
        tmp->values_.resize_and_reset(nnz);
        tmp->set_size(this->get_size());
        exec->run(hybrid::make_convert_to_csr(
            this, ell_row_ptrs.get_const_data(), coo_row_ptrs.get_const_data(),
            tmp.get()));
    }
    result->make_srow();
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::move_to(Csr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::resize(dim<2> new_size,
                                          size_type ell_row_nnz,
                                          size_type coo_nnz)
{
    this->set_size(new_size);
    ell_->resize(new_size, ell_row_nnz);
    coo_->resize(new_size, coo_nnz);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::read(const device_mat_data& data)
{
    auto exec = this->get_executor();
    const auto num_rows = data.get_size()[0];
    const auto num_cols = data.get_size()[1];
    auto local_data = make_temporary_clone(exec, &data);
    array<int64> row_ptrs{exec, num_rows + 1};
    exec->run(hybrid::make_convert_idxs_to_ptrs(
        local_data->get_const_row_idxs(), local_data->get_num_elems(), num_rows,
        row_ptrs.get_data()));
    array<size_type> row_nnz{exec, data.get_size()[0]};
    exec->run(hybrid::make_compute_row_nnz(row_ptrs, row_nnz.get_data()));
    size_type ell_max_nnz{};
    size_type coo_nnz{};
    this->get_strategy()->compute_hybrid_config(row_nnz, &ell_max_nnz,
                                                &coo_nnz);
    if (ell_max_nnz > num_cols) {
        // TODO remove temporary fix after ELL gains true structural zeros
        ell_max_nnz = num_cols;
    }
    array<int64> coo_row_ptrs{exec, num_rows + 1};
    exec->run(hybrid::make_compute_coo_row_ptrs(row_nnz, ell_max_nnz,
                                                coo_row_ptrs.get_data()));
    coo_nnz = exec->copy_val_to_host(coo_row_ptrs.get_const_data() + num_rows);
    this->resize(data.get_size(), ell_max_nnz, coo_nnz);
    exec->run(
        hybrid::make_fill_in_matrix_data(*local_data, row_ptrs.get_const_data(),
                                         coo_row_ptrs.get_const_data(), this));
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::read(device_mat_data&& data)
{
    this->read(data);
    data.empty_out();
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_from_host(this->get_executor(), data));
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::write(mat_data& data) const
{
    std::unique_ptr<const LinOp> op{};
    auto tmp_clone =
        make_temporary_clone(this->get_executor()->get_master(), this);
    auto tmp = tmp_clone.get();
    data = {tmp->get_size(), {}};
    size_type coo_ind = 0;
    auto coo_nnz = tmp->get_coo_num_stored_elements();
    auto coo_vals = tmp->get_const_coo_values();
    auto coo_col_idxs = tmp->get_const_coo_col_idxs();
    auto coo_row_idxs = tmp->get_const_coo_row_idxs();
    for (size_type row = 0; row < tmp->get_size()[0]; ++row) {
        for (size_type i = 0; i < tmp->get_ell_num_stored_elements_per_row();
             ++i) {
            const auto val = tmp->ell_val_at(row, i);
            const auto col = tmp->ell_col_at(row, i);
            if (col != invalid_index<IndexType>()) {
                data.nonzeros.emplace_back(row, col, val);
            }
        }

        while (coo_ind < coo_nnz && coo_row_idxs[coo_ind] == row) {
            data.nonzeros.emplace_back(row, coo_col_idxs[coo_ind],
                                       coo_vals[coo_ind]);
            coo_ind++;
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Hybrid<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(hybrid::make_fill_array(diag->get_values(), diag->get_size()[0],
                                      zero<ValueType>()));
    exec->run(hybrid::make_ell_extract_diagonal(this->get_ell(), lend(diag)));
    exec->run(hybrid::make_coo_extract_diagonal(this->get_coo(), lend(diag)));
    return diag;
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(hybrid::make_inplace_absolute_array(
        this->get_ell_values(), this->get_ell_num_stored_elements()));
    exec->run(hybrid::make_inplace_absolute_array(
        this->get_coo_values(), this->get_coo_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Hybrid<ValueType, IndexType>::absolute_type>
Hybrid<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_hybrid = absolute_type::create(
        exec, this->get_size(), this->get_strategy<absolute_type>());

    abs_hybrid->ell_->copy_from(ell_->compute_absolute());
    abs_hybrid->coo_->copy_from(coo_->compute_absolute());

    return abs_hybrid;
}


#define GKO_DECLARE_HYBRID_MATRIX(ValueType, IndexType) \
    class Hybrid<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_HYBRID_MATRIX);


}  // namespace matrix
}  // namespace gko
