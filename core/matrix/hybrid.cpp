// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/hybrid.hpp>


#include <algorithm>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/array_access.hpp"
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
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // anonymous namespace
}  // namespace hybrid


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
    this->ell_->convert_to(result->ell_);
    this->coo_->convert_to(result->coo_);
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
        exec->run(hybrid::make_prefix_sum_nonnegative(ell_row_ptrs.get_data(),
                                                      num_rows + 1));
        exec->run(hybrid::make_convert_idxs_to_ptrs(
            this->get_const_coo_row_idxs(), this->get_coo_num_stored_elements(),
            num_rows, coo_row_ptrs.get_data()));
        const auto nnz =
            static_cast<size_type>(get_element(ell_row_ptrs, num_rows) +
                                   get_element(coo_row_ptrs, num_rows));
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
        local_data->get_const_row_idxs(), local_data->get_num_stored_elements(),
        num_rows, row_ptrs.get_data()));
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
    coo_nnz = get_element(coo_row_ptrs, num_rows);
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
    exec->run(hybrid::make_ell_extract_diagonal(this->get_ell(), diag.get()));
    exec->run(hybrid::make_coo_extract_diagonal(this->get_coo(), diag.get()));
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

    abs_hybrid->ell_->move_from(ell_->compute_absolute());
    abs_hybrid->coo_->move_from(coo_->compute_absolute());

    return abs_hybrid;
}


#define GKO_DECLARE_HYBRID_MATRIX(ValueType, IndexType) \
    class Hybrid<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_HYBRID_MATRIX);


}  // namespace matrix
}  // namespace gko
