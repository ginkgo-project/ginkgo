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

#include <ginkgo/core/matrix/batch_ell.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/matrix/batch_ell_kernels.hpp"


namespace gko {
namespace matrix {
namespace batch_ell {


GKO_REGISTER_OPERATION(spmv, batch_ell::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, batch_ell::advanced_spmv);
GKO_REGISTER_OPERATION(convert_to_dense, batch_ell::convert_to_dense);
GKO_REGISTER_OPERATION(calculate_total_cols, batch_ell::calculate_total_cols);
GKO_REGISTER_OPERATION(transpose, batch_ell::transpose);
GKO_REGISTER_OPERATION(conj_transpose, batch_ell::conj_transpose);
GKO_REGISTER_OPERATION(batch_scale, batch_ell::batch_scale);
GKO_REGISTER_OPERATION(convert_from_batch_csc,
                       batch_ell::convert_from_batch_csc);
GKO_REGISTER_OPERATION(calculate_max_nnz_per_row,
                       batch_ell::calculate_max_nnz_per_row);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row,
                       batch_ell::calculate_nonzeros_per_row);
GKO_REGISTER_OPERATION(sort_by_column_index, batch_ell::sort_by_column_index);
GKO_REGISTER_OPERATION(is_sorted_by_column_index,
                       batch_ell::is_sorted_by_column_index);
GKO_REGISTER_OPERATION(convert_to_batch_dense,
                       batch_ell::convert_to_batch_dense);
GKO_REGISTER_OPERATION(check_diagonal_entries_exist,
                       batch_ell::check_diagonal_entries_exist);
GKO_REGISTER_OPERATION(add_scaled_identity, batch_ell::add_scaled_identity);


}  // namespace batch_ell


namespace {


template <typename ValueType, typename IndexType>
size_type calculate_max_nnz_per_row(
    const matrix_data<ValueType, IndexType>& data)
{
    size_type nnz = 0;
    IndexType current_row = 0;
    size_type num_stored_elements_per_row = 0;
    for (const auto& elem : data.nonzeros) {
        if (elem.row != current_row) {
            current_row = elem.row;
            num_stored_elements_per_row =
                std::max(num_stored_elements_per_row, nnz);
            nnz = 0;
        }
        nnz += (elem.value != zero<ValueType>());
    }
    return std::max(num_stored_elements_per_row, nnz);
}


}  // namespace


template <typename ValueType, typename IndexType>
void BatchEll<ValueType, IndexType>::create_from_batch_csc_impl(
    const gko::array<ValueType>& values, const gko::array<IndexType>& row_idxs,
    const gko::array<IndexType>& col_ptrs)
{
    this->get_executor()->run(batch_ell::make_convert_from_batch_csc(
        this, values, row_idxs, col_ptrs));
}


template <typename ValueType, typename IndexType>
void BatchEll<ValueType, IndexType>::apply_impl(const BatchLinOp* b,
                                                BatchLinOp* x) const
{
    this->get_executor()->run(batch_ell::make_spmv(
        this, as<BatchDense<ValueType>>(b), as<BatchDense<ValueType>>(x)));
}


template <typename ValueType, typename IndexType>
void BatchEll<ValueType, IndexType>::apply_impl(const BatchLinOp* alpha,
                                                const BatchLinOp* b,
                                                const BatchLinOp* beta,
                                                BatchLinOp* x) const
{
    this->get_executor()->run(batch_ell::make_advanced_spmv(
        as<BatchDense<ValueType>>(alpha), this, as<BatchDense<ValueType>>(b),
        as<BatchDense<ValueType>>(beta), as<BatchDense<ValueType>>(x)));
}


template <typename ValueType, typename IndexType>
void BatchEll<ValueType, IndexType>::convert_to(
    BatchEll<next_precision<ValueType>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->set_size(this->get_size());
    result->stride_ = this->get_stride();
    result->num_stored_elems_per_row_ = this->get_num_stored_elements_per_row();
}


template <typename ValueType, typename IndexType>
void BatchEll<ValueType, IndexType>::move_to(
    BatchEll<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void BatchEll<ValueType, IndexType>::convert_to(
    BatchDense<ValueType>* const result) const
{
    auto temp =
        BatchDense<ValueType>::create(this->get_executor(), this->get_size());
    this->get_executor()->run(
        batch_ell::make_convert_to_batch_dense(this, temp.get()));
    temp->move_to(result);
}


template <typename ValueType, typename IndexType>
void BatchEll<ValueType, IndexType>::move_to(
    BatchDense<ValueType>* const result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void BatchEll<ValueType, IndexType>::read(const std::vector<mat_data>& data)
{
    // Get the number of stored elements of every row.
    auto num_stored_elements_per_row = calculate_max_nnz_per_row(data[0]);
    size_type num_batch_entries = data.size();
    std::vector<size_type> nnz(num_batch_entries, 0);
    for (size_type ind = 0; ind < num_batch_entries; ++ind) {
        for (const auto& elem : data[ind].nonzeros) {
            nnz[ind] += (elem.value != zero<ValueType>());
        }
    }
    GKO_ASSERT(std::equal(nnz.begin() + 1, nnz.end(), nnz.begin()));
    auto tmp = BatchEll::create(
        this->get_executor()->get_master(),
        batch_dim<2>{num_batch_entries, data[0].size},
        batch_stride{num_batch_entries, num_stored_elements_per_row});
    // Get values and column indexes.
    for (size_type id = 0; id < num_batch_entries; ++id) {
        const auto& batch_data = data[id];
        size_type ind = 0;
        size_type n = batch_data.nonzeros.size();
        for (size_type row = 0; row < batch_data.size[0]; row++) {
            size_type col = 0;
            while (ind < n && batch_data.nonzeros[ind].row == row) {
                auto val = batch_data.nonzeros[ind].value;
                if (val != zero<ValueType>()) {
                    tmp->val_at(id, row, col) = val;
                    tmp->col_at(row, col) = batch_data.nonzeros[ind].column;
                    col++;
                }
                ind++;
            }
            for (auto i = col; i < num_stored_elements_per_row; i++) {
                tmp->val_at(id, row, i) = zero<ValueType>();
                tmp->col_at(row, i) = 0;
            }
        }
    }

    // Return the matrix
    tmp->move_to(this);
}


template <typename ValueType, typename IndexType>
void BatchEll<ValueType, IndexType>::write(std::vector<mat_data>& data) const
{
    std::unique_ptr<const BatchLinOp> op{};
    const BatchEll* tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const BatchEll*>(op.get());
    } else {
        tmp = this;
    }
    data = std::vector<mat_data>(tmp->get_num_batch_entries());

    for (size_type batch = 0; batch < tmp->get_num_batch_entries(); ++batch) {
        data[batch] = {tmp->get_size().at(batch), {}};

        for (size_type row = 0; row < tmp->get_size().at(0)[0]; ++row) {
            for (size_type i = 0; i < tmp->num_stored_elems_per_row_.at(batch);
                 ++i) {
                const auto val = tmp->val_at(batch, row, i);
                if (val != zero<ValueType>()) {
                    const auto col = tmp->col_at(row, i);
                    data[batch].nonzeros.emplace_back(row, col, val);
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchEll<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchEll<ValueType, IndexType>::conj_transpose()
    const GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void BatchEll<ValueType, IndexType>::add_scaled_identity_impl(
    const BatchLinOp* const a, const BatchLinOp* const b)
{
    bool has_all_diags = false;
    this->get_executor()->run(
        batch_ell::make_check_diagonal_entries_exist(this, has_all_diags));
    if (!has_all_diags) {
        // TODO: Replace this with proper exception helper after merging
        // non-batched add_scaled_identity PR
        throw std::runtime_error("Matrix does not have all diagonal entries!");
    }
    this->get_executor()->run(batch_ell::make_add_scaled_identity(
        as<const BatchDense<ValueType>>(a), as<const BatchDense<ValueType>>(b),
        this));
}


#define GKO_DECLARE_BATCH_ELL_MATRIX(ValueType) class BatchEll<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_ELL_MATRIX);


}  // namespace matrix
}  // namespace gko
