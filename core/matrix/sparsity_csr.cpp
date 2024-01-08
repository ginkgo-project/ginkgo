// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/array_access.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/sparsity_csr_kernels.hpp"


namespace gko {
namespace matrix {
namespace sparsity_csr {
namespace {


GKO_REGISTER_OPERATION(spmv, sparsity_csr::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, sparsity_csr::advanced_spmv);
GKO_REGISTER_OPERATION(transpose, sparsity_csr::transpose);
GKO_REGISTER_OPERATION(diagonal_element_prefix_sum,
                       sparsity_csr::diagonal_element_prefix_sum);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(fill_in_dense, sparsity_csr::fill_in_dense);
GKO_REGISTER_OPERATION(remove_diagonal_elements,
                       sparsity_csr::remove_diagonal_elements);
GKO_REGISTER_OPERATION(sort_by_column_index,
                       sparsity_csr::sort_by_column_index);
GKO_REGISTER_OPERATION(is_sorted_by_column_index,
                       sparsity_csr::is_sorted_by_column_index);


}  // anonymous namespace
}  // namespace sparsity_csr


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                   LinOp* x) const
{
    mixed_precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(
                sparsity_csr::make_spmv(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                   const LinOp* b,
                                                   const LinOp* beta,
                                                   LinOp* x) const
{
    mixed_precision_dispatch_real_complex<ValueType>(
        [this, alpha, beta](auto dense_b, auto dense_x) {
            auto dense_alpha = make_temporary_conversion<ValueType>(alpha);
            auto dense_beta = make_temporary_conversion<
                typename std::decay_t<decltype(*dense_x)>::value_type>(beta);
            this->get_executor()->run(sparsity_csr::make_advanced_spmv(
                dense_alpha.get(), this, dense_b, dense_beta.get(), dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
SparsityCsr<ValueType, IndexType>& SparsityCsr<ValueType, IndexType>::operator=(
    const SparsityCsr<ValueType, IndexType>& other)
{
    if (&other != this) {
        EnableLinOp<SparsityCsr>::operator=(other);
        value_ = other.value_;
        col_idxs_ = other.col_idxs_;
        row_ptrs_ = other.row_ptrs_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
SparsityCsr<ValueType, IndexType>& SparsityCsr<ValueType, IndexType>::operator=(
    SparsityCsr<ValueType, IndexType>&& other)
{
    if (&other != this) {
        EnableLinOp<SparsityCsr>::operator=(std::move(other));
        value_ = other.value_;
        col_idxs_ = std::move(other.col_idxs_);
        row_ptrs_ = std::move(other.row_ptrs_);
        // restore other invariant
        other.row_ptrs_.resize_and_reset(1);
        other.row_ptrs_.fill(0);
        other.value_.fill(one<ValueType>());
    }
    return *this;
}


template <typename ValueType, typename IndexType>
SparsityCsr<ValueType, IndexType>::SparsityCsr(
    const SparsityCsr<ValueType, IndexType>& other)
    : SparsityCsr{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename IndexType>
SparsityCsr<ValueType, IndexType>::SparsityCsr(
    SparsityCsr<ValueType, IndexType>&& other)
    : SparsityCsr{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType>* result) const
{
    result->row_ptrs_ = this->row_ptrs_;
    result->col_idxs_ = this->col_idxs_;
    result->values_.resize_and_reset(this->get_num_nonzeros());
    result->values_.fill(get_element(this->value_, 0));
    result->set_size(this->get_size());
    result->make_srow();
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::move_to(
    Csr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::convert_to(
    Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp_result = make_temporary_output_clone(exec, result);
    tmp_result->resize(this->get_size());
    tmp_result->fill(zero<ValueType>());
    exec->run(sparsity_csr::make_fill_in_dense(this, tmp_result.get()));
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::read(const device_mat_data& data)
{
    // make a copy, read the data in
    this->read(device_mat_data{this->get_executor(), data});
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::read(device_mat_data&& data)
{
    auto size = data.get_size();
    auto exec = this->get_executor();
    auto arrays = data.empty_out();
    this->row_ptrs_.resize_and_reset(size[0] + 1);
    this->set_size(size);
    this->value_.fill(one<ValueType>());
    this->col_idxs_ = std::move(arrays.col_idxs);
    const auto row_idxs = std::move(arrays.row_idxs);
    auto local_row_idxs = make_temporary_clone(exec, &row_idxs);
    exec->run(sparsity_csr::make_convert_idxs_to_ptrs(
        local_row_idxs->get_const_data(), local_row_idxs->get_size(), size[0],
        this->get_row_ptrs()));
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_from_host(this->get_executor(), data));
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::write(mat_data& data) const
{
    auto tmp = make_temporary_clone(this->get_executor()->get_master(), this);

    data = {tmp->get_size(), {}};

    const auto val = tmp->value_.get_const_data()[0];
    for (size_type row = 0; row < tmp->get_size()[0]; ++row) {
        const auto start = tmp->row_ptrs_.get_const_data()[row];
        const auto end = tmp->row_ptrs_.get_const_data()[row + 1];
        for (auto i = start; i < end; ++i) {
            const auto col = tmp->col_idxs_.get_const_data()[i];
            data.nonzeros.emplace_back(row, col, val);
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> SparsityCsr<ValueType, IndexType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = SparsityCsr::create(exec, gko::transpose(this->get_size()),
                                         this->get_num_nonzeros());

    exec->run(sparsity_csr::make_transpose(this, trans_cpy.get()));
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> SparsityCsr<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<SparsityCsr<ValueType, IndexType>>
SparsityCsr<ValueType, IndexType>::to_adjacency_matrix() const
{
    auto exec = this->get_executor();
    // Adjacency matrix has to be square.
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    const auto num_rows = this->get_size()[0];
    array<IndexType> diag_prefix_sum{exec, num_rows + 1};
    exec->run(sparsity_csr::make_diagonal_element_prefix_sum(
        this, diag_prefix_sum.get_data()));
    const auto num_diagonal_elements =
        static_cast<size_type>(get_element(diag_prefix_sum, num_rows));
    auto adj_mat =
        SparsityCsr::create(exec, this->get_size(),
                            this->get_num_nonzeros() - num_diagonal_elements);

    exec->run(sparsity_csr::make_remove_diagonal_elements(
        this->get_const_row_ptrs(), this->get_const_col_idxs(),
        diag_prefix_sum.get_const_data(), adj_mat.get()));
    return std::move(adj_mat);
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::sort_by_column_index()
{
    auto exec = this->get_executor();
    exec->run(sparsity_csr::make_sort_by_column_index(this));
}


template <typename ValueType, typename IndexType>
bool SparsityCsr<ValueType, IndexType>::is_sorted_by_column_index() const
{
    auto exec = this->get_executor();
    bool is_sorted;
    exec->run(sparsity_csr::make_is_sorted_by_column_index(this, &is_sorted));
    return is_sorted;
}


#define GKO_DECLARE_SPARSITY_MATRIX(ValueType, IndexType) \
    class SparsityCsr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSITY_MATRIX);


}  // namespace matrix
}  // namespace gko
