// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/coo.hpp>


#include <algorithm>
#include <numeric>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/coo_kernels.hpp"


namespace gko {
namespace matrix {
namespace coo {
namespace {


GKO_REGISTER_OPERATION(spmv, coo::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, coo::advanced_spmv);
GKO_REGISTER_OPERATION(spmv2, coo::spmv2);
GKO_REGISTER_OPERATION(advanced_spmv2, coo::advanced_spmv2);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(fill_in_dense, coo::fill_in_dense);
GKO_REGISTER_OPERATION(extract_diagonal, coo::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);
GKO_REGISTER_OPERATION(aos_to_soa, components::aos_to_soa);


}  // anonymous namespace
}  // namespace coo


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(coo::make_spmv(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            this->get_executor()->run(coo::make_advanced_spmv(
                dense_alpha, this, dense_b, dense_beta, dense_x));
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply2_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(coo::make_spmv2(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply2_impl(const LinOp* alpha, const LinOp* b,
                                            LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_x) {
            this->get_executor()->run(
                coo::make_advanced_spmv2(dense_alpha, this, dense_b, dense_x));
        },
        alpha, b, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(
    Coo<next_precision<ValueType>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->row_idxs_ = this->row_idxs_;
    result->col_idxs_ = this->col_idxs_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(
    Coo<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    result->set_size(this->get_size());
    result->row_ptrs_.resize_and_reset(this->get_size()[0] + 1);
    result->col_idxs_ = this->col_idxs_;
    result->values_ = this->values_;
    exec->run(coo::make_convert_idxs_to_ptrs(
        this->get_const_row_idxs(), this->get_num_stored_elements(),
        this->get_size()[0],
        make_temporary_clone(exec, &result->row_ptrs_)->get_data()));
    result->make_srow();
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(Csr<ValueType, IndexType>* result)
{
    auto exec = this->get_executor();
    const auto nnz = this->get_num_stored_elements();
    result->set_size(this->get_size());
    result->row_ptrs_.resize_and_reset(this->get_size()[0] + 1);
    result->col_idxs_ = std::move(this->col_idxs_);
    result->values_ = std::move(this->values_);
    exec->run(coo::make_convert_idxs_to_ptrs(
        this->get_const_row_idxs(), nnz, this->get_size()[0],
        make_temporary_clone(exec, &result->row_ptrs_)->get_data()));
    result->make_srow();
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp_result = make_temporary_output_clone(exec, result);
    tmp_result->resize(this->get_size());
    tmp_result->fill(zero<ValueType>());
    exec->run(coo::make_fill_in_dense(this, tmp_result.get()));
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::resize(dim<2> new_size, size_type nnz)
{
    this->set_size(new_size);
    this->row_idxs_.resize_and_reset(nnz);
    this->col_idxs_.resize_and_reset(nnz);
    this->values_.resize_and_reset(nnz);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::read(const mat_data& data)
{
    auto size = data.size;
    auto exec = this->get_executor();
    this->set_size(size);
    this->row_idxs_.resize_and_reset(data.nonzeros.size());
    this->col_idxs_.resize_and_reset(data.nonzeros.size());
    this->values_.resize_and_reset(data.nonzeros.size());
    device_mat_data view{exec, size, this->row_idxs_.as_view(),
                         this->col_idxs_.as_view(), this->values_.as_view()};
    const auto host_data =
        make_array_view(exec->get_master(), data.nonzeros.size(),
                        const_cast<matrix_data_entry<ValueType, IndexType>*>(
                            data.nonzeros.data()));
    exec->run(
        coo::make_aos_to_soa(*make_temporary_clone(exec, &host_data), view));
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::read(const device_mat_data& data)
{
    this->set_size(data.get_size());
    // copy the arrays from device matrix data into the arrays of
    // this. Compared to the read(device_mat_data&&) version, the internal
    // arrays keep their current ownership status
    this->values_ = make_const_array_view(data.get_executor(),
                                          data.get_num_stored_elements(),
                                          data.get_const_values());
    this->col_idxs_ = make_const_array_view(data.get_executor(),
                                            data.get_num_stored_elements(),
                                            data.get_const_col_idxs());
    this->row_idxs_ = make_const_array_view(data.get_executor(),
                                            data.get_num_stored_elements(),
                                            data.get_const_row_idxs());
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::read(device_mat_data&& data)
{
    this->set_size(data.get_size());
    auto arrays = data.empty_out();
    this->values_ = std::move(arrays.values);
    this->col_idxs_ = std::move(arrays.col_idxs);
    this->row_idxs_ = std::move(arrays.row_idxs);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::write(mat_data& data) const
{
    auto tmp = make_temporary_clone(this->get_executor()->get_master(), this);

    data = {this->get_size(), {}};

    for (size_type i = 0; i < tmp->get_num_stored_elements(); ++i) {
        const auto row = tmp->row_idxs_.get_const_data()[i];
        const auto col = tmp->col_idxs_.get_const_data()[i];
        const auto val = tmp->values_.get_const_data()[i];
        data.nonzeros.emplace_back(row, col, val);
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Coo<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(coo::make_fill_array(diag->get_values(), diag->get_size()[0],
                                   zero<ValueType>()));
    exec->run(coo::make_extract_diagonal(this, diag.get()));
    return diag;
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(coo::make_inplace_absolute_array(
        this->get_values(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Coo<ValueType, IndexType>::absolute_type>
Coo<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_coo = absolute_type::create(exec, this->get_size(),
                                         this->get_num_stored_elements());

    abs_coo->col_idxs_ = col_idxs_;
    abs_coo->row_idxs_ = row_idxs_;
    exec->run(coo::make_outplace_absolute_array(this->get_const_values(),
                                                this->get_num_stored_elements(),
                                                abs_coo->get_values()));

    return abs_coo;
}


#define GKO_DECLARE_COO_MATRIX(ValueType, IndexType) \
    class Coo<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_MATRIX);


}  // namespace matrix
}  // namespace gko
