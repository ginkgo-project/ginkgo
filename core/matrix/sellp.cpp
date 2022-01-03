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

#include <ginkgo/core/matrix/sellp.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/device_matrix_data_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/sellp_kernels.hpp"


namespace gko {
namespace matrix {
namespace sellp {
namespace {


GKO_REGISTER_OPERATION(spmv, sellp::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, sellp::advanced_spmv);
GKO_REGISTER_OPERATION(build_row_ptrs, components::build_row_ptrs);
GKO_REGISTER_OPERATION(compute_slice_sets, sellp::compute_slice_sets);
GKO_REGISTER_OPERATION(fill_in_matrix_data, sellp::fill_in_matrix_data);
GKO_REGISTER_OPERATION(convert_to_dense, sellp::convert_to_dense);
GKO_REGISTER_OPERATION(convert_to_csr, sellp::convert_to_csr);
GKO_REGISTER_OPERATION(count_nonzeros, sellp::count_nonzeros);
GKO_REGISTER_OPERATION(extract_diagonal, sellp::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // anonymous namespace
}  // namespace sellp


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
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(sellp::make_convert_to_dense(this, tmp.get()));
    tmp->move_to(result);
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

    size_type num_stored_nonzeros = 0;
    exec->run(sellp::make_count_nonzeros(this, &num_stored_nonzeros));
    auto tmp = Csr<ValueType, IndexType>::create(
        exec, this->get_size(), num_stored_nonzeros, result->get_strategy());
    exec->run(sellp::make_convert_to_csr(this, tmp.get()));
    tmp->make_srow();
    tmp->move_to(result);
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
    if (this->get_size() != data.size) {
        slice_lengths_.resize_and_reset(ceildiv(data.size[0], slice_size_));
        slice_sets_.resize_and_reset(ceildiv(data.size[0], slice_size_) + 1);
        this->set_size(data.size);
    }
    Array<int64> row_ptrs{exec, data.size[0] + 1};
    auto local_data = make_temporary_clone(exec, &data.nonzeros);
    exec->run(sellp::make_build_row_ptrs(*local_data, data.size[0],
                                         row_ptrs.get_data()));
    exec->run(sellp::make_compute_slice_sets(
        row_ptrs, this->get_slice_size(), this->get_stride_factor(),
        slice_sets_.get_data(), slice_lengths_.get_data()));
    const auto total_cols = exec->copy_val_to_host(
        slice_sets_.get_data() + slice_sets_.get_num_elems() - 1);
    values_.resize_and_reset(total_cols * slice_size_);
    col_idxs_.resize_and_reset(total_cols * slice_size_);
    exec->run(sellp::make_fill_in_matrix_data(*local_data,
                                              row_ptrs.get_const_data(), this));
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_view_from_host(
        this->get_executor(), const_cast<mat_data&>(data)));
}


template <typename ValueType, typename IndexType>
void Sellp<ValueType, IndexType>::write(mat_data& data) const
{
    std::unique_ptr<const LinOp> op{};
    const Sellp* tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const Sellp*>(op.get());
    } else {
        tmp = this;
    }

    data = {tmp->get_size(), {}};

    auto slice_size = tmp->get_slice_size();
    size_type slice_num = static_cast<index_type>(
        (tmp->get_size()[0] + slice_size - 1) / slice_size);
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row_in_slice = 0; row_in_slice < slice_size;
             row_in_slice++) {
            auto row = slice * slice_size + row_in_slice;
            if (row < tmp->get_size()[0]) {
                for (size_type i = 0; i < tmp->get_const_slice_lengths()[slice];
                     i++) {
                    const auto val = tmp->val_at(
                        row_in_slice, tmp->get_const_slice_sets()[slice], i);
                    if (val != zero<ValueType>()) {
                        const auto col =
                            tmp->col_at(row_in_slice,
                                        tmp->get_const_slice_sets()[slice], i);
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
    exec->run(sellp::make_extract_diagonal(this, lend(diag)));
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
