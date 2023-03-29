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

#include <ginkgo/core/matrix/batch_csr.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/matrix/batch_csr_kernels.hpp"


namespace gko {
namespace matrix {
namespace batch_csr {


GKO_REGISTER_OPERATION(spmv, batch_csr::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, batch_csr::advanced_spmv);
GKO_REGISTER_OPERATION(convert_to_dense, batch_csr::convert_to_dense);
GKO_REGISTER_OPERATION(calculate_total_cols, batch_csr::calculate_total_cols);
GKO_REGISTER_OPERATION(transpose, batch_csr::transpose);
GKO_REGISTER_OPERATION(conj_transpose, batch_csr::conj_transpose);
GKO_REGISTER_OPERATION(calculate_max_nnz_per_row,
                       batch_csr::calculate_max_nnz_per_row);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row,
                       batch_csr::calculate_nonzeros_per_row);
GKO_REGISTER_OPERATION(sort_by_column_index, batch_csr::sort_by_column_index);
GKO_REGISTER_OPERATION(is_sorted_by_column_index,
                       batch_csr::is_sorted_by_column_index);
GKO_REGISTER_OPERATION(convert_to_batch_dense,
                       batch_csr::convert_to_batch_dense);
GKO_REGISTER_OPERATION(check_diagonal_entries_exist,
                       batch_csr::check_diagonal_entries_exist);
GKO_REGISTER_OPERATION(add_scaled_identity, batch_csr::add_scaled_identity);


}  // namespace batch_csr


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::apply_impl(const BatchLinOp* b,
                                                BatchLinOp* x) const
{
    using TBatchCsr = BatchCsr<ValueType, IndexType>;
    if (auto b_csr = dynamic_cast<const TBatchCsr*>(b)) {
        // if b is a BatchCSR matrix, we compute an SpGeMM  (Refer to csr matrix
        // apply_impl)
        auto x_csr = as<TBatchCsr>(x);
        // TODO: Implement SpGemm kernel (Note: x must be of the correct size to
        // pass the apply dim match check; the memory allocation(i.e acc. to
        // nnz) is taken care of in the kernel)
        GKO_NOT_IMPLEMENTED;
        // this->get_executor()->run(batch_csr::make_spgemm(this, b_csr,
        // x_csr));
    } else {
        this->get_executor()->run(batch_csr::make_spmv(
            this, as<BatchDense<ValueType>>(b), as<BatchDense<ValueType>>(x)));
    }
}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::apply_impl(const BatchLinOp* alpha,
                                                const BatchLinOp* b,
                                                const BatchLinOp* beta,
                                                BatchLinOp* x) const
{
    this->get_executor()->run(batch_csr::make_advanced_spmv(
        as<BatchDense<ValueType>>(alpha), this, as<BatchDense<ValueType>>(b),
        as<BatchDense<ValueType>>(beta), as<BatchDense<ValueType>>(x)));
}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::convert_to(
    BatchCsr<next_precision<ValueType>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::move_to(
    BatchCsr<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::convert_to(
    BatchDense<ValueType>* const result) const
{
    auto temp =
        BatchDense<ValueType>::create(this->get_executor(), this->get_size());
    this->get_executor()->run(
        batch_csr::make_convert_to_batch_dense(this, temp.get()));
    temp->move_to(result);
}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::move_to(
    BatchDense<ValueType>* const result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::read(const std::vector<mat_data>& data)
{
    size_type num_batch_entries = data.size();
    std::vector<size_type> nnz(num_batch_entries, 0);
    size_type ind = 0;
    for (const auto& batch_data : data) {
        for (const auto& elem : batch_data.nonzeros) {
            nnz[ind] += (elem.value != zero<ValueType>());
        }
        ++ind;
    }
    GKO_ASSERT(std::equal(nnz.begin() + 1, nnz.end(), nnz.begin()));
    auto tmp = BatchCsr::create(this->get_executor()->get_master(),
                                num_batch_entries, data[0].size, nnz[0]);

    size_type id = 0;
    size_type nnz_offset = 0;
    for (const auto& batch_data : data) {
        ind = 0;
        size_type cur_ptr = 0;
        tmp->get_row_ptrs()[0] = cur_ptr;
        for (size_type row = 0; row < batch_data.size[0]; ++row) {
            for (; ind < batch_data.nonzeros.size(); ++ind) {
                if (batch_data.nonzeros[ind].row > row) {
                    break;
                }
                auto val = batch_data.nonzeros[ind].value;
                if (val != zero<ValueType>()) {
                    tmp->get_values()[nnz_offset + cur_ptr] = val;
                    tmp->get_col_idxs()[cur_ptr] =
                        batch_data.nonzeros[ind].column;
                    ++cur_ptr;
                }
            }
            tmp->get_row_ptrs()[row + 1] = cur_ptr;
        }
        nnz_offset += nnz[id];
        ++id;
    }
    tmp->move_to(this);
}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::write(std::vector<mat_data>& data) const
{
    std::unique_ptr<const BatchLinOp> op{};
    const BatchCsr* tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const BatchCsr*>(op.get());
    } else {
        tmp = this;
    }
    data = std::vector<mat_data>(tmp->get_num_batch_entries());

    size_type num_nnz_per_batch =
        tmp->get_num_stored_elements() / tmp->get_num_batch_entries();
    size_type offset = 0;
    for (size_type batch = 0; batch < tmp->get_num_batch_entries(); ++batch) {
        data[batch] = {tmp->get_size().at(batch), {}};

        for (size_type row = 0; row < tmp->get_size().at(0)[0]; ++row) {
            const auto start = tmp->row_ptrs_.get_const_data()[row];
            const auto end = tmp->row_ptrs_.get_const_data()[row + 1];
            for (auto i = start; i < end; ++i) {
                const auto col = tmp->col_idxs_.get_const_data()[i];
                const auto val = tmp->values_.get_const_data()[offset + i];
                data[batch].nonzeros.emplace_back(row, col, val);
            }
        }
        offset += num_nnz_per_batch;
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchCsr<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchCsr<ValueType, IndexType>::conj_transpose()
    const GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::sort_by_column_index() GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
bool BatchCsr<ValueType, IndexType>::is_sorted_by_column_index() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::add_scaled_identity_impl(
    const BatchLinOp* const a, const BatchLinOp* const b)
{
    bool has_all_diags = false;
    this->get_executor()->run(
        batch_csr::make_check_diagonal_entries_exist(this, has_all_diags));
    if (!has_all_diags) {
        // TODO: Replace this with proper exception helper after merging
        // non-batched add_scaled_identity PR
        throw std::runtime_error("Matrix does not have all diagonal entries!");
    }
    this->get_executor()->run(batch_csr::make_add_scaled_identity(
        as<const BatchDense<ValueType>>(a), as<const BatchDense<ValueType>>(b),
        this));
}


#define GKO_DECLARE_BATCH_CSR_MATRIX(ValueType) class BatchCsr<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CSR_MATRIX);


}  // namespace matrix
}  // namespace gko
