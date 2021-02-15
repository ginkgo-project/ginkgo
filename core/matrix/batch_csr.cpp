/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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
#include <ginkgo/core/matrix/identity.hpp>


#include "core/components/absolute_array.hpp"
#include "core/components/fill_array.hpp"
#include "core/matrix/batch_csr_kernels.hpp"


namespace gko {
namespace matrix {
namespace batch_csr {


GKO_REGISTER_OPERATION(spmv, batch_csr::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, batch_csr::advanced_spmv);
GKO_REGISTER_OPERATION(spgemm, batch_csr::spgemm);
GKO_REGISTER_OPERATION(advanced_spgemm, batch_csr::advanced_spgemm);
GKO_REGISTER_OPERATION(spgeam, batch_csr::spgeam);
GKO_REGISTER_OPERATION(convert_to_coo, batch_csr::convert_to_coo);
GKO_REGISTER_OPERATION(convert_to_dense, batch_csr::convert_to_dense);
GKO_REGISTER_OPERATION(convert_to_sellp, batch_csr::convert_to_sellp);
GKO_REGISTER_OPERATION(calculate_total_cols, batch_csr::calculate_total_cols);
GKO_REGISTER_OPERATION(convert_to_ell, batch_csr::convert_to_ell);
GKO_REGISTER_OPERATION(convert_to_hybrid, batch_csr::convert_to_hybrid);
GKO_REGISTER_OPERATION(transpose, batch_csr::transpose);
GKO_REGISTER_OPERATION(conj_transpose, batch_csr::conj_transpose);
GKO_REGISTER_OPERATION(inv_symm_permute, batch_csr::inv_symm_permute);
GKO_REGISTER_OPERATION(row_permute, batch_csr::row_permute);
GKO_REGISTER_OPERATION(inverse_row_permute, batch_csr::inverse_row_permute);
GKO_REGISTER_OPERATION(inverse_column_permute,
                       batch_csr::inverse_column_permute);
GKO_REGISTER_OPERATION(invert_permutation, batch_csr::invert_permutation);
GKO_REGISTER_OPERATION(calculate_max_nnz_per_row,
                       batch_csr::calculate_max_nnz_per_row);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row,
                       batch_csr::calculate_nonzeros_per_row);
GKO_REGISTER_OPERATION(sort_by_column_index, batch_csr::sort_by_column_index);
GKO_REGISTER_OPERATION(is_sorted_by_column_index,
                       batch_csr::is_sorted_by_column_index);
GKO_REGISTER_OPERATION(extract_diagonal, batch_csr::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // namespace batch_csr


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    using ComplexDense = Dense<to_complex<ValueType>>;
//    using TBatchCsr = BatchCsr<ValueType, IndexType>;
//    if (auto b_batch_csr = dynamic_cast<const TBatchCsr *>(b)) {
//        // if b is a BATCH_CSR matrix, we compute a SpGeMM
//        auto x_batch_csr = as<TBatchCsr>(x);
//        this->get_executor()->run(batch_csr::make_spgemm(this, b_batch_csr,
//        x_batch_csr));
//    } else {
//        // otherwise we assume that b is dense and compute a SpMV/SpMM
//        if (dynamic_cast<const Dense<ValueType> *>(b)) {
//            this->get_executor()->run(batch_csr::make_spmv(
//                this, as<Dense<ValueType>>(b), as<Dense<ValueType>>(x)));
//        } else {
//            auto dense_b = as<ComplexDense>(b);
//            auto dense_x = as<ComplexDense>(x);
//            this->apply(dense_b->create_real_view().get(),
//                        dense_x->create_real_view().get());
//        }
//    }
//}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::apply_impl(
    const LinOp *alpha, const LinOp *b, const LinOp *beta,
    LinOp *x) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    using ComplexDense = Dense<to_complex<ValueType>>;
//    using RealDense = Dense<remove_complex<ValueType>>;
//    using TBatchCsr = BatchCsr<ValueType, IndexType>;
//    if (auto b_batch_csr = dynamic_cast<const TBatchCsr *>(b)) {
//        // if b is a BATCH_CSR matrix, we compute a SpGeMM
//        auto x_batch_csr = as<TBatchCsr>(x);
//        auto x_copy = x_batch_csr->clone();
//        this->get_executor()->run(batch_csr::make_advanced_spgemm(
//            as<Dense<ValueType>>(alpha), this, b_batch_csr,
//            as<Dense<ValueType>>(beta), x_copy.get(), x_batch_csr));
//    } else if (dynamic_cast<const Identity<ValueType> *>(b)) {
//        // if b is an identity matrix, we compute an SpGEAM
//        auto x_batch_csr = as<TBatchCsr>(x);
//        auto x_copy = x_batch_csr->clone();
//        this->get_executor()->run(
//            batch_csr::make_spgeam(as<Dense<ValueType>>(alpha), this,
//                             as<Dense<ValueType>>(beta), lend(x_copy),
//                             x_batch_csr));
//    } else {
//        // otherwise we assume that b is dense and compute a SpMV/SpMM
//        if (dynamic_cast<const Dense<ValueType> *>(b)) {
//            this->get_executor()->run(batch_csr::make_advanced_spmv(
//                as<Dense<ValueType>>(alpha), this, as<Dense<ValueType>>(b),
//                as<Dense<ValueType>>(beta), as<Dense<ValueType>>(x)));
//        } else {
//            auto dense_b = as<ComplexDense>(b);
//            auto dense_x = as<ComplexDense>(x);
//            auto dense_alpha = as<RealDense>(alpha);
//            auto dense_beta = as<RealDense>(beta);
//            this->apply(dense_alpha, dense_b->create_real_view().get(),
//                        dense_beta, dense_x->create_real_view().get());
//        }
//    }
//}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::convert_to(
    BatchCsr<next_precision<ValueType>, IndexType> *result) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    result->values_ = this->values_;
//    result->col_idxs_ = this->col_idxs_;
//    result->row_ptrs_ = this->row_ptrs_;
//    result->set_size(this->get_size());
//    convert_strategy_helper(result);
//}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::move_to(
    BatchCsr<next_precision<ValueType>, IndexType> *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    this->convert_to(result);
//}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::read(const mat_data &data)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    size_type nnz = 0;
//    for (const auto &elem : data.nonzeros) {
//        nnz += (elem.value != zero<ValueType>());
//    }
//    auto tmp = BatchCsr::create(this->get_executor()->get_master(), data.size,
//    nnz,
//                           this->get_strategy());
//    size_type ind = 0;
//    size_type cur_ptr = 0;
//    tmp->get_row_ptrs()[0] = cur_ptr;
//    for (size_type row = 0; row < data.size[0]; ++row) {
//        for (; ind < data.nonzeros.size(); ++ind) {
//            if (data.nonzeros[ind].row > row) {
//                break;
//            }
//            auto val = data.nonzeros[ind].value;
//            if (val != zero<ValueType>()) {
//                tmp->get_values()[cur_ptr] = val;
//                tmp->get_col_idxs()[cur_ptr] = data.nonzeros[ind].column;
//                ++cur_ptr;
//            }
//        }
//        tmp->get_row_ptrs()[row + 1] = cur_ptr;
//    }
//    tmp->make_srow();
//    tmp->move_to(this);
//}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::write(mat_data &data) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    std::unique_ptr<const LinOp> op{};
//    const BatchCsr *tmp{};
//    if (this->get_executor()->get_master() != this->get_executor()) {
//        op = this->clone(this->get_executor()->get_master());
//        tmp = static_cast<const BatchCsr *>(op.get());
//    } else {
//        tmp = this;
//    }
//
//    data = {tmp->get_size(), {}};
//
//    for (size_type row = 0; row < tmp->get_size()[0]; ++row) {
//        const auto start = tmp->row_ptrs_.get_const_data()[row];
//        const auto end = tmp->row_ptrs_.get_const_data()[row + 1];
//        for (auto i = start; i < end; ++i) {
//            const auto col = tmp->col_idxs_.get_const_data()[i];
//            const auto val = tmp->values_.get_const_data()[i];
//            data.nonzeros.emplace_back(row, col, val);
//        }
//    }
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> BatchCsr<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto exec = this->get_executor();
//    auto trans_cpy =
//        BatchCsr::create(exec, gko::transpose(this->get_size()),
//                    this->get_num_stored_elements(), this->get_strategy());
//
//    exec->run(batch_csr::make_transpose(this, trans_cpy.get()));
//    trans_cpy->make_srow();
//    return std::move(trans_cpy);
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> BatchCsr<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto exec = this->get_executor();
//    auto trans_cpy =
//        BatchCsr::create(exec, gko::transpose(this->get_size()),
//                    this->get_num_stored_elements(), this->get_strategy());
//
//    exec->run(batch_csr::make_conj_transpose(this, trans_cpy.get()));
//    trans_cpy->make_srow();
//    return std::move(trans_cpy);
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> BatchCsr<ValueType, IndexType>::permute(
    const Array<IndexType> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    GKO_ASSERT_IS_SQUARE_MATRIX(this);
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto permute_cpy =
//        BatchCsr::create(exec, this->get_size(),
//        this->get_num_stored_elements(),
//                    this->get_strategy());
//    Array<IndexType> inv_permutation(exec, this->get_size()[1]);
//
//    exec->run(batch_csr::make_invert_permutation(
//        this->get_size()[1],
//        make_temporary_clone(exec, permutation_indices)->get_const_data(),
//        inv_permutation.get_data()));
//    exec->run(batch_csr::make_inv_symm_permute(inv_permutation.get_const_data(),
//    this,
//                                         permute_cpy.get()));
//    permute_cpy->make_srow();
//    return std::move(permute_cpy);
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> BatchCsr<ValueType, IndexType>::inverse_permute(
    const Array<IndexType> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    GKO_ASSERT_IS_SQUARE_MATRIX(this);
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto permute_cpy =
//        BatchCsr::create(exec, this->get_size(),
//        this->get_num_stored_elements(),
//                    this->get_strategy());
//
//    exec->run(batch_csr::make_inv_symm_permute(
//        make_temporary_clone(exec, permutation_indices)->get_const_data(),
//        this, permute_cpy.get()));
//    permute_cpy->make_srow();
//    return std::move(permute_cpy);
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> BatchCsr<ValueType, IndexType>::row_permute(
    const Array<IndexType> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto permute_cpy =
//        BatchCsr::create(exec, this->get_size(),
//        this->get_num_stored_elements(),
//                    this->get_strategy());
//
//    exec->run(batch_csr::make_row_permute(
//        make_temporary_clone(exec, permutation_indices)->get_const_data(),
//        this, permute_cpy.get()));
//    permute_cpy->make_srow();
//    return std::move(permute_cpy);
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> BatchCsr<ValueType, IndexType>::column_permute(
    const Array<IndexType> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
//    auto exec = this->get_executor();
//    auto permute_cpy =
//        BatchCsr::create(exec, this->get_size(),
//        this->get_num_stored_elements(),
//                    this->get_strategy());
//    Array<IndexType> inv_permutation(exec, this->get_size()[1]);
//
//    exec->run(batch_csr::make_invert_permutation(
//        this->get_size()[1],
//        make_temporary_clone(exec, permutation_indices)->get_const_data(),
//        inv_permutation.get_data()));
//    exec->run(batch_csr::make_inverse_column_permute(inv_permutation.get_const_data(),
//                                               this, permute_cpy.get()));
//    permute_cpy->make_srow();
//    permute_cpy->sort_by_column_index();
//    return std::move(permute_cpy);
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> BatchCsr<ValueType, IndexType>::inverse_row_permute(
    const Array<IndexType> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto inverse_permute_cpy =
//        BatchCsr::create(exec, this->get_size(),
//        this->get_num_stored_elements(),
//                    this->get_strategy());
//
//    exec->run(batch_csr::make_inverse_row_permute(
//        make_temporary_clone(exec, permutation_indices)->get_const_data(),
//        this, inverse_permute_cpy.get()));
//    inverse_permute_cpy->make_srow();
//    return std::move(inverse_permute_cpy);
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> BatchCsr<ValueType, IndexType>::inverse_column_permute(
    const Array<IndexType> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
//    auto exec = this->get_executor();
//    auto inverse_permute_cpy =
//        BatchCsr::create(exec, this->get_size(),
//        this->get_num_stored_elements(),
//                    this->get_strategy());
//
//    exec->run(batch_csr::make_inverse_column_permute(
//        make_temporary_clone(exec, permutation_indices)->get_const_data(),
//        this, inverse_permute_cpy.get()));
//    inverse_permute_cpy->make_srow();
//    inverse_permute_cpy->sort_by_column_index();
//    return std::move(inverse_permute_cpy);
//}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::sort_by_column_index() GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto exec = this->get_executor();
//    exec->run(batch_csr::make_sort_by_column_index(this));
//}


template <typename ValueType, typename IndexType>
bool BatchCsr<ValueType, IndexType>::is_sorted_by_column_index() const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto exec = this->get_executor();
//    bool is_sorted;
//    exec->run(batch_csr::make_is_sorted_by_column_index(this, &is_sorted));
//    return is_sorted;
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
BatchCsr<ValueType, IndexType>::extract_diagonal() const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto exec = this->get_executor();
//
//    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
//    auto diag = Diagonal<ValueType>::create(exec, diag_size);
//    exec->run(batch_csr::make_fill_array(diag->get_values(),
//    diag->get_size()[0],
//                                   zero<ValueType>()));
//    exec->run(batch_csr::make_extract_diagonal(this, lend(diag)));
//    return diag;
//}


template <typename ValueType, typename IndexType>
void BatchCsr<ValueType, IndexType>::compute_absolute_inplace()
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto exec = this->get_executor();
//
//    exec->run(batch_csr::make_inplace_absolute_array(
//        this->get_values(), this->get_num_stored_elements()));
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename BatchCsr<ValueType, IndexType>::absolute_type>
BatchCsr<ValueType, IndexType>::compute_absolute() const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto exec = this->get_executor();
//
//    auto abs_batch_csr = absolute_type::create(exec, this->get_size(),
//                                         this->get_num_stored_elements());
//
//    abs_batch_csr->col_idxs_ = col_idxs_;
//    abs_batch_csr->row_ptrs_ = row_ptrs_;
//    exec->run(batch_csr::make_outplace_absolute_array(this->get_const_values(),
//                                                this->get_num_stored_elements(),
//                                                abs_batch_csr->get_values()));
//
//    convert_strategy_helper(abs_batch_csr.get());
//    return abs_batch_csr;
//}


#define GKO_DECLARE_BATCH_CSR_MATRIX(ValueType, IndexType) \
    class BatchCsr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BATCH_CSR_MATRIX);


}  // namespace matrix
}  // namespace gko
