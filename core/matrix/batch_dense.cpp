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

#include <ginkgo/core/matrix/batch_dense.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/batch_dense_kernels.hpp"


namespace gko {
namespace matrix {
namespace batch_dense {


GKO_REGISTER_OPERATION(simple_apply, batch_dense::simple_apply);
GKO_REGISTER_OPERATION(apply, batch_dense::apply);
GKO_REGISTER_OPERATION(scale, batch_dense::scale);
GKO_REGISTER_OPERATION(add_scaled, batch_dense::add_scaled);
GKO_REGISTER_OPERATION(add_scaled_diag, batch_dense::add_scaled_diag);
GKO_REGISTER_OPERATION(compute_dot, batch_dense::compute_dot);
GKO_REGISTER_OPERATION(compute_norm2, batch_dense::compute_norm2);
GKO_REGISTER_OPERATION(count_nonzeros, batch_dense::count_nonzeros);
GKO_REGISTER_OPERATION(calculate_max_nnz_per_row,
                       batch_dense::calculate_max_nnz_per_row);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row,
                       batch_dense::calculate_nonzeros_per_row);
GKO_REGISTER_OPERATION(calculate_total_cols, batch_dense::calculate_total_cols);
GKO_REGISTER_OPERATION(transpose, batch_dense::transpose);
GKO_REGISTER_OPERATION(conj_transpose, batch_dense::conj_transpose);
GKO_REGISTER_OPERATION(symm_permute, batch_dense::symm_permute);
GKO_REGISTER_OPERATION(inv_symm_permute, batch_dense::inv_symm_permute);
GKO_REGISTER_OPERATION(row_gather, batch_dense::row_gather);
GKO_REGISTER_OPERATION(column_permute, batch_dense::column_permute);
GKO_REGISTER_OPERATION(inverse_row_permute, batch_dense::inverse_row_permute);
GKO_REGISTER_OPERATION(inverse_column_permute,
                       batch_dense::inverse_column_permute);
GKO_REGISTER_OPERATION(extract_diagonal, batch_dense::extract_diagonal);
GKO_REGISTER_OPERATION(inplace_absolute_batch_dense,
                       batch_dense::inplace_absolute_batch_dense);
GKO_REGISTER_OPERATION(outplace_absolute_batch_dense,
                       batch_dense::outplace_absolute_batch_dense);
GKO_REGISTER_OPERATION(make_complex, batch_dense::make_complex);
GKO_REGISTER_OPERATION(get_real, batch_dense::get_real);
GKO_REGISTER_OPERATION(get_imag, batch_dense::get_imag);


}  // namespace batch_dense


template <typename ValueType>
void BatchDense<ValueType>::apply_impl(const LinOp *b,
                                       LinOp *x) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    if (dynamic_cast<const BatchDense<ValueType> *>(b)) {
//        this->get_executor()->run(batch_dense::make_simple_apply(
//            this, as<BatchDense<ValueType>>(b),
//            as<BatchDense<ValueType>>(x)));
//    } else {
//        auto batch_dense_b = as<BatchDense<to_complex<ValueType>>>(b);
//        auto batch_dense_x = as<BatchDense<to_complex<ValueType>>>(x);
//        this->apply(batch_dense_b->create_real_view().get(),
//                    batch_dense_x->create_real_view().get());
//    }
//}


template <typename ValueType>
void BatchDense<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                       const LinOp *beta,
                                       LinOp *x) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    if (dynamic_cast<const BatchDense<ValueType> *>(b)) {
//        this->get_executor()->run(batch_dense::make_apply(
//            as<BatchDense<ValueType>>(alpha), this,
//            as<BatchDense<ValueType>>(b), as<BatchDense<ValueType>>(beta),
//            as<BatchDense<ValueType>>(x)));
//    } else {
//        auto batch_dense_b = as<BatchDense<to_complex<ValueType>>>(b);
//        auto batch_dense_x = as<BatchDense<to_complex<ValueType>>>(x);
//        auto batch_dense_alpha =
//        as<BatchDense<remove_complex<ValueType>>>(alpha); auto
//        batch_dense_beta = as<BatchDense<remove_complex<ValueType>>>(beta);
//        this->apply(batch_dense_alpha,
//        batch_dense_b->create_real_view().get(), batch_dense_beta,
//                    batch_dense_x->create_real_view().get());
//    }
//}


template <typename ValueType>
void BatchDense<ValueType>::scale_impl(const LinOp *alpha) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
//    if (alpha->get_size()[1] != 1) {
//        // different alpha for each column
//        GKO_ASSERT_EQUAL_COLS(this, alpha);
//    }
//    auto exec = this->get_executor();
//    exec->run(batch_dense::make_scale(as<BatchDense<ValueType>>(alpha),
//    this));
//}


template <typename ValueType>
void BatchDense<ValueType>::add_scaled_impl(const LinOp *alpha,
                                            const LinOp *b) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
//    if (alpha->get_size()[1] != 1) {
//        // different alpha for each column
//        GKO_ASSERT_EQUAL_COLS(this, alpha);
//    }
//    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
//    auto exec = this->get_executor();
//
//    if (dynamic_cast<const Diagonal<ValueType> *>(b)) {
//        exec->run(batch_dense::make_add_scaled_diag(
//            as<BatchDense<ValueType>>(alpha),
//            dynamic_cast<const Diagonal<ValueType> *>(b), this));
//        return;
//    }
//
//    exec->run(batch_dense::make_add_scaled(as<BatchDense<ValueType>>(alpha),
//                                     as<BatchDense<ValueType>>(b), this));
//}


template <typename ValueType>
void BatchDense<ValueType>::compute_dot_impl(
    const LinOp *b, LinOp *result) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
//    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
//    auto exec = this->get_executor();
//    exec->run(batch_dense::make_compute_dot(this,
//    as<BatchDense<ValueType>>(b),
//                                      as<BatchDense<ValueType>>(result)));
//}


template <typename ValueType>
void BatchDense<ValueType>::compute_norm2_impl(LinOp *result) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    using NormVector = BatchDense<remove_complex<ValueType>>;
//    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
//    auto exec = this->get_executor();
//    exec->run(batch_dense::make_compute_norm2(as<BatchDense<ValueType>>(this),
//                                        as<NormVector>(result)));
//}


template <typename ValueType>
void BatchDense<ValueType>::convert_to(
    BatchDense<next_precision<ValueType>> *result) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    result->values_ = this->values_;
//    result->stride_ = this->stride_;
//    result->set_size(this->get_size());
//}


template <typename ValueType>
void BatchDense<ValueType>::move_to(
    BatchDense<next_precision<ValueType>> *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    this->convert_to(result);
//}


namespace {


template <typename MatrixType, typename MatrixData>
inline void read_impl(MatrixType *mtx,
                      const MatrixData &data) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto tmp = MatrixType::create(mtx->get_executor()->get_master(),
//    data.size); size_type ind = 0; for (size_type row = 0; row < data.size[0];
//    ++row) {
//        for (size_type col = 0; col < data.size[1]; ++col) {
//            if (ind < data.nonzeros.size() && data.nonzeros[ind].row == row &&
//                data.nonzeros[ind].column == col) {
//                tmp->at(row, col) = data.nonzeros[ind].value;
//                ++ind;
//            } else {
//                tmp->at(row, col) = zero<typename MatrixType::value_type>();
//            }
//        }
//    }
//    tmp->move_to(mtx);
//}


}  // namespace


template <typename ValueType>
void BatchDense<ValueType>::read(const mat_data &data) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    read_impl(this, data);
//}


template <typename ValueType>
void BatchDense<ValueType>::read(const mat_data32 &data) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    read_impl(this, data);
//}


namespace {


template <typename MatrixType, typename MatrixData>
inline void write_impl(const MatrixType *mtx,
                       MatrixData &data) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    std::unique_ptr<const LinOp> op{};
//    const MatrixType *tmp{};
//    if (mtx->get_executor()->get_master() != mtx->get_executor()) {
//        op = mtx->clone(mtx->get_executor()->get_master());
//        tmp = static_cast<const MatrixType *>(op.get());
//    } else {
//        tmp = mtx;
//    }
//
//    data = {mtx->get_size(), {}};
//
//    for (size_type row = 0; row < data.size[0]; ++row) {
//        for (size_type col = 0; col < data.size[1]; ++col) {
//            if (tmp->at(row, col) != zero<typename MatrixType::value_type>())
//            {
//                data.nonzeros.emplace_back(row, col, tmp->at(row, col));
//            }
//        }
//    }
//}


}  // namespace


template <typename ValueType>
void BatchDense<ValueType>::write(mat_data &data) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    write_impl(this, data);
//}


template <typename ValueType>
void BatchDense<ValueType>::write(mat_data32 &data) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    write_impl(this, data);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::transpose() const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//    auto trans_cpy = BatchDense::create(exec,
//    gko::transpose(this->get_size()));
//
//    exec->run(batch_dense::make_transpose(this, trans_cpy.get()));
//
//    return std::move(trans_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//    auto trans_cpy = BatchDense::create(exec,
//    gko::transpose(this->get_size()));
//
//    exec->run(batch_dense::make_conj_transpose(this, trans_cpy.get()));
//    return std::move(trans_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::permute(
    const Array<int32> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_IS_SQUARE_MATRIX(this);
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_symm_permute(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        permute_cpy.get()));
//
//    return std::move(permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::permute(
    const Array<int64> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_IS_SQUARE_MATRIX(this);
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_symm_permute(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        permute_cpy.get()));
//
//    return std::move(permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::inverse_permute(
    const Array<int32> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_IS_SQUARE_MATRIX(this);
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_inv_symm_permute(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        permute_cpy.get()));
//
//    return std::move(permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::inverse_permute(
    const Array<int64> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_IS_SQUARE_MATRIX(this);
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_inv_symm_permute(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        permute_cpy.get()));
//
//    return std::move(permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::row_permute(
    const Array<int32> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_row_gather(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        permute_cpy.get()));
//
//    return std::move(permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::row_permute(
    const Array<int64> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_row_gather(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        permute_cpy.get()));
//
//    return std::move(permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<BatchDense<ValueType>> BatchDense<ValueType>::row_gather(
    const Array<int32> *row_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//    dim<2> out_dim{row_indices->get_num_elems(), this->get_size()[1]};
//    auto row_gathered = BatchDense::create(exec, out_dim);
//
//    exec->run(
//        batch_dense::make_row_gather(make_temporary_clone(exec,
//        row_indices).get(),
//                               this, row_gathered.get()));
//    return row_gathered;
//}


template <typename ValueType>
std::unique_ptr<BatchDense<ValueType>> BatchDense<ValueType>::row_gather(
    const Array<int64> *row_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//    dim<2> out_dim{row_indices->get_num_elems(), this->get_size()[1]};
//    auto row_gathered = BatchDense::create(exec, out_dim);
//
//    exec->run(
//        batch_dense::make_row_gather(make_temporary_clone(exec,
//        row_indices).get(),
//                               this, row_gathered.get()));
//    return row_gathered;
//}


template <typename ValueType>
void BatchDense<ValueType>::row_gather(
    const Array<int32> *row_indices,
    BatchDense<ValueType> *row_gathered) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//    dim<2> expected_dim{row_indices->get_num_elems(), this->get_size()[1]};
//    GKO_ASSERT_EQUAL_DIMENSIONS(expected_dim, row_gathered);
//
//    exec->run(batch_dense::make_row_gather(
//        make_temporary_clone(exec, row_indices).get(), this,
//        make_temporary_clone(exec, row_gathered).get()));
//}


template <typename ValueType>
void BatchDense<ValueType>::row_gather(
    const Array<int64> *row_indices,
    BatchDense<ValueType> *row_gathered) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    dim<2> expected_dim{row_indices->get_num_elems(), this->get_size()[1]};
//    GKO_ASSERT_EQUAL_DIMENSIONS(expected_dim, row_gathered);
//
//    auto exec = this->get_executor();
//
//    this->get_executor()->run(batch_dense::make_row_gather(
//        make_temporary_clone(exec, row_indices).get(), this,
//        make_temporary_clone(exec, row_gathered).get()));
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::column_permute(
    const Array<int32> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
//    auto exec = this->get_executor();
//    auto permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_column_permute(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        permute_cpy.get()));
//
//    return std::move(permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::column_permute(
    const Array<int64> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
//    auto exec = this->get_executor();
//    auto permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_column_permute(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        permute_cpy.get()));
//
//    return std::move(permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::inverse_row_permute(
    const Array<int32> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto inverse_permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_inverse_row_permute(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        inverse_permute_cpy.get()));
//
//    return std::move(inverse_permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::inverse_row_permute(
    const Array<int64> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
//    auto exec = this->get_executor();
//    auto inverse_permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_inverse_row_permute(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        inverse_permute_cpy.get()));
//
//    return std::move(inverse_permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::inverse_column_permute(
    const Array<int32> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
//    auto exec = this->get_executor();
//    auto inverse_permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_inverse_column_permute(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        inverse_permute_cpy.get()));
//
//    return std::move(inverse_permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<LinOp> BatchDense<ValueType>::inverse_column_permute(
    const Array<int64> *permutation_indices) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
//    auto exec = this->get_executor();
//    auto inverse_permute_cpy = BatchDense::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_inverse_column_permute(
//        make_temporary_clone(exec, permutation_indices).get(), this,
//        inverse_permute_cpy.get()));
//
//    return std::move(inverse_permute_cpy);
//}


template <typename ValueType>
std::unique_ptr<Diagonal<ValueType>> BatchDense<ValueType>::extract_diagonal()
    const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//
//    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
//    auto diag = Diagonal<ValueType>::create(exec, diag_size);
//    exec->run(batch_dense::make_extract_diagonal(this, lend(diag)));
//    return diag;
//}


template <typename ValueType>
void BatchDense<ValueType>::compute_absolute_inplace() GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//
//    exec->run(batch_dense::make_inplace_absolute_batch_dense(this));
//}


template <typename ValueType>
std::unique_ptr<typename BatchDense<ValueType>::absolute_type>
BatchDense<ValueType>::compute_absolute() const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//
//    // do not inherit the stride
//    auto abs_batch_dense = absolute_type::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_outplace_absolute_batch_dense(this,
//    abs_batch_dense.get()));
//
//    return abs_batch_dense;
//}


template <typename ValueType>
std::unique_ptr<typename BatchDense<ValueType>::complex_type>
BatchDense<ValueType>::make_complex() const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//
//    auto complex_batch_dense = complex_type::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_make_complex(this,
//    complex_batch_dense.get()));
//
//    return complex_batch_dense;
//}


template <typename ValueType>
void BatchDense<ValueType>::make_complex(
    BatchDense<to_complex<ValueType>> *result) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//
//    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
//
//    exec->run(batch_dense::make_make_complex(
//        this, make_temporary_clone(exec, result).get()));
//}


template <typename ValueType>
std::unique_ptr<typename BatchDense<ValueType>::absolute_type>
BatchDense<ValueType>::get_real() const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//
//    auto real_batch_dense = absolute_type::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_get_real(this, real_batch_dense.get()));
//
//    return real_batch_dense;
//}


template <typename ValueType>
void BatchDense<ValueType>::get_real(
    BatchDense<remove_complex<ValueType>> *result) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//
//    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
//
//    exec->run(
//        batch_dense::make_get_real(this, make_temporary_clone(exec,
//        result).get()));
//}


template <typename ValueType>
std::unique_ptr<typename BatchDense<ValueType>::absolute_type>
BatchDense<ValueType>::get_imag() const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//
//    auto imag_batch_dense = absolute_type::create(exec, this->get_size());
//
//    exec->run(batch_dense::make_get_imag(this, imag_batch_dense.get()));
//
//    return imag_batch_dense;
//}


template <typename ValueType>
void BatchDense<ValueType>::get_imag(
    BatchDense<remove_complex<ValueType>> *result) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto exec = this->get_executor();
//
//    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);
//
//    exec->run(
//        batch_dense::make_get_imag(this, make_temporary_clone(exec,
//        result).get()));
//}


#define GKO_DECLARE_BATCH_DENSE_MATRIX(_type) class BatchDense<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_MATRIX);


}  // namespace matrix


}  // namespace gko
