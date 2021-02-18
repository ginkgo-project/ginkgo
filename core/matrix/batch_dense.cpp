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
void BatchDense<ValueType>::apply_impl(const BatchLinOp *b,
                                       BatchLinOp *x) const GKO_NOT_IMPLEMENTED;
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
void BatchDense<ValueType>::apply_impl(const BatchLinOp *alpha,
                                       const BatchLinOp *b,
                                       const BatchLinOp *beta,
                                       BatchLinOp *x) const GKO_NOT_IMPLEMENTED;
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
void BatchDense<ValueType>::scale_impl(const BatchLinOp *alpha)
    GKO_NOT_IMPLEMENTED;
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
void BatchDense<ValueType>::add_scaled_impl(
    const BatchLinOp *alpha, const BatchLinOp *b) GKO_NOT_IMPLEMENTED;
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
    const BatchLinOp *b, BatchLinOp *result) const GKO_NOT_IMPLEMENTED;
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
void BatchDense<ValueType>::compute_norm2_impl(BatchLinOp *result) const
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
void BatchDense<ValueType>::read(std::vector<const mat_data> &data)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    read_impl(this, data);
//}


template <typename ValueType>
void BatchDense<ValueType>::read(std::vector<const mat_data32> &data)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    read_impl(this, data);
//}


namespace {


template <typename MatrixType, typename MatrixData>
inline void write_impl(const MatrixType *mtx,
                       std::vector<MatrixData> &data) GKO_NOT_IMPLEMENTED;
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
void BatchDense<ValueType>::write(std::vector<mat_data> &data) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    write_impl(this, data);
//}


template <typename ValueType>
void BatchDense<ValueType>::write(std::vector<mat_data32> &data) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    write_impl(this, data);
//}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchDense<ValueType>::transpose() const
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
std::unique_ptr<BatchLinOp> BatchDense<ValueType>::conj_transpose() const
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


#define GKO_DECLARE_BATCH_DENSE_MATRIX(_type) class BatchDense<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_MATRIX);


}  // namespace matrix


}  // namespace gko
