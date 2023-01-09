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

#include <ginkgo/core/matrix/batch_dense.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>


#include "core/matrix/batch_dense_kernels.hpp"


namespace gko {
namespace matrix {
namespace batch_dense {


GKO_REGISTER_OPERATION(simple_apply, batch_dense::simple_apply);
GKO_REGISTER_OPERATION(apply, batch_dense::apply);
GKO_REGISTER_OPERATION(scale, batch_dense::scale);
GKO_REGISTER_OPERATION(add_scaled, batch_dense::add_scaled);
GKO_REGISTER_OPERATION(add_scale, batch_dense::add_scale);
GKO_REGISTER_OPERATION(convergence_add_scaled,
                       batch_dense::convergence_add_scaled);
GKO_REGISTER_OPERATION(add_scaled_diag, batch_dense::add_scaled_diag);
GKO_REGISTER_OPERATION(compute_dot, batch_dense::compute_dot);
GKO_REGISTER_OPERATION(convergence_compute_dot,
                       batch_dense::convergence_compute_dot);
GKO_REGISTER_OPERATION(compute_norm2, batch_dense::compute_norm2);
GKO_REGISTER_OPERATION(convergence_compute_norm2,
                       batch_dense::convergence_compute_norm2);
GKO_REGISTER_OPERATION(copy, batch_dense::copy);
GKO_REGISTER_OPERATION(convergence_copy, batch_dense::convergence_copy);
GKO_REGISTER_OPERATION(convert_to_batch_csr, batch_dense::convert_to_batch_csr);
GKO_REGISTER_OPERATION(count_nonzeros, batch_dense::count_nonzeros);
GKO_REGISTER_OPERATION(calculate_max_nnz_per_row,
                       batch_dense::calculate_max_nnz_per_row);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row,
                       batch_dense::calculate_nonzeros_per_row);
GKO_REGISTER_OPERATION(calculate_total_cols, batch_dense::calculate_total_cols);
GKO_REGISTER_OPERATION(transpose, batch_dense::transpose);
GKO_REGISTER_OPERATION(conj_transpose, batch_dense::conj_transpose);
GKO_REGISTER_OPERATION(add_scaled_identity, batch_dense::add_scaled_identity);


}  // namespace batch_dense


template <typename ValueType>
void BatchDense<ValueType>::apply_impl(const BatchLinOp* b, BatchLinOp* x) const
{
    // TODO: Remove this when non-uniform batching kernels have been
    // implemented
    if (!this->get_size().stores_equal_sizes() ||
        !this->get_stride().stores_equal_strides()) {
        GKO_NOT_IMPLEMENTED;
    }
    this->get_executor()->run(batch_dense::make_simple_apply(
        this, as<BatchDense<ValueType>>(b), as<BatchDense<ValueType>>(x)));
}


template <typename ValueType>
void BatchDense<ValueType>::apply_impl(const BatchLinOp* alpha,
                                       const BatchLinOp* b,
                                       const BatchLinOp* beta,
                                       BatchLinOp* x) const
{
    if (!this->get_size().stores_equal_sizes() ||
        !this->get_stride().stores_equal_strides()) {
        GKO_NOT_IMPLEMENTED;
    }
    if (auto bid = dynamic_cast<const BatchIdentity<ValueType>*>(b)) {
        if (auto xdense = dynamic_cast<BatchDense<ValueType>*>(x)) {
            xdense->add_scale(alpha, this, beta);
        } else {
            GKO_NOT_SUPPORTED(x);
        }
    } else {
        this->get_executor()->run(batch_dense::make_apply(
            as<BatchDense<ValueType>>(alpha), this,
            as<BatchDense<ValueType>>(b), as<BatchDense<ValueType>>(beta),
            as<BatchDense<ValueType>>(x)));
    }
}


template <typename ValueType>
void BatchDense<ValueType>::scale_impl(const BatchLinOp* alpha)
{
    auto batch_alpha = as<BatchDense<ValueType>>(alpha);
    GKO_ASSERT_BATCH_EQUAL_ROWS(
        batch_alpha, batch_dim<2>(this->get_num_batch_entries(), dim<2>(1, 1)));
    for (size_type b = 0; b < batch_alpha->get_num_batch_entries(); ++b) {
        if (batch_alpha->get_size().at(b)[1] != 1) {
            // different alpha for each column
            GKO_ASSERT_BATCH_EQUAL_COLS(this, batch_alpha);
        }
    }
    auto exec = this->get_executor();
    exec->run(batch_dense::make_scale(batch_alpha, this));
}


template <typename ValueType>
void BatchDense<ValueType>::add_scaled_impl(const BatchLinOp* alpha,
                                            const BatchLinOp* b)
{
    auto batch_alpha = as<BatchDense<ValueType>>(alpha);
    auto batch_b = as<BatchDense<ValueType>>(b);
    GKO_ASSERT_BATCH_EQUAL_ROWS(
        batch_alpha, batch_dim<2>(this->get_num_batch_entries(), dim<2>(1, 1)));
    for (size_type b = 0; b < batch_alpha->get_num_batch_entries(); ++b) {
        if (batch_alpha->get_size().at(b)[1] != 1) {
            // different alpha for each column
            GKO_ASSERT_BATCH_EQUAL_COLS(this, batch_alpha);
        }
    }
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(this, batch_b);
    auto exec = this->get_executor();

    exec->run(batch_dense::make_add_scaled(batch_alpha, batch_b, this));
}


template <typename ValueType>
void BatchDense<ValueType>::add_scale(const BatchLinOp* const alpha,
                                      const BatchLinOp* const a,
                                      const BatchLinOp* const beta)
{
    auto batch_alpha = as<BatchDense<ValueType>>(alpha);
    auto batch_beta = as<BatchDense<ValueType>>(beta);
    auto batch_a = as<BatchDense<ValueType>>(a);
    GKO_ASSERT_BATCH_EQUAL_ROWS(
        batch_alpha, batch_dim<2>(this->get_num_batch_entries(), dim<2>(1, 1)));
    if (batch_alpha->get_size().stores_equal_sizes()) {
        if (batch_alpha->get_size().at(0)[1] != 1) {
            // different alpha for each column
            GKO_ASSERT_BATCH_EQUAL_COLS(this, batch_alpha);
        }
    } else {
        for (size_type b = 0; b < batch_alpha->get_num_batch_entries(); ++b) {
            if (batch_alpha->get_size().at(b)[1] != 1) {
                GKO_ASSERT(this->get_size().at(b)[1] ==
                           batch_alpha->get_size().at(b)[1]);
            }
        }
    }
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(this, batch_a);
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(batch_alpha, batch_beta);
    this->get_executor()->run(
        batch_dense::make_add_scale(batch_alpha, batch_a, batch_beta, this));
}


inline const batch_dim<2> get_col_sizes(const batch_dim<2>& sizes)
{
    auto col_sizes = std::vector<dim<2>>(sizes.get_num_batch_entries());
    for (size_type i = 0; i < col_sizes.size(); ++i) {
        col_sizes[i] = dim<2>(1, sizes.at(i)[1]);
    }
    return batch_dim<2>(col_sizes);
}


template <typename ValueType>
void BatchDense<ValueType>::compute_dot_impl(const BatchLinOp* b,
                                             BatchLinOp* result) const
{
    auto batch_result = as<BatchDense<ValueType>>(result);
    auto batch_b = as<BatchDense<ValueType>>(b);
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(this, batch_b);
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(batch_result,
                                      get_col_sizes(this->get_size()));
    auto exec = this->get_executor();
    exec->run(batch_dense::make_compute_dot(this, batch_b, batch_result));
}


template <typename ValueType>
void BatchDense<ValueType>::compute_norm2_impl(BatchLinOp* result) const
{
    using NormVector = BatchDense<remove_complex<ValueType>>;
    auto batch_result = as<NormVector>(result);
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(batch_result,
                                      get_col_sizes(this->get_size()));
    auto exec = this->get_executor();
    exec->run(batch_dense::make_compute_norm2(as<BatchDense<ValueType>>(this),
                                              batch_result));
}


template <typename ValueType>
void BatchDense<ValueType>::convert_to(
    BatchDense<next_precision<ValueType>>* result) const
{
    result->values_ = this->values_;
    result->stride_ = this->stride_;
    result->num_elems_per_batch_cumul_ = this->num_elems_per_batch_cumul_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void BatchDense<ValueType>::move_to(
    BatchDense<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void BatchDense<ValueType>::convert_to(BatchCsr<ValueType, int32>* result) const
{
    auto exec = this->get_executor();

    auto batch_size = this->get_size();
    if (!batch_size.stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }

    auto num_stored_nonzeros =
        array<size_type>{exec->get_master(), this->get_num_batch_entries()};

    exec->get_master()->run(
        batch_dense::make_count_nonzeros(this, num_stored_nonzeros.get_data()));
    gko::dim<2> main_size = this->get_size().at(0);
    const size_type num_nnz =
        num_stored_nonzeros.get_data() ? num_stored_nonzeros.get_data()[0] : 0;
    auto tmp = BatchCsr<ValueType, int32>::create(
        exec, this->get_num_batch_entries(), main_size, num_nnz);
    exec->run(batch_dense::make_convert_to_batch_csr(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType>
void BatchDense<ValueType>::move_to(BatchCsr<ValueType, int32>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
void BatchDense<ValueType>::convert_to(
    BatchDiagonal<ValueType>* const result) const
{
    auto exec = this->get_executor();

    auto batch_size = this->get_size();
    if (!batch_size.stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }
    GKO_ASSERT_BATCH_HAS_SINGLE_COLUMN(this);
    if (this->get_stride().at(0) != 1) {
        GKO_NOT_IMPLEMENTED;
    }
    auto temp = BatchDiagonal<ValueType>::create(
        exec, batch_dim<2>{batch_size.get_num_batch_entries(),
                           dim<2>{batch_size.at(0)[0]}});
    exec->copy(this->get_num_stored_elements(), this->get_const_values(),
               temp->get_values());
    result->copy_from(temp.get());
}


template <typename ValueType>
void BatchDense<ValueType>::move_to(BatchDiagonal<ValueType>* const result)
{
    auto exec = this->get_executor();

    auto batch_size = this->get_size();
    if (!batch_size.stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }
    GKO_ASSERT_BATCH_HAS_SINGLE_COLUMN(this);
    if (this->get_stride().at(0) != 1) {
        GKO_NOT_IMPLEMENTED;
    }
    auto temp = BatchDiagonal<ValueType>::create(
        exec,
        batch_dim<2>{batch_size.get_num_batch_entries(),
                     dim<2>{batch_size.at(0)[0]}},
        std::move(this->values_));
    *result = std::move(*temp);
    // set the size of this to 0
    this->set_size(batch_dim<2>());
}


namespace {


template <typename MatrixType, typename MatrixData>
inline void read_impl(MatrixType* mtx, const std::vector<MatrixData>& data)
{
    auto batch_sizes = std::vector<dim<2>>(data.size());
    size_type ind = 0;
    for (const auto& b : data) {
        batch_sizes[ind] = b.size;
        ++ind;
    }
    auto tmp = MatrixType::create(mtx->get_executor()->get_master(),
                                  batch_dim<2>(batch_sizes));
    for (size_type b = 0; b < data.size(); ++b) {
        size_type ind = 0;
        for (size_type row = 0; row < data[b].size[0]; ++row) {
            for (size_type col = 0; col < data[b].size[1]; ++col) {
                if (ind < data[b].nonzeros.size() &&
                    data[b].nonzeros[ind].row == row &&
                    data[b].nonzeros[ind].column == col) {
                    tmp->at(b, row, col) = data[b].nonzeros[ind].value;
                    ++ind;
                } else {
                    tmp->at(b, row, col) =
                        zero<typename MatrixType::value_type>();
                }
            }
        }
    }
    tmp->move_to(mtx);
}


}  // namespace


template <typename ValueType>
void BatchDense<ValueType>::read(const std::vector<mat_data>& data)
{
    read_impl(this, data);
}


template <typename ValueType>
void BatchDense<ValueType>::read(const std::vector<mat_data32>& data)
{
    read_impl(this, data);
}


namespace {


template <typename MatrixType, typename MatrixData>
inline void write_impl(const MatrixType* mtx, std::vector<MatrixData>& data)
{
    std::unique_ptr<const BatchLinOp> op{};
    const MatrixType* tmp{};
    if (mtx->get_executor()->get_master() != mtx->get_executor()) {
        op = mtx->clone(mtx->get_executor()->get_master());
        tmp = static_cast<const MatrixType*>(op.get());
    } else {
        tmp = mtx;
    }

    data = std::vector<MatrixData>(mtx->get_num_batch_entries());
    for (size_type b = 0; b < mtx->get_num_batch_entries(); ++b) {
        data[b] = {mtx->get_size().at(b), {}};
        for (size_type row = 0; row < data[b].size[0]; ++row) {
            for (size_type col = 0; col < data[b].size[1]; ++col) {
                if (tmp->at(b, row, col) !=
                    zero<typename MatrixType::value_type>()) {
                    data[b].nonzeros.emplace_back(row, col,
                                                  tmp->at(b, row, col));
                }
            }
        }
    }
}


}  // namespace


template <typename ValueType>
void BatchDense<ValueType>::write(std::vector<mat_data>& data) const
{
    write_impl(this, data);
}


template <typename ValueType>
void BatchDense<ValueType>::write(std::vector<mat_data32>& data) const
{
    write_impl(this, data);
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchDense<ValueType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = BatchDense::create(exec, gko::transpose(this->get_size()));

    exec->run(batch_dense::make_transpose(this, trans_cpy.get()));

    return std::move(trans_cpy);
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchDense<ValueType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = BatchDense::create(exec, gko::transpose(this->get_size()));

    exec->run(batch_dense::make_conj_transpose(this, trans_cpy.get()));
    return std::move(trans_cpy);
}


template <typename ValueType>
void BatchDense<ValueType>::add_scaled_identity_impl(const BatchLinOp* const a,
                                                     const BatchLinOp* const b)
{
    this->get_executor()->run(batch_dense::make_add_scaled_identity(
        as<BatchDense<ValueType>>(a), as<BatchDense<ValueType>>(b), this));
}


#define GKO_DECLARE_BATCH_DENSE_MATRIX(_type) class BatchDense<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_MATRIX);


}  // namespace matrix


}  // namespace gko
