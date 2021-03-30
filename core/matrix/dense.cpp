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

#include <ginkgo/core/matrix/dense.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace matrix {
namespace dense {


GKO_REGISTER_OPERATION(simple_apply, dense::simple_apply);
GKO_REGISTER_OPERATION(apply, dense::apply);
GKO_REGISTER_OPERATION(fill, dense::fill);
GKO_REGISTER_OPERATION(scale, dense::scale);
GKO_REGISTER_OPERATION(add_scaled, dense::add_scaled);
GKO_REGISTER_OPERATION(add_scaled_diag, dense::add_scaled_diag);
GKO_REGISTER_OPERATION(compute_dot, dense::compute_dot);
GKO_REGISTER_OPERATION(compute_norm2, dense::compute_norm2);
GKO_REGISTER_OPERATION(count_nonzeros, dense::count_nonzeros);
GKO_REGISTER_OPERATION(calculate_max_nnz_per_row,
                       dense::calculate_max_nnz_per_row);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row,
                       dense::calculate_nonzeros_per_row);
GKO_REGISTER_OPERATION(calculate_total_cols, dense::calculate_total_cols);
GKO_REGISTER_OPERATION(transpose, dense::transpose);
GKO_REGISTER_OPERATION(conj_transpose, dense::conj_transpose);
GKO_REGISTER_OPERATION(symm_permute, dense::symm_permute);
GKO_REGISTER_OPERATION(inv_symm_permute, dense::inv_symm_permute);
GKO_REGISTER_OPERATION(row_gather, dense::row_gather);
GKO_REGISTER_OPERATION(column_permute, dense::column_permute);
GKO_REGISTER_OPERATION(inverse_row_permute, dense::inverse_row_permute);
GKO_REGISTER_OPERATION(inverse_column_permute, dense::inverse_column_permute);
GKO_REGISTER_OPERATION(convert_to_coo, dense::convert_to_coo);
GKO_REGISTER_OPERATION(convert_to_csr, dense::convert_to_csr);
GKO_REGISTER_OPERATION(convert_to_ell, dense::convert_to_ell);
GKO_REGISTER_OPERATION(convert_to_hybrid, dense::convert_to_hybrid);
GKO_REGISTER_OPERATION(convert_to_sellp, dense::convert_to_sellp);
GKO_REGISTER_OPERATION(convert_to_sparsity_csr, dense::convert_to_sparsity_csr);
GKO_REGISTER_OPERATION(extract_diagonal, dense::extract_diagonal);
GKO_REGISTER_OPERATION(inplace_absolute_dense, dense::inplace_absolute_dense);
GKO_REGISTER_OPERATION(outplace_absolute_dense, dense::outplace_absolute_dense);
GKO_REGISTER_OPERATION(make_complex, dense::make_complex);
GKO_REGISTER_OPERATION(get_real, dense::get_real);
GKO_REGISTER_OPERATION(get_imag, dense::get_imag);


}  // namespace dense


namespace {


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Coo<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();

    size_type num_stored_nonzeros = 0;
    exec->run(dense::make_count_nonzeros(source, &num_stored_nonzeros));
    auto tmp = Coo<ValueType, IndexType>::create(exec, source->get_size(),
                                                 num_stored_nonzeros);
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Csr<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();

    size_type num_stored_nonzeros = 0;
    exec->run(dense::make_count_nonzeros(source, &num_stored_nonzeros));
    auto tmp = Csr<ValueType, IndexType>::create(
        exec, source->get_size(), num_stored_nonzeros, result->get_strategy());
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Ell<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();
    size_type num_stored_elements_per_row = 0;
    exec->run(dense::make_calculate_max_nnz_per_row(
        source, &num_stored_elements_per_row));
    const auto max_nnz_per_row = std::max(
        result->get_num_stored_elements_per_row(), num_stored_elements_per_row);
    const auto stride = std::max(result->get_stride(), source->get_size()[0]);
    auto tmp = Ell<ValueType, IndexType>::create(exec, source->get_size(),
                                                 max_nnz_per_row, stride);
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Hybrid<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();
    Array<size_type> row_nnz(exec, source->get_size()[0]);
    exec->run(dense::make_calculate_nonzeros_per_row(source, &row_nnz));
    size_type ell_lim = zero<size_type>();
    size_type coo_lim = zero<size_type>();
    result->get_strategy()->compute_hybrid_config(row_nnz, &ell_lim, &coo_lim);
    const auto max_nnz_per_row =
        std::max(result->get_ell_num_stored_elements_per_row(), ell_lim);
    const auto stride =
        std::max(result->get_ell_stride(), source->get_size()[0]);
    const auto coo_nnz =
        std::max(result->get_coo_num_stored_elements(), coo_lim);
    auto tmp = Hybrid<ValueType, IndexType>::create(
        exec, source->get_size(), max_nnz_per_row, stride, coo_nnz,
        result->get_strategy());
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Sellp<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();
    const auto stride_factor = (result->get_stride_factor() == 0)
                                   ? default_stride_factor
                                   : result->get_stride_factor();
    const auto slice_size = (result->get_slice_size() == 0)
                                ? default_slice_size
                                : result->get_slice_size();
    size_type total_cols = 0;
    exec->run(dense::make_calculate_total_cols(source, &total_cols,
                                               stride_factor, slice_size));
    auto tmp = Sellp<ValueType, IndexType>::create(
        exec, source->get_size(), slice_size, stride_factor, total_cols);
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(SparsityCsr<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();

    size_type num_stored_nonzeros = 0;
    exec->run(dense::make_count_nonzeros(source, &num_stored_nonzeros));
    auto tmp = SparsityCsr<ValueType, IndexType>::create(
        exec, source->get_size(), num_stored_nonzeros);
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


}  // namespace


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(
                dense::make_simple_apply(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                  const LinOp *beta, LinOp *x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            this->get_executor()->run(dense::make_apply(
                dense_alpha, this, dense_b, dense_beta, dense_x));
        },
        alpha, b, beta, x);
}


template <typename ValueType>
void Dense<ValueType>::fill(const ValueType value)
{
    this->get_executor()->run(dense::make_fill(this, value));
}


template <typename ValueType>
void Dense<ValueType>::scale_impl(const LinOp *alpha)
{
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    auto exec = this->get_executor();
    exec->run(dense::make_scale(
        make_temporary_conversion<ValueType>(alpha).get(), this));
}


template <typename ValueType>
void Dense<ValueType>::add_scaled_impl(const LinOp *alpha, const LinOp *b)
{
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    auto exec = this->get_executor();

    if (dynamic_cast<const Diagonal<ValueType> *>(b)) {
        exec->run(dense::make_add_scaled_diag(
            as<Dense<ValueType>>(alpha),
            dynamic_cast<const Diagonal<ValueType> *>(b), this));
    } else {
        exec->run(dense::make_add_scaled(
            make_temporary_conversion<ValueType>(alpha).get(),
            make_temporary_conversion<ValueType>(b).get(), this));
    }
}


template <typename ValueType>
void Dense<ValueType>::compute_dot_impl(const LinOp *b, LinOp *result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    exec->run(dense::make_compute_dot(
        this, make_temporary_conversion<ValueType>(b).get(),
        make_temporary_conversion<ValueType>(result).get()));
}


template <typename ValueType>
void Dense<ValueType>::compute_norm2_impl(LinOp *result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    exec->run(dense::make_compute_norm2(
        this,
        make_temporary_conversion<remove_complex<ValueType>>(result).get()));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(
    Dense<next_precision<ValueType>> *result) const
{
    result->values_ = this->values_;
    result->stride_ = this->stride_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense<next_precision<ValueType>> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Coo<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_coo<const Dense<ValueType> *&,
                                            decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Coo<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Coo<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_coo<const Dense<ValueType> *&,
                                            decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Coo<ValueType, int64> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_csr<const Dense<ValueType> *&,
                                            decltype(result)>);
    result->make_srow();
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_csr<const Dense<ValueType> *&,
                                            decltype(result)>);
    result->make_srow();
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int64> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Ell<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_ell<const Dense<ValueType> *&,
                                            decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Ell<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Ell<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_ell<const Dense<ValueType> *&,
                                            decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Ell<ValueType, int64> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Hybrid<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_hybrid<const Dense<ValueType> *&,
                                               decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Hybrid<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Hybrid<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_hybrid<const Dense<ValueType> *&,
                                               decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Hybrid<ValueType, int64> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Sellp<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_sellp<const Dense<ValueType> *&,
                                              decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Sellp<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Sellp<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_sellp<const Dense<ValueType> *&,
                                              decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Sellp<ValueType, int64> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(SparsityCsr<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_sparsity_csr<const Dense<ValueType> *&,
                                                     decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(SparsityCsr<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(SparsityCsr<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_sparsity_csr<const Dense<ValueType> *&,
                                                     decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(SparsityCsr<ValueType, int64> *result)
{
    this->convert_to(result);
}


namespace {


template <typename MatrixType, typename MatrixData>
inline void read_impl(MatrixType *mtx, const MatrixData &data)
{
    auto tmp = MatrixType::create(mtx->get_executor()->get_master(), data.size);
    size_type ind = 0;
    for (size_type row = 0; row < data.size[0]; ++row) {
        for (size_type col = 0; col < data.size[1]; ++col) {
            if (ind < data.nonzeros.size() && data.nonzeros[ind].row == row &&
                data.nonzeros[ind].column == col) {
                tmp->at(row, col) = data.nonzeros[ind].value;
                ++ind;
            } else {
                tmp->at(row, col) = zero<typename MatrixType::value_type>();
            }
        }
    }
    tmp->move_to(mtx);
}


}  // namespace


template <typename ValueType>
void Dense<ValueType>::read(const mat_data &data)
{
    read_impl(this, data);
}


template <typename ValueType>
void Dense<ValueType>::read(const mat_data32 &data)
{
    read_impl(this, data);
}


namespace {


template <typename MatrixType, typename MatrixData>
inline void write_impl(const MatrixType *mtx, MatrixData &data)
{
    std::unique_ptr<const LinOp> op{};
    const MatrixType *tmp{};
    if (mtx->get_executor()->get_master() != mtx->get_executor()) {
        op = mtx->clone(mtx->get_executor()->get_master());
        tmp = static_cast<const MatrixType *>(op.get());
    } else {
        tmp = mtx;
    }

    data = {mtx->get_size(), {}};

    for (size_type row = 0; row < data.size[0]; ++row) {
        for (size_type col = 0; col < data.size[1]; ++col) {
            if (tmp->at(row, col) != zero<typename MatrixType::value_type>()) {
                data.nonzeros.emplace_back(row, col, tmp->at(row, col));
            }
        }
    }
}


}  // namespace


template <typename ValueType>
void Dense<ValueType>::write(mat_data &data) const
{
    write_impl(this, data);
}


template <typename ValueType>
void Dense<ValueType>::write(mat_data32 &data) const
{
    write_impl(this, data);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Dense::create(exec, gko::transpose(this->get_size()));

    exec->run(dense::make_transpose(this, trans_cpy.get()));

    return std::move(trans_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Dense::create(exec, gko::transpose(this->get_size()));

    exec->run(dense::make_conj_transpose(this, trans_cpy.get()));
    return std::move(trans_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::permute(
    const Array<int32> *permutation_indices) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_symm_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::permute(
    const Array<int64> *permutation_indices) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_symm_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_permute(
    const Array<int32> *permutation_indices) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_inv_symm_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_permute(
    const Array<int64> *permutation_indices) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_inv_symm_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::row_permute(
    const Array<int32> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_row_gather(
        make_temporary_clone(exec, permutation_indices).get(), this,
        permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::row_permute(
    const Array<int64> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_row_gather(
        make_temporary_clone(exec, permutation_indices).get(), this,
        permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::row_gather(
    const Array<int32> *row_indices) const
{
    auto exec = this->get_executor();
    dim<2> out_dim{row_indices->get_num_elems(), this->get_size()[1]};
    auto row_gathered = Dense::create(exec, out_dim);

    exec->run(
        dense::make_row_gather(make_temporary_clone(exec, row_indices).get(),
                               this, row_gathered.get()));
    return row_gathered;
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::row_gather(
    const Array<int64> *row_indices) const
{
    auto exec = this->get_executor();
    dim<2> out_dim{row_indices->get_num_elems(), this->get_size()[1]};
    auto row_gathered = Dense::create(exec, out_dim);

    exec->run(
        dense::make_row_gather(make_temporary_clone(exec, row_indices).get(),
                               this, row_gathered.get()));
    return row_gathered;
}


template <typename ValueType>
void Dense<ValueType>::row_gather(const Array<int32> *row_indices,
                                  Dense<ValueType> *row_gathered) const
{
    auto exec = this->get_executor();
    dim<2> expected_dim{row_indices->get_num_elems(), this->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(expected_dim, row_gathered);

    exec->run(dense::make_row_gather(
        make_temporary_clone(exec, row_indices).get(), this,
        make_temporary_clone(exec, row_gathered).get()));
}


template <typename ValueType>
void Dense<ValueType>::row_gather(const Array<int64> *row_indices,
                                  Dense<ValueType> *row_gathered) const
{
    dim<2> expected_dim{row_indices->get_num_elems(), this->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(expected_dim, row_gathered);

    auto exec = this->get_executor();

    this->get_executor()->run(dense::make_row_gather(
        make_temporary_clone(exec, row_indices).get(), this,
        make_temporary_clone(exec, row_gathered).get()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::column_permute(
    const Array<int32> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_column_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::column_permute(
    const Array<int64> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_column_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_row_permute(
    const Array<int32> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_inverse_row_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        inverse_permute_cpy.get()));

    return std::move(inverse_permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_row_permute(
    const Array<int64> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_inverse_row_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        inverse_permute_cpy.get()));

    return std::move(inverse_permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_column_permute(
    const Array<int32> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_inverse_column_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        inverse_permute_cpy.get()));

    return std::move(inverse_permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_column_permute(
    const Array<int64> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_inverse_column_permute(
        make_temporary_clone(exec, permutation_indices).get(), this,
        inverse_permute_cpy.get()));

    return std::move(inverse_permute_cpy);
}


template <typename ValueType>
std::unique_ptr<Diagonal<ValueType>> Dense<ValueType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(dense::make_extract_diagonal(this, lend(diag)));
    return diag;
}


template <typename ValueType>
void Dense<ValueType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(dense::make_inplace_absolute_dense(this));
}


template <typename ValueType>
std::unique_ptr<typename Dense<ValueType>::absolute_type>
Dense<ValueType>::compute_absolute() const
{
    auto exec = this->get_executor();

    // do not inherit the stride
    auto abs_dense = absolute_type::create(exec, this->get_size());

    exec->run(dense::make_outplace_absolute_dense(this, abs_dense.get()));

    return abs_dense;
}


template <typename ValueType>
std::unique_ptr<typename Dense<ValueType>::complex_type>
Dense<ValueType>::make_complex() const
{
    auto exec = this->get_executor();

    auto complex_dense = complex_type::create(exec, this->get_size());

    exec->run(dense::make_make_complex(this, complex_dense.get()));

    return complex_dense;
}


template <typename ValueType>
void Dense<ValueType>::make_complex(Dense<to_complex<ValueType>> *result) const
{
    auto exec = this->get_executor();

    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);

    exec->run(dense::make_make_complex(
        this, make_temporary_clone(exec, result).get()));
}


template <typename ValueType>
std::unique_ptr<typename Dense<ValueType>::absolute_type>
Dense<ValueType>::get_real() const
{
    auto exec = this->get_executor();

    auto real_dense = absolute_type::create(exec, this->get_size());

    exec->run(dense::make_get_real(this, real_dense.get()));

    return real_dense;
}


template <typename ValueType>
void Dense<ValueType>::get_real(Dense<remove_complex<ValueType>> *result) const
{
    auto exec = this->get_executor();

    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);

    exec->run(
        dense::make_get_real(this, make_temporary_clone(exec, result).get()));
}


template <typename ValueType>
std::unique_ptr<typename Dense<ValueType>::absolute_type>
Dense<ValueType>::get_imag() const
{
    auto exec = this->get_executor();

    auto imag_dense = absolute_type::create(exec, this->get_size());

    exec->run(dense::make_get_imag(this, imag_dense.get()));

    return imag_dense;
}


template <typename ValueType>
void Dense<ValueType>::get_imag(Dense<remove_complex<ValueType>> *result) const
{
    auto exec = this->get_executor();

    GKO_ASSERT_EQUAL_DIMENSIONS(this, result);

    exec->run(
        dense::make_get_imag(this, make_temporary_clone(exec, result).get()));
}


#define GKO_DECLARE_DENSE_MATRIX(_type) class Dense<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_MATRIX);


}  // namespace matrix
}  // namespace gko
