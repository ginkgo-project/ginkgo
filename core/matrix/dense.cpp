/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/matrix/dense.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "core/matrix/dense_kernels.hpp"


#include <algorithm>


namespace gko {
namespace matrix {
namespace dense {


GKO_REGISTER_OPERATION(simple_apply, dense::simple_apply);
GKO_REGISTER_OPERATION(apply, dense::apply);
GKO_REGISTER_OPERATION(scale, dense::scale);
GKO_REGISTER_OPERATION(add_scaled, dense::add_scaled);
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
GKO_REGISTER_OPERATION(convert_to_coo, dense::convert_to_coo);
GKO_REGISTER_OPERATION(convert_to_csr, dense::convert_to_csr);
GKO_REGISTER_OPERATION(move_to_csr, dense::move_to_csr);
GKO_REGISTER_OPERATION(convert_to_ell, dense::convert_to_ell);
GKO_REGISTER_OPERATION(move_to_ell, dense::move_to_ell);
GKO_REGISTER_OPERATION(convert_to_hybrid, dense::convert_to_hybrid);
GKO_REGISTER_OPERATION(move_to_hybrid, dense::move_to_hybrid);
GKO_REGISTER_OPERATION(convert_to_sellp, dense::convert_to_sellp);
GKO_REGISTER_OPERATION(move_to_sellp, dense::move_to_sellp);


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
    exec->run(op(tmp.get(), source));
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
    auto tmp = Csr<ValueType, IndexType>::create(exec, source->get_size(),
                                                 num_stored_nonzeros);
    exec->run(op(tmp.get(), source));
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
    exec->run(op(tmp.get(), source));
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
    exec->run(op(tmp.get(), source));
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
    size_type total_columns = 0;
    exec->run(dense::make_calculate_total_cols(source, &total_columns,
                                               stride_factor));
    const auto total_cols = std::max(result->get_total_cols(), total_columns);
    const auto slice_size = (result->get_slice_size() == 0)
                                ? default_slice_size
                                : result->get_slice_size();
    auto tmp = Sellp<ValueType, IndexType>::create(
        exec, source->get_size(), slice_size, stride_factor, total_cols);
    exec->run(op(tmp.get(), source));
    tmp->move_to(result);
}


}  // namespace


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    this->get_executor()->run(dense::make_simple_apply(
        this, as<Dense<ValueType>>(b), as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                  const LinOp *beta, LinOp *x) const
{
    this->get_executor()->run(dense::make_apply(
        as<Dense<ValueType>>(alpha), this, as<Dense<ValueType>>(b),
        as<Dense<ValueType>>(beta), as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Dense<ValueType>::scale(const LinOp *alpha)
{
    ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        ASSERT_EQUAL_COLS(this, alpha);
    }
    auto exec = this->get_executor();
    if (alpha->get_executor() != exec) NOT_IMPLEMENTED;
    exec->run(dense::make_scale(as<Dense<ValueType>>(alpha), this));
}


template <typename ValueType>
void Dense<ValueType>::add_scaled(const LinOp *alpha, const LinOp *b)
{
    ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        ASSERT_EQUAL_COLS(this, alpha);
    }
    ASSERT_EQUAL_DIMENSIONS(this, b);
    auto exec = this->get_executor();
    if (alpha->get_executor() != exec || b->get_executor() != exec)
        NOT_IMPLEMENTED;
    exec->run(dense::make_add_scaled(as<Dense<ValueType>>(alpha),
                                     as<Dense<ValueType>>(b), this));
}


template <typename ValueType>
void Dense<ValueType>::compute_dot(const LinOp *b, LinOp *result) const
{
    ASSERT_EQUAL_DIMENSIONS(this, b);
    ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (b->get_executor() != exec || result->get_executor() != exec)
        NOT_IMPLEMENTED;
    exec->run(dense::make_compute_dot(this, as<Dense<ValueType>>(b),
                                      as<Dense<ValueType>>(result)));
}


template <typename ValueType>
void Dense<ValueType>::compute_norm2(LinOp *result) const
{
    ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    if (result->get_executor() != exec) NOT_IMPLEMENTED;
    exec->run(dense::make_compute_norm2(as<Dense<ValueType>>(this),
                                        as<Dense<ValueType>>(result)));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Coo<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_coo<decltype(result),
                                            const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Coo<ValueType, int32> *result)
{
    conversion_helper(result, this,
                      dense::template make_convert_to_coo<decltype(result),
                                                          Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Coo<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_coo<decltype(result),
                                            const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Coo<ValueType, int64> *result)
{
    conversion_helper(result, this,
                      dense::template make_convert_to_coo<decltype(result),
                                                          Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_csr<decltype(result),
                                            const Dense<ValueType> *&>);
    result->make_srow();
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int32> *result)
{
    conversion_helper(result, this,
                      dense::template make_move_to_csr<decltype(result),
                                                       Dense<ValueType> *&>);
    result->make_srow();
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_csr<decltype(result),
                                            const Dense<ValueType> *&>);
    result->make_srow();
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int64> *result)
{
    conversion_helper(result, this,
                      dense::template make_move_to_csr<decltype(result),
                                                       Dense<ValueType> *&>);
    result->make_srow();
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Ell<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_ell<decltype(result),
                                            const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Ell<ValueType, int32> *result)
{
    conversion_helper(result, this,
                      dense::template make_move_to_ell<decltype(result),
                                                       Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Ell<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_ell<decltype(result),
                                            const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Ell<ValueType, int64> *result)
{
    conversion_helper(result, this,
                      dense::template make_move_to_ell<decltype(result),
                                                       Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Hybrid<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_hybrid<decltype(result),
                                               const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Hybrid<ValueType, int32> *result)
{
    conversion_helper(result, this,
                      dense::template make_move_to_hybrid<decltype(result),
                                                          Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Hybrid<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_hybrid<decltype(result),
                                               const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Hybrid<ValueType, int64> *result)
{
    conversion_helper(result, this,
                      dense::template make_move_to_hybrid<decltype(result),
                                                          Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Sellp<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_sellp<decltype(result),
                                              const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Sellp<ValueType, int32> *result)
{
    conversion_helper(result, this,
                      dense::template make_move_to_sellp<decltype(result),
                                                         Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Sellp<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_sellp<decltype(result),
                                              const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Sellp<ValueType, int64> *result)
{
    conversion_helper(result, this,
                      dense::template make_move_to_sellp<decltype(result),
                                                         Dense<ValueType> *&>);
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

    exec->run(dense::make_transpose(trans_cpy.get(), this));

    return std::move(trans_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Dense::create(exec, gko::transpose(this->get_size()));

    exec->run(dense::make_conj_transpose(trans_cpy.get(), this));
    return std::move(trans_cpy);
}


#define DECLARE_DENSE_MATRIX(_type) class Dense<_type>;
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(DECLARE_DENSE_MATRIX);
#undef DECLARE_DENSE_MATRIX


}  // namespace matrix


}  // namespace gko
