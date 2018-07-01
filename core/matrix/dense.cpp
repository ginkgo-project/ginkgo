/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

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

#include "core/matrix/dense.hpp"


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/coo.hpp"
#include "core/matrix/csr.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/matrix/ell.hpp"
#include "core/matrix/sellp.hpp"


#include <algorithm>


namespace gko {
namespace matrix {


namespace {


template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(simple_apply, dense::simple_apply<ValueType>);
    GKO_REGISTER_OPERATION(apply, dense::apply<ValueType>);
    GKO_REGISTER_OPERATION(scale, dense::scale<ValueType>);
    GKO_REGISTER_OPERATION(add_scaled, dense::add_scaled<ValueType>);
    GKO_REGISTER_OPERATION(compute_dot, dense::compute_dot<ValueType>);
    GKO_REGISTER_OPERATION(count_nonzeros, dense::count_nonzeros<ValueType>);
    GKO_REGISTER_OPERATION(calculate_max_nonzeros_per_row,
                           dense::calculate_max_nonzeros_per_row<ValueType>);
    GKO_REGISTER_OPERATION(transpose, dense::transpose<ValueType>);
    GKO_REGISTER_OPERATION(conj_transpose, dense::conj_transpose<ValueType>);
};


template <typename... TplArgs>
struct TemplatedOperationCoo {
    GKO_REGISTER_OPERATION(convert_to_coo, dense::convert_to_coo<TplArgs...>);
};


template <typename... TplArgs>
struct TemplatedOperationCsr {
    GKO_REGISTER_OPERATION(convert_to_csr, dense::convert_to_csr<TplArgs...>);
    GKO_REGISTER_OPERATION(move_to_csr, dense::move_to_csr<TplArgs...>);
};


template <typename... TplArgs>
struct TemplatedOperationEll {
    GKO_REGISTER_OPERATION(convert_to_ell, dense::convert_to_ell<TplArgs...>);
    GKO_REGISTER_OPERATION(move_to_ell, dense::move_to_ell<TplArgs...>);
};


template <typename... TplArgs>
struct TemplatedOperationSellp {
    GKO_REGISTER_OPERATION(convert_to_sellp,
        dense::convert_to_sellp<TplArgs...>);
    GKO_REGISTER_OPERATION(move_to_sellp, dense::move_to_sellp<TplArgs...>);
};


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Coo<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();

    size_type num_stored_nonzeros = 0;
    exec->run(TemplatedOperation<ValueType>::make_count_nonzeros_operation(
        source, &num_stored_nonzeros));
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
    exec->run(TemplatedOperation<ValueType>::make_count_nonzeros_operation(
        source, &num_stored_nonzeros));
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
    size_type max_nonzeros_per_row = 0;
    exec->run(TemplatedOperation<ValueType>::
                  make_calculate_max_nonzeros_per_row_operation(
                      source, &max_nonzeros_per_row));
    const auto max_nnz_per_row =
        std::max(result->get_max_nonzeros_per_row(), max_nonzeros_per_row);
    const auto stride =
        std::max(result->get_stride(), source->get_size().num_rows);
    auto tmp = Ell<ValueType, IndexType>::create(exec, source->get_size(),
                                                 max_nnz_per_row, stride);
    exec->run(op(tmp.get(), source));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Sellp<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    NOT_IMPLEMENTED;
}


}  // namespace


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    this->get_executor()->run(
        TemplatedOperation<ValueType>::make_simple_apply_operation(
            this, as<Dense<ValueType>>(b), as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                  const LinOp *beta, LinOp *x) const
{
    this->get_executor()->run(
        TemplatedOperation<ValueType>::make_apply_operation(
            as<Dense<ValueType>>(alpha), this, as<Dense<ValueType>>(b),
            as<Dense<ValueType>>(beta), as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Dense<ValueType>::scale(const LinOp *alpha)
{
    ASSERT_EQUAL_ROWS(alpha, dim(1, 1));
    if (alpha->get_size().num_cols != 1) {
        // different alpha for each column
        ASSERT_EQUAL_COLS(this, alpha);
    }
    auto exec = this->get_executor();
    if (alpha->get_executor() != exec) NOT_IMPLEMENTED;
    exec->run(TemplatedOperation<ValueType>::make_scale_operation(
        as<Dense<ValueType>>(alpha), this));
}


template <typename ValueType>
void Dense<ValueType>::add_scaled(const LinOp *alpha, const LinOp *b)
{
    ASSERT_EQUAL_ROWS(alpha, dim(1, 1));
    if (alpha->get_size().num_cols != 1) {
        // different alpha for each column
        ASSERT_EQUAL_COLS(this, alpha);
    }
    ASSERT_EQUAL_DIMENSIONS(this, b);
    auto exec = this->get_executor();
    if (alpha->get_executor() != exec || b->get_executor() != exec)
        NOT_IMPLEMENTED;
    exec->run(TemplatedOperation<ValueType>::make_add_scaled_operation(
        as<Dense<ValueType>>(alpha), as<Dense<ValueType>>(b), this));
}


template <typename ValueType>
void Dense<ValueType>::compute_dot(const LinOp *b, LinOp *result) const
{
    ASSERT_EQUAL_DIMENSIONS(this, b);
    ASSERT_EQUAL_DIMENSIONS(result, dim(1, this->get_size().num_cols));
    auto exec = this->get_executor();
    if (b->get_executor() != exec || result->get_executor() != exec)
        NOT_IMPLEMENTED;
    exec->run(TemplatedOperation<ValueType>::make_compute_dot_operation(
        this, as<Dense<ValueType>>(b), as<Dense<ValueType>>(result)));
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Coo<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        TemplatedOperationCoo<ValueType, int32>::
            template make_convert_to_coo_operation<decltype(result),
                                                   const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Coo<ValueType, int32> *result)
{
    conversion_helper(
        result, this,
        TemplatedOperationCoo<ValueType, int32>::
            template make_convert_to_coo_operation<decltype(result),
                                                   Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Coo<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        TemplatedOperationCoo<ValueType, int64>::
            template make_convert_to_coo_operation<decltype(result),
                                                   const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Coo<ValueType, int64> *result)
{
    conversion_helper(
        result, this,
        TemplatedOperationCoo<ValueType, int64>::
            template make_convert_to_coo_operation<decltype(result),
                                                   Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        TemplatedOperationCsr<ValueType, int32>::
            template make_convert_to_csr_operation<decltype(result),
                                                   const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int32> *result)
{
    conversion_helper(
        result, this,
        TemplatedOperationCsr<ValueType, int32>::
            template make_move_to_csr_operation<decltype(result),
                                                Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        TemplatedOperationCsr<ValueType, int64>::
            template make_convert_to_csr_operation<decltype(result),
                                                   const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int64> *result)
{
    conversion_helper(
        result, this,
        TemplatedOperationCsr<ValueType, int64>::
            template make_move_to_csr_operation<decltype(result),
                                                Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Ell<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        TemplatedOperationEll<ValueType, int32>::
            template make_convert_to_ell_operation<decltype(result),
                                                   const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Ell<ValueType, int32> *result)
{
    conversion_helper(
        result, this,
        TemplatedOperationEll<ValueType, int32>::
            template make_move_to_ell_operation<decltype(result),
                                                Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Ell<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        TemplatedOperationEll<ValueType, int64>::
            template make_convert_to_ell_operation<decltype(result),
                                                   const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Ell<ValueType, int64> *result)
{
    conversion_helper(
        result, this,
        TemplatedOperationEll<ValueType, int64>::
            template make_move_to_ell_operation<decltype(result),
                                                Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Sellp<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        TemplatedOperationSellp<ValueType, int32>::
            template make_convert_to_sellp_operation<decltype(result),
                                                   const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Sellp<ValueType, int32> *result)
{
    conversion_helper(
        result, this,
        TemplatedOperationSellp<ValueType, int32>::
            template make_move_to_sellp_operation<decltype(result),
                                                Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Sellp<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        TemplatedOperationSellp<ValueType, int64>::
            template make_convert_to_sellp_operation<decltype(result),
                                                   const Dense<ValueType> *&>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Sellp<ValueType, int64> *result)
{
    conversion_helper(
        result, this,
        TemplatedOperationSellp<ValueType, int64>::
            template make_move_to_sellp_operation<decltype(result),
                                                Dense<ValueType> *&>);
}


namespace {


template <typename MatrixType, typename MatrixData>
inline void read_impl(MatrixType *mtx, const MatrixData &data)
{
    auto tmp = MatrixType::create(mtx->get_executor()->get_master(), data.size);
    size_type ind = 0;
    for (size_type row = 0; row < data.size.num_rows; ++row) {
        for (size_type col = 0; col < data.size.num_cols; ++col) {
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

    for (size_type row = 0; row < data.size.num_rows; ++row) {
        for (size_type col = 0; col < data.size.num_cols; ++col) {
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

    exec->run(TemplatedOperation<ValueType>::make_transpose_operation(
        trans_cpy.get(), this));

    return std::move(trans_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Dense::create(exec, gko::transpose(this->get_size()));

    exec->run(TemplatedOperation<ValueType>::make_conj_transpose_operation(
        trans_cpy.get(), this));
    return std::move(trans_cpy);
}


#define DECLARE_DENSE_MATRIX(_type) class Dense<_type>;
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(DECLARE_DENSE_MATRIX);
#undef DECLARE_DENSE_MATRIX


}  // namespace matrix


}  // namespace gko
