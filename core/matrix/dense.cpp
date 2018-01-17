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
#include "core/base/utils.hpp"
#include "core/matrix/csr.hpp"
#include "core/matrix/dense_kernels.hpp"

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
};


template <typename... TplArgs>
struct TemplatedOperationCsr {
    GKO_REGISTER_OPERATION(convert_to_csr, dense::convert_to_csr<TplArgs...>);
    GKO_REGISTER_OPERATION(move_to_csr, dense::move_to_csr<TplArgs...>);
};
}  // namespace


template <typename ValueType>
void Dense<ValueType>::copy_from(const LinOp *other)
{
    as<ConvertibleTo<Dense<ValueType>>>(other)->convert_to(this);
}


template <typename ValueType>
void Dense<ValueType>::copy_from(std::unique_ptr<LinOp> other)
{
    as<ConvertibleTo<Dense<ValueType>>>(other.get())->move_to(this);
}


template <typename ValueType>
void Dense<ValueType>::apply(const LinOp *b, LinOp *x) const
{
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_EQUAL_COLS(b, x);
    auto exec = this->get_executor();
    if (b->get_executor() != exec || x->get_executor() != exec) NOT_IMPLEMENTED;
    exec->run(TemplatedOperation<ValueType>::make_simple_apply_operation(
        this, as<Dense<ValueType>>(b), as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Dense<ValueType>::apply(const LinOp *alpha, const LinOp *b,
                             const LinOp *beta, LinOp *x) const
{
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_EQUAL_COLS(b, x);
    ASSERT_EQUAL_DIMENSIONS(alpha, size(1, 1));
    ASSERT_EQUAL_DIMENSIONS(beta, size(1, 1));
    auto exec = this->get_executor();
    if (alpha->get_executor() != exec || b->get_executor() != exec ||
        beta->get_executor() != exec || x->get_executor() != exec)
        NOT_IMPLEMENTED;
    exec->run(TemplatedOperation<ValueType>::make_apply_operation(
        as<Dense<ValueType>>(alpha), this, as<Dense<ValueType>>(b),
        as<Dense<ValueType>>(beta), as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Dense<ValueType>::scale(const LinOp *alpha)
{
    ASSERT_EQUAL_ROWS(alpha, size(1, 1));
    if (alpha->get_num_cols() != 1) {
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
    ASSERT_EQUAL_ROWS(alpha, size(1, 1));
    if (alpha->get_num_cols() != 1) {
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
    ASSERT_EQUAL_DIMENSIONS(result, size(1, this->get_num_cols()));
    auto exec = this->get_executor();
    if (b->get_executor() != exec || result->get_executor() != exec)
        NOT_IMPLEMENTED;
    exec->run(TemplatedOperation<ValueType>::make_compute_dot_operation(
        this, as<Dense<ValueType>>(b), as<Dense<ValueType>>(result)));
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::clone_type() const
{
    return std::unique_ptr<Dense>(new Dense(this->get_executor(), 0, 0, 0));
}


template <typename ValueType>
void Dense<ValueType>::clear()
{
    this->set_dimensions(0, 0, 0);
    values_.clear();
    padding_ = 0;
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Dense *result) const
{
    result->set_dimensions(this);
    result->values_ = values_;
    result->padding_ = padding_;
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense *result)
{
    result->set_dimensions(this);
    result->values_ = std::move(values_);
    result->padding_ = padding_;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>> create_receiving_mtx(
    const Dense<ValueType> *source)
{
    auto exec = source->get_executor();

    Array<size_type> num_stored_nonzeros(exec, 1);
    exec->run(TemplatedOperation<ValueType>::make_count_nonzeros_operation(
        source, num_stored_nonzeros.get_data()));
    auto tmp = Csr<ValueType, IndexType>::create(
        exec, source->get_num_rows(), source->get_num_cols(),
        *num_stored_nonzeros.get_data());
    return tmp;
}

template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int32> *result) const
{
    auto tmp = create_receiving_mtx<ValueType, int32>(this);
    this->get_executor()->run(
        TemplatedOperationCsr<ValueType, int32>::make_convert_to_csr_operation(
            tmp.get(), this));
    tmp->move_to(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int32> *result)
{
    auto tmp = create_receiving_mtx<ValueType, int32>(this);
    this->get_executor()->run(
        TemplatedOperationCsr<ValueType, int32>::make_convert_to_csr_operation(
            tmp.get(), this));
    tmp->move_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int64> *result) const
{
    auto tmp = create_receiving_mtx<ValueType, int64>(this);
    this->get_executor()->run(
        TemplatedOperationCsr<ValueType, int64>::make_convert_to_csr_operation(
            tmp.get(), this));
    tmp->move_to(result);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int64> *result)
{
    auto tmp = create_receiving_mtx<ValueType, int64>(this);
    this->get_executor()->run(
        TemplatedOperationCsr<ValueType, int64>::make_move_to_csr_operation(
            tmp.get(), this));
    tmp->move_to(result);
}


#define DECLARE_DENSE_MATRIX(_type) class Dense<_type>;
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(DECLARE_DENSE_MATRIX);
#undef DECLARE_DENSE_MATRIX


}  // namespace matrix


}  // namespace gko
