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

#include "core/matrix/ell.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace matrix {


namespace {


template <typename... TplArgs>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(spmv, ell::spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(advanced_spmv, ell::advanced_spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(convert_to_dense, ell::convert_to_dense<TplArgs...>);
    GKO_REGISTER_OPERATION(move_to_dense, ell::move_to_dense<TplArgs...>);
};


}  // namespace


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::copy_from(const LinOp *other)
{
    as<ConvertibleTo<Ell<ValueType, IndexType>>>(other)->convert_to(this);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::copy_from(std::unique_ptr<LinOp> other)
{
    as<ConvertibleTo<Ell<ValueType, IndexType>>>(other.get())->move_to(this);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply(const LinOp *b, LinOp *x) const
{
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_EQUAL_COLS(b, x);
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_spmv_operation(
            this, as<Dense>(b), as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply(const LinOp *alpha, const LinOp *b,
                                      const LinOp *beta, LinOp *x) const
{
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_EQUAL_COLS(b, x);
    ASSERT_EQUAL_DIMENSIONS(alpha, size(1, 1));
    ASSERT_EQUAL_DIMENSIONS(beta, size(1, 1));
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_advanced_spmv_operation(
            as<Dense>(alpha), this, as<Dense>(b), as<Dense>(beta),
            as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Ell<ValueType, IndexType>::clone_type() const
{
    return std::unique_ptr<LinOp>(
        new Ell(this->get_executor(), this->get_num_rows(),
                this->get_num_cols(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::clear()
{
    this->set_dimensions(0, 0, 0);
    values_.clear();
    col_idxs_.clear();
    row_ptrs_.clear();
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(Ell *other) const
{
    other->set_dimensions(this);
    other->values_ = values_;
    other->col_idxs_ = col_idxs_;
    // other->row_ptrs_ = row_ptrs_;
    other->max_nnz_row_ = max_nnz_row_;
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(Ell *other)
{
    other->set_dimensions(this);
    other->values_ = std::move(values_);
    other->col_idxs_ = std::move(col_idxs_);
    // other->row_ptrs_ = std::move(row_ptrs_);
    other->max_nnz_row_ = std::move(max_nnz_row_);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(
        exec, this->get_num_rows(), this->get_num_cols(), this->get_num_cols());
    exec->run(TemplatedOperation<
              ValueType, IndexType>::make_convert_to_dense_operation(tmp.get(),
                                                                     this));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(Dense<ValueType> *result)
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(
        exec, this->get_num_rows(), this->get_num_cols(), this->get_num_cols());
    exec->run(
        TemplatedOperation<ValueType, IndexType>::make_move_to_dense_operation(
            tmp.get(), this));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::read_from_mtx(const std::string &filename)
{
    auto data = read_raw_from_mtx<ValueType, IndexType>(filename);
    size_type nnz = 0;
    for (const auto &elem : data.nonzeros) {
        nnz += (std::get<2>(elem) != zero<ValueType>());
    }
    auto tmp = create(this->get_executor()->get_master(), data.num_rows,
                      data.num_cols, nnz);
    size_type ind = 0;
    size_type cur_ptr = 0;
    tmp->get_row_ptrs()[0] = cur_ptr;
    for (size_type row = 0; row < data.num_rows; ++row) {
        for (; ind < data.nonzeros.size(); ++ind) {
            if (std::get<0>(data.nonzeros[ind]) > row) {
                break;
            }
            auto val = std::get<2>(data.nonzeros[ind]);
            if (val != zero<ValueType>()) {
                tmp->get_values()[cur_ptr] = val;
                tmp->get_col_idxs()[cur_ptr] = std::get<1>(data.nonzeros[ind]);
                ++cur_ptr;
            }
        }
        tmp->get_row_ptrs()[row + 1] = cur_ptr;
    }
    tmp->move_to(this);
}


#define DECLARE_ELL_MATRIX(ValueType, IndexType) class Ell<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_ELL_MATRIX);
#undef DECLARE_ELL_MATRIX


}  // namespace matrix
}  // namespace gko
