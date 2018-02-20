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

#include "core/matrix/coo.hpp"
#include "core/matrix/csr.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/coo_kernels.hpp"
#include "core/matrix/dense.hpp"


#include <algorithm>
#include <numeric>


namespace gko {
namespace matrix {


namespace {


template <typename... TplArgs>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(spmv, coo::spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(advanced_spmv, coo::advanced_spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(convert_row_idxs_to_ptrs,
                           coo::convert_row_idxs_to_ptrs<TplArgs...>);
    GKO_REGISTER_OPERATION(transpose, coo::transpose<TplArgs...>);
    GKO_REGISTER_OPERATION(conj_transpose, coo::conj_transpose<TplArgs...>);
};


}  // namespace


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply(const LinOp *b, LinOp *x) const
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
void Coo<ValueType, IndexType>::apply(const LinOp *alpha, const LinOp *b,
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
std::unique_ptr<Csr<ValueType, IndexType>>
Coo<ValueType, IndexType>::conversion_helper() const
{
    auto exec = this->get_executor();
    auto tmp = Csr<ValueType, IndexType>::create(
        exec, this->get_num_rows(), this->get_num_cols(),
        this->get_num_stored_elements());
    exec->run(
        TemplatedOperation<IndexType>::make_convert_row_idxs_to_ptrs_operation(
            this->get_const_row_idxs(), this->get_num_stored_elements(),
            tmp->get_row_ptrs(), this->get_num_rows() + 1));
    return std::move(tmp);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType> *result) const
{
    auto tmp = conversion_helper();
    tmp->values_ = this->values_;
    tmp->col_idxs_ = this->col_idxs_;
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(Csr<ValueType, IndexType> *result)
{
    auto tmp = conversion_helper();
    tmp->values_ = std::move(this->values_);
    tmp->col_idxs_ = std::move(this->col_idxs_);
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::read_from_mtx(const std::string &filename)
{
    auto data = read_raw_from_mtx<ValueType, IndexType>(filename);
    size_type nnz = 0;
    for (const auto &elem : data.nonzeros) {
        nnz += (std::get<2>(elem) != zero<ValueType>());
    }
    auto tmp = create(this->get_executor()->get_master(), data.num_rows,
                      data.num_cols, nnz);
    size_type elt = 0;
    for (const auto &elem : data.nonzeros) {
        auto val = std::get<2>(elem);
        if (val != zero<ValueType>()) {
            tmp->get_row_idxs()[elt] = std::get<0>(elem);
            tmp->get_col_idxs()[elt] = std::get<1>(elem);
            tmp->get_values()[elt] = std::get<2>(elem);
            elt++;
        }
    }
    this->copy_from(std::move(tmp));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Coo<ValueType, IndexType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = create(exec, this->get_num_cols(), this->get_num_rows(),
                            this->get_num_stored_elements());

    exec->run(
        TemplatedOperation<ValueType, IndexType>::make_transpose_operation(
            trans_cpy.get(), this));
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Coo<ValueType, IndexType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = create(exec, this->get_num_cols(), this->get_num_rows(),
                            this->get_num_stored_elements());

    exec->run(
        TemplatedOperation<ValueType, IndexType>::make_conj_transpose_operation(
            trans_cpy.get(), this));
    return std::move(trans_cpy);
}


#define DECLARE_COO_MATRIX(ValueType, IndexType) class Coo<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_COO_MATRIX);
#undef DECLARE_COO_MATRIX


}  // namespace matrix
}  // namespace gko
