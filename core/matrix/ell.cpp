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


#include <algorithm>


#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/dense.hpp"
#include "core/matrix/ell_kernels.hpp"


namespace gko {
namespace matrix {


namespace {


template <typename... TplArgs>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(spmv, ell::spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(advanced_spmv, ell::advanced_spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(convert_to_dense, ell::convert_to_dense<TplArgs...>);
};


template <typename ValueType, typename IndexType>
size_type calculate_max_nonzeros_per_row(
    const matrix_data<ValueType, IndexType> &data)
{
    size_type nnz = 0;
    IndexType current_row = 0;
    size_type max_nonzeros_per_row = 0;
    for (const auto &elem : data.nonzeros) {
        if (elem.row != current_row) {
            max_nonzeros_per_row = std::max(max_nonzeros_per_row, nnz);
            nnz = 0;
        }
        nnz += (elem.value != zero<ValueType>());
    }
    return max_nonzeros_per_row;
}


}  // namespace


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_spmv_operation(
            this, as<Dense>(b), as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta, LinOp *x) const
{
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_advanced_spmv_operation(
            as<Dense>(alpha), this, as<Dense>(b), as<Dense>(beta),
            as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_dimensions().num_rows,
                                        this->get_dimensions().num_cols,
                                        this->get_dimensions().num_cols);
    exec->run(TemplatedOperation<
              ValueType, IndexType>::make_convert_to_dense_operation(tmp.get(),
                                                                     this));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(Dense<ValueType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::read(const mat_data &data)
{
    // Get the maximum number of nonzero elements of every row.
    auto max_nonzeros_per_row = calculate_max_nonzeros_per_row(data);

    // Create an ELLPACK format matrix based on the sizes.
    auto tmp = create(this->get_executor()->get_master(), data.num_rows,
                      data.num_cols, max_nonzeros_per_row, data.num_rows);

    // Get values and column indexes.
    size_type ind = 0;
    size_type n = data.nonzeros.size();
    auto vals = tmp->get_values();
    auto col_idxs = tmp->get_col_idxs();
    for (size_type row = 0; row < data.num_rows; row++) {
        size_type col = 0;
        while (ind < n && data.nonzeros[ind].row == row) {
            auto val = data.nonzeros[ind].value;
            if (val != zero<ValueType>()) {
                tmp->val_at(row, col) = val;
                tmp->col_at(row, col) = data.nonzeros[ind].column;
                col++;
            }
            ind++;
        }
        for (auto i = col; i < max_nonzeros_per_row; i++) {
            tmp->val_at(row, i) = zero<ValueType>();
            tmp->col_at(row, i) = 0;
        }
    }

    // Return the matrix
    tmp->move_to(this);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::write(mat_data &data) const
{
    std::unique_ptr<const LinOp> op{};
    const Ell *tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const Ell *>(op.get());
    } else {
        tmp = this;
    }

    data = {tmp->get_dimensions().num_rows, tmp->get_dimensions().num_cols, {}};

    for (size_type row = 0; row < tmp->get_dimensions().num_rows; ++row) {
        for (size_type i = 0; i < tmp->max_nonzeros_per_row_; ++i) {
            const auto val = tmp->val_at(row, i);
            if (val != zero<ValueType>()) {
                const auto col = tmp->col_at(row, i);
                data.nonzeros.emplace_back(row, col, val);
            }
        }
    }
}


#define DECLARE_ELL_MATRIX(ValueType, IndexType) class Ell<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_ELL_MATRIX);
#undef DECLARE_ELL_MATRIX


}  // namespace matrix
}  // namespace gko
