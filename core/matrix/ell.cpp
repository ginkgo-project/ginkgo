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

#include <ginkgo/core/matrix/ell.hpp>


#include <algorithm>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/absolute_array.hpp"
#include "core/components/fill_array.hpp"
#include "core/matrix/ell_kernels.hpp"


namespace gko {
namespace matrix {
namespace ell {


GKO_REGISTER_OPERATION(spmv, ell::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, ell::advanced_spmv);
GKO_REGISTER_OPERATION(convert_to_dense, ell::convert_to_dense);
GKO_REGISTER_OPERATION(convert_to_csr, ell::convert_to_csr);
GKO_REGISTER_OPERATION(count_nonzeros, ell::count_nonzeros);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row,
                       ell::calculate_nonzeros_per_row);
GKO_REGISTER_OPERATION(extract_diagonal, ell::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // namespace ell


namespace {


template <typename ValueType, typename IndexType>
size_type calculate_max_nnz_per_row(
    const matrix_data<ValueType, IndexType> &data)
{
    size_type nnz = 0;
    IndexType current_row = 0;
    size_type num_stored_elements_per_row = 0;
    for (const auto &elem : data.nonzeros) {
        if (elem.row != current_row) {
            current_row = elem.row;
            num_stored_elements_per_row =
                std::max(num_stored_elements_per_row, nnz);
            nnz = 0;
        }
        nnz += (elem.value != zero<ValueType>());
    }
    return std::max(num_stored_elements_per_row, nnz);
}


}  // namespace


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using ComplexDense = Dense<to_complex<ValueType>>;

    if (dynamic_cast<const Dense<ValueType> *>(b)) {
        this->get_executor()->run(ell::make_spmv(this, as<Dense<ValueType>>(b),
                                                 as<Dense<ValueType>>(x)));
    } else {
        auto dense_b = as<ComplexDense>(b);
        auto dense_x = as<ComplexDense>(x);
        this->apply(dense_b->create_real_view().get(),
                    dense_x->create_real_view().get());
    }
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta, LinOp *x) const
{
    using ComplexDense = Dense<to_complex<ValueType>>;
    using RealDense = Dense<remove_complex<ValueType>>;

    if (dynamic_cast<const Dense<ValueType> *>(b)) {
        this->get_executor()->run(ell::make_advanced_spmv(
            as<Dense<ValueType>>(alpha), this, as<Dense<ValueType>>(b),
            as<Dense<ValueType>>(beta), as<Dense<ValueType>>(x)));
    } else {
        auto dense_b = as<ComplexDense>(b);
        auto dense_x = as<ComplexDense>(x);
        auto dense_alpha = as<RealDense>(alpha);
        auto dense_beta = as<RealDense>(beta);
        this->apply(dense_alpha, dense_b->create_real_view().get(), dense_beta,
                    dense_x->create_real_view().get());
    }
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(
    Ell<next_precision<ValueType>, IndexType> *result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->num_stored_elements_per_row_ = this->num_stored_elements_per_row_;
    result->stride_ = this->stride_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(
    Ell<next_precision<ValueType>, IndexType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(ell::make_convert_to_dense(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(Dense<ValueType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType> *result) const
{
    auto exec = this->get_executor();

    size_type num_stored_elements = 0;
    exec->run(ell::make_count_nonzeros(this, &num_stored_elements));

    auto tmp = Csr<ValueType, IndexType>::create(
        exec, this->get_size(), num_stored_elements, result->get_strategy());
    exec->run(ell::make_convert_to_csr(this, tmp.get()));

    tmp->make_srow();
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(Csr<ValueType, IndexType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::read(const mat_data &data)
{
    // Get the number of stored elements of every row.
    auto num_stored_elements_per_row = calculate_max_nnz_per_row(data);

    // Create an ELLPACK format matrix based on the sizes.
    auto tmp = Ell::create(this->get_executor()->get_master(), data.size,
                           num_stored_elements_per_row, data.size[0]);

    // Get values and column indexes.
    size_type ind = 0;
    size_type n = data.nonzeros.size();
    auto vals = tmp->get_values();
    auto col_idxs = tmp->get_col_idxs();
    for (size_type row = 0; row < data.size[0]; row++) {
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
        for (auto i = col; i < num_stored_elements_per_row; i++) {
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

    data = {tmp->get_size(), {}};

    for (size_type row = 0; row < tmp->get_size()[0]; ++row) {
        for (size_type i = 0; i < tmp->num_stored_elements_per_row_; ++i) {
            const auto val = tmp->val_at(row, i);
            if (val != zero<ValueType>()) {
                const auto col = tmp->col_at(row, i);
                data.nonzeros.emplace_back(row, col, val);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Ell<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(ell::make_fill_array(diag->get_values(), diag->get_size()[0],
                                   zero<ValueType>()));
    exec->run(ell::make_extract_diagonal(this, lend(diag)));
    return diag;
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(ell::make_inplace_absolute_array(
        this->get_values(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Ell<ValueType, IndexType>::absolute_type>
Ell<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_ell = absolute_type::create(
        exec, this->get_size(), this->get_num_stored_elements_per_row(),
        this->get_stride());

    abs_ell->col_idxs_ = col_idxs_;
    exec->run(ell::make_outplace_absolute_array(this->get_const_values(),
                                                this->get_num_stored_elements(),
                                                abs_ell->get_values()));

    return abs_ell;
}


#define GKO_DECLARE_ELL_MATRIX(ValueType, IndexType) \
    class Ell<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_MATRIX);


}  // namespace matrix
}  // namespace gko
