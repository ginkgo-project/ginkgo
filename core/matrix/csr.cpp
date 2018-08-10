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

#include "core/matrix/csr.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/coo.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace matrix {


namespace {


template <typename... TplArgs>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(spmv, csr::spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(advanced_spmv, csr::advanced_spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(convert_row_ptrs_to_idxs,
                           csr::convert_row_ptrs_to_idxs<TplArgs...>);
    GKO_REGISTER_OPERATION(convert_to_dense, csr::convert_to_dense<TplArgs...>);
    GKO_REGISTER_OPERATION(move_to_dense, csr::move_to_dense<TplArgs...>);
    GKO_REGISTER_OPERATION(transpose, csr::transpose<TplArgs...>);
    GKO_REGISTER_OPERATION(conj_transpose, csr::conj_transpose<TplArgs...>);
};


}  // namespace


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_spmv_operation(
            this, as<Dense>(b), as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta, LinOp *x) const
{
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_advanced_spmv_operation(
            as<Dense>(alpha), this, as<Dense>(b), as<Dense>(beta),
            as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Coo<ValueType, IndexType>> Csr<ValueType, IndexType>::make_coo()
    const
{
    auto exec = this->get_executor();
    auto tmp = Coo<ValueType, IndexType>::create(
        exec, this->get_size(), this->get_num_stored_elements());
    exec->run(
        TemplatedOperation<IndexType>::make_convert_row_ptrs_to_idxs_operation(
            this->get_const_row_ptrs(), this->get_size()[0],
            tmp->get_row_idxs()));
    return tmp;
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Coo<ValueType, IndexType> *result) const
{
    auto tmp = this->make_coo();
    tmp->values_ = this->values_;
    tmp->col_idxs_ = this->col_idxs_;
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Coo<ValueType, IndexType> *result)
{
    auto tmp = this->make_coo();
    tmp->values_ = std::move(this->values_);
    tmp->col_idxs_ = std::move(this->col_idxs_);
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(TemplatedOperation<
              ValueType, IndexType>::make_convert_to_dense_operation(tmp.get(),
                                                                     this));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Dense<ValueType> *result)
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(
        TemplatedOperation<ValueType, IndexType>::make_move_to_dense_operation(
            tmp.get(), this));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::read(const mat_data &data)
{
    size_type nnz = 0;
    for (const auto &elem : data.nonzeros) {
        nnz += (elem.value != zero<ValueType>());
    }
    auto tmp = Csr::create(this->get_executor()->get_master(), data.size, nnz);
    size_type ind = 0;
    size_type cur_ptr = 0;
    tmp->get_row_ptrs()[0] = cur_ptr;
    for (size_type row = 0; row < data.size[0]; ++row) {
        for (; ind < data.nonzeros.size(); ++ind) {
            if (data.nonzeros[ind].row > row) {
                break;
            }
            auto val = data.nonzeros[ind].value;
            if (val != zero<ValueType>()) {
                tmp->get_values()[cur_ptr] = val;
                tmp->get_col_idxs()[cur_ptr] = data.nonzeros[ind].column;
                ++cur_ptr;
            }
        }
        tmp->get_row_ptrs()[row + 1] = cur_ptr;
    }
    tmp->move_to(this);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::write(mat_data &data) const
{
    std::unique_ptr<const LinOp> op{};
    const Csr *tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const Csr *>(op.get());
    } else {
        tmp = this;
    }

    data = {tmp->get_size(), {}};

    for (size_type row = 0; row < tmp->get_size()[0]; ++row) {
        const auto start = tmp->row_ptrs_.get_const_data()[row];
        const auto end = tmp->row_ptrs_.get_const_data()[row + 1];
        for (auto i = start; i < end; ++i) {
            const auto col = tmp->col_idxs_.get_const_data()[i];
            const auto val = tmp->values_.get_const_data()[i];
            data.nonzeros.emplace_back(row, col, val);
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Csr::create(exec, gko::transpose(this->get_size()),
                                 this->get_num_stored_elements());

    exec->run(
        TemplatedOperation<ValueType, IndexType>::make_transpose_operation(
            trans_cpy.get(), this));
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Csr::create(exec, gko::transpose(this->get_size()),
                                 this->get_num_stored_elements());

    exec->run(
        TemplatedOperation<ValueType, IndexType>::make_conj_transpose_operation(
            trans_cpy.get(), this));
    return std::move(trans_cpy);
}


#define DECLARE_CSR_MATRIX(ValueType, IndexType) class Csr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_CSR_MATRIX);
#undef DECLARE_CSR_MATRIX


}  // namespace matrix
}  // namespace gko
