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

#include <ginkgo/core/matrix/bccoo.hpp>


#include <algorithm>
#include <numeric>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/bccoo_kernels.hpp"


namespace gko {
namespace matrix {


namespace bccoo {


GKO_REGISTER_OPERATION(spmv, bccoo::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, bccoo::advanced_spmv);
GKO_REGISTER_OPERATION(spmv2, bccoo::spmv2);
GKO_REGISTER_OPERATION(advanced_spmv2, bccoo::advanced_spmv2);
GKO_REGISTER_OPERATION(convert_to_coo, bccoo::convert_to_coo);
GKO_REGISTER_OPERATION(convert_to_csr, bccoo::convert_to_csr);
GKO_REGISTER_OPERATION(convert_to_dense, bccoo::convert_to_dense);
GKO_REGISTER_OPERATION(extract_diagonal, bccoo::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // namespace bccoo


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    precision_dispatch_real_complex<ValueType>(
//        [this](auto dense_b, auto dense_x) {
//            this->get_executor()->run(bccoo::make_spmv(this, dense_b,
//            dense_x));
//        },
//        b, x);
//}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                             const LinOp* beta, LinOp* x) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    precision_dispatch_real_complex<ValueType>(
//        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x)
//        {
//            this->get_executor()->run(bccoo::make_advanced_spmv(
//                dense_alpha, this, dense_b, dense_beta, dense_x));
//        },
//        alpha, b, beta, x);
//}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::apply2_impl(const LinOp* b, LinOp* x) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    precision_dispatch_real_complex<ValueType>(
//        [this](auto dense_b, auto dense_x) {
//            this->get_executor()->run(bccoo::make_spmv2(this, dense_b,
//            dense_x));
//        },
//        b, x);
//}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::apply2_impl(
    const LinOp* alpha, const LinOp* b, LinOp* x) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    precision_dispatch_real_complex<ValueType>(
//        [this](auto dense_alpha, auto dense_b, auto dense_x) {
//            this->get_executor()->run(
//                bccoo::make_advanced_spmv2(dense_alpha, this, dense_b,
//                dense_x));
//        },
//        alpha, b, x);
//}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::convert_to(
    Bccoo<next_precision<ValueType>, IndexType>* result) const
// GKO_NOT_IMPLEMENTED;
/* */
{
    // TODO (script:bccoo): change the code imported from matrix/coo if needed
    result->data_ = this->data_;
    result->rows_ = this->rows_;
    result->offsets_ = this->offsets_;
    result->set_size(this->get_size());
}
/* */


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::move_to(
    Bccoo<next_precision<ValueType>, IndexType>* result) GKO_NOT_IMPLEMENTED;
/*
{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
    this->convert_to(result);
}
*/


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::convert_to(
    Coo<ValueType, IndexType>* result) const GKO_NOT_IMPLEMENTED;
/*
{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
    auto exec = this->get_executor();
    auto tmp = Coo<ValueType, IndexType>::create(
        exec, this->get_size(), this->get_num_stored_elements());
    tmp->values_ = this->values_;
    tmp->col_idxs_ = this->col_idxs_;
    exec->run(bccoo::make_convert_to_coo(this, tmp.get()));
    tmp->make_srow();
    tmp->move_to(result);
}
*/


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::move_to(Coo<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;
/*
{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
    auto exec = this->get_executor();
    auto tmp = Coo<ValueType, IndexType>::create(
        exec, this->get_size(), this->get_num_stored_elements(),
        result->get_strategy());
    tmp->values_ = std::move(this->values_);
    tmp->col_idxs_ = std::move(this->col_idxs_);
    exec->run(bccoo::make_convert_to_coo(this, tmp.get()));
    tmp->make_srow();
    tmp->move_to(result);
}
*/


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType>* result) const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    auto exec = this->get_executor();
//    auto tmp = Csr<ValueType, IndexType>::create(
//        exec, this->get_size(), this->get_num_stored_elements(),
//        result->get_strategy());
//    tmp->values_ = this->values_;
//    tmp->col_idxs_ = this->col_idxs_;
//    exec->run(bccoo::make_convert_to_csr(this, tmp.get()));
//    tmp->make_srow();
//    tmp->move_to(result);
//}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::move_to(Csr<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    auto exec = this->get_executor();
//    auto tmp = Csr<ValueType, IndexType>::create(
//        exec, this->get_size(), this->get_num_stored_elements(),
//        result->get_strategy());
//    tmp->values_ = std::move(this->values_);
//    tmp->col_idxs_ = std::move(this->col_idxs_);
//    exec->run(bccoo::make_convert_to_csr(this, tmp.get()));
//    tmp->make_srow();
//    tmp->move_to(result);
//}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    auto exec = this->get_executor();
//    auto tmp = Dense<ValueType>::create(exec, this->get_size());
//    exec->run(bccoo::make_convert_to_dense(this, tmp.get()));
//    tmp->move_to(result);
//}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::move_to(Dense<ValueType>* result)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    this->convert_to(result);
//}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::read(const mat_data& data)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    size_type nnz = 0;
//    for (const auto &elem : data.nonzeros) {
//        nnz += (elem.value != zero<ValueType>());
//    }
//    auto tmp = Bccoo::create(this->get_executor()->get_master(), data.size,
//    nnz); size_type elt = 0; for (const auto &elem : data.nonzeros) {
//        auto val = elem.value;
//        if (val != zero<ValueType>()) {
//            tmp->get_row_idxs()[elt] = elem.row;
//            tmp->get_col_idxs()[elt] = elem.column;
//            tmp->get_values()[elt] = elem.value;
//            elt++;
//        }
//    }
//    this->copy_from(std::move(tmp));
//}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::write(mat_data& data) const
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    std::unique_ptr<const LinOp> op{};
//    const Bccoo *tmp{};
//    if (this->get_executor()->get_master() != this->get_executor()) {
//        op = this->clone(this->get_executor()->get_master());
//        tmp = static_cast<const Bccoo *>(op.get());
//    } else {
//        tmp = this;
//    }
//
//    data = {this->get_size(), {}};
//
//    for (size_type i = 0; i < tmp->get_num_stored_elements(); ++i) {
//        const auto row = tmp->row_idxs_.get_const_data()[i];
//        const auto col = tmp->col_idxs_.get_const_data()[i];
//        const auto val = tmp->values_.get_const_data()[i];
//        data.nonzeros.emplace_back(row, col, val);
//    }
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Bccoo<ValueType, IndexType>::extract_diagonal() const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    auto exec = this->get_executor();
//
//    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
//    auto diag = Diagonal<ValueType>::create(exec, diag_size);
//    exec->run(bccoo::make_fill_array(diag->get_values(), diag->get_size()[0],
//                                   zero<ValueType>()));
//    exec->run(bccoo::make_extract_diagonal(this, lend(diag)));
//    return diag;
//}


template <typename ValueType, typename IndexType>
void Bccoo<ValueType, IndexType>::compute_absolute_inplace()
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    auto exec = this->get_executor();
//
//    exec->run(bccoo::make_inplace_absolute_array(
//        this->get_values(), this->get_num_stored_elements()));
//}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Bccoo<ValueType, IndexType>::absolute_type>
Bccoo<ValueType, IndexType>::compute_absolute() const GKO_NOT_IMPLEMENTED;
//{
// TODO (script:bccoo): change the code imported from matrix/coo if needed
//    auto exec = this->get_executor();
//
//    auto abs_bccoo = absolute_type::create(exec, this->get_size(),
//                                         this->get_num_stored_elements());
//
//    abs_bccoo->col_idxs_ = col_idxs_;
//    abs_bccoo->row_idxs_ = row_idxs_;
//    exec->run(bccoo::make_outplace_absolute_array(this->get_const_values(),
//                                                this->get_num_stored_elements(),
//                                                abs_bccoo->get_values()));
//
//    return abs_bccoo;
//}


#define GKO_DECLARE_BCCOO_MATRIX(ValueType, IndexType) \
    class Bccoo<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_MATRIX);


}  // namespace matrix
}  // namespace gko
