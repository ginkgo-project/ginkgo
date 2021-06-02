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

#include <ginkgo/core/matrix/coo.hpp>


#include <algorithm>
#include <numeric>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/bccoo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/absolute_array.hpp"
#include "core/components/fill_array.hpp"
#include "core/matrix/coo_kernels.hpp"


namespace gko {
namespace matrix {


namespace coo {


GKO_REGISTER_OPERATION(spmv, coo::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, coo::advanced_spmv);
GKO_REGISTER_OPERATION(spmv2, coo::spmv2);
GKO_REGISTER_OPERATION(advanced_spmv2, coo::advanced_spmv2);
GKO_REGISTER_OPERATION(convert_to_bccoo, coo::convert_to_bccoo);
GKO_REGISTER_OPERATION(convert_to_csr, coo::convert_to_csr);
GKO_REGISTER_OPERATION(convert_to_dense, coo::convert_to_dense);
GKO_REGISTER_OPERATION(extract_diagonal, coo::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);
GKO_REGISTER_OPERATION(mem_size_bccoo, coo::mem_size_bccoo);


}  // namespace coo


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(coo::make_spmv(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta, LinOp *x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            this->get_executor()->run(coo::make_advanced_spmv(
                dense_alpha, this, dense_b, dense_beta, dense_x));
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply2_impl(const LinOp *b, LinOp *x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(coo::make_spmv2(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply2_impl(const LinOp *alpha, const LinOp *b,
                                            LinOp *x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_x) {
            this->get_executor()->run(
                coo::make_advanced_spmv2(dense_alpha, this, dense_b, dense_x));
        },
        alpha, b, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(
    Coo<next_precision<ValueType>, IndexType> *result) const
{
    result->values_ = this->values_;
    result->row_idxs_ = this->row_idxs_;
    result->col_idxs_ = this->col_idxs_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(
    Coo<next_precision<ValueType>, IndexType> *result)
{
    this->convert_to(result);
}


/*
template <typename ValueType, typename IndexType>
inline size_type mem_size_bccoo(std::shared_ptr<const ReferenceExecutor> exec,
                                const IndexType *row_idxs,
                                const IndexType *col_idxs,
                                const size_type num_rows,
                                IndexType rows, IndexType offsets,
                                const size_type block_size,
                                                                                                                                size_type *mem_size) // GKO_NOT_IMPLEMENTED;

{
    size_type num_stored_elements = row_idxs.size();
    size_type num_blocks = rows.size();
    size_type p = 0;
                offsets[0] = 0;
    for (size_type b = 0; b < num_blocks; b++) {
        size_type k = b * block_size;
        size_type r = row_idxs[k];
        size_type c = 0;
                                rows[b] = r;
        for (size_type l = 0; l < block_size && k < num_stored_elements; l++,
k++) { if (row_idxs[k] != r) { // new row r = row_idxs[k]; c = 0; p++;
            }
            size_type d = col_idxs[k] - c;
            if (d < 0x7d) {
                p++;
            } else if (d < 0xffff) {
                p += 3;
            } else {
                p += 5;
            }
            c = col_idxs[k];
        }
                                offsets[b+1] = p;
    }
    return p;
}
*/


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(
    Bccoo<ValueType, IndexType> *result) const  // GKO_NOT_IMPLEMENTED;
/* */
{
    auto exec = this->get_executor();
    auto num_stored_elements = this->get_num_stored_elements();

    // const auto block_size = 1024;
    const size_type block_size = 1024;
    //		const auto block_size = Bccoo<ValueType,
    // IndexType>::compute_block_size(
    // result->get_executor(), this.size(), num_stored_elements);
    const auto num_blocks = ceildiv(num_stored_elements, block_size);

    Array<IndexType> rows_(exec, num_blocks);
    Array<IndexType> offsets_(exec, num_blocks + 1);

    size_type mem_size{};
    if (exec == exec->get_master()) {
        exec->run(coo::make_mem_size_bccoo(
            //			mem_size_bccoo(
            this->get_const_row_idxs(), this->get_const_col_idxs(),
            //				this->get_size()[0], rows_, offsets_,
            // block_size, &mem_size);
            this->get_size()[0], rows_.get_data(), offsets_.get_data(),
            num_stored_elements, num_blocks, block_size, &mem_size));
    } else {
        auto host_coo = clone(exec->get_master(), this);
        exec->run(coo::make_mem_size_bccoo(
            host_coo->get_const_row_idxs(), host_coo->get_const_col_idxs(),
            host_coo->get_size()[0], rows_.get_data(), offsets_.get_data(),
            num_stored_elements, num_blocks, block_size, &mem_size));
    }

    Array<uint8> data_(exec, mem_size);

    auto tmp = Bccoo<ValueType, IndexType>::create(
        exec, this->get_size(), data_, offsets_, rows_, num_stored_elements,
        block_size);

    exec->run(coo::make_convert_to_bccoo(this, tmp.get()));
    tmp->move_to(result);
}
/* */

template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(Bccoo<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;
/*
{
    auto exec = this->get_executor();
    auto tmp = Csr<ValueType, IndexType>::create(
        exec, this->get_size(), this->get_num_stored_elements(),
        result->get_strategy());
    tmp->values_ = std::move(this->values_);
    tmp->col_idxs_ = std::move(this->col_idxs_);
    exec->run(coo::make_convert_to_csr(this, tmp.get()));
    tmp->make_srow();
    tmp->move_to(result);
}
*/

template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Csr<ValueType, IndexType>::create(
        exec, this->get_size(), this->get_num_stored_elements(),
        result->get_strategy());
    tmp->values_ = this->values_;
    tmp->col_idxs_ = this->col_idxs_;
    exec->run(coo::make_convert_to_csr(this, tmp.get()));
    tmp->make_srow();
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(Csr<ValueType, IndexType> *result)
{
    auto exec = this->get_executor();
    auto tmp = Csr<ValueType, IndexType>::create(
        exec, this->get_size(), this->get_num_stored_elements(),
        result->get_strategy());
    tmp->values_ = std::move(this->values_);
    tmp->col_idxs_ = std::move(this->col_idxs_);
    exec->run(coo::make_convert_to_csr(this, tmp.get()));
    tmp->make_srow();
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(coo::make_convert_to_dense(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(Dense<ValueType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::read(const mat_data &data)
{
    size_type nnz = 0;
    for (const auto &elem : data.nonzeros) {
        nnz += (elem.value != zero<ValueType>());
    }
    auto tmp = Coo::create(this->get_executor()->get_master(), data.size, nnz);
    size_type elt = 0;
    for (const auto &elem : data.nonzeros) {
        auto val = elem.value;
        if (val != zero<ValueType>()) {
            tmp->get_row_idxs()[elt] = elem.row;
            tmp->get_col_idxs()[elt] = elem.column;
            tmp->get_values()[elt] = elem.value;
            elt++;
        }
    }
    this->copy_from(std::move(tmp));
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::write(mat_data &data) const
{
    std::unique_ptr<const LinOp> op{};
    const Coo *tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const Coo *>(op.get());
    } else {
        tmp = this;
    }

    data = {this->get_size(), {}};

    for (size_type i = 0; i < tmp->get_num_stored_elements(); ++i) {
        const auto row = tmp->row_idxs_.get_const_data()[i];
        const auto col = tmp->col_idxs_.get_const_data()[i];
        const auto val = tmp->values_.get_const_data()[i];
        data.nonzeros.emplace_back(row, col, val);
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Coo<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(coo::make_fill_array(diag->get_values(), diag->get_size()[0],
                                   zero<ValueType>()));
    exec->run(coo::make_extract_diagonal(this, lend(diag)));
    return diag;
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(coo::make_inplace_absolute_array(
        this->get_values(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Coo<ValueType, IndexType>::absolute_type>
Coo<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_coo = absolute_type::create(exec, this->get_size(),
                                         this->get_num_stored_elements());

    abs_coo->col_idxs_ = col_idxs_;
    abs_coo->row_idxs_ = row_idxs_;
    exec->run(coo::make_outplace_absolute_array(this->get_const_values(),
                                                this->get_num_stored_elements(),
                                                abs_coo->get_values()));

    return abs_coo;
}


#define GKO_DECLARE_COO_MATRIX(ValueType, IndexType) \
    class Coo<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_MATRIX);


}  // namespace matrix
}  // namespace gko
