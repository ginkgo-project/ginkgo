/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/sparsity_csr_kernels.hpp"


namespace gko {
namespace matrix {
namespace sparsity_csr {


GKO_REGISTER_OPERATION(spmv, sparsity_csr::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, sparsity_csr::advanced_spmv);
GKO_REGISTER_OPERATION(transpose, sparsity_csr::transpose);
GKO_REGISTER_OPERATION(count_num_diagonal_elements,
                       sparsity_csr::count_num_diagonal_elements);
GKO_REGISTER_OPERATION(remove_diagonal_elements,
                       sparsity_csr::remove_diagonal_elements);
GKO_REGISTER_OPERATION(sort_by_column_index,
                       sparsity_csr::sort_by_column_index);
GKO_REGISTER_OPERATION(is_sorted_by_column_index,
                       sparsity_csr::is_sorted_by_column_index);


}  // namespace sparsity_csr


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::apply_impl(const LinOp *b,
                                                   LinOp *x) const
{
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        sparsity_csr::make_spmv(this, as<Dense>(b), as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::apply_impl(const LinOp *alpha,
                                                   const LinOp *b,
                                                   const LinOp *beta,
                                                   LinOp *x) const
{
    using Dense = Dense<ValueType>;
    this->get_executor()->run(sparsity_csr::make_advanced_spmv(
        as<Dense>(alpha), this, as<Dense>(b), as<Dense>(beta), as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::read(const mat_data &data)
{
    size_type nnz = 0;
    for (const auto &elem : data.nonzeros) {
        nnz += (elem.value != zero<ValueType>());
    }
    auto tmp =
        SparsityCsr::create(this->get_executor()->get_master(), data.size, nnz);
    size_type ind = 0;
    size_type cur_ptr = 0;
    tmp->get_row_ptrs()[0] = cur_ptr;
    tmp->get_value()[0] = one<ValueType>();
    for (size_type row = 0; row < data.size[0]; ++row) {
        for (; ind < data.nonzeros.size(); ++ind) {
            if (data.nonzeros[ind].row > row) {
                break;
            }
            auto val = data.nonzeros[ind].value;
            if (val != zero<ValueType>()) {
                tmp->get_col_idxs()[cur_ptr] = data.nonzeros[ind].column;
                ++cur_ptr;
            }
        }
        tmp->get_row_ptrs()[row + 1] = cur_ptr;
    }
    tmp->move_to(this);
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::write(mat_data &data) const
{
    std::unique_ptr<const LinOp> op{};
    const SparsityCsr *tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const SparsityCsr *>(op.get());
    } else {
        tmp = this;
    }

    data = {tmp->get_size(), {}};

    const auto val = tmp->value_.get_const_data()[0];
    for (size_type row = 0; row < tmp->get_size()[0]; ++row) {
        const auto start = tmp->row_ptrs_.get_const_data()[row];
        const auto end = tmp->row_ptrs_.get_const_data()[row + 1];
        for (auto i = start; i < end; ++i) {
            const auto col = tmp->col_idxs_.get_const_data()[i];
            data.nonzeros.emplace_back(row, col, val);
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> SparsityCsr<ValueType, IndexType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = SparsityCsr::create(exec, gko::transpose(this->get_size()),
                                         this->get_num_nonzeros());

    exec->run(sparsity_csr::make_transpose(trans_cpy.get(), this));
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> SparsityCsr<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<SparsityCsr<ValueType, IndexType>>
SparsityCsr<ValueType, IndexType>::to_adjacency_matrix() const
{
    auto exec = this->get_executor();
    // Adjacency matrix has to be square.
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    size_type num_diagonal_elements = 0;
    exec->run(sparsity_csr::make_count_num_diagonal_elements(
        this, &num_diagonal_elements));
    auto adj_mat =
        SparsityCsr::create(exec, this->get_size(),
                            this->get_num_nonzeros() - num_diagonal_elements);

    exec->run(sparsity_csr::make_remove_diagonal_elements(
        adj_mat.get(), this->get_const_row_ptrs(), this->get_const_col_idxs()));
    return std::move(adj_mat);
}


template <typename ValueType, typename IndexType>
void SparsityCsr<ValueType, IndexType>::sort_by_column_index()
{
    auto exec = this->get_executor();
    exec->run(sparsity_csr::make_sort_by_column_index(this));
}


template <typename ValueType, typename IndexType>
bool SparsityCsr<ValueType, IndexType>::is_sorted_by_column_index() const
{
    auto exec = this->get_executor();
    bool is_sorted;
    exec->run(sparsity_csr::make_is_sorted_by_column_index(this, &is_sorted));
    return is_sorted;
}


#define GKO_DECLARE_SPARSITY_MATRIX(ValueType, IndexType) \
    class SparsityCsr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSITY_MATRIX);


}  // namespace matrix
}  // namespace gko
