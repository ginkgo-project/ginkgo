/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/matrix/fbcsr.hpp>


#include <limits>
#include <map>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/absolute_array.hpp"
#include "core/components/fill_array.hpp"
#include "core/components/fixed_block.hpp"
#include "core/matrix/fbcsr_kernels.hpp"


namespace gko {
namespace matrix {
namespace fbcsr {


GKO_REGISTER_OPERATION(spmv, fbcsr::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, fbcsr::advanced_spmv);
GKO_REGISTER_OPERATION(convert_to_csr, fbcsr::convert_to_csr);
GKO_REGISTER_OPERATION(convert_to_dense, fbcsr::convert_to_dense);
GKO_REGISTER_OPERATION(transpose, fbcsr::transpose);
GKO_REGISTER_OPERATION(conj_transpose, fbcsr::conj_transpose);
GKO_REGISTER_OPERATION(calculate_max_nnz_per_row,
                       fbcsr::calculate_max_nnz_per_row);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row,
                       fbcsr::calculate_nonzeros_per_row);
GKO_REGISTER_OPERATION(is_sorted_by_column_index,
                       fbcsr::is_sorted_by_column_index);
GKO_REGISTER_OPERATION(extract_diagonal, fbcsr::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // namespace fbcsr


template <typename ValueType, typename IndexType>
Fbcsr<ValueType, IndexType>::Fbcsr(const std::shared_ptr<const Executor> exec,
                                   const dim<2> &size,
                                   const size_type num_nonzeros,
                                   const int block_size)
    : EnableLinOp<Fbcsr>(exec, size),
      bs_{block_size},
      nbcols_{blockutils::getNumBlocks(block_size, size[1])},
      values_(exec, num_nonzeros),
      col_idxs_(exec, blockutils::getNumBlocks(block_size * block_size,
                                               num_nonzeros)),
      row_ptrs_(exec, blockutils::getNumBlocks(block_size, size[0]) + 1)
{}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::apply_impl(const LinOp *const b,
                                             LinOp *const x) const
{
    using Dense = Dense<ValueType>;
    if (auto b_fbcsr = dynamic_cast<const Fbcsr<ValueType, IndexType> *>(b)) {
        // if b is a FBCSR matrix, we compute a SpGeMM
        throw /*::gko::*/ NotImplemented(__FILE__, __LINE__,
                                         "SpGeMM for Fbcsr");
    } else {
        // otherwise we assume that b is dense and compute a SpMV/SpMM
        this->get_executor()->run(
            fbcsr::make_spmv(this, as<Dense>(b), as<Dense>(x)));
    }
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::apply_impl(const LinOp *const alpha,
                                             const LinOp *const b,
                                             const LinOp *const beta,
                                             LinOp *const x) const
{
    using Dense = Dense<ValueType>;
    if (auto b_fbcsr = dynamic_cast<const Fbcsr<ValueType, IndexType> *>(b)) {
        // if b is a FBCSR matrix, we compute a SpGeMM
        throw NotImplemented(__FILE__, __LINE__, "Adv SpGeMM for Fbcsr");
    } else if (dynamic_cast<const Identity<ValueType> *>(b)) {
        // if b is an identity matrix, we compute an SpGEAM
        throw NotImplemented(__FILE__, __LINE__, "Adv SpGEAM for Fbcsr");
    } else {
        // otherwise we assume that b is dense and compute a SpMV/SpMM
        this->get_executor()->run(
            fbcsr::make_advanced_spmv(as<Dense>(alpha), this, as<Dense>(b),
                                      as<Dense>(beta), as<Dense>(x)));
    }
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::convert_to(
    Fbcsr<ValueType, IndexType> *const result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    result->set_size(this->get_size());
    result->bs_ = this->bs_;
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::move_to(
    Fbcsr<ValueType, IndexType> *const result)
{
    EnableLinOp<Fbcsr>::move_to(result);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::convert_to(
    Fbcsr<next_precision<ValueType>, IndexType> *const result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    result->set_size(this->get_size());
    result->bs_ = this->bs_;
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::move_to(
    Fbcsr<next_precision<ValueType>, IndexType> *const result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::convert_to(
    Dense<ValueType> *const result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(fbcsr::make_convert_to_dense(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::move_to(Dense<ValueType> *const result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType> *const result) const
{
    auto exec = this->get_executor();
    auto tmp = Csr<ValueType, IndexType>::create(
        exec, this->get_size(), this->get_num_stored_elements(),
        result->get_strategy());
    exec->run(fbcsr::make_convert_to_csr(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::move_to(
    Csr<ValueType, IndexType> *const result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::convert_to(
    SparsityCsr<ValueType, IndexType> *const result) const
{
    using blockutils::getNumBlocks;
    auto exec = this->get_executor();
    auto tmp = SparsityCsr<ValueType, IndexType>::create(
        exec,
        gko::dim<2>{getNumBlocks(bs_, this->get_size()[0]),
                    getNumBlocks(bs_, this->get_size()[1])},
        getNumBlocks(bs_ * bs_, this->get_num_stored_elements()));

    tmp->col_idxs_ = this->col_idxs_;
    tmp->row_ptrs_ = this->row_ptrs_;
    tmp->value_ = Array<ValueType>(exec, {one<ValueType>()});
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::move_to(
    SparsityCsr<ValueType, IndexType> *const result)
{
    this->convert_to(result);
}


/* Within blocks, the storage order is row-major.
 * Currently, this implementation is sequential and has complexity O(n log n)
 * assuming nnz = O(n).
 * Can this be changed to a parallel O(n) implementation?
 */
template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::read(const mat_data &data)
{
    if (data.nonzeros.size() > std::numeric_limits<index_type>::max())
        throw std::range_error(std::string("file: ") + __FILE__ + ":" +
                               std::to_string(__LINE__) +
                               ": List of nonzeros is too big!");

    const index_type nnz = static_cast<index_type>(data.nonzeros.size());

    const int bs = this->bs_;

    using Blk_t = blockutils::DenseBlock<value_type>;

    struct FbEntry {
        index_type block_row;
        index_type block_column;
    };

    struct FbLess {
        bool operator()(const FbEntry &a, const FbEntry &b) const
        {
            if (a.block_row != b.block_row)
                return a.block_row < b.block_row;
            else
                return a.block_column < b.block_column;
        }
    };

    auto create_block_map = [nnz, bs](const mat_data &data) {
        std::map<FbEntry, Blk_t, FbLess> blocks;
        for (index_type inz = 0; inz < nnz; inz++) {
            const index_type row = data.nonzeros[inz].row;
            const index_type col = data.nonzeros[inz].column;
            const value_type val = data.nonzeros[inz].value;

            const int localrow = static_cast<int>(row % bs);
            const int localcol = static_cast<int>(col % bs);
            const index_type blockrow = row / bs;
            const index_type blockcol = col / bs;

            Blk_t &nnzblk = blocks[{blockrow, blockcol}];
            if (nnzblk.size() == 0) {
                nnzblk.resize(bs, bs);
                nnzblk.zero();
                nnzblk(localrow, localcol) = val;
            } else {
                if (nnzblk(localrow, localcol) != gko::zero<value_type>())
                    throw Error(__FILE__, __LINE__,
                                "Error: re-visited the same non-zero!");
                nnzblk(localrow, localcol) = val;
            }
        }
        return blocks;
    };

    const std::map<FbEntry, Blk_t, FbLess> blocks = create_block_map(data);

    auto tmp = Fbcsr::create(this->get_executor()->get_master(), data.size,
                             blocks.size() * bs * bs, bs);

    tmp->row_ptrs_.get_data()[0] = 0;
    index_type cur_brow = 0, cur_bnz = 0,
               cur_bcol = blocks.begin()->first.block_column;
    const index_type num_brows = data.size[0] / bs;

    blockutils::DenseBlocksView<value_type, index_type> values(
        tmp->values_.get_data(), bs, bs);

    for (auto it = blocks.begin(); it != blocks.end(); it++) {
        GKO_ENSURE_IN_BOUNDS(cur_brow, num_brows);

        tmp->col_idxs_.get_data()[cur_bnz] = it->first.block_column;

        for (int ibr = 0; ibr < bs; ibr++)
            for (int jbr = 0; jbr < bs; jbr++)
                values(cur_bnz, ibr, jbr) = it->second(ibr, jbr);

        if (it->first.block_row > cur_brow) {
            tmp->row_ptrs_.get_data()[++cur_brow] = cur_bnz;
        } else {
            assert(cur_brow == it->first.block_row);
            assert(cur_bcol <= it->first.block_column);
        }

        cur_bcol = it->first.block_column;
        cur_bnz++;
    }

    tmp->row_ptrs_.get_data()[++cur_brow] =
        static_cast<index_type>(blocks.size());

    assert(cur_brow == tmp->get_size()[0] / bs);

    tmp->move_to(this);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::write(mat_data &data) const
{
    std::unique_ptr<const LinOp> op{};
    const Fbcsr *tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const Fbcsr *>(op.get());
    } else {
        tmp = this;
    }

    data = {tmp->get_size(), {}};

    const blockutils::DenseBlocksView<const value_type, index_type> vblocks(
        tmp->values_.get_const_data(), bs_, bs_);

    for (size_type brow = 0; brow < tmp->get_size()[0] / bs_; ++brow) {
        const auto start = tmp->row_ptrs_.get_const_data()[brow];
        const auto end = tmp->row_ptrs_.get_const_data()[brow + 1];

        for (auto inz = start; inz < end; ++inz) {
            for (int ib = 0; ib < bs_; ib++) {
                const auto row = brow * bs_ + ib;
                for (int jb = 0; jb < bs_; jb++) {
                    const auto col =
                        tmp->col_idxs_.get_const_data()[inz] * bs_ + jb;
                    const auto val = vblocks(inz, ib, jb);
                    data.nonzeros.emplace_back(row, col, val);
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Fbcsr<ValueType, IndexType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Fbcsr::create(exec, gko::transpose(this->get_size()),
                                   this->get_num_stored_elements(), bs_);

    exec->run(fbcsr::make_transpose(this, trans_cpy.get()));
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Fbcsr<ValueType, IndexType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Fbcsr::create(exec, gko::transpose(this->get_size()),
                                   this->get_num_stored_elements(), bs_);

    exec->run(fbcsr::make_conj_transpose(this, trans_cpy.get()));
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::sort_by_column_index() GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
bool Fbcsr<ValueType, IndexType>::is_sorted_by_column_index() const
{
    auto exec = this->get_executor();
    bool is_sorted;
    exec->run(fbcsr::make_is_sorted_by_column_index(this, &is_sorted));
    return is_sorted;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Fbcsr<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(fbcsr::make_fill_array(diag->get_values(), diag->get_size()[0],
                                     zero<ValueType>()));
    exec->run(fbcsr::make_extract_diagonal(this, lend(diag)));
    return diag;
}


template <typename ValueType, typename IndexType>
void Fbcsr<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(fbcsr::make_inplace_absolute_array(
        this->get_values(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Fbcsr<ValueType, IndexType>::absolute_type>
Fbcsr<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_fbcsr = absolute_type::create(exec, this->get_size(),
                                           this->get_num_stored_elements(),
                                           this->get_block_size());

    abs_fbcsr->col_idxs_ = col_idxs_;
    abs_fbcsr->row_ptrs_ = row_ptrs_;
    exec->run(fbcsr::make_outplace_absolute_array(
        this->get_const_values(), this->get_num_stored_elements(),
        abs_fbcsr->get_values()));

    return abs_fbcsr;
}


#define GKO_DECLARE_FBCSR_MATRIX(ValueType, IndexType) \
    class Fbcsr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_MATRIX);


}  // namespace matrix
}  // namespace gko
