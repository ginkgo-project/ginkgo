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

#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/block_matrix.hpp>
#include <ginkgo/core/matrix/dense.hpp>

namespace gko {
namespace matrix {

namespace {


std::unique_ptr<BlockMatrix> block_vector_view(LinOp* v,
                                               std::vector<span> block_sizes)
{
    auto v_submatrix = dynamic_cast<SubmatrixViewCreateable*>(v);
    std::vector<std::vector<std::shared_ptr<LinOp>>> sub_blocks;
    for (const auto& rows : block_sizes) {
        sub_blocks.emplace_back(
            1, v_submatrix->create_submatrix(rows, {0, v->get_size()[1]}));
    }
    return BlockMatrix::create(v->get_executor(), v->get_size(), sub_blocks);
}

std::unique_ptr<BlockMatrix, std::function<void(BlockMatrix*)>> block_vector(
    LinOp* v, const std::vector<span>& blocks)
{
    if (auto p = dynamic_cast<BlockMatrix*>(v)) {
        return {p, [](auto ptr) {}};
    } else if (dynamic_cast<SubmatrixViewCreateable*>(v)) {
        return block_vector_view(v, blocks);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename SubmatrixCreateableType>
std::vector<std::vector<std::shared_ptr<LinOp>>> create_sub_blocks(
    SubmatrixCreateableType* monolith, const std::vector<gko::span>& blocks)
{
    std::vector<std::vector<std::shared_ptr<LinOp>>> sub_blocks;
    for (const auto& rows : blocks) {
        std::vector<std::shared_ptr<LinOp>> linop_row;
        for (const auto& cols : blocks) {
            linop_row.push_back(monolith->create_submatrix(rows, cols));
        }
        sub_blocks.emplace_back(std::move(linop_row));
    }
    return sub_blocks;
}

}  // namespace


BlockMatrix::BlockMatrix(
    std::shared_ptr<const Executor> exec, const dim<2> size,
    const std::vector<std::vector<std::shared_ptr<LinOp>>>& blocks)
    : EnableLinOp<BlockMatrix>(exec, size),
      block_size_(blocks.size(), begin(blocks)->size()),
      spans_(),
      blocks_(blocks.size())
{
    for (size_t row = 0; row < blocks.size(); ++row) {
        blocks_[row] = std::vector<std::shared_ptr<LinOp>>(begin(blocks)[row]);
        auto span_begin = row > 0 ? spans_[row - 1].end : 0;
        auto span_size =
            blocks[row].empty() ? 0 : blocks[row][0]->get_size()[0];
        spans_.emplace_back(span_begin, span_begin + span_size);
        GKO_ASSERT_EQ(block_size_[1], blocks_[row].size());
    }
}


BlockMatrix::BlockMatrix(std::shared_ptr<const Executor> exec,
                         SubmatrixCreateable* monolithic_op,
                         const std::vector<gko::span>& blocks)
    : EnableLinOp<BlockMatrix>(exec, as<LinOp>(monolithic_op)->get_size()),
      block_size_(blocks.size(), blocks.size()),
      spans_(blocks),
      blocks_(create_sub_blocks(monolithic_op, spans_))
{}


BlockMatrix::BlockMatrix(std::shared_ptr<const Executor> exec,
                         SubmatrixViewCreateable* monolithic_op,
                         const std::vector<gko::span>& blocks)
    : EnableLinOp<BlockMatrix>(exec, as<LinOp>(monolithic_op)->get_size()),
      block_size_(blocks.size(), blocks.size()),
      spans_(blocks),
      blocks_(create_sub_blocks(monolithic_op, spans_))
{}


void BlockMatrix::apply_impl(const LinOp* b, LinOp* x) const
{
    auto block_b = block_vector(const_cast<LinOp*>(b), spans_);
    auto block_x = block_vector(x, spans_);

    auto one = gko::initialize<Dense<double>>({1}, this->get_executor());
    auto zero = gko::initialize<Dense<double>>({0}, this->get_executor());

    for (size_t block_row = 0; block_row < block_size_[0]; ++block_row) {
        for (size_t block_col = 0; block_col < block_size_[1]; ++block_col) {
            if (block_col == 0) {
                blocks_[block_row][block_col]->apply(
                    block_b->blocks_[block_col][0].get(),
                    block_x->blocks_[block_row][0].get());
            } else {
                blocks_[block_row][block_col]->apply(
                    one.get(), block_b->blocks_[block_col][0].get(), one.get(),
                    block_x->blocks_[block_row][0].get());
            }
        }
    }
}


void BlockMatrix::apply_impl(const LinOp* alpha, const LinOp* b,
                             const LinOp* beta, LinOp* x) const
{
    auto block_b = block_vector(const_cast<LinOp*>(b), spans_);
    auto block_x = block_vector(x, spans_);

    auto one = gko::initialize<Dense<double>>({1}, this->get_executor());
    auto zero = gko::initialize<Dense<double>>({0}, this->get_executor());

    for (size_t block_row = 0; block_row < block_size_[0]; ++block_row) {
        for (size_t block_col = 0; block_col < block_size_[1]; ++block_col) {
            if (block_col == 0) {
                blocks_[block_row][block_col]->apply(
                    alpha, block_b->blocks_[block_col][0].get(), beta,
                    block_x->blocks_[block_row][0].get());
            } else {
                blocks_[block_row][block_col]->apply(
                    alpha, block_b->blocks_[block_col][0].get(), one.get(),
                    block_x->blocks_[block_row][0].get());
            }
        }
    }
}


}  // namespace matrix
}  // namespace gko
