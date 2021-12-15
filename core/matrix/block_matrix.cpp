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
#include <utility>

namespace gko {
namespace matrix {

namespace {


template <typename ValueType, typename IndexType>
std::unique_ptr<BlockMatrix> block_vector_view(
    distributed::Vector<ValueType, IndexType>* v,
    const std::vector<span>& block_sizes,
    const std::vector<dim<2>>& global_sizes)
{
    auto exec = v->get_executor();

    span columns{0, v->get_size()[1]};

    std::vector<std::vector<std::shared_ptr<LinOp>>> sub_blocks;

    auto global_offset =
        v->get_partition()->get_global_offset(v->get_communicator()->rank());
    for (size_type i = 0; i < block_sizes.size(); ++i) {
        const auto& rows = block_sizes[i];
        gko::span local_rows{rows.begin - global_offset,
                             rows.end - global_offset};

        auto sub_vector = v->get_local()->create_submatrix(local_rows, columns);

        auto sub = gko::share(distributed::Vector<ValueType, IndexType>::create(
            exec, v->get_communicator(), v->get_partition(), global_sizes[i],
            dim<2>{rows.length(), columns.length()}));
        *(sub->get_local()) = std::move(*sub_vector.get());
        sub_blocks.emplace_back(1, std::move(sub));
    }
    return BlockMatrix::create(v->get_executor(), v->get_size(),
                               std::move(sub_blocks), block_sizes);
}


std::unique_ptr<BlockMatrix> block_vector_view(
    LinOp* v, const std::vector<span>& block_sizes)
{
    auto v_submatrix = dynamic_cast<SubmatrixViewCreateable*>(v);
    std::vector<std::vector<std::shared_ptr<LinOp>>> sub_blocks;
    for (const auto& rows : block_sizes) {
        sub_blocks.emplace_back(
            1, v_submatrix->create_submatrix(rows, {0, v->get_size()[1]}));
    }
    return BlockMatrix::create(v->get_executor(), v->get_size(),
                               std::move(sub_blocks), block_sizes);
}


std::unique_ptr<BlockMatrix, std::function<void(BlockMatrix*)>> block_vector(
    LinOp* v, const std::vector<span>& blocks,
    const std::vector<dim<2>>& global_sizes)
{
    if (auto p = dynamic_cast<BlockMatrix*>(v)) {
        return {p, [](auto ptr) {}};
    } else if (dynamic_cast<distributed::DistributedBase*>(v)) {
        if (auto p = dynamic_cast<distributed::Vector<float, int32>*>(v)) {
            return block_vector_view(p, blocks, global_sizes);
        } else if (auto p =
                       dynamic_cast<distributed::Vector<float, int64>*>(v)) {
            return block_vector_view(p, blocks, global_sizes);
        } else if (auto p =
                       dynamic_cast<distributed::Vector<double, int32>*>(v)) {
            return block_vector_view(p, blocks, global_sizes);
        } else if (auto p =
                       dynamic_cast<distributed::Vector<double, int64>*>(v)) {
            return block_vector_view(p, blocks, global_sizes);
        } else if (auto p = dynamic_cast<
                       distributed::Vector<std::complex<float>, int32>*>(v)) {
            return block_vector_view(p, blocks, global_sizes);
        } else if (auto p = dynamic_cast<
                       distributed::Vector<std::complex<float>, int64>*>(v)) {
            return block_vector_view(p, blocks, global_sizes);
        } else if (auto p = dynamic_cast<
                       distributed::Vector<std::complex<double>, int32>*>(v)) {
            return block_vector_view(p, blocks, global_sizes);
        } else if (auto p = dynamic_cast<
                       distributed::Vector<std::complex<double>, int64>*>(v)) {
            return block_vector_view(p, blocks, global_sizes);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
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
    std::vector<std::vector<std::shared_ptr<LinOp>>> blocks,
    std::vector<gko::span> block_spans)
    : EnableLinOp<BlockMatrix>(exec, size),
      block_size_(blocks.size(), begin(blocks)->size()),
      spans_(std::move(block_spans)),
      blocks_(std::move(blocks))
{}


BlockMatrix::BlockMatrix(std::shared_ptr<const Executor> exec,
                         SubmatrixCreateable* monolithic_op,
                         std::vector<gko::span> blocks)
    : BlockMatrix(std::move(exec),
                  dynamic_cast<LinOp*>(monolithic_op)->get_size(),
                  create_sub_blocks(monolithic_op, blocks), std::move(blocks))
{}


BlockMatrix::BlockMatrix(std::shared_ptr<const Executor> exec,
                         SubmatrixViewCreateable* monolithic_op,
                         std::vector<gko::span> blocks)
    : BlockMatrix(std::move(exec),
                  dynamic_cast<LinOp*>(monolithic_op)->get_size(),
                  create_sub_blocks(monolithic_op, blocks), std::move(blocks))
{}


void BlockMatrix::apply_impl(const LinOp* b, LinOp* x) const
{
    std::vector<dim<2>> vector_sizes(blocks_.size());
    for (size_t block_row = 0; block_row < block_size_[0]; ++block_row) {
        vector_sizes[block_row] = {blocks_[block_row][0]->get_size()[0],
                                   b->get_size()[1]};
    }

    auto block_b = block_vector(const_cast<LinOp*>(b), spans_, vector_sizes);
    auto block_x = block_vector(x, spans_, vector_sizes);

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
    std::vector<dim<2>> vector_sizes(blocks_.size());
    for (size_t block_row = 0; block_row < block_size_[0]; ++block_row) {
        vector_sizes[block_row] = {blocks_[block_row][0]->get_size()[0],
                                   b->get_size()[1]};
    }

    auto block_b = block_vector(const_cast<LinOp*>(b), spans_, vector_sizes);
    auto block_x = block_vector(x, spans_, vector_sizes);

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
