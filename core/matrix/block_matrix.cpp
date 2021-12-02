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

#include <ginkgo/core/matrix/block_matrix.hpp>
#include <ginkgo/core/matrix/dense.hpp>

namespace gko {
namespace matrix {

namespace {


template <typename DenseType>
std::unique_ptr<std::conditional_t<std::is_const_v<DenseType>,
                                   const BlockMatrix, BlockMatrix>>
dense_to_block(DenseType* v, const std::vector<size_type>& block_sizes)
{
    using value_type = typename DenseType::value_type;
    auto v_without_const = const_cast<std::remove_const_t<DenseType>*>(v);
    std::vector<std::vector<std::shared_ptr<LinOp>>> blocks(block_sizes.size());
    std::vector<int32> block_offsets(block_sizes.size() + 1, 0);
    std::partial_sum(begin(block_sizes), end(block_sizes),
                     begin(block_offsets) + 1);
    for (size_t i = 0; i < block_sizes.size(); ++i) {
        gko::span rows(block_offsets[i], block_offsets[i + 1]);
        blocks[i] = std::vector<std::shared_ptr<LinOp>>{gko::share(
            v_without_const->create_submatrix(rows, v->get_size()[1] - 1))};
    }
    return BlockMatrix::create(v->get_executor(), v->get_size(), blocks);
}


template <bool is_const, typename ValueType>
using DenseType =
    std::conditional_t<is_const, const gko::matrix::Dense<ValueType>,
                       gko::matrix::Dense<ValueType>>;
template <bool is_const>
using BlockType = std::conditional_t<is_const, const gko::matrix::BlockMatrix,
                                     gko::matrix::BlockMatrix>;


template <typename LinOpType>
std::unique_ptr<std::conditional_t<std::is_const_v<LinOpType>,
                                   const BlockMatrix, BlockMatrix>,
                std::function<void(BlockType<std::is_const_v<LinOpType>>*)>>
as_block_vector(LinOpType* v, const std::vector<size_type>& block_sizes)
{
    constexpr bool is_const = std::is_const_v<LinOpType>;
    if (auto block_dense = dynamic_cast<BlockType<is_const>*>(v)) {
        return std::unique_ptr<
            BlockType<is_const>,
            std::function<void(BlockType<std::is_const_v<LinOpType>>*)>>(
            block_dense, [](BlockType<is_const>*) {});
    } else if (auto dense = dynamic_cast<DenseType<is_const, float>*>(v)) {
        return dense_to_block(dense, block_sizes);
    } else if (auto dense = dynamic_cast<DenseType<is_const, double>*>(v)) {
        return dense_to_block(dense, block_sizes);
    } else if (auto dense =
                   dynamic_cast<DenseType<is_const, std::complex<float>>*>(v)) {
        return dense_to_block(dense, block_sizes);
    } else if (auto dense =
                   dynamic_cast<DenseType<is_const, std::complex<double>>*>(
                       v)) {
        return dense_to_block(dense, block_sizes);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


}  // namespace


void BlockMatrix::apply_impl(const LinOp* b, LinOp* x) const
{
    auto block_b = as_block_vector(b, size_per_block_);
    auto block_x = as_block_vector(x, size_per_block_);

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
    auto block_b = as_block_vector(b, size_per_block_);
    auto block_x = as_block_vector(x, size_per_block_);

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
