// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/block_operator.hpp>


#include <utility>


#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace {


template <typename Fn>
auto dispatch_dense(Fn&& fn, LinOp* v)
{
    return run<matrix::Dense<float>*, matrix::Dense<double>*,
               matrix::Dense<std::complex<float>>*,
               matrix::Dense<std::complex<double>>*>(v, std::forward<Fn>(fn));
}


template <typename LinOpType>
auto create_vector_blocks(LinOpType* vector,
                          const std::vector<detail::value_span>& spans)
{
    return [=](size_type i) {
        return dispatch_dense(
            [&](auto* dense) -> std::unique_ptr<LinOpType> {
                GKO_ENSURE_IN_BOUNDS(i, spans.size());
                return dense->create_submatrix(spans[i],
                                               {0, dense->get_size()[1]});
            },
            const_cast<LinOp*>(vector));
    };
}


const LinOp* find_non_zero_in_row(
    const std::vector<std::vector<std::shared_ptr<const LinOp>>>& blocks,
    size_type row)
{
    auto it = std::find_if(blocks[row].begin(), blocks[row].end(),
                           [](const auto& b) { return b.get() != nullptr; });
    GKO_THROW_IF_INVALID(it != blocks[row].end(),
                         "Encountered row with only nullptrs.");
    return it->get();
}


const LinOp* find_non_zero_in_col(
    const std::vector<std::vector<std::shared_ptr<const LinOp>>>& blocks,
    size_type col)
{
    auto it = std::find_if(blocks.begin(), blocks.end(), [col](const auto& b) {
        return b[col].get() != nullptr;
    });
    GKO_THROW_IF_INVALID(it != blocks.end(),
                         "Encountered columns with only nullptrs.");
    return it->at(col).get();
}


void validate_blocks(
    const std::vector<std::vector<std::shared_ptr<const LinOp>>>& blocks)
{
    GKO_THROW_IF_INVALID(blocks.empty() || !blocks.front().empty(),
                         "Blocks must either be empty, or a 2D std::vector.");
    // all rows have same number of columns
    for (size_type row = 1; row < blocks.size(); ++row) {
        GKO_ASSERT_EQ(blocks[row].size(), blocks.front().size());
    }
    // within each row and each column the blocks have the same number of rows
    // and columns respectively
    for (size_type row = 0; row < blocks.size(); ++row) {
        auto non_zero_row = find_non_zero_in_row(blocks, row);
        for (size_type col = 0; col < blocks.front().size(); ++col) {
            auto non_zero_col = find_non_zero_in_col(blocks, col);
            if (blocks[row][col]) {
                GKO_ASSERT_EQUAL_COLS(blocks[row][col], non_zero_col);
                GKO_ASSERT_EQUAL_ROWS(blocks[row][col], non_zero_row);
            }
        }
    }
}


template <typename Fn>
std::vector<detail::value_span> compute_local_spans(
    size_type num_blocks,
    const std::vector<std::vector<std::shared_ptr<const LinOp>>>& blocks,
    Fn&& get_size)
{
    validate_blocks(blocks);
    std::vector<detail::value_span> local_spans;
    size_type offset = 0;
    for (size_type i = 0; i < num_blocks; ++i) {
        auto local_size = get_size(i);
        local_spans.emplace_back(offset, offset + local_size);
        offset += local_size;
    }
    return local_spans;
}


dim<2> compute_global_size(
    const std::vector<std::vector<std::shared_ptr<const LinOp>>>& blocks)
{
    validate_blocks(blocks);
    if (blocks.empty()) {
        return {};
    }
    size_type num_rows = 0;
    for (size_type row = 0; row < blocks.size(); ++row) {
        num_rows += find_non_zero_in_row(blocks, row)->get_size()[0];
    }
    size_type num_cols = 0;
    for (size_type col = 0; col < blocks.front().size(); ++col) {
        num_cols += find_non_zero_in_col(blocks, col)->get_size()[1];
    }
    return {num_rows, num_cols};
}


}  // namespace


std::unique_ptr<BlockOperator> BlockOperator::create(
    std::shared_ptr<const Executor> exec)
{
    return std::unique_ptr<BlockOperator>(new BlockOperator(std::move(exec)));
}


std::unique_ptr<BlockOperator> BlockOperator::create(
    std::shared_ptr<const Executor> exec,
    std::vector<std::vector<std::shared_ptr<const LinOp>>> blocks)
{
    return std::unique_ptr<BlockOperator>(
        new BlockOperator(std::move(exec), std::move(blocks)));
}


BlockOperator::BlockOperator(std::shared_ptr<const Executor> exec)
    : EnableLinOp<BlockOperator>(std::move(exec))
{}


BlockOperator::BlockOperator(
    std::shared_ptr<const Executor> exec,
    std::vector<std::vector<std::shared_ptr<const LinOp>>> blocks)
    : EnableLinOp<BlockOperator>(exec, compute_global_size(blocks)),
      block_size_(blocks.empty()
                      ? dim<2>{}
                      : dim<2>(blocks.size(), blocks.front().size())),
      row_spans_(compute_local_spans(
          block_size_[0], blocks,
          [&](auto i) {
              return find_non_zero_in_row(blocks, i)->get_size()[0];
          })),
      col_spans_(compute_local_spans(block_size_[1], blocks, [&](auto i) {
          return find_non_zero_in_col(blocks, i)->get_size()[1];
      }))
{
    for (auto& row : blocks) {
        for (auto& block : row) {
            if (block && block->get_executor() != exec) {
                blocks_.push_back(gko::clone(exec, block));
            } else {
                blocks_.push_back(std::move(block));
            }
        }
    }
}


void init_one_cache(std::shared_ptr<const Executor> exec,
                    const detail::DenseCache<default_precision>& one_cache)
{
    if (one_cache.get() == nullptr) {
        one_cache.init(std::move(exec), {1, 1});
        one_cache->fill(one<default_precision>());
    }
}


void BlockOperator::apply_impl(const LinOp* b, LinOp* x) const
{
    auto block_b = create_vector_blocks(b, col_spans_);
    auto block_x = create_vector_blocks(x, row_spans_);

    init_one_cache(this->get_executor(), one_);
    for (size_type row = 0; row < block_size_[0]; ++row) {
        bool first_in_row = true;
        for (size_type col = 0; col < block_size_[1]; ++col) {
            if (!block_at(row, col)) {
                continue;
            }
            if (first_in_row) {
                block_at(row, col)->apply(block_b(col), block_x(row));
                first_in_row = false;
            } else {
                block_at(row, col)->apply(one_.get(), block_b(col), one_.get(),
                                          block_x(row));
            }
        }
    }
}


void BlockOperator::apply_impl(const LinOp* alpha, const LinOp* b,
                               const LinOp* beta, LinOp* x) const
{
    auto block_b = create_vector_blocks(b, col_spans_);
    auto block_x = create_vector_blocks(x, row_spans_);

    init_one_cache(this->get_executor(), one_);
    for (size_type row = 0; row < block_size_[0]; ++row) {
        bool first_in_row = true;
        for (size_type col = 0; col < block_size_[1]; ++col) {
            if (!block_at(row, col)) {
                continue;
            }
            if (first_in_row) {
                block_at(row, col)->apply(alpha, block_b(col), beta,
                                          block_x(row));
                first_in_row = false;
            } else {
                block_at(row, col)->apply(alpha, block_b(col), one_.get(),
                                          block_x(row));
            }
        }
    }
}


BlockOperator::BlockOperator(const BlockOperator& other)
    : EnableLinOp<BlockOperator>(other.get_executor())
{
    *this = other;
}


BlockOperator::BlockOperator(BlockOperator&& other) noexcept
    : EnableLinOp<BlockOperator>(other.get_executor())
{
    *this = std::move(other);
}


BlockOperator& BlockOperator::operator=(const BlockOperator& other)
{
    if (this != &other) {
        auto exec = this->get_executor();

        set_size(other.get_size());
        block_size_ = other.get_block_size();
        col_spans_ = other.col_spans_;
        row_spans_ = other.row_spans_;
        blocks_.clear();
        for (const auto& block : other.blocks_) {
            blocks_.emplace_back(block == nullptr ? nullptr
                                                  : gko::clone(exec, block));
        }
    }
    return *this;
}


BlockOperator& BlockOperator::operator=(BlockOperator&& other)
{
    if (this != &other) {
        auto exec = this->get_executor();

        set_size(other.get_size());
        other.set_size({});

        block_size_ = std::exchange(other.block_size_, dim<2>{});
        col_spans_ = std::move(other.col_spans_);
        row_spans_ = std::move(other.row_spans_);
        blocks_ = std::move(other.blocks_);
        if (exec != other.get_executor()) {
            for (auto& block : blocks_) {
                if (block != nullptr) {
                    block = gko::clone(exec, block);
                }
            }
        }
    }
    return *this;
}


}  // namespace gko
