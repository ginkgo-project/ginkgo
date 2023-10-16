/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>

#include <utility>

#include <pdexa-ext/ginkgo/core/base/block_operator.hpp>


namespace gko {
namespace {

template<typename ValueType>
std::unique_ptr<BlockOperator>
block_vector_view(gko::matrix::Dense<ValueType>* v, const std::vector<value_span>& spans) {
  auto exec = v->get_executor();

  span columns{0, v->get_size()[1]};

  std::vector<std::vector<std::shared_ptr<LinOp>>> blocks;

  for (const auto& rows : spans) {
    blocks.emplace_back(1, v->create_submatrix(rows, columns));
  }

  return BlockOperator::create(v->get_executor(), std::move(blocks));
}

template<typename Fn>
auto dispatch_dense(Fn&& fn, LinOp* v) {
  if (auto p = dynamic_cast<matrix::Dense<float>*>(v)) {
    return fn(p);
  } else if (auto p = dynamic_cast<matrix::Dense<double>*>(v)) {
    return fn(p);
  } else if (auto p = dynamic_cast<matrix::Dense<std::complex<float>>*>(v)) {
    return fn(p);
  } else if (auto p = dynamic_cast<matrix::Dense<std::complex<double>>*>(v)) {
    return fn(p);
  } else {
    GKO_NOT_IMPLEMENTED;
  }
}

std::unique_ptr<BlockOperator>
block_vector(LinOp* v, const std::vector<value_span>& blocks) {
  return dispatch_dense([&blocks](auto* dense) {
                          return block_vector_view(dense, blocks);
                        },
                        v);
}

std::unique_ptr<const BlockOperator>
block_vector(const LinOp* v, const std::vector<value_span>& blocks) {
  auto non_const = block_vector(const_cast<LinOp*>(v), blocks);
  return std::unique_ptr<const BlockOperator>{non_const.release()};
}

const LinOp* find_non_zero_in_row(const std::vector<std::vector<std::shared_ptr<LinOp>>>& blocks, size_type row) {
  auto it = std::find_if(blocks[row].begin(), blocks[row].end(), [](const auto& b) { return b.get() != nullptr; });
  GKO_ASSERT(it != blocks[row].end());
  return it->get();
}

const LinOp* find_non_zero_in_col(const std::vector<std::vector<std::shared_ptr<LinOp>>>& blocks, size_type col) {
  auto it = std::find_if(blocks.begin(), blocks.end(), [col](const auto& b) { return b[col].get() != nullptr; });
  GKO_ASSERT(it != blocks.end());
  return it->at(col).get();
}

void validate_blocks(const std::vector<std::vector<std::shared_ptr<LinOp>>>& blocks) {
  GKO_ASSERT(blocks.empty() || !blocks.front().empty());
  // all rows have same number of columns
  for (size_type row = 0; row < blocks.size(); ++row) {
    GKO_ASSERT_EQ(blocks[row].size(), blocks.front().size());
  }
  // within each row and each column the blocks have the same number of rows and columns respectively
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

template<typename Fn>
std::vector<value_span>
compute_local_spans(size_type num_blocks, const std::vector<std::vector<std::shared_ptr<LinOp>>>& blocks,
                    Fn&& get_size) {
  validate_blocks(blocks);
  std::vector<value_span> local_spans;
  size_type offset = 0;
  for (size_type i = 0; i < num_blocks; ++i) {
    auto local_size = get_size(i);
    local_spans.emplace_back(offset, offset + local_size);
    offset += local_size;
  }
  return local_spans;
}

dim<2> compute_global_size(const std::vector<std::vector<std::shared_ptr<LinOp>>>& blocks) {
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


std::unique_ptr<BlockOperator> BlockOperator::create(std::shared_ptr<const Executor> exec) {
  return std::unique_ptr<BlockOperator>(new BlockOperator(std::move(exec)));
}

std::unique_ptr<BlockOperator> BlockOperator::create(std::shared_ptr<const Executor> exec,
                                                     std::vector<std::vector<std::shared_ptr<LinOp>>> blocks) {
  return std::unique_ptr<BlockOperator>(new BlockOperator(std::move(exec), std::move(blocks)));
}

BlockOperator::BlockOperator(std::shared_ptr<const Executor> exec)
    : EnableLinOp<BlockOperator>(std::move(exec)) {}

BlockOperator::BlockOperator(std::shared_ptr<const Executor> exec,
                             std::vector<std::vector<std::shared_ptr<LinOp>>> blocks)
    : EnableLinOp<BlockOperator>(exec, compute_global_size(blocks))
    , block_size_(blocks.empty() ? dim<2>{} : dim<2>(blocks.size(), blocks.front().size()))
    , row_spans_(compute_local_spans(block_size_[0],
                                     blocks,
                                     [&](auto i) { return find_non_zero_in_row(blocks, i)->get_size()[0]; }))
    , col_spans_(compute_local_spans(block_size_[1],
                                     blocks,
                                     [&](auto i) { return find_non_zero_in_col(blocks, i)->get_size()[1]; })) {
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

void validate_application_block_parameters(ptr_param<const BlockOperator> op, ptr_param<const BlockOperator> b,
                                           ptr_param<const BlockOperator> x) {
  dim<2> single_col{0, 1};
  GKO_ASSERT_CONFORMANT(op->get_block_size(), b->get_block_size());
  GKO_ASSERT_EQUAL_ROWS(op->get_block_size(), x->get_block_size());
  GKO_ASSERT_EQUAL_COLS(single_col, b->get_block_size());
  GKO_ASSERT_EQUAL_COLS(single_col, x->get_block_size());
}

void BlockOperator::apply_impl(const LinOp* b, LinOp* x) const {
  auto block_b = block_vector(b, col_spans_);
  auto block_x = block_vector(x, row_spans_);

  validate_application_block_parameters(this, block_b, block_x);

  auto one = gko::initialize<matrix::Dense<double>>({1}, this->get_executor());
  for (size_type row = 0; row < block_size_[0]; ++row) {
    dispatch_dense([](auto* dense) { dense->fill(0.0); }, block_x->block_at(row, 0));
    for (size_type col = 0; col < block_size_[1]; ++col) {
      if (!block_at(row, col)) {
        continue;
      }
      block_at(row, col)->apply(one.get(), block_b->block_at(col, 0), one.get(), block_x->block_at(row, 0));
    }
  }
}

void BlockOperator::apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const {

  auto block_b = block_vector(b, col_spans_);
  auto block_x = block_vector(x, row_spans_);

  validate_application_block_parameters(this, block_b, block_x);

  auto one = gko::initialize<matrix::Dense<double>>({1}, this->get_executor());
  for (size_type row = 0; row < block_size_[0]; ++row) {
    dispatch_dense([beta](auto* dense) { dense->scale(beta); }, block_x->block_at(row, 0));
    for (size_type col = 0; col < block_size_[1]; ++col) {
      if (!block_at(row, col)) {
        continue;
      }
      block_at(row, col)->apply(alpha, block_b->block_at(col, 0), one.get(), block_x->block_at(row, 0));
    }
  }
}

}  // namespace gko
