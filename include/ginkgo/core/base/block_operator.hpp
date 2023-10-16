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

#ifndef GINKGO_BLOCK_MATRIX_HPP
#define GINKGO_BLOCK_MATRIX_HPP

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>


namespace gko {

/**
 * Necessary since the normal gko::span is not copy/movable
 */
struct value_span {
  /**
   * Creates a span representing a point `point`.
   *
   * The `begin` of this span is set to `point`, and the `end` to `point + 1`.
   *
   * @param point  the point which the span represents
   */
  constexpr value_span(size_type point) noexcept
      : value_span{point, point + 1} {}

  /**
   * Creates a span.
   *
   * @param begin  the beginning of the span
   * @param end  the end of the span
   */
  constexpr value_span(size_type begin, size_type end) noexcept
      : begin{begin}, end{end} {}

  constexpr operator span() const { return {begin, end}; }

  constexpr value_span(const span& s) noexcept
      : value_span(s.begin, s.end) {}

  /**
   * Checks if a span is valid.
   *
   * @return true if and only if `this->begin <= this->end`
   */
  constexpr bool is_valid() const { return begin <= end; }

  /**
   * Returns the length of a span.
   *
   * @return `this->end - this->begin`
   */
  constexpr size_type length() const { return end - begin; }

  /**
   * Beginning of the span.
   */
  size_type begin;

  /**
   * End of the span.
   */
  size_type end;
};

class BlockOperator final : public EnableLinOp<BlockOperator> {
  friend class EnablePolymorphicObject<BlockOperator, LinOp>;

public:
  dim<2> get_block_size() const { return block_size_; }

  const LinOp* block_at(size_type i, size_type j) const {
    return blocks_[i * block_size_[1] + j].get();
  }

  LinOp* block_at(size_type i, size_type j) {
    return blocks_[i * block_size_[1] + j].get();
  }

  static std::unique_ptr<BlockOperator> create(std::shared_ptr<const Executor> exec);

  static std::unique_ptr<BlockOperator> create(std::shared_ptr<const Executor> exec,
                                               std::vector<std::vector<std::shared_ptr<LinOp>>> blocks);

private:
  explicit BlockOperator(std::shared_ptr<const Executor> exec);

  BlockOperator(std::shared_ptr<const Executor> exec,
                std::vector<std::vector<std::shared_ptr<LinOp>>> blocks);

  void apply_impl(const LinOp* b, LinOp* x) const override;

  void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const override;

  dim<2> block_size_;
  std::vector<value_span> row_spans_;
  std::vector<value_span> col_spans_;
  std::vector<std::shared_ptr<LinOp>> blocks_;
};



}  // namespace gko

#endif  // GINKGO_BLOCK_MATRIX_HPP
