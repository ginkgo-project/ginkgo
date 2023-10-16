// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_BLOCK_MATRIX_HPP
#define GINKGO_BLOCK_MATRIX_HPP

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>


namespace gko {
namespace detail {


/**
 * Carbon copy og gko::span, except that the members are not const.
 * Necessary since the normal gko::span is not copy/movable.
 */
struct value_span {
    constexpr value_span(size_type point) noexcept
        : value_span{point, point + 1}
    {}

    constexpr value_span(size_type begin, size_type end) noexcept
        : begin{begin}, end{end}
    {}

    constexpr operator span() const { return {begin, end}; }

    constexpr value_span(const span& s) noexcept : value_span(s.begin, s.end) {}

    constexpr bool is_valid() const { return begin <= end; }

    constexpr size_type length() const { return end - begin; }

    size_type begin;
    size_type end;
};


}  // namespace detail


/**
 * A BlockOperator represents a linear operator that is partitioned into
 * multiple blocks.
 *
 * For example, a BlockOperator can be used to define the operator:
 * ```
 * | A B |
 * | C D |
 * ```
 * where `A, B, C, D` itself are matrices of compatible size. This can be
 * created with:
 * ```c++
 * std::shared_ptr<LinOp> A = ...;
 * std::shared_ptr<LinOp> B = ...;
 * std::shared_ptr<LinOp> C = ...;
 * std::shared_ptr<LinOp> D = ...;
 * auto bop = BlockOperator::create(exec, {{A, B}, {C, D}});
 * ```
 * The requirements on the individual blocks passed to the create method are:
 * - In each block-row, all blocks have the same number of rows
 * - In each block-column, all blocks have the same number of columns
 * - Each block-row must have the same number of blocks
 * It is possible to set blocks to zero, by passing in a nullptr. But every
 * block-row and block-column must contain at least one non-nullptr block.
 *
 * The constructor will store all passed in blocks on the same executor as the
 * BlockOperator, which will requires copying any block that is associated with
 * a different executor.
 */
class BlockOperator final : public EnableLinOp<BlockOperator> {
    friend class EnablePolymorphicObject<BlockOperator, LinOp>;

public:
    /**
     * Get the block dimension of this, i.e. the number of blocks per row and
     * column.
     *
     * @return  The block dimension of this.
     */
    dim<2> get_block_size() const { return block_size_; }

    /**
     * Const access to a specific block.
     *
     * @param i  block row.
     * @param j  block column.
     *
     * @return  the block stored at (i, j).
     */
    const LinOp* block_at(size_type i, size_type j) const
    {
        GKO_ENSURE_IN_DIMENSION_BOUNDS(i, j, block_size_);
        return blocks_[i * block_size_[1] + j].get();
    }

    /**
     * Create empty BlockOperator.
     *
     * @param exec  the executor of this.
     *
     * @return  empty BlockOperator.
     */
    static std::unique_ptr<BlockOperator> create(
        std::shared_ptr<const Executor> exec);

    /**
     * Create BlockOperator from the given blocks.
     *
     * @param exec  the executor of this.
     * @param blocks  the blocks of this operator. The blocks will be used in a
     *                row-major form.
     *
     * @return  BlockOperator with the given blocks.
     */
    static std::unique_ptr<BlockOperator> create(
        std::shared_ptr<const Executor> exec,
        std::vector<std::vector<std::shared_ptr<const LinOp>>> blocks);

    /**
     * Copy constructs a BlockOperator. The executor of other is used for this.
     * The blocks of other are deep-copied into this, using clone.
     */
    BlockOperator(const BlockOperator& other);

    /**
     * Move constructs a BlockOperator. The executor of other is used for this.
     * All remaining data of other is moved into this. After this operation,
     * other will be empty.
     */
    BlockOperator(BlockOperator&& other) noexcept;

    /**
     * Copy assigns a BlockOperator. The executor of this is not modified.
     * The blocks of other are deep-copied into this, using clone.
     */
    BlockOperator& operator=(const BlockOperator& other);

    /**
     * Move assigns a BlockOperator. The executor of this is not modified.
     * All data of other (except its executor) is moved into this. If the
     * executor of this and other differ, the blocks will be copied to the
     * executor of this. After this operation, other will be empty.
     */
    BlockOperator& operator=(BlockOperator&& other);

private:
    explicit BlockOperator(std::shared_ptr<const Executor> exec);

    BlockOperator(
        std::shared_ptr<const Executor> exec,
        std::vector<std::vector<std::shared_ptr<const LinOp>>> blocks);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    dim<2> block_size_;
    std::vector<detail::value_span> row_spans_;
    std::vector<detail::value_span> col_spans_;
    std::vector<std::shared_ptr<const LinOp>> blocks_;
    /**
     * @internal Using a fixed precision here may lead to temporary
     *           conversions, since there is no value_type information
     *           of the stored block available.
     *
     * @todo fix when better value_type information is available
     */
    detail::DenseCache<default_precision> one_;
};


}  // namespace gko

#endif  // GINKGO_BLOCK_MATRIX_HPP
