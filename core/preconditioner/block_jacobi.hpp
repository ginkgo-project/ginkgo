/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_HPP_
#define GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_HPP_


#include "core/base/array.hpp"
#include "core/base/convertible.hpp"
#include "core/base/lin_op.hpp"


namespace gko {
namespace preconditioner {


template <typename, typename>
class BlockJacobiFactory;


/**
 * A block-Jacobi preconditioner is a block-diagonal linear operator, obtained
 * by inverting the diagonal blocks of another operator.
 *
 * The BlockJacobi class implements this inversion of the diagonal blocks using
 * Gauss-Jordan elimination with column pivoting, and stores the inverse
 * explicitly in a customized format.
 *
 * If the diagonal blocks of the matrix are not explicitly set by the user, the
 * implementation will try to automatically detect the blocks by first finding
 * the natural blocks of the matrix, and then applying the supervariable
 * agglomeration procedure on these blocks.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  integral type used to store pointers to the start of each
 *                    block
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class BlockJacobi : public BasicLinOp<BlockJacobi<ValueType, IndexType>> {
    friend class BasicLinOp<BlockJacobi>;
    friend class BlockJacobiFactory<ValueType, IndexType>;

public:
    using BasicLinOp<BlockJacobi>::convert_to;
    using BasicLinOp<BlockJacobi>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    /**
     * Returns the number of blocks of the operator.
     *
     * @return the number of blocks of the operator
     */
    size_type get_num_blocks() const noexcept { return num_blocks_; }

    /**
     * Returns the maximum allowed block size of each block.
     *
     * @return the maximum allowed block size of each block
     */
    uint32 get_max_block_size() const noexcept { return max_block_size_; }

    /**
     * Returns the array of pointers to the start of diagonal blocks.
     *
     * @return the array of pointers to the start of diagonal blocks
     */
    IndexType *get_block_pointers() noexcept
    {
        return block_pointers_.get_data();
    }

    /**
     * Returns the array of pointers to the start of diagonal blocks.
     *
     * @return the array of pointers to the start of diagonal blocks
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const IndexType *get_const_block_pointers() const noexcept
    {
        return block_pointers_.get_const_data();
    }

    /**
     * Returns the padding used for storing the blocks.
     *
     * @return the padding used for storing the blocks
     */
    size_type get_padding() const noexcept { return max_block_size_; }

    /**
     * Returns the pointer to the memory used for storing the block data.
     *
     * Element (`i`, `j`) of block `b` is stored in position
     * `(get_block_pointers()[b] + i) * padding + j` of the array.
     *
     * @return the pointer to the memory used for storing the block data
     */
    ValueType *get_blocks() noexcept { return blocks_.get_data(); }

    /**
     * Returns the pointer to the memory used for storing the block data.
     *
     * Element (`i`, `j`) of block `b` is stored in position
     * `(get_block_pointers()[b] + i) * padding + j` of the array.
     *
     * @return the pointer to the memory used for storing the block data
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const ValueType *get_const_blocks() const noexcept
    {
        return blocks_.get_const_data();
    }

protected:
    explicit BlockJacobi(std::shared_ptr<const Executor> exec)
        : BasicLinOp<BlockJacobi>(exec, 0, 0, 0),
          block_pointers_(exec),
          blocks_(exec)
    {}

    BlockJacobi(std::shared_ptr<const Executor> exec,
                const LinOp *system_matrix, uint32 max_block_size,
                const Array<IndexType> &block_pointers)
        : BasicLinOp<BlockJacobi>(exec, 0, 0, 0),
          num_blocks_(block_pointers.get_num_elems() - 1),
          max_block_size_(max_block_size),
          block_pointers_(block_pointers),
          blocks_(exec)
    {
        block_pointers_.set_executor(this->get_executor());
        this->generate(system_matrix);
    }

    static std::unique_ptr<BlockJacobi> create(
        std::shared_ptr<const Executor> exec, const LinOp *system_matrix,
        uint32 max_block_size, const Array<IndexType> &block_pointers)
    {
        return std::unique_ptr<BlockJacobi>(new BlockJacobi(
            std::move(exec), system_matrix, max_block_size, block_pointers));
    }

    void generate(const LinOp *system_matrix);

private:
    size_type num_blocks_{};
    uint32 max_block_size_{};
    Array<IndexType> block_pointers_;
    Array<ValueType> blocks_;
};


/**
 * This factory is used to create a block-Jacobi preconditioner from the
 * operator A, by inverting the diagonal blocks of the operator.
 *
 * The factory generates a BlockJacobi object witch stores the preconditioner.
 * For more details see the documentation for BlockJacobi.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  integral type used to store pointers to the start of each
 *                    block
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class BlockJacobiFactory : public LinOpFactory {
public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Creates a new block-Jacobi factory.
     *
     * @param exec  the executor where the block-Jacobi preconditioner will be
     *              stored
     * @param max_block_size  maximum allowed size of diagonal blocks
     *                        (a tighter bound on the size can positively affect
     *                        the performance of the preconditioner generation
     *                        and application)
     *
     * @return a unique pointer to the newly created factory
     */
    static std::unique_ptr<BlockJacobiFactory> create(
        std::shared_ptr<const Executor> exec, uint32 max_block_size)
    {
        return std::unique_ptr<BlockJacobiFactory>(
            new BlockJacobiFactory(std::move(exec), max_block_size));
    }

    std::unique_ptr<LinOp> generate(
        std::shared_ptr<const LinOp> base) const override;

    /**
     * Returns the maximum allowed size of diagonal blocks.
     *
     * @return the maximum allowed size of diagonal blocks
     */
    uint32 get_max_block_size() const noexcept { return max_block_size_; }

    /**
     * Sets the array of block pointer which will be used to determine the
     * blocks for the matrix.
     *
     * @param  block_pointers  The array of block pointer, the value at position
     *                         `i` of the array should be set to the row
     *                         where the `i`-th block starts. In addition, the
     *                         last value of the array should be set to the
     *                         number of rows of the matrix. Thus, the total
     *                         number of blocks is
     *                         `block_pointers.get_num_elems() - 1`.
     */
    void set_block_pointers(const Array<IndexType> &block_pointers)
    {
        block_pointers_ = block_pointers;
    }

    /**
     * @copydoc set_block_pointers(const Array<IndexType> &)
     */
    void set_block_pointers(Array<IndexType> &&block_pointers)
    {
        block_pointers_ = std::move(block_pointers);
    }

    /**
     * Returns a reference to the array of block pointers
     *
     * @return the array of block pointers
     */
    const Array<IndexType> &get_block_pointers() const noexcept
    {
        return block_pointers_;
    }

protected:
    BlockJacobiFactory(std::shared_ptr<const Executor> exec,
                       uint32 max_block_size)
        : LinOpFactory(exec),
          max_block_size_(max_block_size),
          block_pointers_(exec)
    {}

private:
    uint32 max_block_size_;
    Array<IndexType> block_pointers_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_HPP_
